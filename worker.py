import os
import gc
import a3c
import gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
from scipy.signal import lfilter

DISCOUNT = 0.993
DEATH_PENALTY = -200
LOSE_GAME_PENALTY = -500
WIN_GAME_REWARD = 500
REWARD_SCALING = 1 / 200
is_chief_worker = False


class StubProgress:

    def update(self):
        pass

    def close(self):
        pass


def log(statement, *args):
    if is_chief_worker:
        if len(args) == 0:
            print(statement)
        else:
            print(statement % args)


def initialize_variables(session, network, read_queue, peer_queues):
    if is_chief_worker:
        variables, step = session.run([network.FlattenedVariableOp, network.TrainStep])
        list(filter(lambda queue: queue.put((variables, step)), peer_queues))
    else:
        variables, step = read_queue.get()
        session.run(network.AssignVariableOp, feed_dict={network.VariablePH: variables})

    if step == 0:
        session.run(network.AssignOldPolicyOp)


def run_episode(session, environment, network):
    total_reward = 0
    timesteps = 0
    observations = []
    actions = []
    oldprobabilities = []
    rewards = []
    values = []
    observation = environment.reset()
    done = False

    lives = 3
    session.run(network.ResetStateOp)
    while not done:
        observations.append(observation)
        action, value, oldprobability = session.run(network.EvaluatePolicyOp,
                                                    feed_dict={network.InputsPH: [observation]})
        action = action[0]
        value = value[0]
        oldprobability = oldprobability[0]
        observation, reward, done, info = environment.step(action)

        local_reward = reward
        if not done:
            if info['ale.lives'] < lives:
                local_reward = local_reward + DEATH_PENALTY
        else:
            if info['ale.lives'] > 0:
                local_reward = local_reward + WIN_GAME_REWARD
            else:
                local_reward = local_reward + LOSE_GAME_PENALTY

        actions.append(action)
        oldprobabilities.append(oldprobability)
        rewards.append(local_reward)
        values.append(value)
        total_reward = total_reward + reward
        timesteps = timesteps + 1
        lives = info['ale.lives']

    _, value, _ = session.run(network.EvaluatePolicyOp, feed_dict={network.InputsPH: [observation]})
    values.append(value)

    observations = np.array(observations)
    actions = np.array(actions)
    oldprobabilities = np.array(oldprobabilities)
    rewards = np.array(rewards)
    values = np.array(values)

    rewards = rewards * REWARD_SCALING
    advantages = rewards + DISCOUNT * values[1:]
    advantages = advantages - values[:-1]
    advantages = lfilter([1], [1, -DISCOUNT], x=advantages[::-1])[::-1]

    session.run(network.AccumulatorOp, feed_dict={network.InputsPH: observations,
                                                  network.RewardsPH: rewards,
                                                  network.ActionsPH: actions,
                                                  network.OldProbabilityPH: oldprobabilities,
                                                  network.AdvantagePH: advantages})

    return total_reward, timesteps


def solicit_gradients(read_queue, peer_queue, gradient, timesteps, mean_reward):
    list(filter(lambda queue: queue.put((gradient, timesteps, mean_reward)), peer_queue))
    results = list(map(lambda x: read_queue.get(), range(len(peer_queue))))

    gradients = []
    timesteps = timesteps
    for result in results:
        gradients.append(result[0])
        timesteps = timesteps + result[1]
        mean_reward = mean_reward + result[2]

    return np.array(gradients), timesteps, mean_reward / (len(peer_queue) + 1)


def start(game_id, read_queue, peer_queues, is_chief, num_episodes_per_update, device=None):
    with tf.device(device):
        global is_chief_worker
        is_chief_worker = is_chief

        worker_count = len(peer_queues) + 1
        network = a3c.A3CNetwork(64, 512, DISCOUNT, 0.2)

        memory_limit = worker_count / worker_count

        mean_reward_ph = tf.placeholder(shape=[], dtype=tf.float32)
        tf.summary.scalar("AvgReward/Episode", mean_reward_ph)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("Checkpoints")

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=memory_limit))
        with tf.train.MonitoredTrainingSession(checkpoint_dir="Checkpoints" if is_chief_worker else None,
                                               save_checkpoint_secs=None,
                                               save_checkpoint_steps=1 if is_chief_worker else None,
                                               save_summaries_secs=None,
                                               save_summaries_steps=None,
                                               config=config) as sess:
            tf.get_default_graph().finalize()
            initialize_variables(sess, network, read_queue, peer_queues)
            log("Completed initialization process\n")

            env = gym.make(game_id)

            sess.run(network.ResetAccumulatorOp)
            while True:
                ctimestamp = datetime.now()
                total_reward = 0
                timesteps = 0

                progress = StubProgress()
                if is_chief_worker:
                    progress = tqdm(total=num_episodes_per_update)

                for index in range(num_episodes_per_update):
                    result = run_episode(sess, env, network)
                    total_reward = total_reward + result[0]
                    timesteps = timesteps + result[1]
                    progress.update()
                progress.close()

                stimestamp = datetime.now()

                mean_reward = total_reward / num_episodes_per_update
                gradients, timesteps, mean_reward = solicit_gradients(read_queue, peer_queues,
                                                                      sess.run(network.FlattenedAccumulatorOp),
                                                                      timesteps,
                                                                      mean_reward)
                summary, step, _ = sess.run([summary_op, network.TrainStep, network.ApplyGradientOp],
                                            feed_dict={network.TotalTimestepsPH: timesteps,
                                                       network.GradientPH: gradients,
                                                       mean_reward_ph: mean_reward})
                sess.run(network.ResetAccumulatorOp)

                log("Completed %i episodes in %i steps, time taken %i seconds", (index + 1) * worker_count, timesteps,
                    (datetime.now() - ctimestamp).total_seconds())
                log("Solicited and applied gradient from %i other peers, time taken %i seconds", len(peer_queues),
                    (datetime.now() - stimestamp).total_seconds())
                log("Update completed with mean reward of %.2f\n", mean_reward)
                if is_chief_worker:
                    summary_writer.add_summary(summary, step)

                gc.collect()
