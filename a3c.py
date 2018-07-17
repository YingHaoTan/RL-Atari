import tensorflow as tf
import tensorflow.contrib.cudnn_rnn as rnn
import tensorflow.contrib.layers as layers
import tensorflow.contrib.framework as framework
from tensorflow import initializers as init
from functools import reduce


class A3CNetwork:

    def __init__(self, cnn_batch_size, fc_batch_size, discount, proximal_epsilon, scope="A3CNetwork"):
        self.ConvolutionBatchSize = cnn_batch_size
        self.FullyConnectedBatchSize = fc_batch_size
        self.Discount = discount
        self.ProximalEpsilon = proximal_epsilon
        self.Scope = scope
        self.TrainStep = tf.train.create_global_step()
        self.InputsPH = tf.placeholder(shape=[None, 210, 160, 3], dtype=tf.uint8)
        self.RewardsPH = tf.placeholder(shape=[None], dtype=tf.float32)
        self.ActionsPH = tf.placeholder(shape=[None], dtype=tf.int32)
        self.OldProbabilityPH = tf.placeholder(shape=[None], dtype=tf.float32)
        self.AdvantagePH = tf.placeholder(shape=[None], dtype=tf.float32)
        self.TotalTimestepsPH = tf.placeholder(shape=[], dtype=tf.int32)
        self.InputShape = tf.shape(self.InputsPH)
        self.__c_lstm_node__ = rnn.CudnnLSTM(1, 256, name="Current/" + self.Scope)
        self.__o_lstm_node__ = rnn.CudnnLSTM(1, 256, name="Old/" + self.Scope)

        self.EvaluatePolicyOp, self.ResetStateOp = self.__build_prediction_network__()
        phs_and_ops = self.__build_training_network__()
        self.GradientPH = phs_and_ops[0]
        self.VariablePH = phs_and_ops[1]
        self.ApplyGradientOp = phs_and_ops[2]
        self.AccumulatorOp = phs_and_ops[3]
        self.FlattenedAccumulatorOp = phs_and_ops[4]
        self.AssignVariableOp = phs_and_ops[5]
        self.FlattenedVariableOp = phs_and_ops[6]
        self.ResetAccumulatorOp = phs_and_ops[7]
        self.AssignOldPolicyOp = phs_and_ops[8]

    def __build_prediction_network__(self):
        with tf.device("/cpu:0"):
            with tf.variable_scope(self.Scope):
                c_state = tf.get_variable("CState", shape=[1, 1, 256], dtype=tf.float32, initializer=init.zeros())
                h_state = tf.get_variable("HState", shape=[1, 1, 256], dtype=tf.float32, initializer=init.zeros())
                o_c_state = tf.get_variable("OCState", shape=[1, 1, 256], dtype=tf.float32, initializer=init.zeros())
                o_h_state = tf.get_variable("OHState", shape=[1, 1, 256], dtype=tf.float32, initializer=init.zeros())

        with tf.variable_scope("Current"):
            outputs = self.__map_cnn__(self.InputsPH)
            outputs, nstates = self.__c_lstm_node__(tf.reshape(outputs, [-1, 1, outputs.shape[-1]]), (c_state, h_state))
            outputs = tf.reshape(outputs, [-1, outputs.shape[-1]])
            policy = self.__map_policy__(outputs)
            value = tf.squeeze(self.__map_value_net__(outputs), 1)
            action = tf.squeeze(tf.multinomial(tf.log(policy), 1, output_dtype=tf.int32), 1)

        with tf.variable_scope("Old"):
            outputs = self.__map_cnn__(self.InputsPH)
            outputs, ostates = self.__c_lstm_node__(tf.reshape(outputs, [-1, 1, outputs.shape[-1]]),
                                                    (o_c_state, o_h_state))
            outputs = self.__map_policy__(tf.reshape(outputs, [-1, outputs.shape[-1]]))
            oprobability = tf.gather_nd(outputs,
                                        tf.stack([tf.zeros(shape=tf.shape(action), dtype=tf.int32), action], 1))

        with tf.control_dependencies([tf.assign(c_state, nstates[0]), tf.assign(h_state, nstates[1]),
                                      tf.assign(o_c_state, ostates[0]), tf.assign(o_h_state, ostates[1])]):
            action = tf.identity(action)
            value = tf.identity(value)
            oprobability = tf.identity(oprobability)

        return [action, value, oprobability], tf.group([c_state.initializer, h_state.initializer])

    def __build_training_network__(self):
        with tf.variable_scope("Current"):
            outputs = self.__build_training_cnn__(self.InputsPH)
            outputs, states = self.__c_lstm_node__(outputs)
            npolicies, nvalues = self.__build_training_final_layers__(outputs)

        variables = tf.trainable_variables("Current/" + self.Scope)
        accumulators, reset_gradients = self.__build_gradient_accumulators__()
        loss, ploss, vloss, hloss, accumulate_loss_ops, reset_loss = self.__build_loss__(npolicies, nvalues)
        gradients = tf.gradients(loss, variables)
        accumulator_op = tf.group([*[tf.assign_add(accumulators[index], tf.cast(gradients[index], tf.float64))
                                     for index in range(len(gradients))],
                                   accumulate_loss_ops])

        num_accumulators = reduce(lambda x, y: x + y, map(lambda x: x.shape.num_elements(), accumulators))
        grad_placeholder = tf.placeholder(shape=[None, num_accumulators], dtype=tf.float64)
        var_placeholder = tf.placeholder(shape=num_accumulators, dtype=tf.float32)
        flatten_accumulators = tf.concat([tf.reshape(accumulator, [-1]) for accumulator in accumulators], axis=0)
        stacked_accumulators = tf.concat([grad_placeholder, tf.reshape(flatten_accumulators, [1, -1])], axis=0)
        timestep_fp = tf.cast(self.TotalTimestepsPH, tf.float64)
        stacked_accumulators = tf.reduce_sum(stacked_accumulators, axis=0) / timestep_fp
        stacked_accumulators, gnorm = tf.clip_by_global_norm([stacked_accumulators], 1e3)
        stacked_accumulators = tf.cast(stacked_accumulators[0], tf.float32)
        flatten_variables = tf.concat([tf.reshape(variable, [-1]) for variable in variables], axis=0)

        offset = 0
        grads_and_vars = []
        assign_ops = []
        for index in range(len(variables)):
            accumulator = accumulators[index]
            var = variables[index]
            endindex = offset + accumulator.shape.num_elements()
            grads_and_vars.append((tf.reshape(stacked_accumulators[offset: endindex], accumulator.shape), var))
            assign_ops.append(tf.assign(var, tf.reshape(var_placeholder[offset: endindex], accumulator.shape)))

            offset = endindex

        old_variables = tf.trainable_variables("Old/" + self.Scope)
        assign_old_variables = [tf.assign(old_variables[index], variables[index])
                                for index in range(len(old_variables))]
        with tf.control_dependencies(assign_old_variables):
            apply_op = tf.train.RMSPropOptimizer(1e-4).apply_gradients(grads_and_vars=grads_and_vars,
                                                                       global_step=self.TrainStep)
        tf.summary.scalar("GlobalNorm", gnorm)
        with tf.name_scope("Loss"):
            ploss = ploss / timestep_fp
            vloss = vloss / timestep_fp
            hloss = hloss / timestep_fp
            loss = ploss + vloss + hloss
            tf.summary.scalar("General", loss)
            tf.summary.scalar("Policy", ploss)
            tf.summary.scalar("Value", vloss)
            tf.summary.scalar("Entropy", hloss)
        with tf.control_dependencies([apply_op, loss]):
            apply_op = tf.identity(gnorm)

        return grad_placeholder, var_placeholder, apply_op, accumulator_op, \
            flatten_accumulators, tf.group(assign_ops), flatten_variables, tf.group([reset_gradients, reset_loss]), \
            assign_old_variables

    def __build_loss__(self, npolicies, nvalues):
        with tf.device("/cpu:0"):
            with tf.variable_scope(self.Scope):
                ploss_accumulator = tf.get_variable(name="PolicyLoss", shape=[],
                                                    dtype=tf.float64, initializer=init.zeros())
                vloss_accumulator = tf.get_variable(name="ValueLoss", shape=[],
                                                    dtype=tf.float64, initializer=init.zeros())
                hloss_accumulator = tf.get_variable(name="EntropyLoss", shape=[],
                                                    dtype=tf.float64, initializer=init.zeros())

        likelihood = tf.stack([tf.range(tf.shape(self.ActionsPH)[0], dtype=tf.int32), self.ActionsPH], 1)
        likelihood = tf.gather_nd(npolicies, likelihood) / self.OldProbabilityPH
        clipped_likelihood = tf.clip_by_value(likelihood, 1.0 - self.ProximalEpsilon, 1.0 + self.ProximalEpsilon)
        policy_loss = -0.01 * tf.minimum(likelihood * self.AdvantagePH, clipped_likelihood * self.AdvantagePH)
        value_loss = 0.25 * (self.RewardsPH - nvalues) ** 2
        entropy_loss = tf.reduce_sum(npolicies * tf.log(tf.where(npolicies == 0.0,
                                                                 tf.ones(tf.shape(npolicies)),
                                                                 npolicies)),
                                     axis=1)
        loss = tf.reduce_sum(policy_loss + value_loss + entropy_loss)
        accum_ploss = tf.assign_add(ploss_accumulator, tf.cast(tf.reduce_sum(policy_loss), tf.float64))
        accum_vloss = tf.assign_add(vloss_accumulator, tf.cast(tf.reduce_sum(value_loss), tf.float64))
        accum_hloss = tf.assign_add(hloss_accumulator, tf.cast(tf.reduce_sum(entropy_loss), tf.float64))

        loss = tf.identity(loss)

        return loss, ploss_accumulator, vloss_accumulator, hloss_accumulator, [accum_ploss, accum_vloss, accum_hloss], \
            tf.group([ploss_accumulator.initializer, vloss_accumulator.initializer, hloss_accumulator.initializer])

    def __build_gradient_accumulators__(self):
        gradient_accumulators = []
        reset_gradients = []
        with tf.variable_scope(self.Scope):
            for var in tf.trainable_variables("Current/" + self.Scope):
                shape = 3409920 if var.shape.dims is None else var.shape

                accumulator = tf.get_variable(name=var.name.replace("/", "_").replace(":", "-"),
                                              shape=shape, dtype=tf.float64, initializer=init.zeros())
                gradient_accumulators.append(accumulator)
                reset_gradients.append(accumulator.initializer)

        return gradient_accumulators, tf.group(reset_gradients)

    def __build_training_cnn__(self, inputs):
        padding = self.ConvolutionBatchSize - (self.InputShape[0] % self.ConvolutionBatchSize)
        conformable_input = tf.cond(tf.equal(padding, 0),
                                    lambda: inputs,
                                    lambda: tf.concat([inputs,
                                                       tf.zeros(shape=(padding, 210, 160, 3), dtype=tf.uint8)],
                                                      axis=0))
        conformable_input = tf.reshape(conformable_input, shape=[-1, self.ConvolutionBatchSize, 210, 160, 3])
        outputs = tf.map_fn(self.__map_cnn__, conformable_input, dtype=tf.float32,
                            parallel_iterations=1, swap_memory=True)
        return tf.reshape(outputs, [-1, 1, outputs.shape[-1]])[:self.InputShape[0]]

    def __map_cnn__(self, inputs):
        with tf.variable_scope(self.Scope, reuse=tf.AUTO_REUSE):
            with framework.arg_scope([layers.conv2d, layers.max_pool2d, layers.batch_norm],
                                     data_format="NCHW"):
                with framework.arg_scope([layers.conv2d], activation_fn=tf.nn.selu,
                                         weights_initializer=tf.initializers.variance_scaling()):
                    cnn = tf.cast(tf.transpose(inputs, [0, 3, 1, 2]), tf.float32)
                    cnn = layers.conv2d(cnn, 16, 5, 2)
                    cnn = layers.max_pool2d(cnn, 2)
                    cnn = layers.conv2d(cnn, 32, 3)
                    cnn = layers.max_pool2d(cnn, 2)
                    cnn = layers.conv2d(cnn, 64, 3)
                    cnn = layers.max_pool2d(cnn, 2)
                    cnn = layers.conv2d(cnn, 128, 3)
                    cnn = layers.max_pool2d(cnn, 2)
                    cnn = layers.conv2d(cnn, 256, 3)
                    cnn = layers.max_pool2d(cnn, 2)
                    cnn = layers.conv2d(cnn, 512, 3)
                    cnn = tf.reshape(cnn, [-1, cnn.shape[1] * cnn.shape[2] * cnn.shape[3]])
        return cnn

    def __build_training_final_layers__(self, inputs):
        padding = self.FullyConnectedBatchSize - (self.InputShape[0] % self.FullyConnectedBatchSize)
        conformable_input = tf.reshape(inputs, [-1, inputs.shape[-1]])
        conformable_input = tf.cond(tf.equal(padding, 0),
                                    lambda: conformable_input,
                                    lambda: tf.concat([conformable_input,
                                                       tf.zeros(shape=(padding, inputs.shape[-1]), dtype=tf.float32)],
                                                      axis=0))
        conformable_input = tf.reshape(conformable_input, shape=[-1, self.FullyConnectedBatchSize, inputs.shape[-1]])

        policies = tf.map_fn(self.__map_policy__, conformable_input, parallel_iterations=1, swap_memory=True)
        policies = tf.reshape(policies, [-1, 6])[:self.InputShape[0]]

        values = tf.map_fn(self.__map_value_net__, conformable_input, parallel_iterations=1, swap_memory=True)
        values = tf.reshape(values, [-1, 1])[:self.InputShape[0]]

        return policies, values

    def __map_policy__(self, inputs):
        with tf.variable_scope(self.Scope, reuse=tf.AUTO_REUSE):
            policies = layers.fully_connected(inputs, 6, activation_fn=tf.nn.softmax, scope="PolicyFCNet")

        return policies

    def __map_value_net__(self, inputs):
        with tf.variable_scope(self.Scope, reuse=tf.AUTO_REUSE):
            values = layers.fully_connected(inputs, 1, activation_fn=None, scope="ValueFCNet")

        return values
