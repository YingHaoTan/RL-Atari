import multiprocessing
import worker

if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count() - 1
    queues = [multiprocessing.Queue() for _ in range(cpu_count)]
    processes = [multiprocessing.Process(target=worker.start,
                                         args=("SpaceInvaders-v0",
                                               queues[index],
                                               queues[:index] + queues[index + 1:],
                                               index == 0,
                                               80))
                 for index in range(cpu_count)]

    list(filter(lambda process: process.start(), processes))
    list(filter(lambda process: process.join(), processes))
