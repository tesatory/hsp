import time
from utils import *
import torch
import torch.multiprocessing as mp

class ThreadedWorker(mp.Process):
    def __init__(self, id, trainer_maker, comm, *args, **kwargs):
        self.id = id
        super(ThreadedWorker, self).__init__()
        self.trainer = trainer_maker()
        self.comm = comm

    def run(self):
        while True:
            task = self.comm.recv()
            if task == 'quit':
                return
            elif task == 'run_batch':
                batch, stat = self.trainer.run_batch()
                self.trainer.optimizer.zero_grad()
                s = self.trainer.compute_grad(batch)
                merge_stat(s, stat)
                self.comm.send(stat)
            elif task == 'send_grads':
                grads = [p._grad.data for p in self.trainer.params]
                self.comm.send(grads)


class ThreadedTrainer(object):
    def __init__(self, args, trainer_maker):
        self.args = args
        self.comms = []
        self.trainer = trainer_maker()
        # itself will do the same job as workers
        self.nworkers = args.nthreads - 1
        for i in range(self.nworkers):
            comm, comm_remote = mp.Pipe()
            self.comms.append(comm)
            worker = ThreadedWorker(i, trainer_maker, comm_remote)
            worker.start()
        self.grads = None
        self.worker_grads = None

    def quit(self):
        for comm in self.comms:
            comm.send('quit')

    def obtain_grad_pointers(self):
        # only need perform this once
        if self.grads is None:
            self.grads = [p._grad.data for p in self.trainer.params]

        if self.worker_grads is None:
            self.worker_grads = []
            for comm in self.comms:
                comm.send('send_grads')
                self.worker_grads.append(comm.recv())

    def train_batch(self):
        # run workers in parallel
        for comm in self.comms:
            comm.send('run_batch')

        # run its own trainer
        batch, stat = self.trainer.run_batch()
        self.trainer.optimizer.zero_grad()
        s = self.trainer.compute_grad(batch)
        merge_stat(s, stat)

        # check if workers are finished
        for comm in self.comms:
            s = comm.recv()
            merge_stat(s, stat)

        # add gradients of workers
        self.obtain_grad_pointers()
        for i in range(len(self.grads)):
            for g in self.worker_grads:
                self.grads[i] += g[i]
            self.grads[i] /= stat['num_steps']

        if not self.args.freeze:
            self.trainer.optimizer.step()
        return stat

    def state_dict(self):
        return self.trainer.state_dict()

    def load_state_dict(self, state):
        self.trainer.load_state_dict(state)
