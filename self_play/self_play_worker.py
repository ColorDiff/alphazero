from time import sleep, time

import torch.multiprocessing as mp

from self_play.eval_process import EvalProcess
from self_play.self_play_process import SelfPlayProcess, VectorizedSelfPlayProcess
from threading import Thread, Lock
import torch


class SelfPlayWorker(Thread):

    def __init__(self, game_name, game_kwargs, mcts_kwargs, net_kwargs,
                 num_self_play_worker_procs, client, device):
        super(SelfPlayWorker, self).__init__()
        self.num_self_play_worker_procs = num_self_play_worker_procs
        self.game_name = game_name
        self.game_kwargs = game_kwargs
        self.mcts_kwargs = mcts_kwargs
        self.net_kwargs = net_kwargs
        self.device = device
        self.running = False
        self.client = client

    def _init(self):
        ctx = mp.get_context('spawn')

        # Start self play workers
        self.self_play_workers = []
        conns = []
        for i in range(self.num_self_play_worker_procs):
            a, b = ctx.Pipe()
            parent, child = ctx.Pipe()
            p = VectorizedSelfPlayProcess(self.game_name, self.game_kwargs, self.mcts_kwargs,
                                          vector_len=64, eval_conn=a, parent_conn=child)
            ps = ctx.Process(target=p.run)
            ps.start()
            conns.append(b)
            self.self_play_workers.append([ps, parent])

        # Start the eval worker
        self.eval_worker_conn, child = ctx.Pipe()
        eval_worker = EvalProcess(self.game_name, self.game_kwargs, self.net_kwargs, conns, child, self.device)
        self.eval_ps = ctx.Process(target=eval_worker.run)
        self.eval_ps.start()

    @staticmethod
    def _add_self_play_data_callback(msg):
        print('Got answer from server: {}'.format(msg))

    def _get_parameters_callback(self, msg):
        print("Got parameters from server. Status: {}".format(msg['status']))
        if msg['status'] != 'OK':
            return
        state_dict = msg['data']
        if state_dict is None:
            print("Parameters returned were empty.")
            return
        self.eval_worker_conn.send(('update_params', state_dict))

    def run(self):
        self._init()
        self.running = True
        last = 0
        try:
            while self.running:
                # Gather and send self-play data
                for worker, conn in self.self_play_workers:
                    if conn.poll():
                        print('Adding self-play data')
                        obs, mask, probs, reward = conn.recv()
                        self.client.add_self_play_data(self._add_self_play_data_callback, [obs, mask, probs, reward])

                # Update parameters
                now = time()
                if now - last > 60:
                    last = now
                    self.client.get_parameters(self._get_parameters_callback)
                sleep(0.1)
        finally:
            self.stop()

    def stop(self):
        self.running = False
        for worker, conn in self.self_play_workers:
            conn.send('stop')
            worker.join()
        self.eval_worker_conn.send(('stop', None))
        self.eval_ps.join()
