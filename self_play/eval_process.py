from threading import Thread, Lock
from time import sleep

import torch.multiprocessing as mp

from games import make_game
from net import Policy
import torch


class EvalProcess(mp.Process):

    MAX_BATCH_SIZE = 4096
    BATCH_SIZE_BUFFER = 128

    def __init__(self, game_name, game_kwargs, net_kwargs, mcts_conns, parent_conn, device='cuda:0'):
        super().__init__()
        self.game_name, self.game_kwargs = game_name, game_kwargs
        self.net_kwargs = net_kwargs
        self.mcts_conns = mcts_conns
        self.running = False
        self.device = device
        self._params = None
        self.parent_conn = parent_conn

    def _init(self):
        game = make_game(self.game_name, **self.game_kwargs)
        self.net = Policy(game.obs_space_shape, game.num_actions, **self.net_kwargs).to(self.device)
        self.net.eval()
        self.obs_buffer = torch.zeros((self.MAX_BATCH_SIZE + self.BATCH_SIZE_BUFFER, *game.obs_space_shape),
                                      dtype=torch.float32)
        self.action_mask_buffer = torch.zeros((self.MAX_BATCH_SIZE + self.BATCH_SIZE_BUFFER, game.num_actions),
                                              dtype=torch.bool)
        self.t = Thread(target=self._listen, daemon=True)
        self.lock = Lock()

    def _listen(self):
        while self.running:
            cmd, other = self.parent_conn.recv()
            if cmd == 'stop':
                self.running = False
                print("Shutting down EvalProcess")
                break
            elif cmd == 'update_params':
                with self.lock:
                    try:
                        self.net.load_state_dict(other)
                    except Exception as e:
                        print("Error while loading state dict: {}".format(e))
                print("Updated parameters")

    def run(self):
        self._init()
        self.running = True
        self.t.start()
        while self.running:
            # Aggregate batches
            pos = 0
            route_back = []
            for i, conn in enumerate(self.mcts_conns):
                if conn.poll():
                    obs_batch, action_mask_batch = conn.recv()
                    start, end = pos, pos + len(obs_batch)
                    self.obs_buffer[start:end] = obs_batch
                    self.action_mask_buffer[start:end] = action_mask_batch
                    route_back.append([i, start, end])
                    pos = end
                    if end > self.MAX_BATCH_SIZE:
                        break
            # Run evaluation
            if pos == 0:
                sleep(0.01)
                continue
            p, v = self.eval_batch(self.obs_buffer[:pos], self.action_mask_buffer[:pos])
            # Send back results
            for conn_i, start, end in route_back:
                self.mcts_conns[conn_i].send([p[start:end], v[start:end]])

            if self._params is not None:
                self.net.load_state_dict(self._params)
                self._params = None
        self.t.join()

    @torch.no_grad()
    def eval_batch(self, obs_batch, action_mask_batch):
        with self.lock:
            p, v = self.net(obs_batch.to(self.device))
        p[~action_mask_batch.to(self.device)] = -float('inf')
        p = torch.softmax(p, -1)
        return p.cpu(), v.cpu()
