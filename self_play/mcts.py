from games.interface import Game
import numpy as np
import torch
from typing import Optional, List


class _Node:

    def __init__(self, state, valid_actions, prior_probs: np.ndarray, ply, is_terminal, value_mul):
        self.ply = ply
        self.state = state
        self.is_terminal = is_terminal
        self.value_mul = value_mul
        self.actions = valid_actions

        self.n = np.zeros(len(self.actions), dtype=np.float32)
        self.w = np.zeros_like(self.n)
        self.q = np.zeros_like(self.n)
        # assert is_terminal or 0.01 > abs(sum(prior_probs) - 1), sum(prior_probs)
        self.p = prior_probs

        self.children: List[Optional[_Node]] = [None for _ in range(len(self.actions))]


class MCTS:

    def __init__(self, game: Game, eval_conn, virtual_threads=4, ply_deterministic=10,
                 c_puct=5, alpha=0.3, eta=0.25, n_vl=3):
        # Store the game and set the initial state
        self.game = game
        # Store connection to the NN evaluator
        self.eval_conn = eval_conn
        # Store parameters
        self.ply_deterministic = ply_deterministic
        self.c_puct = c_puct
        self.n_vl = n_vl
        self.alpha = alpha
        self.eta = eta
        self.n_virtual_threads = virtual_threads
        # Initialize root
        self._init_root()
        # Buffers
        self.obs_batch = np.zeros((virtual_threads, *self.game.obs_space_shape), dtype=np.float32)
        self.action_mask_batch = np.zeros((virtual_threads, self.game.num_actions), dtype=bool)
        self._add_noise()

    def _init_root(self):
        # Initializes the root with from current game state
        prior_probs, _ = self._request_evaluation(self.game.get_obs()[None, ...],
                                                  self.game.valid_action_mask()[None, ...])
        self.root = _Node(self.game.get_state(),
                          self.game.valid_actions(), prior_probs[0][self.game.valid_action_mask()],
                          0, self.game.is_terminal, 1 if self.game.is_first_player_to_move else -1)

    def select(self):
        trails = []
        self.game.set_state(self.root.state)
        # is_first_player = self.game.is_first_player_to_move
        for i in range(self.n_virtual_threads):
            node = self.root
            action_idx = self._select_action(node)  # select_action in the root node
            node.n[action_idx] += self.n_vl
            node.w[action_idx] -= self.n_vl
            node.q[action_idx] = node.w[action_idx] / node.n[action_idx]
            trail = [(node, action_idx)]
            while node.children[action_idx] is not None and not node.children[action_idx].is_terminal:
                # Until we found a leaf or terminal state
                # Traverse down the tree
                node = node.children[action_idx]
                action_idx = self._select_action(node)
                node.n[action_idx] += self.n_vl
                node.w[action_idx] -= self.n_vl
                node.q[action_idx] = node.w[action_idx] / node.n[action_idx]
                trail.append((node, action_idx))

            # Update the game state
            self.game.set_state(node.state)
            # Simulate the action to get to the new leaf state
            self.game.step(node.actions[action_idx])

            # Store mcts simulation state
            trails.append(trail)
            self.obs_batch[i] = self.game.get_obs()
            self.action_mask_batch[i] = self.game.valid_action_mask()
        return trails, self.obs_batch, self.action_mask_batch

    def run_simulations(self):
        trails, self.obs_batch, self.action_mask_batch = self.select()
        # Request batched neural network evaluation
        prior_probs, values = self._request_evaluation(self.obs_batch, self.action_mask_batch)
        self.backup(prior_probs, values, trails)

    def backup(self, prior_probs, values, trails):
        for i in range(self.n_virtual_threads):
            leaf, action_idx = trails[i][-1]
            # Restore game state
            self.game.set_state(leaf.state)
            self.game.step(leaf.actions[action_idx])
            # Create new node if it doesn't already exist
            # (happens when in the same node is evaluated multiple times in a batch)
            if leaf.children[action_idx] is None:
                leaf.children[action_idx] = _Node(self.game.get_state(), self.game.valid_actions(),
                                                  prior_probs[i][self.game.valid_action_mask()],
                                                  leaf.ply + 1, self.game.is_terminal,
                                                  1 if self.game.is_first_player_to_move else -1)
            if self.game.is_terminal:
                val_incr = self.game.get_first_player_reward
            else:
                val_incr = values[i]
            # Backup
            for node, action_idx in trails[i]:
                node.n[action_idx] += 1 - self.n_vl
                node.w[action_idx] += (val_incr * node.value_mul) + self.n_vl
                node.q[action_idx] = node.w[action_idx] / node.n[action_idx]

    def step(self, action_index, is_index=True):
        if not is_index:
            action_index = int(np.argmax(self.root.actions == action_index))
        #  Dirichlet Noise in root node.
        child = self.root.children[action_index]
        if child is None:
            state = self.game.get_state()
            self.game.set_state(self.root.state)
            self.game.step(self.root.actions[action_index])
            ply = self.root.ply
            self._init_root()
            self.root.ply = ply + 1
            self.game.set_state(state)
        else:
            self.root = child
        self._add_noise()

    def get_probs(self):
        if self.root.ply > self.ply_deterministic:
            probs = np.zeros_like(self.root.p)
            probs[np.argmax(self.root.n)] = 1
        else:
            probs = self.root.n.astype(np.float64) / self.root.n.astype(np.float64).sum()
        return probs

    def get_action(self):
        # Called after N times run_simulation has been called.
        probs = self.get_probs()
        action_index = np.random.choice(np.arange(len(self.root.actions)), size=1, p=probs)[0]
        return self.root.actions[action_index], action_index, probs.astype(np.float32)

    def _add_noise(self):
        alpha = np.ones_like(self.root.p) * self.alpha
        self.root.p = (1 - self.eta) * self.root.p + self.eta * np.random.dirichlet(alpha, 1)[0]

    def _select_action(self, node) -> int:
        return np.argmax(node.q + self.c_puct * node.p * np.sqrt(node.n.sum() + 1) / (node.n + 1))

    def _request_evaluation(self, obs, valid_action_mask):
        self.eval_conn.send([torch.from_numpy(obs), torch.from_numpy(valid_action_mask)])
        p, v = self.eval_conn.recv()
        return p.numpy(), v.numpy()


if __name__ == '__main__':
    from games.connect_x import ConnectX
    from net import Policy
    from threading import Thread

    g = ConnectX(7, 6, 2)

    class Connection(Thread):

        def __init__(self):
            super().__init__()
            self.net = Policy(g.obs_space_shape, g.num_actions, 3, 256)
            self.q = []
            self.res_q = []
            self.running = False

        def send(self, x):
            self.q.append(x)

        def run(self):
            self.running = True
            while self.running:
                with torch.no_grad():
                    if len(self.q) > 0:
                        o, m = self.q[0]
                        self.q = self.q[1:]
                        p, v = self.net(o)
                        p[~m] = -float('inf')
                        p = torch.softmax(p, -1)
                        self.res_q.append([p, v])

        def recv(self):
            while len(self.res_q) == 0:
                pass
            res = self.res_q[0]
            self.res_q = self.res_q[1:]
            return res

    conn = Connection()
    conn.start()

    g.reset()
    tree = MCTS(g, conn, 8)
    for i in range(128//8):
        tree.run_simulations()
    action = tree.get_action()
    tree.step(action)
    tree.run_simulations()
    conn.running = False
    conn.join()
