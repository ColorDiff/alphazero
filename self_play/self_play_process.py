import numpy as np
import torch.multiprocessing as mp

from games import make_game
from self_play.mcts import MCTS
import torch
import threading


# TODO: resignation
class SelfPlayProcess(mp.Process):
    MAX_GAME_LEN = 300

    def __init__(self, game_name: str, game_kwargs: dict, mcts_kwargs: dict, eval_conn, parent_conn):
        super().__init__()
        self.game_name, self.game_kwargs = game_name, game_kwargs
        self.num_simulations = mcts_kwargs['num_simulations']
        self.mcts_kwargs = mcts_kwargs.copy()
        del self.mcts_kwargs['num_simulations']
        self.running = False
        self.eval_conn = eval_conn
        self.parent_conn = parent_conn

    def _init(self):
        self.game = make_game(self.game_name, **self.game_kwargs)
        self.obs_buffer = torch.zeros((self.MAX_GAME_LEN, *self.game.obs_space_shape), dtype=torch.float32)
        self.prob_buffer = torch.zeros((self.MAX_GAME_LEN, self.game.num_actions), dtype=torch.float32)
        self.mask_buffer = torch.zeros((self.MAX_GAME_LEN, self.game.num_actions), dtype=torch.float32)
        self.z_buffer = torch.zeros((self.MAX_GAME_LEN, 1), dtype=torch.float32)
        self.t = threading.Thread(target=self._listen, daemon=True)

    def _send_data(self, game_length):
        print('Game done: length (ply): {}, winner: {}'.format(game_length, self.z_buffer[0].item()))
        self.parent_conn.send((self.obs_buffer[:game_length].clone(),
                               self.mask_buffer[:game_length].clone(),
                               self.prob_buffer[:game_length].clone(),
                               self.z_buffer[:game_length].clone()))

    def _listen(self):
        while self.running:
            cmd = self.parent_conn.recv()
            if cmd == 'stop':
                self.running = False
                print("SelfPlayProcess shutting down")

    def run(self):
        self._init()
        self.running = True
        self.t.start()
        while self.running:
            self.game.reset()
            state = self.game.get_state()
            trees = [MCTS(self.game, self.eval_conn, **self.mcts_kwargs),
                     MCTS(self.game, self.eval_conn, **self.mcts_kwargs)]
            game_length = 0
            while not self.game.is_terminal and self.running:
                tree = trees[int(self.game.is_first_player_to_move)]
                for i_sim in range(self.num_simulations // self.mcts_kwargs['virtual_threads'] + 1):
                    tree.run_simulations()
                # Collect self-play data
                self.game.set_state(state)
                obs, z_mul = self.game.get_obs(), 1 if self.game.is_first_player_to_move else -1
                action_mask = self.game.valid_action_mask()
                action, action_idx, probs = tree.get_action()
                # Store data
                self.obs_buffer[game_length] = torch.from_numpy(obs)
                self.prob_buffer[game_length] *= 0  # Reset the other entries
                self.prob_buffer[game_length, action_mask] = torch.from_numpy(probs)
                self.mask_buffer[game_length] = torch.from_numpy(action_mask)
                self.z_buffer[game_length] = z_mul
                # Advance game
                self.game.step(action)
                state = self.game.get_state()
                trees[0].step(action_idx)
                trees[1].step(action_idx)
                game_length += 1
                if game_length == self.MAX_GAME_LEN:
                    break

            # Game end, game is in terminal state or aborted due to overstepping MAX_GAME_LEN
            if not self.game.is_terminal:
                self.z_buffer *= 0
            else:
                self.z_buffer *= self.game.get_first_player_reward
            self._send_data(game_length)
        self.t.join()


class VectorizedSelfPlayProcess(mp.Process):
    MAX_GAME_LEN = 300

    def __init__(self, game_name: str, game_kwargs: dict, mcts_kwargs: dict, vector_len, eval_conn, parent_conn):
        super().__init__()
        self.vector_len = vector_len
        self.game_name, self.game_kwargs = game_name, game_kwargs
        self.num_simulations = mcts_kwargs['num_simulations']
        self.mcts_kwargs = mcts_kwargs.copy()
        del self.mcts_kwargs['num_simulations']
        self.running = False
        self.eval_conn = eval_conn
        self.parent_conn = parent_conn

    def _init(self):
        self.games = [make_game(self.game_name, **self.game_kwargs) for _ in range(self.vector_len)]
        self.obs_buffer = torch.zeros((self.vector_len, self.MAX_GAME_LEN, *self.games[0].obs_space_shape),
                                      dtype=torch.float32)
        self.prob_buffer = torch.zeros((self.vector_len, self.MAX_GAME_LEN, self.games[0].num_actions),
                                       dtype=torch.float32)
        self.mask_buffer = torch.zeros((self.vector_len, self.MAX_GAME_LEN, self.games[0].num_actions),
                                       dtype=torch.float32)
        self.z_buffer = torch.zeros((self.vector_len, self.MAX_GAME_LEN, 1), dtype=torch.float32)
        self.t = threading.Thread(target=self._listen, daemon=True)

    def _send_data(self, idx, game_length):
        print('Game done: length (ply): {}, winner: {}'.format(game_length, self.z_buffer[idx, 0].item()))
        self.parent_conn.send((self.obs_buffer[idx, :game_length].clone(),
                               self.mask_buffer[idx, :game_length].clone(),
                               self.prob_buffer[idx, :game_length].clone(),
                               self.z_buffer[idx, :game_length].clone()))

    def _listen(self):
        while self.running:
            cmd = self.parent_conn.recv()
            if cmd == 'stop':
                self.running = False
                print("SelfPlayProcess shutting down")

    def run(self):
        self._init()
        self.running = True
        self.t.start()
        # Init other buffers and game instances
        game_lengths = []
        states = []
        trees = []
        trails = []
        for i_game in range(self.vector_len):
            self.games[i_game].reset()
            game_lengths.append(0)
            states.append(self.games[i_game].get_state())
            trails.append(None)
            trees.append([MCTS(self.games[i_game], self.eval_conn, **self.mcts_kwargs),
                          MCTS(self.games[i_game], self.eval_conn, **self.mcts_kwargs)])
        # More Buffers
        virtual_threads = self.mcts_kwargs['virtual_threads']
        obs_batch_buffer = np.zeros((self.vector_len * virtual_threads, *trees[0][0].obs_batch.shape[1:]),
                                    dtype=np.float32)
        action_mask_buffer = np.zeros((self.vector_len * virtual_threads, *trees[0][0].action_mask_batch.shape[1:]),
                                      dtype=np.float32)
        while self.running:

            # Batched simulation phase
            for i_simulation in range(self.num_simulations // self.mcts_kwargs['virtual_threads'] + 1):
                # Batched Selection phase
                for i_game in range(self.vector_len):
                    trail, obs_batch, mask_batch = trees[i_game][int(self.games[i_game].is_first_player_to_move)] \
                        .select()
                    trails[i_game] = trail
                    obs_batch_buffer[i_game * virtual_threads: (i_game + 1) * virtual_threads] = obs_batch
                    action_mask_buffer[i_game * virtual_threads: (i_game + 1) * virtual_threads] = mask_batch

                # Batched network evaluation phase
                self.eval_conn.send([torch.from_numpy(obs_batch_buffer), torch.from_numpy(action_mask_buffer)])
                ps, vs = self.eval_conn.recv()
                ps, vs = ps.numpy(), vs.numpy()

                # Batched backup phase
                for i_game in range(self.vector_len):
                    trees[i_game][int(self.games[i_game].is_first_player_to_move)] \
                        .backup(ps[i_game * virtual_threads: (i_game + 1) * virtual_threads],
                                vs[i_game * virtual_threads: (i_game + 1) * virtual_threads],
                                trails[i_game])

            # Store transitions and step games
            for i_game in range(self.vector_len):
                game = self.games[i_game]
                game_length = game_lengths[i_game]
                game.set_state(states[i_game])

                obs, z_mul = game.get_obs(), 1 if game.is_first_player_to_move else -1
                action_mask = game.valid_action_mask()
                action, action_idx, probs = trees[i_game][int(game.is_first_player_to_move)].get_action()
                # Store data
                self.obs_buffer[i_game, game_length] = torch.from_numpy(obs)
                self.prob_buffer[i_game, game_length] *= 0  # Reset the other entries
                self.prob_buffer[i_game, game_length, action_mask] = torch.from_numpy(probs)
                self.mask_buffer[i_game, game_length] = torch.from_numpy(action_mask)
                self.z_buffer[i_game, game_length] = z_mul
                # Advance game and step
                game.step(action)
                game_lengths[i_game] += 1
                states[i_game] = game.get_state()
                trees[i_game][0].step(action_idx)
                trees[i_game][1].step(action_idx)
                # Handle game over
                if game.is_terminal or game_length == self.MAX_GAME_LEN:
                    if not game.is_terminal:
                        self.z_buffer[i_game] *= 0
                    else:
                        self.z_buffer[i_game] *= game.get_first_player_reward
                    self._send_data(i_game, game_lengths[i_game])
                    game.reset()
                    game_lengths[i_game] = 0
                    states[i_game] = game.get_state()
                    trees[i_game] = [MCTS(game, self.eval_conn, **self.mcts_kwargs),
                                     MCTS(game, self.eval_conn, **self.mcts_kwargs)]

        self.t.join()
