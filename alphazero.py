from time import time

from games import make_game
from games.interface import Game
from net import load_ckp
from self_play.mcts import MCTS
import torch
import numpy as np


class Conn:

    def __init__(self, agent):
        self.agent = agent
        self.reqs = []

    def send(self, x):
        self.reqs.append(x)

    def recv(self):
        res = self._eval_positions(*self.reqs[0])
        self.reqs = self.reqs[1:]
        return res

    @torch.no_grad()
    def _eval_positions(self, obs, action_masks):
        p, v = self.agent.model(obs.to(self.agent.device))
        p[~action_masks.to(self.agent.device)] = -float('inf')
        p = torch.softmax(p, -1)
        return p.cpu(), v.cpu()


class AlphaZero:

    def __init__(self, ckp_file, device='cuda'):
        self.model, self.mcts_kwargs, self.game_name, self.game_kwargs = load_ckp(ckp_file)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.tree = None
        self.conn = Conn(self)

    def on_game_start(self, game: Game, ply_deterministic=None):
        name, kwargs = game.describe()
        assert name == self.game_name, '{} != {}'.format(name, self.game_name)
        assert kwargs == self.game_kwargs, '{} != {}'.format(kwargs, self.game_kwargs)
        state = game.get_state()
        if ply_deterministic is None:
            ply_deterministic = self.mcts_kwargs['ply_deterministic']
        self.tree = MCTS(game, self.conn, virtual_threads=self.mcts_kwargs['virtual_threads'],
                         ply_deterministic=ply_deterministic, c_puct=self.mcts_kwargs['c_puct'],
                         alpha=1, eta=0, n_vl=self.mcts_kwargs['n_vl'])
        game.set_state(state)

    def move(self, action):
        self.tree.step(action, is_index=False)

    def get_action(self, game, thinking_time: float):
        # Thinking time is the minimum time used.
        # More complex time management will require an estimate of a simulation run's time consumption.
        assert thinking_time > 0
        start = time()
        state = game.get_state()
        while time() - start < thinking_time:
            self.tree.run_simulations()
        action, _, probs = self.tree.get_action()
        game.set_state(self.tree.root.state)
        print('Evaluation:', ['{:.3f}'.format(float(item)) for item in self.tree.root.q])
        print('Visit counts:', [int(item) for item in self.tree.root.n])
        print('Prior probs:', ['{:.3f}'.format(float(item)) for item in self.tree.root.p])
        print("Root actions:", self.tree.root.actions)
        game.set_state(state)
        return action

    def play(self):
        # Make game
        game = make_game(self.game_name, **self.game_kwargs)
        game.reset()
        self.on_game_start(game, ply_deterministic=-1)  # Always select the best move

        def get_input(prompt, valid_values, typ):
            while True:
                try:
                    r = typ(input(prompt))
                    if r in valid_values:
                        return r
                    else:
                        raise Exception()
                except Exception:
                    print("Invalid value, expected one of {}".format(valid_values))

        # Player order
        print("Game: {}, {}".format(self.game_name, self.game_kwargs))
        res = get_input("Do you want to go first? [y/n]", ['y', 'n'], str)
        thinking_time = get_input('How long should AlphaZero think per turn? [s]', np.arange(0.5, 60.0, 0.5), float)
        agent_is_first_player = res == 'n'
        print(game)
        while not game.is_terminal:
            print("=" * 20)
            if agent_is_first_player == game.is_first_player_to_move:
                action = self.get_action(game, thinking_time)
                game.step(action)
                self.move(action)
                print("AlphaZero moved {}".format(action))
            else:
                action = get_input('Select a move:', [int(item) for item in game.valid_actions()], int)
                game.step(action)
                self.move(action)
                print("You moved {}".format(action))
            print(game)


if __name__ == '__main__':
    import sys
    agent = AlphaZero(sys.argv[1], 'cuda')
    agent.play()
