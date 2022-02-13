import argparse
import math
from time import sleep
import numpy as np
from trueskill import Rating, quality_1vs1, rate_1vs1
from alphazero import AlphaZero
from games import make_game
import pickle
import os
import glob


class Evaluator:

    def __init__(self, game_name, game_kwargs, thinking_time: float = 5,
                 ply_deterministic=None, device='cuda', verbose=True):
        self.players = []
        self.ckps = []
        self.game_name = game_name
        self.game_kwargs = game_kwargs
        self.thinking_time = thinking_time
        self.device = device
        self.verbose = verbose
        self.ply_deterministic = ply_deterministic

    def add_player(self, ckp, num_matches):
        self.players.append(Rating())
        self.ckps.append(ckp)
        self.eval_player(len(self.players) - 1, num_matches)

    def eval_player(self, i_player, num_matches):
        assert 0 <= i_player < len(self.players)
        self._log("Evaluating player: {} on {} matches".format(self.ckps[i_player], num_matches))
        if len(self.players) == 1:
            self._log('No other player is in the league currently. Skipping.')
            return
        for n_match in range(num_matches):
            idx, quality = self._find_best_match(i_player)
            self._log('Best match for player is {} with quality {}'.format(self.players[idx], quality))
            results = self.play_match(i_player, idx)
            for winner_i, looser_i, draw in results:
                self.players[winner_i], self.players[looser_i] = rate_1vs1(self.players[winner_i],
                                                                           self.players[looser_i], draw)
            self._log("Player {} rating after match: {}".format(self.ckps[i_player], self.players[i_player]))
            # Sort the players be average rank for logging
            self._log('=' * 5 + 'League ranking' + '=' * 5, True)
            for i, (ckp, player) in enumerate(self.get_ranking()):
                self._log('{}. Checkpoint: {} has rating {}'.format(i, ckp, player), True)

    def get_ranking(self):
        avg_true_skills = []
        for i, player in enumerate(self.players):
            avg_true_skills.append([player.mu, player, self.ckps[i]])
        return list(map(lambda x: [x[2], x[1]], sorted(avg_true_skills, key=lambda x: x[0])))

    def play_match(self, i, j):
        player_i, player_j = AlphaZero(self.ckps[i], self.device), AlphaZero(self.ckps[j], self.device)
        game = make_game(self.game_name, **self.game_kwargs)
        # Each match consists of playing two games in which each player goes first.
        result_1 = self._convert_result(i, j, self._play_game(game, player_i, player_j))
        result_2 = self._convert_result(j, i, self._play_game(game, player_j, player_i))
        return result_1, result_2

    def is_registered(self, ckp):
        return ckp in self.ckps

    def _log(self, msg, always=False):
        if self.verbose or always:
            print(msg)

    @staticmethod
    def _convert_result(first_player, second_player, result):
        draw = False
        winner = second_player
        looser = first_player
        if result == 1:
            winner = first_player
            looser = second_player
        elif result == 0:
            draw = True
        return [winner, looser, draw]

    def _play_game(self, game, first_player: AlphaZero, second_player: AlphaZero) -> int:
        self._log('=' * 5 + 'Playing game' + '=' * 5)
        game.reset()
        first_player.on_game_start(game, self.ply_deterministic)
        second_player.on_game_start(game, self.ply_deterministic)
        while not game.is_terminal:
            self._log(game)
            if game.is_first_player_to_move:
                action = first_player.get_action(game, thinking_time=self.thinking_time)
            else:
                action = second_player.get_action(game, thinking_time=self.thinking_time)
            game.step(action)
            first_player.move(action)
            second_player.move(action)
        result = game.get_first_player_reward
        self._log('First player reward: {}'.format(result))
        return result

    def _find_best_match(self, i_player):
        # idx, max_quality = -1, -math.inf
        # for i in range(len(self.players)):
        #     if i != i_player:
        #         quality = quality_1vs1(self.players[i], self.players[i_player])
        #         if quality > max_quality:
        #             max_quality = quality
        #             idx = i
        idx = np.random.choice([i for i in range(len(self.players)) if i != i_player])
        max_quality = 0
        return idx, max_quality

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file):
        with open(file, 'rb') as f:
            return pickle.load(f)


def main(args):
    EVAL_FILE = 'eval.pkl'
    # Change folder
    old_dir = os.getcwd()
    os.chdir(args.ckp_folder)

    def get_ckps():
        return [file for file in glob.glob('./' + args.ckp_pattern)]

    ckps = get_ckps()
    while len(ckps) < 2:
        print("Waiting for at least 2 checkpoints to exist")
        sleep(10)
        ckps = get_ckps()

    if os.path.exists(EVAL_FILE):       # Load from disk if exists
        evaluator = Evaluator.load(EVAL_FILE)
        evaluator.ply_deterministic = args.ply_deterministic
        evaluator.device = args.device
        evaluator.verbose = args.verbose
    else:                               # Otherwise create a new evaluator
        from net import load_ckp
        _, _, game_name, game_kwargs = load_ckp(ckps[0])
        evaluator = Evaluator(game_name, game_kwargs, args.thinking_time, args.ply_deterministic, args.device,
                              args.verbose)

    # Scan for new checkpoints and evaluate them
    try:
        while True:
            ckps = get_ckps()
            for ckp in ckps:
                if not evaluator.is_registered(ckp):
                    print("Found new checkpoint to evaluate: {}".format(ckp))
                    evaluator.add_player(ckp, args.num_matches)
            sleep(10)
    finally:
        evaluator.save(EVAL_FILE)
        os.chdir(old_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckp_folder', required=True, help='The folder in which checkpoints '
                                                            'are written that should be evaluated. Not recursive.')
    parser.add_argument('--ckp_pattern', default='ckp_*.pth', help='The extension which identifies checkpoints.')
    parser.add_argument('--ply_deterministic', default=4, type=int, help='How many steps to sample actions.')
    parser.add_argument('--thinking_time', default=5, type=float, help='How long the agents are allowed'
                                                                       ' to think per move in seconds.')
    parser.add_argument('--num_matches', default=20, type=int, help='Number of matches to play to determine the '
                                                                    'checkpoint\'s strength relative to all others.')
    parser.add_argument('--device', default='cuda', type=str, help='The compute device to use.')
    parser.add_argument('--verbose', action='store_true', help='Be verbose and print the evaluation.')
    parser.set_defaults(verbose=False)
    main(parser.parse_args())
