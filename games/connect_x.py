import numpy as np
from games.interface import Game
from copy import deepcopy


class ConnectX(Game):

    FIRST_PLAYER = 1
    FIRST_PLAYER_SYMBOL = 'X'
    DRAW = 0
    SECOND_PLAYER = -1
    SECOND_PLAYER_SYMBOL = 'O'

    def __init__(self, width, height, x):
        self._winner = 0
        self._p_idx = 0
        self._done = False
        self._set_config(width, height, x)
        self._heights = None
        self._state = None

    def describe(self):
        return ConnectX.__name__, dict(width=self._width, height=self._height, x=self._x)

    def reset(self):
        self._set_config(self._width, self._height, self._x)

    def _set_config(self, width, height, x):
        assert width > 0 and height > 0 and x > 0
        # assert x <= width and x <= height
        # Properties
        self._width = width
        self._height = height
        self._x = x
        self._idxs_offsets = np.arange(self._x)
        # State
        self._state = np.zeros((self._width, self._height, 2), dtype=np.int32)
        self._heights = np.zeros(self._width, dtype=np.int32)
        self._obs_buffer = np.zeros((self._width, self._height, 3), dtype=np.float32)
        # Other properties
        self._p_idx = 0
        self._winner = 0
        self._done = False

    @property
    def num_actions(self):
        return self._width

    def valid_action_mask(self) -> np.ndarray:
        return self._heights < self._height

    def valid_actions(self):
        return np.arange(self._width)[self.valid_action_mask()]

    def step(self, action):
        assert self._heights[action] < self._height
        assert 0 <= action < self._width
        assert not self._done
        self._state[action, self._heights[action], self._p_idx] = 1
        self._heights[action] += 1

        # Check horizontal connections
        self._done = False
        view = self._state[..., self._p_idx]
        non_zero = np.argwhere(view)
        for x, y in non_zero:
            non_valid_horizontal_start = x > self._width - self._x
            non_valid_vertical_start = y > self._height - self._x
            non_valid_reverse_start = y < self._x - 1
            if not non_valid_horizontal_start and view[x:x+self._x, y].sum() == self._x:        # Horizontal
                self._done = True
                break
            if not non_valid_vertical_start and view[x, y:y+self._x].sum() == self._x:          # Vertical
                self._done = True
                break
            if not non_valid_horizontal_start and not non_valid_vertical_start and \
                    view[x + self._idxs_offsets, y + self._idxs_offsets].sum() == self._x:      # Diagonal up
                self._done = True
                break
            # Diagonal down
            if not non_valid_horizontal_start and not non_valid_reverse_start and \
                    view[x + self._idxs_offsets, y - self._idxs_offsets].sum() == self._x:
                self._done = True
                break
        if self._done:
            self._winner = self.FIRST_PLAYER if self._p_idx == 0 else self.SECOND_PLAYER
        elif self.valid_action_mask().sum() == 0:
            self._done = True
            self._winner = self.DRAW
        self._p_idx = (self._p_idx + 1) % 2

    @property
    def is_first_player_to_move(self):
        return self._p_idx == 0

    def __str__(self):
        def fill(c):
            return '{} '.format(self.FIRST_PLAYER_SYMBOL if c[0] else (self.SECOND_PLAYER_SYMBOL if c[1] else ' '))
        lines = ['| ' for _ in range(self._height)]
        for y in list(range(self._height)):
            for x in range(self._width):
                lines[self._height - 1 - y] += fill(self._state[x, y])
        for i in range(len(lines)):
            lines[i] += '|'
        total_width = (2 * self._width + 3)
        config_str = 'Connect {}'.format(self._x)
        s = '-' * ((total_width - len(config_str)) // 2) + config_str
        s += '-' * int(np.ceil((total_width - len(config_str)) / 2))
        lines.append(s)
        s = '{} '.format(self.FIRST_PLAYER_SYMBOL if self.is_first_player_to_move else self.SECOND_PLAYER_SYMBOL)
        s2 = '  '
        for i, valid in enumerate(self.valid_action_mask()):
            s += '{}'.format(str(i).ljust(2))
            s2 += 'v ' if valid else '  '
        lines = [s, s2] + lines
        if self._done:
            if self._winner == self.FIRST_PLAYER:
                lines.append(self.FIRST_PLAYER_SYMBOL + ' WINS!')
            elif self._winner == self.SECOND_PLAYER:
                lines.append(self.SECOND_PLAYER_SYMBOL + ' WINS!')
            else:
                lines.append("DRAW!")
        return '\n'.join(lines)

    def render(self):
        print(str(self))

    @property
    def is_terminal(self):
        return self._done

    @property
    def obs_space_shape(self):
        return [self._width, self._height, 3]

    def get_obs(self):
        # Observation relativized to the player that is to move.
        if self._p_idx == 1:    # Second player
            self._obs_buffer[..., :2] = self._state[..., ::-1]
        else:   # First player
            self._obs_buffer[..., :2] = self._state
        self._obs_buffer[..., 2] = self._p_idx
        return self._obs_buffer

    @property
    def get_first_player_reward(self):
        assert self.is_terminal
        return self._winner

    def get_state(self):
        return deepcopy(self._state), deepcopy(self._heights), self._done, self._winner, self._p_idx

    def set_state(self, state):
        state, heights, self._done, self._winner, self._p_idx = state
        self._state[:] = state
        self._heights[:] = heights


def play_game(width=7, height=6, x=4):

    game = ConnectX(width, height, x)
    while True:
        game.reset(), game.render()
        while not game.is_terminal:
            while True:
                try:
                    action = int(input('Select action:'))
                except Exception:
                    action = -1
                if action in game.valid_actions():
                    break
                print("Invalid action. Please try again!")
            game.step(action), game.render()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('ConnectX')
    parser.add_argument('--width', default=7, type=int)
    parser.add_argument('--height', default=6, type=int)
    parser.add_argument('--x', default=2, type=int)
    args = parser.parse_args()
    play_game(args.width, args.height, args.x)
