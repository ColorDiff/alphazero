import numpy as np
from games.interface import Game


class CrossKalah(Game):
    HOUSE_RANGE = [3, 21]
    SEED_RANGE = [3, 21]

    DRAW = 0
    NORTH_WINS = -1
    SOUTH_WINS = 1

    def __init__(self, houses, seeds):
        self._houses = houses
        self._seeds = seeds
        self._winner = self.DRAW
        self._done = False
        self._s_to_move = True

    def describe(self):
        return CrossKalah.__name__, dict(houses=self._houses, seeds=self._seeds)

    @property
    def num_actions(self):
        return self._houses

    def reset_random(self, houses=None, seeds=None, init_seed=False):
        if init_seed:
            np.random.seed(None)
        self._set_config(houses, seeds)

    def reset(self):
        return self.reset_random(self._houses, self._seeds, init_seed=False)

    def _set_config(self, houses, seeds):
        if houses is None:
            houses = np.random.randint(*self.HOUSE_RANGE)
        if seeds is None:
            seeds = np.random.randint(*self.SEED_RANGE)
        assert self.HOUSE_RANGE[1] >= houses >= self.HOUSE_RANGE[0]
        assert self.SEED_RANGE[1] >= seeds >= self.SEED_RANGE[0]
        self._houses = houses
        self._seeds = seeds
        self._obs_buffer = np.zeros(self.obs_space_shape, dtype=np.float32)
        self._winner = self.DRAW
        self._done = False
        # State in Kalah (3, 3):
        #               1  2  3                         3  2  1
        # [0,           3, 3, 3,        0,              3, 3, 3]
        # Bank North,   Houses South    Bank South,     Houses North
        self._state = np.ones((self._houses * 2 + 2,), dtype=np.uint32) * seeds
        self._bank_n_idx = 0
        self._bank_s_idx = 1 + self._houses
        self._state[self._bank_n_idx] = 0
        self._state[self._bank_s_idx] = 0
        self._houses_s_mask = np.zeros_like(self._state).astype(bool)
        self._houses_s_mask[1: 1 + self._houses] = True
        self._houses_n_mask = np.zeros_like(self._state).astype(bool)
        self._houses_n_mask[self._houses + 2:] = True
        self._south_mask = np.ones_like(self._state).astype(bool)
        self._south_mask[self._bank_n_idx] = False
        self._north_mask = np.ones_like(self._south_mask).astype(bool)
        self._north_mask[self._bank_s_idx] = False
        self._s_to_move = True

    @property
    def is_first_player_to_move(self):
        return self._s_to_move

    def valid_action_mask(self):
        if self._s_to_move:
            return self._state[self._houses_s_mask] > 0
        else:
            return self._state[self._houses_n_mask] > 0

    def valid_actions(self):
        return np.arange(self._houses)[self.valid_action_mask()]

    def get_state(self):
        state = np.zeros(len(self._state) + 3, dtype=np.int32)
        state[:len(self._state)] = self._state
        state[-3] = self._s_to_move * 1
        state[-2] = self._winner
        state[-1] = self._done * 1
        return state

    @property
    def is_terminal(self) -> bool:
        return self._done

    def get_score(self, south: bool):
        return self._winner * (1 if south else -1)

    def set_state(self, state):
        assert len(state) == len(self._state) + 3
        self._s_to_move = bool(state[-3])
        self._winner = state[-2]
        self._done = bool(state[-1])
        self._state[:] = state[:-3]

    def step(self, action):
        assert not self._done
        assert 0 <= action < self._houses, action
        next_to_move = not self._s_to_move
        if self._s_to_move:
            bank_idx = self._houses
            mask = self._south_mask
            offset = 0
        else:
            bank_idx = 0
            mask = self._north_mask
            offset = self._houses + 1
        view = self._state[mask]
        idx = action + offset
        to_move = view[idx]
        assert to_move != 0
        # Remove seeds
        view[idx] = 0
        # Determine indices eligible for increment
        if to_move % 2 == 0:
            add_idxs = np.arange(idx + 1, idx + 1 + to_move) % len(view)
        else:  # Cross-Kalah rule, balancing the game
            add_idxs = np.arange(idx - to_move, idx)[::-1] % len(view)
        # Store the index and seed count of the last increment prior to adding
        last_idx = add_idxs[-1]
        last_idx_count = view[last_idx]
        # Do the increments
        np.add.at(view, add_idxs, 1)
        # Special rules:
        # 1. If the last seed is placed in the own bank, move again
        if last_idx == bank_idx:
            next_to_move = not next_to_move
        # 2. If the last seed is placed in an empty, owned house opposing a non-empty house,
        # capture the opponents seeds in that house
        elif last_idx_count == 0 and view[last_idx] == 1:
            if self._s_to_move:
                if last_idx < self._houses:
                    opposing_idx = len(view) - 1 - last_idx
                    if view[opposing_idx] != 0:
                        view[bank_idx] += view[opposing_idx] + 1
                        view[last_idx] = 0
                        view[opposing_idx] = 0
            else:
                if last_idx > self._houses:
                    opposing_idx = len(view) - last_idx
                    if view[opposing_idx] != 0:
                        view[bank_idx] += view[opposing_idx] + 1
                        view[last_idx] = 0
                        view[opposing_idx] = 0
        # Set the new board values
        self._state[mask] = view
        # Check win conditions
        done = False
        if self._state[self._houses_s_mask].sum() == 0:
            self._state[self._bank_n_idx] += self._state[self._houses_n_mask].sum()
            self._state[self._houses_n_mask] = 0
            done = True
        elif self._state[self._houses_n_mask].sum() == 0:
            self._state[self._bank_s_idx] += self._state[self._houses_s_mask].sum()
            self._state[self._houses_s_mask] = 0
            done = True
        winner = self.DRAW
        if self._state[self._bank_n_idx] > self._houses * self._seeds:
            winner = self.NORTH_WINS
        elif self._state[self._bank_s_idx] > self._houses * self._seeds:
            winner = self.SOUTH_WINS
        # Update player to move
        self._winner = winner
        self._done = done
        self._s_to_move = next_to_move

    def __str__(self):
        char_width = len(str(max(self._state)))
        width = (char_width + 3) * (self._houses + 2) + 1
        empty = ' ' + ' ' * (char_width + 2)
        border = '|' + '-' * (char_width + 2)

        def fill(x, movable=False):
            if movable:
                return '|>{}<'.format(str(x).rjust(char_width))
            return '| {} '.format(str(x).rjust(char_width))

        lines = ['' for _ in range(7)]
        lines[0] = empty + border * self._houses
        lines[1] = empty
        for i, count in enumerate(self._state[self._houses_n_mask][::-1]):
            lines[1] += fill(count, not self._s_to_move and self.valid_action_mask()[::-1][i])
        lines[2] += border + border * self._houses + border
        lines[3] = fill(self._state[self._bank_n_idx]) + '|' + ' ' * (width - 2 * len(empty) - 2)
        lines[3] += fill(self._state[self._bank_s_idx])
        lines[4] = border * (self._houses + 2)
        lines[5] = empty
        for i, count in enumerate(self._state[self._houses_s_mask]):
            lines[5] += fill(count, self._s_to_move and self.valid_action_mask()[i])
        lines[6] = empty + border * self._houses
        for i in range(len(lines)):
            lines[i] += '|'
        if self._done:
            if self._winner == self.DRAW:
                s = 'Draw!'
            else:
                s = ('South Wins!' if self._winner == self.SOUTH_WINS else 'North Wins!')
            lines.append(empty + s)
        return '\n'.join(lines)

    def render(self):
        print(str(self))

    @property
    def obs_space_shape(self):
        return [self._houses + 1, self._seeds * self._houses + 1, 3]

    def get_obs(self):
        idxs = np.arange(self._houses)
        houses_south = self._state[self._houses_s_mask].reshape(-1, 1).clip(0, self._seeds * self._houses)
        houses_north = self._state[self._houses_n_mask].reshape(-1, 1).clip(0, self._seeds * self._houses)
        bank_s = self._state[self._bank_s_idx].clip(0, self._seeds * self._houses)
        bank_n = self._state[self._bank_n_idx].clip(0, self._seeds * self._houses)
        if self._s_to_move:
            self._obs_buffer[idxs, houses_south[idxs], 0] = 1
            self._obs_buffer[idxs, houses_north[idxs], 1] = 1
            self._obs_buffer[self._houses, bank_s, 0] = 1
            self._obs_buffer[self._houses, bank_n, 1] = 1
        else:
            self._obs_buffer[idxs, houses_south[idxs], 1] = 1
            self._obs_buffer[idxs, houses_north[idxs], 0] = 1
            self._obs_buffer[self._houses, bank_s, 1] = 1
            self._obs_buffer[self._houses, bank_n, 0] = 1
        self._obs_buffer[:, :, 2] = self._s_to_move * 1
        return self._obs_buffer

    @property
    def get_first_player_reward(self):
        assert self.is_terminal
        return self._winner


if __name__ == '__main__':
    k = CrossKalah(3, 3)
    k.reset(), k.render()
    k.step(0), k.render()
    k.step(1), k.render()
    k.step(0), k.render()
    k.step(0), k.render()
    k.step(2), k.render()
    k.step(1), k.render()
    k.step(0), k.render()
    k.step(2), k.render()
    k.step(2), k.render()
    # k.step(2), k.render()
    k.step(0), k.render()
    k.step(2), k.render()
    k.step(2), k.render()
    k.step(1), k.render()
    k.step(1), k.render()
    k.step(1), k.render()
    k.step(0), k.render()
    k.step(2), k.render()
    k.step(1), k.render()
    k.step(2), k.render()
    k.step(2), k.render()
    print("Saving last state")
    s = k.get_state()
    print(s)
    k.step(0), k.render()
    print("Restoring state. Result:")
    k.set_state(s), k.render()
    k.step(0), k.render()
    print(k.get_state())
    print(k.get_state())
    k.set_state(s)
    try:
        k.step(1)
        print("Expected an Error, but step continued")
    except AssertionError:
        pass
    k.step(0)
    try:
        k.step(0)
        print("Expected an error, but step continued")
    except AssertionError:
        pass
    print(k.get_obs()[..., 1], k.obs_space_shape)
