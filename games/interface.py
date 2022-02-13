from abc import ABC, abstractmethod


class Game(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def valid_actions(self):
        pass

    @abstractmethod
    def valid_action_mask(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @property
    @abstractmethod
    def is_first_player_to_move(self):
        pass

    @property
    @abstractmethod
    def is_terminal(self):
        pass

    @property
    @abstractmethod
    def obs_space_shape(self):
        pass

    @abstractmethod
    def get_obs(self):
        pass

    @property
    @abstractmethod
    def get_first_player_reward(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def set_state(self, state):
        pass

    @property
    @abstractmethod
    def num_actions(self):
        pass

    def describe(self):
        pass
