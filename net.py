from torch import nn
import torch

from games import make_game


class ResBlock(nn.Module):

    def __init__(self, num_filters):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(num_filters, num_filters, 3, 1, 1),
                                    nn.BatchNorm2d(num_filters),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(num_filters, num_filters, 3, 1, 1),
                                    nn.BatchNorm2d(num_filters))

    def forward(self, x):
        res = self.layers(x)
        return torch.relu(res + x)


class Policy(nn.Module):

    def __init__(self, obs_space_shape, num_actions, num_layers=19, num_filters=256):
        super().__init__()
        width, height = obs_space_shape[1:]
        self.conv_in = nn.Sequential(nn.Conv2d(obs_space_shape[0], num_filters, 3, 1, 1),
                                     nn.BatchNorm2d(num_filters),
                                     nn.ReLU(inplace=True))
        self.res_blocks = nn.Sequential(*[ResBlock(num_filters) for _ in range(num_layers)])
        self.policy_head = nn.Sequential(nn.Conv2d(num_filters, 2, 1, 1, 0),
                                         nn.BatchNorm2d(2),
                                         nn.ReLU(inplace=True),
                                         nn.Flatten(),
                                         nn.Linear(width * height * 2, num_actions))
        self.value_head = nn.Sequential(nn.Conv2d(num_filters, 1, 1, 1, 0),
                                        nn.BatchNorm2d(1),
                                        nn.ReLU(inplace=True),
                                        nn.Flatten(),
                                        nn.Linear(width * height * 1, 256),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(256, 1),
                                        nn.Tanh())

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_blocks(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v

    @classmethod
    def from_ckp(cls, ckp):
        game_name = ckp['other'][3]
        game_kwargs = ckp['other'][4]
        net_kwargs = ckp['other'][5]
        state_dict = ckp['params']
        game = make_game(game_name, **game_kwargs)
        model = cls(game.obs_space_shape, game.num_actions, **net_kwargs)
        model.load_state_dict(state_dict)
        return model


def load_ckp(file):
    ckp = torch.load(file, map_location='cpu')
    model = Policy.from_ckp(ckp)
    game_name = ckp['other'][3]
    game_kwargs = ckp['other'][4]
    if len(ckp['other']) == 7:
        mcts_kwargs = dict(num_simulations=1024, virtual_threads=8, ply_deterministic=4,
                           c_puct=5, alpha=0.5, eta=0.25, n_vl=3)
    else:
        mcts_kwargs = ckp['other'][7]
    return model, mcts_kwargs, game_name, game_kwargs


if __name__ == '__main__':
    net = Policy([2, 10, 10], 10, 5, 256)
    p_, v_ = net(torch.randn(1, 2, 10, 10).float())
    print(p_.shape, v_.shape)
