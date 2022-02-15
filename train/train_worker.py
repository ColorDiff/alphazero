import os
import threading
from time import sleep, time

import torch.multiprocessing as mp
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from torch.optim import SGD, lr_scheduler

from games import make_game
from net import Policy, load_ckp
from threading import Thread


class TrainWorker(Thread):

    def __init__(self, game_name, game_kwargs, net_kwargs, train_config, mcts_config,
                 out_folder, client, device='cuda', init_from_ckp=None):
        super().__init__()
        self.game_name, self.game_kwargs = game_name, game_kwargs
        self.net_kwargs = net_kwargs
        self.train_conf = train_config
        self.out_folder = out_folder
        self.mcts_config = mcts_config
        os.makedirs(out_folder, exist_ok=True)
        self.device = device
        self.client = client
        self.running = False
        self.init_from = init_from_ckp

    def _init(self):
        game = make_game(self.game_name, **self.game_kwargs)
        if self.init_from is None:
            self.obs_buffer = torch.zeros((self.train_conf['buffer_size'], *game.obs_space_shape), dtype=torch.float32)
            self.action_mask_buffer = torch.zeros(self.train_conf['buffer_size'], game.num_actions, dtype=torch.bool)
            self.probs_buffer = torch.zeros((self.train_conf['buffer_size'], game.num_actions), dtype=torch.float32)
            self.reward_buffer = torch.zeros((self.train_conf['buffer_size'], 1), dtype=torch.float32)
            self.last_added = torch.zeros((self.train_conf['buffer_size'],), dtype=torch.float32)
            self.buffer_size = self.train_conf['buffer_size']
            self.buffer_end = 0
            self.buffer_i = 0
            self.sample_allowance = 0
            self.i_iter = -1
            self.net = Policy(game.obs_space_shape, game.num_actions, **self.net_kwargs).to(self.device)
            ckp = None
        else:
            ckp = torch.load(self.init_from, map_location='cpu')
            self.net = Policy.from_ckp(ckp).to(self.device)
            self.buffer_end, self.buffer_i, self.i_iter, self.game_name, self.game_kwargs, \
            self.net_kwargs, self.train_conf, self.mcts_config, self.sample_allowance = ckp['other']
            buffers = torch.load(os.path.join(os.path.dirname(self.init_from), 'buffers.pth'))
            self.obs_buffer = buffers['obs']
            self.probs_buffer = buffers['probs']
            self.action_mask_buffer = buffers['masks']
            self.reward_buffer = buffers['rewards']
            self.last_added = torch.zeros(len(self.reward_buffer), dtype=torch.float32)
        self.batch_size = self.train_conf['batch_size']
        self.num_batches_per_step = self.train_conf['num_batches_per_step']
        self.min_buffer_size = self.train_conf['min_buffer_size']
        self.num_iter_per_publish = self.train_conf['num_iter_per_publish']
        self.num_iter_per_ckp = self.train_conf['num_iter_per_ckp']
        self.max_iter = self.train_conf['max_iter']
        self.steps_per_obs = self.train_conf['steps_per_observation']
        self.net.train()
        self.opt = SGD(self.net.parameters(), lr=self.train_conf['lr'], momentum=self.train_conf['momentum'],
                       weight_decay=self.train_conf['weight_decay'])
        self.lr_scheduler = lr_scheduler.MultiStepLR(self.opt, self.train_conf['lr_decay_step'], 0.1)
        if ckp is not None:
            self.opt.load_state_dict(ckp['opt'])
            self.lr_scheduler.load_state_dict(ckp['scheduler'])
        self.writer = SummaryWriter(self.out_folder)
        self.lock = threading.Lock()
        self.last_publish = None
        self.last_save = None
        self._publish_parameters()

    def _insert_data_callback(self, msg):
        print("Received message from server with status: {}".format(msg['status']))
        if msg['status'] != 'OK':
            print("Ignoring data.")
            return
        for item in msg['data']:
            obs, mask, probs, reward = item
            print('Fetched self-play from server. got: {} items'.format(len(obs)))
            with self.lock:
                start, end = self.buffer_i, self.buffer_i + len(obs)
                if end > self.buffer_size:
                    start, end = 0, len(obs)
                self.obs_buffer[start:end] = obs
                self.probs_buffer[start:end] = probs
                self.reward_buffer[start:end] = reward
                self.action_mask_buffer[start:end] = mask
                self.last_added[start:end] = self.i_iter
                self.buffer_i = end
                self.buffer_end = max(self.buffer_end, self.buffer_i)
                self.sample_allowance += len(obs) * self.steps_per_obs * (self.buffer_end / self.buffer_size)

    def _sample_buffer(self):
        idxs = torch.randint(0, self.buffer_end, (self.batch_size * self.num_batches_per_step,))
        for i in range(self.num_batches_per_step):
            start, stop = i * self.batch_size, (i + 1) * self.batch_size
            with self.lock:
                res = [self.obs_buffer[idxs[start:stop]], self.action_mask_buffer[idxs[start:stop]],
                       self.probs_buffer[idxs[start:stop]], self.reward_buffer[idxs[start:stop]]]
                yield [item.to(self.device) for item in res]

    def _train_step(self):
        self.opt.zero_grad()
        agg_ce_loss = 0
        agg_mse_loss = 0
        for obs, mask, probs, rewards in self._sample_buffer():
            # Forward network
            p, v = self.net(obs)
            # MSE Loss
            mse_loss = F.mse_loss(v, rewards) / self.num_batches_per_step
            # Masked CE Loss
            p[~mask] = -torch.inf
            ce_loss = torch.nan_to_num(probs * -torch.log_softmax(p, dim=-1),
                                       nan=0, posinf=0, neginf=0).sum(dim=-1).mean() / self.num_batches_per_step
            loss = mse_loss + ce_loss
            loss.backward()
            # Track loss
            agg_mse_loss += mse_loss.item()
            agg_ce_loss += ce_loss.item()
            self.sample_allowance -= len(obs)
        print('CE loss:', agg_ce_loss, 'MSE Loss:', agg_mse_loss)
        self.opt.step()
        self.lr_scheduler.step()
        self.i_iter += 1
        # Write metrics to tensorboard
        self.writer.add_scalar('loss/ce_loss', agg_ce_loss, self.i_iter)
        self.writer.add_scalar('loss/mse_loss', agg_mse_loss, self.i_iter)
        self.writer.add_scalar('loss/loss', agg_mse_loss + agg_ce_loss, self.i_iter)
        self.writer.add_scalar('data/buffer_size', self.buffer_end, self.i_iter)
        self.writer.add_scalar('data/avg_buffer_age', (self.i_iter - self.last_added).mean(), self.i_iter)

    def _publish_parameters_callback(self, msg):
        print("Publish parameters: Received answer from server {}".format(msg))

    def _publish_parameters(self):
        if self.last_publish == self.i_iter:
            return
        self.last_publish = self.i_iter
        params = {k: v.cpu() for k, v in self.net.state_dict().items()}
        self.client.put_parameters(self._publish_parameters_callback, params)

    def _save_state(self):
        if self.last_save == self.i_iter:
            return
        self.last_save = self.i_iter
        params = self.net.state_dict()
        opt_state = self.opt.state_dict()
        scheduler_state = self.lr_scheduler.state_dict()
        other = [self.buffer_end,
                 self.buffer_i,
                 self.i_iter,
                 self.game_name,
                 self.game_kwargs,
                 self.net_kwargs,
                 self.train_conf,
                 self.mcts_config,
                 self.sample_allowance]
        path = os.path.join(self.out_folder, 'ckp_{}.pth'.format(self.i_iter // self.num_iter_per_ckp))
        torch.save({'params': params,
                    'opt': opt_state,
                    'scheduler': scheduler_state,
                    'other': other}, path)
        torch.save({'obs', self.obs_buffer,
                    'probs', self.probs_buffer,
                    'masks', self.action_mask_buffer,
                    'rewards', self.reward_buffer},
                   os.path.join(self.out_folder, 'buffers.pth'))
        print("Written ckp to: {}".format(path))

    def stop(self):
        self.running = False
        self.join()

    def run(self):
        self._init()
        self.running = True
        last = 0
        empty = False
        while self.running or self.i_iter < self.max_iter:
            if self.buffer_end >= self.min_buffer_size and \
                    self.sample_allowance > self.batch_size * self.num_batches_per_step:
                if self.buffer_end == self.min_buffer_size:
                    self.sample_allowance = 0
                self._train_step()
            else:
                print("Waiting for buffer to fill: [{}/{}], or sample allowance: "
                      "[{:.1f}/{}]".format(self.buffer_end, self.min_buffer_size,
                                           self.sample_allowance, self.batch_size * self.num_batches_per_step))
                sleep(0.5)

            # after some steps publish new parameters
            if self.i_iter % self.num_iter_per_publish == 0:
                self._publish_parameters()

            # after some steps, save a checkpoint and the buffer states.
            if self.i_iter % self.num_iter_per_ckp == 0:
                self._save_state()

            now = time()
            if not empty or now - last > 0.05:
                last = now
                print("Requesting self-play data from server")
                self.client.get_self_play_data(self._insert_data_callback)
