from threading import Timer

import argparse

from self_play.self_play_worker import SelfPlayWorker
from train.train_worker import TrainWorker
from comm import run_server, AlphaZeroClient
from getpass import getpass
import yaml


def main(args):
    if args.start_server:
        run_server(int(args.port))
        return

    num_self_play_worker_procs = args.n_proc

    with open(args.conf, 'r') as f:
        d = yaml.safe_load(f)
    game_name = d['game_name']
    game_kwargs = d['game_kwargs']
    net_kwargs = d['net_kwargs']
    mcts_kwargs = d['mcts_kwargs']
    train_kwargs = d['train_kwargs']

    client = AlphaZeroClient(args.ip, int(args.port), args.user, getpass())

    if args.trainer:
        runner = TrainWorker(game_name, game_kwargs, net_kwargs, train_kwargs, mcts_kwargs,
                             args.out_folder, client, args.device, args.init_from_ckp)
        runner.start()
    else:
        runner = SelfPlayWorker(game_name, game_kwargs, mcts_kwargs, net_kwargs,
                                num_self_play_worker_procs, client, args.device)
        runner.start()

    def stop():
        runner.stop()
        client.stop()

    # Shutdown
    try:
        if args.hours:
            print("Stopping in {} hours".format(args.hours))
            t = Timer(3600 * args.hours, stop)
            t.start()
        client.start()
    finally:
        stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_server', action='store_true', help='Starts the data broker.')
    parser.add_argument('--trainer', action='store_true', help='Specify if this process is the trainer')
    parser.add_argument('--ip', default='localhost', help='IP of the data broker')
    parser.add_argument('--user', default=None, type=str,
                        help='Username used to log into the data broker machine for scp.')
    parser.add_argument('--port', default='29500', help='Port of the data broker.')
    parser.add_argument('--n_proc', default=8, type=int, help='Number of self-play processes')
    parser.add_argument('--conf', default='./conf.yml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out_folder', default='./results')
    parser.add_argument('--init_from_ckp', default=None, type=str, help='only applies to the trainer.')
    parser.add_argument('--hours', default=None, type=int, help='Run for N hours, then shut down.')
    parser.set_defaults(trainer=False, start_server=False)
    a = parser.parse_args()
    if not a.start_server:
        assert a.user is not None, 'Please specify a username with which you can log into the data-brokers machine.'
    main(a)
