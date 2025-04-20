import argparse
from train.ppo_train import train
from train.evaluate import evaluate
import gymnasium
from importlib.metadata import version as pkg_version
import sys

gymnasium.__version__ = pkg_version("gymnasium")

sys.modules['gym'] = gymnasium

def main():
    parser = argparse.ArgumentParser(
        description='PPO-based MovieLens recommendation'
    )
    parser.add_argument(
        '--mode', choices=['train', 'eval'], default='train',
        help='train or evaluate the PPO agent'
    )
    parser.add_argument(
        '--timesteps', type=int, default=200000,
        help='number of training timesteps'
    )
    parser.add_argument(
        '--episodes', type=int, default=10,
        help='number of evaluation episodes'
    )
    parser.add_argument(
        '--max_steps', type=int, default=500,
        help='steps per episode'
    )
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    else:
        avg_reward = evaluate()
        print(f"Average reward over {args.episodes} episodes: {avg_reward:.2f}")

if __name__ == '__main__':
    main()