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
        # run evaluation and grab all the metrics
        metrics = evaluate(
            episodes=args.episodes,
            max_steps=args.max_steps,
            k=10,
        )

        print(f"Average reward over {args.episodes} episodes: {metrics['avg_reward']:.2f}")
        print(f"Precision@10:               {metrics['precision']:.4f}")
        print(f"Recall@10:                  {metrics['recall']:.4f}")
        print(f"NDCG@10:                    {metrics['ndcg']:.4f}")
        print(f"Mean Average Precision@10:  {metrics['map']:.4f}")

if __name__ == '__main__':
    main()