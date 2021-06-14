import argparse
from simulator import Simulator
from naive_policy import NaivePolicy
from dqn_policy import DQNTrainer
from reward_manager import RewardManager

def main(args):
    reward_man = RewardManager(args)

    if args.policy == "naive":
        policy = NaivePolicy(args, reward_man)
    else:
        # Create the policy
        policy = DQNTrainer(args, reward_man)

    # Create the simulator
    sim = Simulator(args, policy)

    # Run the simulation
    sim.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a fulfillment simulation.')
    parser.add_argument('--cord_bounds', type=int, default=10,
                    help='Max bounds for cordinates.')
    parser.add_argument("--num_skus", type=int, default=2,
                    help="Number of unique products SKUs.")
    parser.add_argument("--max_inv_prod", type=int, default=10,
                    help="Max inventory for each product across all inventory nodes.")
    parser.add_argument("--min_inv_prod", type=int, default=0,
                    help="Min inventory for each product across all inventory nodes.")
    parser.add_argument("--num_inv_nodes", type=int, default=2,
                    help="Number of inventory nodes.")
    parser.add_argument("--demand_lam", type=float, default=1.0,
                    help="Lambda parameter for sampling demand from demand poisson distribution.")
    parser.add_argument("--T_max", type=int, default=128,
                    help="Max number of orders")
    parser.add_argument("--reward_alpha", type=float, default=0.5,
                    help="Reward item discount.")
    parser.add_argument("--emb_size", type=int, default=256,
                    help="Embedding size.")
    parser.add_argument("--hidden_size", type=int, default=256,
                    help="Number of hidden units used for NN policy.")
    parser.add_argument("--epsilon", type=float, default=0.9,
                    help="Initial epsilon used for epsilon-greedy in DQN.")
    parser.add_argument("--min_epsilon", type=float, default=0.05,
                    help="Minimum epsilon value used for epsilon-greedy in DQN.")
    parser.add_argument("--epsilon_decay", type=int, default=1024,
                    help="Epsilon decay step.")
    parser.add_argument("--lr", type=float, default=6e-4,
                    help="Learning rate used for DRL models.")
    parser.add_argument("--lr_gamma", type=float, default=0.999,
                    help="Learning rate decay factor.")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                    help="Minimum learning rate.")
    parser.add_argument("--no_lr_decay", action="store_true",
                    help="Don't use lr decau.")
    parser.add_argument("--max_grad_norm", type=float, default=2.0,
                    help="Maximum gradient norm.")
    parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size used for training.")
    parser.add_argument("--mem_cap", type=int, default=100000,
                    help="Replay memory capacity.")
    parser.add_argument("--gamma", type=float, default=0.99,
                    help="Gamma value for discounting reward.")
    parser.add_argument("--episodes", type=int, default=1024,
                    help="Number of episodes.")
    parser.add_argument("--save_dir", default="models",
                    help="Directory to save the models.")
    parser.add_argument("--load", action="store_true",
                    help="Load saved models.")
    parser.add_argument("--no_per", action="store_true",
                    help="Don't use Prioritized Experience Replay (PER) for DQN model.")
    parser.add_argument("--per_beta", type=float, default=0.4,
                    help="Beta used for proportional priority.")
    parser.add_argument("--eps", type=float, default=1e-6,
                    help="Epsilon used for proportional priority.")
    parser.add_argument("--per_alpha", type=float, default=0.6,
                    help="Alpha used for proportional priority.")
    parser.add_argument("--tgt_update_step", type=int, default=5,
                    help="Number of training batches before target is updated.")
    parser.add_argument("--plot", action="store_true",
                    help="Plot training results.")
    parser.add_argument("--reward_smooth_w", type=int, default=32,
                    help="Window size for reward smoothing plot.")
    parser.add_argument("--policy", default="naive",
                    help="Policy to use (e.g., naive, dqn) .")


    args = parser.parse_args()
    main(args)
