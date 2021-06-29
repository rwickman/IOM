import argparse

from simulator import Simulator
from naive_policy import NaivePolicy
from dqn_policy import DQNTrainer
from actor_critic_policy import ActorCriticPolicy
from primal_dual_policy import PrimalDual
from reward_manager import RewardManager
from evaluator import Evaluator
from visual import Visual

def main(args):
    reward_man = RewardManager(args)
    if args.eval:
        sim = Simulator(args, None)
        if args.vis:
            visual = Visual(args, sim._inv_nodes)
        else:
            visual = None

        eval = Evaluator(args, reward_man, sim, visual)
        eval.run()
    else:
        # Create the policy
        if args.policy == "naive":
            policy = NaivePolicy(args, reward_man)
        elif args.policy == "dqn":
            if args.no_per:
                args.policy = args.policy + "_no_per"

            policy = DQNTrainer(args, reward_man)
        elif args.policy == "ac":
            policy = ActorCriticPolicy(args, reward_man)
        else:
            policy = PrimalDual(args, reward_man)

        # Create the simulator
        sim = Simulator(args, policy)

        # Run the simulation
        sim.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a fulfillment simulation.')
    parser.add_argument('--coord_bounds', type=int, default=10,
                    help='Max bounds for coordinates.')
    parser.add_argument("--num_skus", type=int, default=2,
                    help="Number of unique products SKUs.")
    parser.add_argument("--max_inv_prod", type=int, default=20,
                    help="Max inventory for each product across all inventory nodes.")
    parser.add_argument("--min_inv_prod", type=int, default=1,
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
    parser.add_argument("--gae_lam", type=float, default=0.95,
                    help="GAE lambda.")
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
    parser.add_argument("--eps", type=float, default=1e-16,
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
                    help="Policy to use (e.g., naive, dqn, primal) .")
    parser.add_argument("--loc_json", default=None,
                    help="JSON containing location coordinates for every inventory node.")

    pd_args = parser.add_argument_group("Primal-Dual")
    pd_args.add_argument("--kappa", type=float, default=2,
                    help="Kappa value used in Primal-Dual Urban algorithm.")

    eval_args = parser.add_argument_group("Evaluation")
    eval_args.add_argument("--eval", action="store_true",
                    help="Evaluate the policies.")
    eval_args.add_argument("--policy_dir", default="../policies",
                    help="Directory containing the model policies to load during evaluation.")
    eval_args.add_argument("--eval_episodes", type=int, default=512,
                    help="Number of evaluation episodes.")
    eval_args.add_argument("--num_bar_ep", type=int, default=5,
                    help="Number of episodes to plot on average reward episode figure.")


    vis_args = parser.add_argument_group("Visual")
    vis_args.add_argument("--vis", action="store_true",
                    help="Visualize the policies.")
    vis_args.add_argument("--screen_size", type=int, default=1024,
                    help="Screen size of the Pygame window.")
    vis_args.add_argument("--screen_padding", type=int, default=128,
                    help="Padding around fulfillment grid for information to fit.")
    vis_args.add_argument("--font_size", type=int, default=24,
                    help="Padding around fulfillment grid for information to fit.")
    
    ac_args = parser.add_argument_group("Actor Critic")
    ac_args.add_argument("--noptepochs", type=int, default=1,
                    help="Number of epochs to train batch of episodes")
    ac_args.add_argument("--critic_lam", type=float, default=0.5,
                    help="Critic loss weighting")
    ac_args.add_argument("--min_exps", type=int, default=512,
                    help="The minimum number of timesteps to run before training over stored experience.")
    ac_args.add_argument("--ppo_clip", type=float, default=0.2,
                    help="PPO surrogate loss clipping.")
    ac_args.add_argument("--vpg", action="store_true",
                    help="Use vanilla policy gradient loss for actor-critic policy.")
    ac_args.add_argument("--reward_scale_factor", type=float, default=0.01,
                    help="Reward scaling factor that may helps the policy learn faster.")
    args = parser.parse_args()
    main(args)