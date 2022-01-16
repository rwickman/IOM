import argparse

from simulator import Simulator
from naive_policy import NaivePolicy
from dqn_policy import DQNTrainer
from dqn_emb_policy import DQNEmbTrainer
from actor_critic_policy import ActorCriticPolicy
from primal_dual_policy import PrimalDual
from reward_manager import RewardManager
from evaluator import Evaluator
from visual import Visual
from dataset_simulation import DatasetSimulation

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
        if args.use_dataset:
            dataset_sim = DatasetSimulation()
            print("dataset_sim.num_skus: ", dataset_sim.num_skus)
            args.num_skus = dataset_sim.num_skus
        else:
            dataset_sim = None

        # Create the policy
        if args.policy == "naive":
            policy = NaivePolicy(args, reward_man)
        elif args.policy == "dqn":
            if args.no_per:
                args.policy = args.policy + "_no_per"

            policy = DQNTrainer(args, reward_man)
        elif "dqn_emb" in args.policy:
            policy = DQNEmbTrainer(args, reward_man)
        elif "ac" in args.policy:
            policy = ActorCriticPolicy(args, reward_man)
        elif args.policy == "primal":
            policy = PrimalDual(args, reward_man)
        else:
            raise Exception("Invalid policy name.")

        # Create the simulator
        sim = Simulator(args, policy, dataset_sim)
        
        # Run the simulation
        sim.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a fulfillment simulation.')

    sim_args = parser.add_argument_group("Simulator")
    sim_args.add_argument("--demand_beta_a", type=float, default=0.1,
                    help="a parameter value used in the beta distribution to generate demand around a city.")
    sim_args.add_argument("--demand_beta_b", type=float, default=1.0,
                    help="b parameter value used in the beta distribution to generate demand around a city.")
    sim_args.add_argument("--city_radius", type=float, default=4,
                    help="Radius of the given city such that demand is generated.")
    sim_args.add_argument("--city_loc", default=None,
                    help="JSON city locations.")
    sim_args.add_argument("--inv_loc", default=None,
                    help="JSON containing location coordinates for every inventory node.")
    sim_args.add_argument("--ramp_max_prod", action="store_true",
                    help="Slowly ramp up max inventory of SKUs at inventory nodes.")
    sim_args.add_argument("--ramp_eps", type=int, default=256,
                    help="Amount of episodes till max inventory of SKUs at inventory nodes can be generated.")
    sim_args.add_argument("--rand_max_prod", action="store_true",
                    help="Before each episode randomly generate max inventory per SKU for every inventory node between [1, max_inv_prod].")
    sim_args.add_argument('--coord_bounds', type=int, default=10,
                    help='Max bounds for coordinates.')
    sim_args.add_argument("--num_skus", type=int, default=2,
                    help="Number of unique products SKUs.")
    sim_args.add_argument("--max_inv_prod", type=int, default=20,
                    help="Max inventory for each product across all inventory nodes.")
    sim_args.add_argument("--min_inv_prod", type=int, default=0,
                    help="Min inventory for each product across all inventory nodes.")
    sim_args.add_argument("--num_inv_nodes", type=int, default=2,
                    help="Number of inventory nodes.")
    sim_args.add_argument("--demand_lam", type=float, default=1.0,
                    help="Lambda parameter for sampling demand from demand poisson distribution.")
    sim_args.add_argument("--order_line_lam", type=float, default=1.0,
                    help="Lambda parameter for sampling the number of order lines (i.e., SKUs) from poisson distribution.")
    sim_args.add_argument("--inv_sku_lam", type=float, default=None,
                    help="Lambda parameter for sampling the number of SKUs at inventory node from poisson distribution.")
    sim_args.add_argument("--rand_inv_sku_lam", action="store_true",
                    help="Use a random sku lamabda value for each epsiode between [1, num_skus] (this overrides inv_sku_lam).",)
    
    sim_args.add_argument("--order_max", type=int, default=128,
                    help="Max number of orders in an episode during training.")
    sim_args.add_argument("--eval_order_max", type=int, default=None,
                    help="Max number of orders in an episode during evaluation.")

    sim_args.add_argument("--use_dataset", action="store_true",
                    help="Use a dataset to generate demand.")

    parser.add_argument("--reward_alpha", type=float, default=0.5,
                    help="Reward item discount.")
    parser.add_argument("--hidden_size", type=int, default=128,
                    help="Number of hidden units used for NN policy.")
    parser.add_argument("--num_hidden", type=int, default=2,
                    help="Number of hidden layers for NN policy.")
    parser.add_argument("--epsilon", type=float, default=0.95,
                    help="Initial epsilon used for epsilon-greedy in DQN.")
    parser.add_argument("--min_epsilon", type=float, default=0.01,
                    help="Minimum epsilon value used for epsilon-greedy in DQN.")
    parser.add_argument("--epsilon_decay", type=int, default=1024,
                    help="Epsilon decay step used for decaying the epsilon value in epsilon-greedy exploration.")
    parser.add_argument("--lr", type=float, default=6e-4,
                    help="Learning rate used for DRL models.")
    parser.add_argument("--lr_gamma", type=float, default=0.999,
                    help="Learning rate decay factor.")
    parser.add_argument("--min_lr", type=float, default=5e-6,
                    help="Minimum learning rate.")
    parser.add_argument("--no_lr_decay", action="store_true",
                    help="Don't use lr decay.")
    parser.add_argument("--max_grad_norm", type=float, default=2.0,
                    help="Maximum gradient norm.")
    parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size used for training.")

    parser.add_argument("--gamma", type=float, default=0.99,
                    help="Gamma value for discounting reward.")
    parser.add_argument("--gae_lam", type=float, default=0.95,
                    help="GAE lambda.")
    parser.add_argument("--episodes", type=int, default=1024,
                    help="Number of episodes.")
    parser.add_argument("--save_dir", default="models",
                    help="Directory to save the models.")
    parser.add_argument("--load", action="store_true",
                    help="Load saved models and arguments.")
    parser.add_argument("--load_arg_keys", nargs="*",
                    default=["hidden_size", "emb_size", "num_hidden", "dff", "num_heads", "num_enc_layers"],
                    help="Arguemnts to load when loading model.")

    parser.add_argument("--plot", action="store_true",
                    help="Plot training results.")
    parser.add_argument("--reward_smooth_w", type=int, default=32,
                    help="Window size for reward smoothing plot.")
    parser.add_argument("--policy", default="naive",
                    help="Policy to use (e.g., naive, dqn, primal) .")
    parser.add_argument("--train_iter", type=int, default=1,
                    help="Number of train steps after each episode.")
    
    
    imit_args = parser.add_argument_group("Imitation Learning")
    imit_args.add_argument("--expert_pretrain", type=int, default=0,
                    help="Number of train steps to pretrain and collected experiences using expert (e.g., expert_pretrain/train_iter number of episodes).")
    imit_args.add_argument("--expert_dir", default=None,
                    help="Directory of the expert agent, if not specified imitation learning will not be used.")
    imit_args.add_argument("--expert_margin", type=float, default=0.1,
                    help="Margin value used for expert margin classification loss.")                
    imit_args.add_argument("--expert_lam", type=float, default=0.01,
                    help="Weight of the expert margin classification loss.")
    imit_args.add_argument("--expert_epsilon", type=float, default=0.0,
                    help="Epsilon value added to priority value when using PER.")


    dqn_args = parser.add_argument_group("DQN")
    dqn_args.add_argument("--emb_size", type=int, default=64,
        help="Number of transformer encoder and decoder layers.")
    dqn_args.add_argument("--no_per", action="store_true",
                    help="Don't use Prioritized Experience Replay (PER) for DQN model.")
    dqn_args.add_argument("--per_beta", type=float, default=0.4,
                    help="Beta used for proportional priority.")
    dqn_args.add_argument("--eps", type=float, default=1e-9,
                    help="Epsilon used for proportional priority.")
    dqn_args.add_argument("--per_alpha", type=float, default=0.6,
                    help="Alpha used for proportional priority.")
    dqn_args.add_argument("--tgt_update_step", type=int, default=1,
                    help="Number of training batches before target is updated.")
    dqn_args.add_argument("--mem_cap", type=int, default=32768,
                    help="Replay memory capacity.")
    dqn_args.add_argument("--expert_mem_cap", type=int, default=16384,
                    help="Number of expert experiences in replay memory.")
    
    dqn_args.add_argument("--dqn_steps", type=int, default=1,
                    help="Number of steps to use for multistep DQN.")
    dqn_args.add_argument("--tgt_tau", type=float, default=0.05,
                    help="The tau value to control the update rate of the target DQN parameters.")

    pd_args = parser.add_argument_group("Primal-Dual")
    pd_args.add_argument("--kappa", type=float, default=0.25,
                    help="Kappa value used in Primal-Dual Urban algorithm.")

    eval_args = parser.add_argument_group("Evaluation")
    eval_args.add_argument("--eval", action="store_true",
                    help="Evaluate the policies.")
    eval_args.add_argument("--policy_dir", default="../policies",
                    help="Directory containing the model policies to load during evaluation.")
    eval_args.add_argument("--eval_episodes", type=int, default=128,
                    help="Number of evaluation episodes.")
    eval_args.add_argument("--num_bar_ep", type=int, default=5,
                    help="Number of episodes to plot on average reward episode figure.")
    eval_args.add_argument("--no_rand_fulfill_eval", action="store_true",
                    help="Don't plot the random results in evaluation.")
    eval_args.add_argument("--no_naive_fulfill_eval", action="store_true",
                    help="Don't plot the naive results in evaluation.")



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
    ac_args.add_argument("--ac_epochs", type=int, default=1,
                    help="Number of epochs to train batch of episodes")
    ac_args.add_argument("--critic_lam", type=float, default=0.5,
                    help="Critic loss weighting")
    ac_args.add_argument("--min_exps", type=int, default=4096,
                    help="The minimum number of timesteps to run before training over stored experience.")
    ac_args.add_argument("--ppo_clip", type=float, default=0.2,
                    help="PPO surrogate loss clipping.")
    ac_args.add_argument("--vpg", action="store_true",
                    help="Use vanilla policy gradient loss for actor-critic policy.")
    ac_args.add_argument("--reward_scale_factor", type=float, default=0.01,
                    help="Reward scaling factor that may helps the policy learn faster.")
    
    tran_args = parser.add_argument_group("Transformer")
    tran_args.add_argument("--num_enc_layers", type=int, default=2,
        help="Number of transformer encoder and decoder layers.")
    tran_args.add_argument("--num_heads", type=int, default=4,
        help="Number attention heads.")
    tran_args.add_argument("--max_pos_enc", type=int, default=10000,
        help="Maximum positional encoding for encoder.")
    tran_args.add_argument("--drop_rate", type=float, default=0.05,
        help="Dropout rate.")
    parser.add_argument("--dff", type=int, default=128,
        help="Number of units in the pointwise FFN .")
    
    args = parser.parse_args()
    main(args)