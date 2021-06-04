import argparse
from simulator import Simulator
from naive_policy import NaivePolicy
from reward_manager import RewardManager

def main(args):
    reward_man = RewardManager(args)
    policy = NaivePolicy(args, reward_man)
    sim = Simulator(args, policy)

    # Run the simulation
    sim.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a fulfillment simulation.')
    parser.add_argument('--cord_bounds', type=int, default=10,
                    help='Max bounds for cordinates.')
    parser.add_argument("--num_skus", type=int, default=2,
                    help="Number of unique products SKUs.")
    parser.add_argument("--max_prod_inv", type=int, default=10,
                    help="Max inventory for each product across all inventory nodes.")
    parser.add_argument("--min_prod_inv", type=int, default=0,
                    help="Min inventory for each product across all inventory nodes.")
    parser.add_argument("--num_inv_nodes", type=int, default=2,
                    help="Number of inventory nodes.")
    parser.add_argument("--demand_lam", type=float, default=1.0,
                    help="Lambda parameter for sampling demand from demand poisson distribution.")
    parser.add_argument("--T_max", type=int, default=10,
                    help="Max number of orders")
    parser.add_argument("--reward_alpha", type=float, default=0.5,
                    help="Reward item discount.")
    parser.add_argument("--emb_size", type=int, default=256,
                    help="Embedding size.")
    parser.add_argument("--hidden_size", type=int, default=256,
                    help="Number of hidden units used for NN policy.")


                    

    args = parser.parse_args()
    main(args)
