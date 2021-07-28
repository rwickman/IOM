import json, os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm

from simulator import Simulator, InventoryProduct, DemandNode
from reward_manager import RewardManager
from naive_policy import NaivePolicy
from random_policy import RandomPolicy
from dqn_policy import DQNTrainer
from dqn_att_policy import DQNAttTrainer
from dqn_emb_policy import DQNEmbTrainer
from primal_dual_policy import PrimalDual
from actor_critic_policy import ActorCriticPolicy
from visual import Visual

sns.set(style="darkgrid", font_scale=1.5)

class EvaluationResults:
    def __init__(self):
        # For all rewards
        self.rewards_dict = {}
        
        # For reward averages across episodes
        self.ep_reward_avgs_dict = {}

    def add_rewards(self, policy_name: str, rewards: list[float]):
        if policy_name not in self.rewards_dict:
            self.rewards_dict[policy_name] = []

        self.rewards_dict[policy_name].extend(rewards)

    def add_ep_rewards(self, policy_name: str, ep_rewards: list[float]):
        if policy_name not in self.ep_reward_avgs_dict:
            self.ep_reward_avgs_dict[policy_name] = []
        self.ep_reward_avgs_dict[policy_name].append(sum(ep_rewards) / len(ep_rewards))


class Evaluator:
    def __init__(self,
                args,
                reward_man: RewardManager,
                sim: Simulator,
                visual: Visual = None):

        self.args = args
        self.reward_man = reward_man    
        self.sim = sim
        self.visual = visual

        # Remove reward scaling
        self.reward_man._reward_scale_factor = 1

        # Set the initial inventory
        self._inv_dict = self._init_inv()

        # Load the policies
        self._policies = self._load_policies()

    def _init_inv(self) -> dict:
        """Create dict storing the inventory so can restock same way for all policies in an episode."""
        inv_dict = {}
        for inv_node_id, inv_node in self.sim._inv_node_man._inv_nodes_dict.items():
            inv_dict[inv_node_id] = []
            for inv_prod in inv_node.inv.items():
                # Copy the inventory node
                inv_dict[inv_node_id].append(inv_prod.copy())
        
        return inv_dict

    def _restock_nodes(self):
        """Restock the inventory nodes for testing next policy."""
        self.sim._inv_node_man.empty()
        assert self.sim._inv_node_man.inv.inv_size == 0
        

        for inv_node_id, inv_prods in self._inv_dict.items():
            for inv_prod in inv_prods:
                self.sim._inv_node_man.add_product(inv_node_id, inv_prod.copy())

    def _gen_demand_nodes(self) -> list[DemandNode]:
        """Generate list of demand nodes for an evaluation episode."""
        stock = self.sim._inv_node_man.stock

        # Get non-zero items
        stock = [item for item in stock if item.quantity > 0] 

        demand_nodes = []
        while len(stock) > 0:
            if self.args.eval_order_max and len(demand_nodes) >= self.args.eval_order_max:
                break
            

            demand_node = self.sim._gen_demand_node(stock)
            demand_nodes.append(demand_node)

            # Update stock
            for inv_prod in demand_node.inv.items():
                if inv_prod.quantity > 0:
                    # Remove the quantity from stock 
                    for item in stock:
                        if item.sku_id == inv_prod.sku_id:
                            item.quantity -= inv_prod.quantity
                            # sanity-check
                            assert inv_prod.quantity >= 0 

                            # Remove the item from the stock
                            if item.quantity <= 0:
                                stock.remove(item)

        return demand_nodes

    def _load_policies(self) -> list:
        """Load all the policies."""
        policies = {}
        policies["naive"] = NaivePolicy(self.args, self.reward_man)
        policies["random"] = RandomPolicy(self.args, self.reward_man)

        dirs = os.listdir(self.args.policy_dir)
        for policy_dir in dirs:
            policy_dir = os.path.join(self.args.policy_dir, policy_dir)
            # Verify it is a directory
            if os.path.isdir(policy_dir):
                train_dict_path = os.path.join(policy_dir, "train_dict.json") 
                # Verfiy the train JSON exists
                if os.path.exists(train_dict_path):

                    with open(train_dict_path) as f:
                        train_dict = json.load(f)

                        # # Copy old arguments
                        # exempt_keys = set(self.args.exempt_load_args)
                        # old_args = train_dict["args"]
                        # cur_args_dict = vars(self.args)
                        # for key, val in old_args.items():
                        #     if key not in exempt_keys:
                        #         cur_args_dict[key] = val

                        # Temporarily set the save_dir to this path
                        self.args.save_dir = policy_dir

                        # Make sure to load the parameters
                        self.args.load = True
                        
                        if train_dict["policy_name"] == "dqn" or train_dict["policy_name"] == "dqn_no_per":
                            policies[train_dict["policy_name"]] = DQNTrainer(self.args, self.reward_man)
                        elif "primal" in train_dict["policy_name"]:
                            policies[train_dict["policy_name"]] = PrimalDual(self.args, self.reward_man)
                        elif "ac" in train_dict["policy_name"]:
                            policies[train_dict["policy_name"]] = ActorCriticPolicy(self.args, self.reward_man)
                            policies[train_dict["policy_name"]]._actor_critic.eval()
                        elif train_dict["policy_name"] == "dqnatt":
                            policies[train_dict["policy_name"]] = DQNAttTrainer(self.args, self.reward_man)
                            policies[train_dict["policy_name"]]._dqn.eval()
                        elif "dqn_emb" in train_dict["policy_name"]:
                            policies[train_dict["policy_name"]] = DQNEmbTrainer(self.args, self.reward_man)
                            policies[train_dict["policy_name"]]._dqn.eval()
                        else:
                            raise Exception(f'Could not handle {train_dict["policy_name"]} policy!')
        return policies

    def reset(self):
        """Reset for next episode."""
        self.sim._reset()
        self._inv_dict = self._init_inv()

    def plot_results(self, eval_results: EvaluationResults):        
        # Plot the bar graphs
        fig, ax = plt.subplots(1)
        for i, policy_tuple in enumerate(eval_results.rewards_dict.items()):
            policy_name, rewards = policy_tuple
            avg_reward = sum(rewards) / len(rewards)
            ax.bar(policy_name, -1 *avg_reward, label=policy_name, zorder=3)
            print(f"{policy_name} Average Reward: ", avg_reward)
        # Add legend 
        ax.set(
            ylabel="Average Cost",
            title=f"Policy Results with {self.args.num_inv_nodes} Inventory Nodes and {self.args.num_skus} SKUs")

        # Add grid behind bars
        ax.grid(zorder=0)

        plt.show()

        # Plot episode averages
        x = np.arange(self.args.num_bar_ep)
        ep_list  = [[i] for i in range(min(self.args.num_bar_ep, self.args.eval_episodes))]
        columns = ["Episode"]
        for i, policy_tuple in enumerate(eval_results.ep_reward_avgs_dict.items()):
            policy_name, rewards = policy_tuple
            for j in range(len(ep_list)):
                ep_list[j].append(-1 * rewards[j])
            columns.append(policy_name)

        df = pd.DataFrame(ep_list, columns=columns)
        df.plot(
            x="Episode",
            kind="bar",
            stacked=False,
            title="Average Episode Cost")
        plt.ylabel("Average Cost")
        # plt.tight_layout()
        plt.show()

        # Compute
        print("\n")
        for i, policy_tuple in enumerate(eval_results.ep_reward_avgs_dict.items()):
            policy_name, rewards = policy_tuple
            print(f"{policy_name} Max Avg Episode Reward: {max(rewards)}")
            print(f"{policy_name} Min Avg Episode Reward: {min(rewards)}")
            print(f"{policy_name} Mean Avg Episode Reward: {np.mean(rewards)}")
            print(f"{policy_name} Std Avg Episode Reward: {np.std(rewards)}\n")

    def run(self):
        eval_results = EvaluationResults()
        for i in tqdm(range(self.args.eval_episodes)):
            # Generate demand nodes for this episode
            demand_nodes = self._gen_demand_nodes()
            for policy_name, policy in self._policies.items():

                # Run an episode
                ep_rewards = []

                for demand_node in demand_nodes:
                    # Get order results for policy
                    if "dqn" in policy_name:
                        policy_results = policy(self.sim._inv_nodes, demand_node, argmax=True)
                    else:
                        policy_results = policy(self.sim._inv_nodes, demand_node)

                    if self.visual:
                        self.visual.render_order(demand_node, policy_results, policy_name)

                    # Remove products in the order from inventory
                    self.sim.remove_products(policy_results)

                    # Save current episode reward results for this policy  
                    order_rewards = [exp.reward for exp in policy_results.exps]

                    eval_results.add_rewards(
                        policy_name,
                        order_rewards)
                    ep_rewards.extend(order_rewards)

                eval_results.add_ep_rewards(policy_name, ep_rewards)
                # Sanity-check to verify every item was fulfilled
                #

                # Restock the inv nodes
                self._restock_nodes()

                if self.visual:
                    self.visual.reset()
            
            # Reset for next eval episode
            if i + 1 < self.args.eval_episodes:
                self.reset()
        
        self.plot_results(eval_results)