import json, os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from config import device
from argparse import Namespace

from simulator import Simulator  
from nodes import  DemandNode
from reward_manager import RewardManager
from naive_policy import NaivePolicy
from random_policy import RandomPolicy
from dqn_policy import DQNTrainer
from dqn_emb_policy import DQNEmbTrainer
from value_lookhead_policy import ValueLookaheadPolicy
from value_lookhead_emb_policy import ValueLookaheadEmbPolicy

from dqn_lookhead_policy import DQNLookaheadTrainer
from primal_dual_policy import PrimalDual
from actor_critic_policy import ActorCriticPolicy
from visual import Visual
from dataset_simulator import DatasetSimulator

sns.set(style="darkgrid", font_scale=1.5)

class EvaluationResults:
    """Stores evaluation results for all the policies."""
    def __init__(self):
        # For all rewards
        self.rewards_dict = {}
        
        # For reward averages across episodes
        self.ep_reward_avgs_dict = {}

    def add_rewards(self, policy_name: str, rewards: list):
        """Add rewards.
        
        Args:
            policy_name: the policy that produced the rewards.
            rewards: the rewards from a fulfillment decision.
        """
        if policy_name not in self.rewards_dict:
            self.rewards_dict[policy_name] = []

        self.rewards_dict[policy_name].append(rewards)

    def add_ep_rewards(self, policy_name: str, ep_rewards: list):
        """Add rewards from an episode.
        
        Args:
            policy_name: the policy that produced the rewards.
            ep_rewards: the rewards from the episode.
        """
        if policy_name not in self.ep_reward_avgs_dict:
            self.ep_reward_avgs_dict[policy_name] = []
        self.ep_reward_avgs_dict[policy_name].append(sum(ep_rewards) / len(ep_rewards))


class Evaluator:
    def __init__(self,
                args,
                reward_man: RewardManager,
                sim: Simulator,
                dataset_sim: DatasetSimulator = None,
                visual: Visual = None):

        self.args = args
        self.reward_man = reward_man    
        self.sim = sim
        self.visual = visual
        self.dataset_sim = dataset_sim

        # Remove reward scaling
        #self.reward_man._reward_scale_factor = 1

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

    def _gen_demand_nodes(self) -> list:
        """Generate list of demand nodes for an evaluation episode."""
        print("Generating demand nodes.")
        stock = self.sim._inv_node_man.stock

        # Get non-zero items
        stock = [item for item in stock if item.quantity > 0] 

        demand_nodes = []
        
        while len(stock) > 0:
            if self.args.eval_order_max and len(demand_nodes) >= self.args.eval_order_max:
                break
            
            if self.dataset_sim is not None:
                inv_list = [0] * self.args.num_skus
                for item in stock:
                    inv_list[item.sku_id] = item.quantity
                demand_node = self.dataset_sim.gen_demand_node(inv_list)
            else:
                demand_node = self.sim._gen_demand_node(stock)
            demand_nodes.append(demand_node)
            # TODO: FIX THIS TO USE DatasetSimulator
            ## Will have to fix gen_demand_node in DS because it updates the demand probs along side

            # Update stock
            was_removed = False
            for inv_prod in demand_node.inv.items():
                if inv_prod.quantity > 0:
                    # Remove the quantity from stock 
                    for item in stock:
                        if item.sku_id == inv_prod.sku_id:
                            was_removed = True
                            item.quantity -= inv_prod.quantity
                            # sanity-check
                            assert inv_prod.quantity >= 0 

                            # Remove the item from the stock
                            if item.quantity <= 0:
                                stock.remove(item)
            assert was_removed
        #self.dataset_sim.init_sku_distr(stock)
        print("Done generating demand nodes.")
        return demand_nodes

    def _load_policies(self) -> list:
        """Load all the policies for evaluation.
        
        Returns:
            a dict str names to Policy objects.
        """
        policies = {}
        if not self.args.no_naive_fulfill_eval:
            policies["naive"] = NaivePolicy(self.args, self.reward_man)
        if not self.args.no_rand_fulfill_eval:
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
                        elif "dqn_emb" in train_dict["policy_name"]:
                            policies[train_dict["policy_name"]] = DQNEmbTrainer(self.args, self.reward_man)
                            policies[train_dict["policy_name"]]._dqn.eval()
                        elif "val_lookahead_emb" in train_dict["policy_name"].lower():
                            print("USING VALUE LOOKAHEAD")
                            policies[train_dict["policy_name"]] = ValueLookaheadEmbPolicy(self.args, self.reward_man)
                            policies[train_dict["policy_name"]]._val_model                        

                            args =  Namespace(**vars(self.args))
                            args.gamma = 0.0
                        
                            policies[train_dict["policy_name"]+ "_no_gamma"] = ValueLookaheadEmbPolicy(args, self.reward_man)
                            policies[train_dict["policy_name"]+ "_no_gamma"]._val_model.eval()

                        elif "val_lookahead" in train_dict["policy_name"].lower():
                            print("USING VALUE LOOKAHEAD")
                            policies[train_dict["policy_name"]] = ValueLookaheadPolicy(self.args, self.reward_man)
                            policies[train_dict["policy_name"]]._val_model.eval()
                            
                            args =  Namespace(**vars(self.args))
                            args.gamma = 0.0
                        
                            policies[train_dict["policy_name"]+ "_no_gamma"] = ValueLookaheadPolicy(args, self.reward_man)
                            policies[train_dict["policy_name"]+ "_no_gamma"]._val_model.eval()
                        elif "lookahead" in train_dict["policy_name"].lower():
                            print("USING LOOKAHEAD")
                            policies[train_dict["policy_name"]] = DQNLookaheadTrainer(self.args, self.reward_man)
                            policies[train_dict["policy_name"]]._dqn.eval()
                            args =  Namespace(**vars(self.args))
                            args.gamma = 0.0
                            policies[train_dict["policy_name"] + "_no_gamma"] = DQNLookaheadTrainer(args, self.reward_man)
                            policies[train_dict["policy_name"] + "_no_gamma"]._dqn.eval()
                        
                        else:
                            raise Exception(f'Could not handle {train_dict["policy_name"]} policy!')
        return policies

    def reset(self):
        """Reset for next episode."""
        self.sim._reset()
        self._inv_dict = self._init_inv()

    def plot_results(self, eval_results: EvaluationResults):
        """Plot the evaluation results.
        
        Args:
            eval_results: the evaluation results.
        """        
        # Plot the bar graphs
        fig, ax = plt.subplots(1)
        for i, policy_tuple in enumerate(eval_results.rewards_dict.items()):
            policy_name, rewards = policy_tuple
            avg_reward = sum(rewards) / len(rewards)
            ret = 0

            for i, reward in enumerate(rewards):

                ret += reward * self.args.gamma ** i 
            print("RETURN: ", ret)
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
        """Evaluate the policies."""

        eval_results = EvaluationResults()
        for i in tqdm(range(self.args.eval_episodes)):
            # Generate demand nodes for this episode
            demand_nodes = self._gen_demand_nodes()
            sku_distrs = []
            cur_policy_i = 0
            for policy_name, policy in self._policies.items():
                print("policy_name", policy_name)

                # Run an episode
                ep_rewards = []

                for j, demand_node in enumerate(demand_nodes):
                    # Get order results for policy
                    if "dqn" in policy_name or "lookahead" in policy_name:
                        if self.dataset_sim is not None:
                            # self.dataset_sim.init_sku_distr(self.sim._inv_node_man.stock)
                            #print("self.dataset_sim.cur_sku_distr", self.dataset_sim.cur_sku_distr.max(), self.dataset_sim.cur_sku_distr.min())
                            if j + 1 < len(sku_distrs):
                                next_sku_distr = sku_distrs[j+1]
                                
                                # print("next_sku_distr", next_sku_distr, next_sku_distr.max(), next_sku_distr.min(), next_sku_distr.mean(), next_sku_distr.sum())
                            else:
                                next_sku_distr = torch.zeros_like(sku_distrs[j])
                                

                            policy_results = policy(self.sim._inv_nodes, demand_node, sku_distrs[j], next_sku_distr, argmax=True)
                        else:
                            policy_results = policy(self.sim._inv_nodes, demand_node, torch.tensor([1/self.args.num_skus]).repeat(self.args.num_skus).to(device), argmax=True)
                    else:
                        if cur_policy_i == 0:
                            self.dataset_sim.init_sku_distr(self.sim._inv_node_man.stock)
                            sku_distrs.append(self.dataset_sim.cur_sku_distr.float().clone())

                        policy_results = policy(self.sim._inv_nodes, demand_node)

                    if self.visual:
                        self.visual.render_order(demand_node, policy_results, policy_name)

                    # Remove products in the order from inventory
                    self.sim.remove_products(policy_results)

                    # Save current episode reward results for this policy
                    order_reward = sum([exp.reward for exp in policy_results.exps])

                    eval_results.add_rewards(
                        policy_name,
                        order_reward)
                    ep_rewards.append(order_reward)

                eval_results.add_ep_rewards(policy_name, ep_rewards)
                # Sanity-check to verify every item was fulfilled
                #

                # Restock the inv nodes
                self._restock_nodes()
                cur_policy_i += 1

                if self.visual:
                    self.visual.reset()
            
            # Reset for next eval episode
            if i + 1 < self.args.eval_episodes:
                self.reset()
        
        self.plot_results(eval_results)