"""
Implementation of a primal–dual algorithm for online order 
fulfillment with multiproduct orders from the paper
Primal–Dual Algorithms for Order Fulfillment at Urban Outfitters, Inc.
"""
import os
import torch

from reward_manager import RewardManager
from fulfillment_plan import FulfillmentPlan
from policy import PolicyResults, Policy, Experience
from nodes import DemandNode, InventoryProduct

class PrimalDual(Policy):
    def __init__(self, args, reward_man: RewardManager, is_expert=False):
        super().__init__(args, True)
        self._reward_man = reward_man
        self._is_expert = is_expert

        if self._is_expert:
            self._model_file = os.path.join(self.args.expert_dir, "dual_lams.pt")
            self.load()
        else:
            self._model_file = os.path.join(self.args.save_dir, "dual_lams.pt")
            if self.args.load:
                self.load()
            else:
                # Initialize dual variables
                self._dual_lams = torch.zeros((self.args.num_inv_nodes, self.args.num_skus))
            
        # Hyperparameters defined in the paper
        self._alpha_1 = 1 / (1 + torch.log(torch.tensor(self.args.kappa)))
        self._alpha_2 = 0
        self._beta_bar = max(self.args.min_inv_prod, 1)
        self._beta = self.args.kappa / ( (1 + 1 / self._beta_bar) ** (self._beta_bar/self._alpha_1)  - 1)
        self._init_inv = None
        self._exps = []

    def __call__(self, 
            inv_nodes: list,
            demand_node: DemandNode) -> PolicyResults:
        """Create a fulfillment decision for the DemandNode using the primal-dual policy.
        
        Args:
            list of InventoryNodes.
            deamnd_node: the DeamndNode representing the current order.
        
        Returns:
            the fulfillment decision results.
        """
        # Add initial inventory
        if self._init_inv is None:
            self._init_inv = torch.zeros((self.args.num_inv_nodes, self.args.num_skus))
            # Create initial inventory vector
            for inv_node in inv_nodes:
                for sku_id in range(self.args.num_skus):
                    self._init_inv[inv_node.inv_node_id, sku_id] = inv_node.inv.product_quantity(sku_id)

        # Store experiences per timestep
        exps = []

        # Keep up with fulfillment requests for every inventory node
        fulfill_plan = FulfillmentPlan()
        
        for inv_prod in demand_node.inv.items():
            for _ in range(inv_prod.quantity):

                # Attempt to allocate item to every inventory node
                best_pd_obj_val = None
                best_inv_node_id = None
                best_reward = None
                for inv_node in inv_nodes:
                    # Number of products of this type at node
                    cur_quant = inv_node.inv.product_quantity(inv_prod.sku_id)

                    # Amount currently allocated to node
                    fulfill_quant = fulfill_plan.fulfill_quantity(
                        inv_node.inv_node_id,
                        inv_prod.sku_id)

                    # Check if you some amount can still be allocated to this inventory node
                    if cur_quant - fulfill_quant > 0:
                        cur_reward = self._reward_man.get_reward(
                            inv_node,
                            demand_node,
                            fulfill_plan)

                        # Primal-dual objective value
                        pd_obj_val = cur_reward - self._dual_lams[inv_node.inv_node_id, inv_prod.sku_id]

                        # Check if should update greedy best decision
                        if best_pd_obj_val is None or pd_obj_val > best_pd_obj_val:
                            best_pd_obj_val = pd_obj_val
                            best_inv_node_id = inv_node.inv_node_id
                            best_reward = cur_reward

                # Check if none of the inventory nodes had inventory for this item 
                if best_pd_obj_val is None:
                    raise Exception("Invalid inventory for demand.")

                # Create the demand for one product with SKU at greedy best
                best_inv_prod = InventoryProduct(inv_prod.sku_id, 1)

                # Fulfill order
                fulfill_plan.add_product(best_inv_node_id, best_inv_prod)

                # Create experience, using SKU ID as state
                exps.append(
                    Experience(inv_prod.sku_id, best_inv_node_id, best_reward))

        if not self._is_expert and not self.args.eval:
            self._exps.extend(exps)
         
        return PolicyResults(fulfill_plan, exps)

    def train(self):
        """Update the dual lambda values."""
        for exp in self._exps:
            denom = self._alpha_1 * max(self._init_inv[exp.action, exp.state], 1) + self._alpha_2
            
            term_1 = self._dual_lams[exp.action, exp.state] * (1 + 1/denom)
            term_2 = self._beta * (exp.reward / denom)
            self._dual_lams[exp.action, exp.state] = term_1 + term_2

        # Reset the experiences
        self._exps = []                   
        self._init_inv = None    

    def is_train_ready(self) -> bool:
        """Check if variables are ready to get updated."""
        return len(self._exps) > 0

    def save(self):
        """Save the dual variables."""
        model_dict = {
            "dual_lams" : self._dual_lams
        }
        torch.save(model_dict, self._model_file)
        
    def load(self):
        """Load the dual variables."""
        model_dict = torch.load(self._model_file)
        self._dual_lams = model_dict["dual_lams"] 

    def reset(self):
        pass
