
from abc import ABC, abstractmethod
import torch

from nodes import InventoryNode, DemandNode, InventoryProduct
from fulfillment_plan import FulfillmentPlan
from policy import PolicyResults, Policy, Experience
from reward_manager import RewardManager
from config import device


class RLPolicy(Policy, ABC):
    """Superclass for all RLPolcies"""
    def __init__(self, args, reward_man: RewardManager):
        super().__init__(args, True, load_args=True)
        self._reward_man = reward_man
        # Inventory nodes + demand left + currently allocated items + one-hot encoding for item to allocate + locations coordinates
        self.inp_size = self.args.num_inv_nodes * self.args.num_skus + \
                    self.args.num_skus + \
                    self.args.num_inv_nodes * self.args.num_skus + \
                    self.args.num_skus + \
                    self.args.num_inv_nodes * 2  + 2

        self._hyper_dict  = {
            "max_inv_prod" : self.args.max_inv_prod,
            "coord_bounds" : self.args.coord_bounds
        }

        self._train_step = 0

    @abstractmethod
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Make a prediction on the state using DNN."""
        pass

    @abstractmethod
    def sample_action(self, model_pred: torch.Tensor, argmax=False) -> int:
        """Sample an action."""
        pass

    @abstractmethod
    def is_train_ready(self) -> bool:
        """Returns if model is ready to be trained."""
        pass

    @abstractmethod
    def train(self) -> float:
        """Train the model."""
        pass

    def _create_state(self,
                inv: torch.Tensor,
                inv_locs: torch.Tensor,
                demand: torch.Tensor,
                demand_loc: torch.Tensor,
                cur_fulfill: torch.Tensor,
                item_hot: torch.Tensor,
                sku_distr) -> torch.Tensor:
        """Create the normalized state from the given state elements.
        
        
        Args:
            inv: tensor which contains rows for inventory nodes and columsn for SKU quantity
            inv_locs: tensor which contains 
        """
        
        # Concatenate together and scale elements
        return torch.cat((
            inv.flatten() / self._hyper_dict["max_inv_prod"],
            inv_locs.flatten() / self._hyper_dict["coord_bounds"],
            demand / self._hyper_dict["max_inv_prod"],
            demand_loc / self._hyper_dict["coord_bounds"],
            cur_fulfill.flatten() / self._hyper_dict["max_inv_prod"],
            item_hot))

    def _get_valid_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Get valid actions for the current state vector."""
        print("GETTING VALID ACTIONS IN RL POLICY")
        # Derive the inventory and SKU ID from the state vector 
        inv = state[:self.args.num_inv_nodes * self.args.num_skus].reshape(self.args.num_inv_nodes, self.args.num_skus)
        sku_id = int(state[-self.args.num_skus:].nonzero())
        valid_actions = (inv[:, sku_id] > 0).nonzero().flatten()

        return valid_actions

    def __call__(self,
                inv_nodes: list,
                demand_node: DemandNode,
                sku_distr: torch.Tensor,
                argmax=False) -> PolicyResults:
        """Create a fulfillment decision for the DemandNode using a RL based policy.
        
        Args:
            list of InventoryNodes.
            demand_node: the DemandNode representing the current order.
        
        Returns:
            the fulfillment decision results.
        """
        run_expert = self._train_step < self.args.expert_pretrain
        if run_expert:
            expert_plan = self._expert_policy(inv_nodes, demand_node)
        
        inv_locs = torch.zeros(self.args.num_inv_nodes, 2).to(device)

        inv = torch.zeros(self.args.num_inv_nodes, self.args.num_skus).to(device)

        # Create current inventory vector
        for inv_node in inv_nodes:
            inv_locs[inv_node.inv_node_id, 0] = inv_node.loc.coords.x
            inv_locs[inv_node.inv_node_id, 1] = inv_node.loc.coords.y
            inv[inv_node.inv_node_id] = torch.tensor(inv_node.inv.inv_list).to(device)
            # for sku_id in range(self.args.num_skus):
            #     inv[inv_node.inv_node_id, sku_id] = inv_node.inv.product_quantity(sku_id)

        # Create demand vector
        demand_loc = torch.zeros(2).to(device)
        demand_loc[0] = demand_node.loc.coords.x
        demand_loc[1] = demand_node.loc.coords.y
        # demand = torch.zeros(self.args.num_skus).to(device)
        demand = torch.tensor(demand_node.inv.inv_list).to(device)
        # for sku_id in range(self.args.num_skus):
        #     demand[sku_id] = demand_node.inv.product_quantity(sku_id)
   
        # Keep up with fulfillment requests for every inventory node
        fulfill_plan = FulfillmentPlan()

        # Make item fulfillment actions
        cur_fulfill = torch.zeros(self.args.num_inv_nodes, self.args.num_skus).to(device)
        exps  = []
        for i, inv_prod in enumerate(demand_node.inv.items()):
            for j in range(inv_prod.quantity):
                # Create one hot encoded vector for the current item selection
                item_hot = torch.zeros(self.args.num_skus).to(device)
                item_hot[inv_prod.sku_id] = 1

                # Create the current state
                state = self._create_state(
                    inv,
                    inv_locs,
                    demand,
                    demand_loc,
                    cur_fulfill,
                    item_hot,
                    sku_distr)
                                
                if i != 0 or j != 0:
                    # Update the previous experience next state
                    exps[-1].next_state = state

                # Run through policy
                model_pred = self.predict(state)

                # Get indices of nodes that have nonzero inventory
                valid_idxs = self._get_valid_actions(state)

                # Select an inventory node
                if run_expert:
                    # Sample an action according to expert
                    action = expert_plan.exps[len(exps)].action
                else:
                    # Sample action using RL policy
                    action = self.sample_action(model_pred.squeeze()[valid_idxs], argmax)
                    action = int(valid_idxs[action])
                
                # Get the reward for this timestep                
                reward = self._reward_man.get_reward(
                                inv_nodes[action],
                                demand_node,
                                fulfill_plan)

                # Add experience
                exps.append(
                    Experience(state, action, reward, is_expert=run_expert))

                # Increase items fulfilled at this location
                cur_fulfill[action, inv_prod.sku_id] += 1
                fulfill_plan.add_product(action, InventoryProduct(inv_prod.sku_id, 1))

                # Decrease inventory at node
                inv[action, inv_prod.sku_id] -= 1
                demand[inv_prod.sku_id] -= 1

        # Create the results from the order
        results = PolicyResults(fulfill_plan, exps)

        return results