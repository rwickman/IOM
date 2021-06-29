
from abc import ABC, abstractmethod
import torch

from simulator import InventoryNode, DemandNode, InventoryProduct
from fulfillment_plan import FulfillmentPlan
from policy import PolicyResults, Policy, Experience

class RLPolicy(Policy, ABC):
    def __init__(self, args):
        super().__init__(args, True)

        # Inventory nodes + demand left + currently allocated items + one-hot encoding for item to allocate + locations coordinates
        self.inp_size = self.args.num_inv_nodes * self.args.num_skus + \
                    self.args.num_skus + \
                    self.args.num_inv_nodes * self.args.num_skus + \
                    self.args.num_skus + \
                    self.args.num_inv_nodes * 2  + 2


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


    def _get_valid_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Get valid actions for the current state vector."""
        #Derive the inventory and SKU ID from the state vector 
        inv = state[:self.args.num_inv_nodes * self.args.num_skus].reshape(self.args.num_inv_nodes, self.args.num_skus)
        sku_id = int(state[-self.args.num_skus:].nonzero())

        valid_actions = (inv[:, sku_id] > 0).nonzero().flatten()
        return valid_actions 

    def __call__(self,
                inv_nodes: list[InventoryNode],
                demand_node: DemandNode,
                argmax=False) -> PolicyResults:
        inv_locs = torch.zeros(self.args.num_inv_nodes, 2)

        inv = torch.zeros(self.args.num_inv_nodes, self.args.num_skus)

        # Create current inventory vector
        for inv_node in inv_nodes:
            inv_locs[inv_node.inv_node_id, 0] = inv_node.loc.coords.x
            inv_locs[inv_node.inv_node_id, 1] = inv_node.loc.coords.y
            for sku_id in range(self.args.num_skus):
                inv[inv_node.inv_node_id, sku_id] = inv_node.inv.product_quantity(sku_id)

        # Create demand vector
        demand_loc = torch.zeros(2)
        demand_loc[0] = demand_node.loc.coords.x
        demand_loc[1] = demand_node.loc.coords.y
        demand = torch.zeros(self.args.num_skus)
        for sku_id in range(self.args.num_skus):
            demand[sku_id] = demand_node.inv.product_quantity(sku_id)
   
        # Keep up with fulfillment requests for every inventory node
        fulfill_plan = FulfillmentPlan()

        # Make item fulfillment actions
        cur_fulfill = torch.zeros(self.args.num_inv_nodes, self.args.num_skus)
        exps  = []
        for i, inv_prod in enumerate(demand_node.inv.items()):
            for j in range(inv_prod.quantity):
                # Create one hot encoded vector for the current item selection
                item_hot = torch.zeros(self.args.num_skus)
                item_hot[inv_prod.sku_id] = 1

                # Create the current state
                state = torch.cat(
                    (inv.flatten(), inv_locs.flatten(), demand, demand_loc, cur_fulfill.flatten(), item_hot))

                if i != 0 or j != 0:
                    # Update the previous experience next state
                    exps[-1].next_state = state

                # Run through policy
                model_pred = self.predict(state)

                # Get indicies of nodes that have nonzero inventory
                valid_idxs = (inv[:, inv_prod.sku_id] > 0).nonzero().flatten()

                # Select an inventory node
                action = self.sample_action(model_pred[valid_idxs], argmax)
                action = int(valid_idxs[action])

                reward = self._reward_man.get_reward(
                                inv_nodes[action],
                                demand_node,
                                fulfill_plan)

                # Add experience
                exps.append(
                    Experience(state.clone(), action, reward))

                # Increase items fulfilled at this location
                cur_fulfill[action, inv_prod.sku_id] += 1
                fulfill_plan.add_product(action, InventoryProduct(inv_prod.sku_id, 1))

                # Decrease inventory at node
                inv[action, inv_prod.sku_id] -= 1
                demand[inv_prod.sku_id] -= 1

        # Create the results from the order
        results = PolicyResults(fulfill_plan, exps)

        return results