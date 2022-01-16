import random

from nodes import InventoryNode, DemandNode, InventoryProduct
from fulfillment_plan import FulfillmentPlan
from policy import PolicyResults, Policy, Experience

class RandomPolicy(Policy):
    """Policy for randomly fulfilling orders."""
    def __init__(self, args, reward_man):
        super().__init__(args, is_trainable=False)
        self._reward_man = reward_man
    
    def __call__(self,
                inv_nodes: list,
                demand_node: DemandNode) -> PolicyResults:
        """Create a fulfillment decision for the DemandNode using a random policy.
        
        Args:
            list of InventoryNodes.
            deamnd_node: the DeamndNode representing the current order.
        
        Returns:
            the fulfillment decision results.
        """
        # Keep up with fulfillment requests for every inventory node
        fulfill_plan = FulfillmentPlan()
        exps = []

        for inv_prod in demand_node.inv.items():
            for _ in range(inv_prod.quantity):
                potential_inv_nodes = []
                
                # Evaluate cost of fulfilling to each inventory node if available
                for inv_node in inv_nodes:
                    # Number of products of this type at node
                    cur_quant = inv_node.inv.product_quantity(inv_prod.sku_id)
                    # Get all inventory nodes with some stock still left

                    # Amount currently allocated to node
                    fulfill_quant = fulfill_plan.fulfill_quantity(
                        inv_node.inv_node_id,
                        inv_prod.sku_id)
                    

                    # Check if you some amount can still be allocated to this inventory node
                    if cur_quant - fulfill_quant > 0:
                        potential_inv_nodes.append(inv_node.inv_node_id)
                
                rand_inv_node_id = random.sample(potential_inv_nodes, 1)[0]

                # Create the demand for one product with SKU at greedy best
                rand_inv_prod = InventoryProduct(inv_prod.sku_id, 1)
                
                # Fulfill order
                cur_reward = self._reward_man.get_reward(
                            inv_nodes[rand_inv_node_id],
                            demand_node,
                            fulfill_plan)

                fulfill_plan.add_product(rand_inv_node_id, rand_inv_prod)
                
                exps.append(
                    Experience(None, rand_inv_node_id, cur_reward))
        
        return PolicyResults(fulfill_plan, exps)