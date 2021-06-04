from simulator import InventoryNode, DemandNode, InventoryProduct
from fulfillment_plan import FulfillmentPlan
from policy import PolicyResults

class NaivePolicy:
    def __init__(self, args, reward_man):
        self.args = args
        self._reward_man = reward_man

    def __call__(self,
                inv_nodes: list[InventoryNode],
                demand_node: list[DemandNode]) -> FulfillmentPlan:
        
        # Keep up with what products have already been allocated
        fulfill_dict = {}

        # Keep up with fulfillment requests for every inventory node
        fulfill_plan = FulfillmentPlan()
        rewards = []

        for _ in range(demand_node.inv.inv_size):
            max_inv_node_id = max_reward = max_sku_id = None

            for inv_prod in demand_node.inv.items():
            
                if inv_prod.sku_id not in fulfill_dict:
                    fulfill_dict[inv_prod.sku_id] = 0
                
                # Check if there are products of this type that still need to be allocated
                if inv_prod.quantity - fulfill_dict[inv_prod.sku_id] > 0:
                    
                    # Evaluate cost of fulfilling to each inventory node if available
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
                                fulfill_plan,
                                inv_node.inv_node_id,
                                inv_prod)

                            if max_reward is None or cur_reward > max_reward:
                                max_reward = cur_reward
                                max_inv_node_id = inv_node.inv_node_id
                                max_sku_id = inv_prod.sku_id

            # Create the demand for one product with SKU at greedy best
            max_inv_prod = InventoryProduct(max_sku_id, 1)
            #print("NAIVE POLICY", max_inv_node_id, max_inv_prod)
            
            fulfill_plan.add_product(max_inv_node_id, max_inv_prod)

            fulfill_dict[max_sku_id] += 1
            rewards.append(max_reward)

        
        return PolicyResults(fulfill_plan, rewards)


                    





                
                

            
            
            
            