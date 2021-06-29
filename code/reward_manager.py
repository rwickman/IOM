from simulator import DemandNode, InventoryNode, InventoryProduct
from fulfillment_plan import FulfillmentPlan

class RewardManager:
    def __init__(self, args):
        self.args = args

    def get_reward(self,
                    inv_node: InventoryNode,
                    demand_node: DemandNode,
                    fulfill_plan: FulfillmentPlan):
        """Get current reward.
            NOTE: This is called before the fulfillment decision in a timestep.
        """
        inv_fulfill = fulfill_plan.get_fulfillment(inv_node.inv_node_id)
        dist = inv_node.loc.get_distance(demand_node.loc)

        if inv_fulfill:
            return -dist * self.args.reward_alpha ** inv_fulfill.inv.inv_size
        else:
            return -dist