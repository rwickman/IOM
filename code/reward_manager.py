from nodes import DemandNode, InventoryNode
from fulfillment_plan import FulfillmentPlan


class RewardManager:
    """Handles the reward."""
    def __init__(self, args):
        self.args = args
        # Reward scaling factor to make learning easier as large values take longer to converge
        self._reward_scale_factor = 1/((2 * self.args.coord_bounds) ** 2)
        print("self.args.coord_bounds", self.args.coord_bounds)
        print("max distance: ", (2 * self.args.coord_bounds) ** 2)
        self.prev_rewards = []
        self.wind_idx = 0
        self.reward_wind_sizes = 1024

    def get_reward(self,
                    inv_node: InventoryNode,
                    demand_node: DemandNode,
                    fulfill_plan: FulfillmentPlan):
        """Get current reward.
            NOTE: This is called before the fulfillment decision in a timestep.
        """
        inv_fulfill = fulfill_plan.get_fulfillment(inv_node.inv_node_id)
        dist = self._reward_scale_factor  * inv_node.loc.get_distance(demand_node.loc)
        
        if inv_fulfill:
            # this inventory node already has items routed to it
            return -dist * self.args.reward_alpha ** inv_fulfill.inv.inv_size
        else:
            # first item routed to an inventory node
            return -dist
        
    def scale_reward(self, reward):
        if len(self.prev_rewards) < self.reward_wind_sizes:
            self.prev_rewards.append(reward)
        else:
            self.prev_rewards[self.wind_idx] = reward
            self.wind_idx = (self.wind_idx + 1) % self.reward_wind_sizes


        if len(self.prev_rewards) > 1:
            return reward - sum(self.prev_rewards) / len(self.prev_rewards)
        else:
            return reward
