import unittest
import sys
import os

# Add code path to sys path
file_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(file_path, "../code")
sys.path.append(code_path)

from naive_policy import NaivePolicy
from reward_manager import RewardManager
from simulator import Cordinates, Location, InventoryProduct, InventoryNode, DemandNode
from fake_args import FakeArgs
        

class TestNaivePolicy(unittest.TestCase):
    def test_two_item(self):
        args = FakeArgs()
        
        reward_man = RewardManager(args)
        policy = NaivePolicy(args, reward_man)

        # Create fake inventory nodes
        loc = Location(Cordinates(0, 0))
        inv_node_id_1 = 0
        inv_prod_1 = InventoryProduct(0, 2)
        inv_prod_2 = InventoryProduct(1, 1)
        inv_prods = [inv_prod_1, inv_prod_2]
        inv_node_1 = InventoryNode(inv_prods, loc, inv_node_id_1)

        loc = Location(Cordinates(10, 10))
        inv_node_id_2 = 1
        inv_prod_2 = InventoryProduct(1, 2)
        inv_prods = [inv_prod_2]
        inv_node_2 = InventoryNode(inv_prods, loc, inv_node_id_2)

        inv_nodes = [inv_node_1, inv_node_2]

        # Create fake demand node
        loc = Location(Cordinates(2, 1))
        inv_prod_1 = InventoryProduct(0, 1)
        inv_prod_2 = InventoryProduct(1, 2)
        inv_prods = [inv_prod_1, inv_prod_2]
        demand_node = DemandNode(inv_prods, loc)
        
        # Test distance
        exp_dist_1 = 2.23606797749979
        exp_dist_2 = 12.041594578792296
        self.assertEqual(demand_node.loc.get_distance(inv_node_1.loc), exp_dist_1)
        self.assertEqual(demand_node.loc.get_distance(inv_node_2.loc), exp_dist_2)

        # Test rewards are correct
        policy_results = policy(inv_nodes, demand_node)
        self.assertEqual(policy_results.rewards[0], -exp_dist_1)
        self.assertEqual(policy_results.rewards[1], -exp_dist_1 * args.reward_alpha)
        self.assertEqual(policy_results.rewards[2], -exp_dist_2)

        # Test the fulfillmemt quantities are correct
        exp_node_0_sku_0_quant = 1
        exp_node_0_sku_1_quant = 1
        exp_node_1_sku_0_quant = 0
        exp_node_1_sku_1_quant = 1
        
        fulfill = policy_results.fulfill_plan.get_fulfillment(inv_node_id_1)
        self.assertEqual(fulfill.inv_node_id, inv_node_id_1)
        self.assertEqual(fulfill.inv.product_quantity(0), exp_node_0_sku_0_quant)
        self.assertEqual(fulfill.inv.product_quantity(1), exp_node_0_sku_1_quant)
        
        fulfill = policy_results.fulfill_plan.get_fulfillment(inv_node_id_2)
        self.assertEqual(fulfill.inv_node_id, inv_node_id_2)
        self.assertEqual(fulfill.inv.product_quantity(0), exp_node_1_sku_0_quant)
        self.assertEqual(fulfill.inv.product_quantity(1), exp_node_1_sku_1_quant)


if __name__ == '__main__':
    unittest.main()