import os, sys, unittest

# Add code path to sys path
file_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(file_path, "../code")
sys.path.append(code_path)

from reward_manager import RewardManager
from evaluator import Evaluator
from simulator import Simulator, InventoryProduct, DemandNode, Location, Coordinates
from fake_args import FakeArgs

class FakeGenDemandNode:
    def __init__(self):
        self.num_calls = 0
        self.inv_prods = [
            [InventoryProduct(0,1)],
            [InventoryProduct(0,1), InventoryProduct(1,1)],
            [InventoryProduct(1,1)]
        ]
    def __call__(self, stock): 
        loc = Location(
            Coordinates(self.num_calls,self.num_calls))

        if self.num_calls < len(self.inv_prods):
            demand_node = DemandNode(
                self.inv_prods[self.num_calls],
                loc)
        else:
            demand_node = DemandNode(
                [],
                loc)
            

        self.num_calls += 1
        return demand_node



class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.args = FakeArgs()
        self.reward_man = RewardManager(self.args)
        self.sim = Simulator(self.args)
        self.eval = Evaluator(self.args, self.reward_man, self.sim)

    def test_init_inv(self):        
        expected_inv_dict = {
            0 : [InventoryProduct(0, 1), InventoryProduct(1, 1)],
            1: [InventoryProduct(0, 1), InventoryProduct(1, 1)]
        }
        inv_dict = self.eval._init_inv()

        self.assertDictEqual(inv_dict, expected_inv_dict)

    def test_restock_nodes(self):
        expected_inv_dict = {
            0 : [InventoryProduct(0, 1), InventoryProduct(1, 1)],
            1: [InventoryProduct(0, 1), InventoryProduct(1, 1)]
        }

        self.eval._restock_nodes()
        self.assertDictEqual(self.eval._inv_dict, expected_inv_dict)

    def test_gen_demand_nodes(self):
        fake_gen = FakeGenDemandNode()

        self.sim._gen_demand_node = fake_gen

        demand_nodes = self.eval._gen_demand_nodes()
        self.assertEqual(len(demand_nodes), 3)

        # Verify all the demand nodes are correct
        for i in range(len(demand_nodes)):
            for j, inv_prod in enumerate(demand_nodes[i].inv.items()):
                self.assertGreaterEqual(len(fake_gen.inv_prods[i]), j)
                self.assertEqual(inv_prod, fake_gen.inv_prods[i][j])