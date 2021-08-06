import os, sys, unittest 

# Add code path to sys path
file_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(file_path, "../code")
sys.path.append(code_path)

from policy import PolicyResults
from reward_manager import RewardManager
from simulator import *
from fake_args import FakeArgs
from fake_ep import FakeEpisode


class FakePolicy:
    def __init__(self, args, reward_man: RewardManager):
        self.args = args
        self._reward_man = reward_man
    
    def __call__(self, 
                inv_nodes: list[InventoryNode],
                demand_node: DemandNode) -> PolicyResults:
        pass

class TestSimulator(unittest.TestCase):

    def setUp(self):
        self.args = FakeArgs()
        self.sim = Simulator(self.args, None)

    def test_gen_inv_node(self):
        fake_coords = Coordinates(0, 0)
        fake_loc = Location(fake_coords)
        expected_id = len(self.sim._inv_nodes)
        inv_node = self.sim._gen_inv_node(fake_loc)

        # Verify ID is correct 
        self.assertEqual(inv_node.inv_node_id, expected_id)

        # Verify it was given stock
        for inv_prod in inv_node.inv.items():
            self.assertEqual(inv_prod.quantity, 1)

    def test_rand_loc(self):
        loc = self.sim._rand_loc()
        self.assertTrue(loc.coords.x >= -self.args.coord_bounds and loc.coords.x <= self.args.coord_bounds)
        self.assertTrue(loc.coords.y >= -self.args.coord_bounds and loc.coords.y <= self.args.coord_bounds)

        self.args.coord_bounds = 0
        loc = self.sim._rand_loc()
        self.assertEqual(loc.coords.x, 0)
        self.assertEqual(loc.coords.y, 0)

    def test_gen_inv_node_stock(self):
        inv_prods = self.sim._gen_inv_node_stock(self.args.max_inv_prod)

        # Check inventory created for every item
        self.assertEqual(len(inv_prods), self.args.num_skus)
        inv_prods = sorted(inv_prods, key=lambda x: x.sku_id)
        # Verify correct quantity was created
        for sku_id, inv_prod in enumerate(inv_prods):
            self.assertEqual(inv_prod.sku_id, sku_id)
            self.assertEqual(inv_prod.quantity, 1)

    def test_restock_inv(self):
        # First empty stock
        self.sim._inv_node_man.empty()
        self.assertEqual(self.sim._inv_node_man.inv.inv_size, 0)

        # Then restock and verify it is correct
        self.sim._restock_inv()
        self.assertTrue(self.sim._inv_node_man.inv.inv_size > 0)
        # Assumes min_inv_prod = max_inv_prod = 1
        for inv_node in self.sim._inv_nodes:
            for inv_prod in inv_node.inv.items():
                self.assertEqual(inv_prod.quantity, 1)


