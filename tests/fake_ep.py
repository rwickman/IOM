import torch, os, sys
# Add code path to sys path
file_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(file_path, "../code")
sys.path.append(code_path)

from policy import Experience
from simulator import Cordinates, Location, InventoryProduct, InventoryNode, DemandNode
from fake_args import FakeArgs

class FakeEpisode:
    def __init__(self):
        self.args = FakeArgs()

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
        self.inv_nodes = [inv_node_1, inv_node_2]

        # Create fake demand node
        loc = Location(Cordinates(2, 1))
        inv_prod_1 = InventoryProduct(0, 1)
        inv_prod_2 = InventoryProduct(1, 2)
        inv_prods = [inv_prod_1, inv_prod_2]
        self.demand_node = DemandNode(inv_prods, loc)

        # Distance b/w demand node and inventory node 1
        self.dist_1 = 2.23606797749979
        # Distance b/w demand node and inventory node 2
        self.dist_2 = 12.041594578792296

        # Create the rewards
        self.rewards = [
            -self.dist_1,
            -self.dist_1 * self.args.reward_alpha,
            -self.dist_2
        ]

        # Create the states
        inv = [2, 1, 0, 2]
        demand = [1, 2]
        cur_fulfill = [0,0,0,0]
        item_hot = [1,0]
        state_1 = torch.tensor(inv + demand + cur_fulfill + item_hot)

        inv = [1, 1, 0, 2]
        demand = [0,2]
        cur_fulfill = [1, 0, 0, 0]
        item_hot = [0, 1]
        state_2 = torch.tensor(inv + demand + cur_fulfill + item_hot)


        inv = [1, 0, 0, 2]
        demand = [0,1]
        cur_fulfill = [1, 1, 0, 0]
        item_hot = [0, 1]
        state_3 = torch.tensor(inv + demand + cur_fulfill + item_hot)

        self.states = [state_1, state_2, state_3]

        # Create the actions
        self.actions = [0, 0, 1]

        self.sku_ids = [0, 1, 1]

        # Create fake experiences
        self.exps = []
        for i in range(len(self.states)):
            self.exps.append(
                Experience(
                    self.states[i],
                    self.actions[i],
                    self.rewards[i],
                    None if i + 1 >= len(self.states) else self.states[i+1]
            ))