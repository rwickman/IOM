import torch, os, sys
# Add code path to sys path
file_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(file_path, "../code")
sys.path.append(code_path)

from policy import Experience
from simulator import Coordinates, Location, InventoryProduct, InventoryNode, DemandNode
from fake_args import FakeArgs

class FakeEpisode:
    def __init__(self):
        self.args = FakeArgs()

        # Create fake inventory nodes
        loc = Location(Coordinates(0, 0))
        inv_node_id_1 = 0
        inv_prod_1 = InventoryProduct(0, 2)
        inv_prod_2 = InventoryProduct(1, 1)
        inv_prods = [inv_prod_1, inv_prod_2]
        inv_node_1 = InventoryNode(inv_prods, loc, inv_node_id_1)

        loc = Location(Coordinates(10, 10))
        inv_node_id_2 = 1
        inv_prod_2 = InventoryProduct(1, 2)
        inv_prods = [inv_prod_2]
        inv_node_2 = InventoryNode(inv_prods, loc, inv_node_id_2)
        self.inv_nodes = [inv_node_1, inv_node_2]

        # Create fake demand node
        loc = Location(Coordinates(2, 1))
        inv_prod_1 = InventoryProduct(0, 1)
        inv_prod_2 = InventoryProduct(1, 2)
        inv_prods = [inv_prod_1, inv_prod_2]
        self.demand_node = DemandNode(inv_prods, loc)
        
        self.reward_scale_factor = 1/((2 * self.args.coord_bounds) **2)

        # Distance b/w demand node and inventory node 1
        self.dist_1 = 2.23606797749979 
        # Distance b/w demand node and inventory node 2
        self.dist_2 = 12.041594578792296

        # Create the rewards
        self.rewards = [
            -self.dist_1 * self.reward_scale_factor,
            -self.dist_1 * self.args.reward_alpha * self.reward_scale_factor,
            -self.dist_2 * self.reward_scale_factor
        ]

        inv_coords = [0,0,10,10]
        demand_coords = [2,1]

        # Create the states
        inv = [2, 1, 0, 2]
        demand = [1, 2]
        cur_fulfill = [0,0,0,0]
        item_hot = [1,0]
        state_1 = torch.tensor(inv + inv_coords + demand + demand_coords + cur_fulfill + item_hot, dtype=torch.float32)

        inv = [1, 1, 0, 2]
        demand = [0, 2]
        cur_fulfill = [1, 0, 0, 0]
        item_hot = [0, 1]
        state_2 = torch.tensor(inv + inv_coords + demand + demand_coords + cur_fulfill + item_hot, dtype=torch.float32)


        inv = [1, 0, 0, 2]
        demand = [0, 1]
        cur_fulfill = [1, 1, 0, 0]
        item_hot = [0, 1]
        state_3 = torch.tensor(inv + inv_coords + demand + demand_coords + cur_fulfill + item_hot, dtype=torch.float32)

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
    
    def norm_states(self):
        for state in self.states:
            state[4] = state[4]/self.args.coord_bounds
            state[5] = state[5]/self.args.coord_bounds
            state[6] = state[6]/self.args.coord_bounds
            state[7] = state[7]/self.args.coord_bounds
            state[10] = state[10]/self.args.coord_bounds
            state[11] = state[11]/self.args.coord_bounds




class FakeEpisode_2:
    def __init__(self):
        self.args = FakeArgs()
        self.reward_scale_factor = 1/((2 * self.args.coord_bounds) **2)
        self.num_skus = 3
        self.num_inv_nodes = 4
        
        self.inv = torch.tensor([
            [0,1,3],
            [0,2,4],
            [1,0,1],
            [4,1,2]], dtype=torch.float32)
        self.locs = torch.tensor([
            [0,0],
            [10,10],
            [-10,-10],
            [2,2]
        ], dtype=torch.float32)

        # Create the inventory nodes
        self.inv_nodes = []
        for i in range(self.num_inv_nodes):
            cur_inv_prods = []
            for j in range(self.num_skus):
                if self.inv[i][j] > 0:
                    cur_inv_prods.append(
                        InventoryProduct(j, self.inv[i,j]))
            loc = Location(Coordinates(float(self.locs[i, 0]), float(self.locs[i, 1])))
            self.inv_nodes.append(
                InventoryNode(cur_inv_prods, loc, i))


        # Create fake demand node
        self.demand_loc_coords = torch.tensor([2.0,1.0])
        self.demand_loc = Location(
            Coordinates(
                float(self.demand_loc_coords[0]),
                float(self.demand_loc_coords[1])))

        self.demand = torch.tensor(
            [2,0,1]
        )
        demand_inv_prods = []
        for i in range(3):
            demand_inv_prods.append(InventoryProduct(i, self.demand[i]))


        self.demand_node = DemandNode(
            demand_inv_prods, self.demand_loc)
        