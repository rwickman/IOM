import random
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import json
import os

@dataclass
class InventoryProduct:
    # Unique ID
    sku_id: int
    # Number of products with this SKU
    quantity: int

@dataclass
class Cordinates:
    x: float
    y: float
    
class Location:
    def __init__(self, cords: Cordinates):
        self._cords = cords

    def get_distance(self, other_loc):
        """Get euclidean distace between cordinates."""
        return ((self._cords.x - other_loc.cords.x) ** 2 + (self._cords.y - other_loc.cords.y) ** 2) ** 0.5

    @property
    def cords(self):
        return self._cords

class Node:
    def __init__(self, inv_prods: list, loc: Location):
        self._inv = Inventory(inv_prods)
        self._loc = loc

    @property
    def inv(self):
        return self._inv
    
    @property
    def loc(self):
        return self._loc

class DemandNode(Node):
    def __init__(self, inv_prods: list, loc: Location):
        super().__init__(inv_prods, loc)

class Inventory:
    def __init__(self, inv_prods: list):
        self._inv_dict = {}
        # Number of items currently in inventory
        self._inv_size = 0
        
        # Populate the initial inventory
        if inv_prods:    
            for inv_prod in inv_prods:
                self.add_product(inv_prod)
    
    def add_product(self, inv_prod: InventoryProduct):
        """Add products and quantites of each."""
        if inv_prod.quantity < 0:
            raise Exception("Product quantity cant be less than 0.")

        if inv_prod.sku_id is None:
            raise Exception("Invalid SKU ID.")

        if inv_prod.sku_id not in self._inv_dict:
            self._inv_dict[inv_prod.sku_id] = inv_prod.quantity
        else:
            self._inv_dict[inv_prod.sku_id] += inv_prod.quantity
        
        # Update inventory size
        self._inv_size += inv_prod.quantity
    
    def remove_product(self, inv_prod: InventoryProduct):
        if inv_prod.sku_id not in self._inv_dict or self._inv_dict[inv_prod.sku_id] < inv_prod.quantity: 
            raise Exception("Tried to remove unavailable product.")
        
        self._inv_dict[inv_prod.sku_id] -= inv_prod.quantity
        self._inv_size -= inv_prod.quantity

        # Sanity check    
        assert self._inv_size >= 0
    
    
    def product_quantity(self, sku_id: int) -> int:
        if sku_id not in self._inv_dict:
            return 0
        else:
            return self._inv_dict[sku_id]

    @property
    def inv_size(self):
        return self._inv_size
    
    @property
    def sku_ids(self):
        return self._inv_dict.keys()
    
    def items(self):
        for sku_id, quantity in self._inv_dict.items(): 
            yield InventoryProduct(sku_id, quantity)


class InventoryNode(Node):
    def __init__(self, inv_prods: list, loc: Location, inv_node_id: int):
        super().__init__(inv_prods, loc)
        self._inv_node_id = inv_node_id
    
    @property
    def inv_node_id(self) -> int:
        return self._inv_node_id

class InventoryNodeManager:
    """Manage the total inventory over all the inventory nodes."""
    def __init__(self, inv_nodes: list[InventoryNode]): 
        self._inv_nodes_dict = {}
        self._init_inv(inv_nodes)
        
    def _init_inv(self, inv_nodes: list[InventoryNode]):
        """Create an Inventory object that accumulates inventory accross all nodes."""
        inv_prods = []
        for inv_node in inv_nodes:
            # Add to inventory node to dict
            self._inv_nodes_dict[inv_node.inv_node_id] = inv_node

            # Add its inventory to the overall inventory
            for sku_id in inv_node.inv.sku_ids:
                # print(f"sku_id {sku_id} with {inv_node.inv.product_quantity(sku_id)}")
                inv_quantity = inv_node.inv.product_quantity(sku_id)
                if inv_quantity > 0:
                    inv_prods.append(
                        InventoryProduct(
                            sku_id,
                            inv_quantity))
        
        self._inv = Inventory(inv_prods)

    @property
    def stock(self):
        inv_prods = []
        for inv_prod in self._inv.items():
            inv_prods.append(inv_prod)
        return inv_prods

    @property
    def sku_ids(self):
        return self._inv.keys()
    
    @property
    def inv(self):
        return self._inv

    def product_quantity(self, sku_id: int) -> int:
        return self._inv.product_quantity(sku_id)
        
    
    def add_product(self, inv_node_id: int, inv_prod: InventoryProduct):
        self._inv_nodes_dict[inv_node_id].inv.add_product(inv_prod)
        self._inv.add_product(inv_prod)

    def remove_product(self, inv_node_id: int, inv_prod: InventoryProduct):
        self._inv_nodes_dict[inv_node_id].inv.remove_product(inv_prod)

        self._inv.remove_product(inv_prod)


class Simulator:
    def __init__(self, args, policy):
        self.args = args
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self._policy = policy
        self._train_file = os.path.join(
            self.args.save_dir,
            "train_dict.json")

        self._train_dict = {
            "ep_reward_avgs" : [],
            "policy_losses" : [],
            "inv_node_cords" : []
        }

        if self.args.load:
            self._load()

        self._init_inv_nodes()

    def _init_inv_nodes(self):
        """Initialize the inventory nodes."""
        self._inv_nodes = []
        for i in range(self.args.num_inv_nodes):        
            if self.args.load:
                # Create using saved location
                cords = Cordinates(
                    self._train_dict["inv_node_cords"][i][0],
                    self._train_dict["inv_node_cords"][i][1])
                loc = Location(cords)
                inv_node = self._gen_inv_node(loc)
            else:
                inv_node = self._gen_inv_node()
                # Save new location to restore later
                self._train_dict["inv_node_cords"].append(
                    [inv_node.loc.cords.x, inv_node.loc.cords.y])  

            self._inv_nodes.append(inv_node)
        
        self._inv_node_man = InventoryNodeManager(self._inv_nodes)

    def _reset(self):
        self._restock_inv()
        # Tell the policy it is over
        self._policy.reset()

    def _gen_inv_node(self, loc: Location = None) -> InventoryNode:
        if loc is None:
            loc = self._rand_loc()
        
        inv_node_id = len(self._inv_nodes)

        # Generate the inventory for this node
        inv_prods = self._gen_inv_node_stock()

        return InventoryNode(inv_prods, loc, inv_node_id)

    def _gen_inv_node_stock(self):
        # Generate the inventory for this node
        inv_prods = []
        for i in range(self.args.num_skus):
            # Generate a random quantity for this SKU
            rand_quant = random.randint(self.args.min_inv_prod, self.args.max_inv_prod)
            # Make product 
            inv_prods.append(InventoryProduct(i, rand_quant))

        return inv_prods

    def _restock_inv(self):
        for i in range(self.args.num_inv_nodes):
            inv_prods = self._gen_inv_node_stock()
            for inv_prod in inv_prods:
                self._inv_node_man.add_product(i, inv_prod)

    def _gen_demand_node(self):
        """Generate a demand node."""
        # Create random location
        loc = self._rand_loc()

        # Choose a random sku_id to allow for at least one order
        stock = self._inv_node_man.stock

        # Get non-zero items
        stock = [item for item in stock if item.quantity > 0]
        random.shuffle(stock)

        inv_prods = []
        for i, item in enumerate(stock):
            num_demand = np.random.poisson(self.args.demand_lam)
            # Make the demand for the first item at least one
            if i == 0:
                num_demand = max(1, num_demand)

            # Clip demand at stock inventory limit
            num_demand = min(num_demand, item.quantity)

            inv_prods.append(InventoryProduct(item.sku_id, num_demand))
              
        return DemandNode(inv_prods, loc)

    def _rand_loc(self) -> Location:
        rand_x = random.uniform(-self.args.cord_bounds, self.args.cord_bounds)
        rand_y = random.uniform(-self.args.cord_bounds, self.args.cord_bounds)
        cords = Cordinates(rand_x, rand_y)
        loc = Location(cords)
        return loc

    def _save(self):
        with open(self._train_file, "w") as f:
            json.dump(self._train_dict, f)
    
    def _load(self):
        if not os.path.exists(self._train_file):
            raise Exception(f"Cannot load because train file {self._train_file} does not exists.")
        with open(self._train_file) as f:
            self._train_dict = json.load(f)        

    def plot_results(self):
        def moving_average(x):
            return np.convolve(x, np.ones(self.args.reward_smooth_w), 'valid') / self.args.reward_smooth_w
        
        fig, axs = plt.subplots(2)
        y = moving_average(self._train_dict["ep_reward_avgs"])
        axs[0].plot(y)
        axs[0].set(
            title="Episode Average Reward",
            xlabel="Episode",
            ylabel="Average Reward"
        )

        axs[1].plot(self._train_dict["policy_losses"])
        axs[1].set(
            title="Policy MSE Loss",
            xlabel="Episode",
            ylabel="Average Loss"
        )

        plt.show()

    def run(self):
        for e_i in range(self.args.episodes):
            rewards = []
            for t in range(self.args.T_max):
                if self._inv_node_man.inv.inv_size <= 0:
                    break

                demand_node = self._gen_demand_node()

                # Get the fulfillment plan
                policy_results = self._policy(self._inv_nodes, demand_node)

                # Remove the products from the inventory nodes
                for fulfillment in policy_results.fulfill_plan.fulfillments():
                    for inv_prod in fulfillment.inv.items():
                        self._inv_node_man.remove_product(
                            fulfillment.inv_node_id,
                            inv_prod)

                rewards.extend(
                    [exp.reward for exp in policy_results.exps])

            # Reset the simulator for the next episode
            self._reset()

            if len(rewards) == 0:
                continue

            reward_avg = sum(rewards) / len(rewards)
            print("reward_avg", reward_avg)
            self._train_dict["ep_reward_avgs"].append(reward_avg)
                        
            # Train if this is a trainable policy
            if self._policy.is_trainable and self._policy.is_train_ready():
                loss = self._policy.train()                
                
                self._train_dict["policy_losses"].append(loss)

        if self._policy.is_trainable:
            self._policy.save()
        self._save()
        if self.args.plot:
            self.plot_results()