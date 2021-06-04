import random
from dataclasses import dataclass
import numpy as np


@dataclass
class InventoryProduct:
    # Unique ID, SKU is str instead of int to model real-life
    sku_id: str
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
class Product:
    def __init__(self, name: str, price: float):
        self._name = name

    @property
    def name(self):
        """Name of product."""
        return self._name
    
class Order:
    def __init__(self, products: list):
        self._products = products
    
    @property
    def total_price(self):
        return sum([product.price for product in self._products])


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

        if not inv_prod.sku_id:
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
    
    
    def product_quantity(self, sku_id: str) -> int:
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
    def __init__(self, inv_prods: list, loc: Location, inv_node_id: str):
        super().__init__(inv_prods, loc)
        self._inv_node_id = inv_node_id
    
    @property
    def inv_node_id(self) -> str:
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

    def product_quantity(self, sku_id: str) -> int:
        return self._inv.product_quantity(sku_id)
        
    
    def add_product(self, inv_node_id: str, inv_prod: InventoryProduct):
        self._inv_nodes_dict[inv_node_id].add_product(inv_prod)
        self._inv.add_product(inv_prod)

    def remove_product(self, inv_node_id: str, inv_prod: InventoryProduct):
        self._inv_nodes_dict[inv_node_id].inv.remove_product(inv_prod)

        self._inv.remove_product(inv_prod)


class Simulator:
    def __init__(self, args, policy):
        self.args = args
        self._policy = policy

        self._inv_nodes = []
        for i in range(self.args.num_inv_nodes):
            self._inv_nodes.append(self._gen_inv_node())
        self._inv_node_man = InventoryNodeManager(self._inv_nodes)

    def _gen_inv_node(self) -> InventoryNode:
        loc = self._rand_loc()
        inv_node_id = str(len(self._inv_nodes))

        # Generate the inventory for this node
        inv_prods = []
        for i in range(self.args.num_skus):
            # Generate a random quantity for this SKU
            rand_quant = random.randint(self.args.min_prod_inv, self.args.max_prod_inv)
            # Make product 
            inv_prods.append(InventoryProduct(str(i), rand_quant))

        return InventoryNode(inv_prods, loc, inv_node_id)
         
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

    def run(self):
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

            print(self._inv_node_man.inv.inv_size)
            print(policy_results.rewards)