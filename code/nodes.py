"""
Classes relating to inventory and demand nodes.
"""

from dataclasses import dataclass

@dataclass
class InventoryProduct:
    # Unique ID
    sku_id: int
    # Number of products with this SKU
    quantity: int

    def copy(self):
        """Make a copy of this InventoryProduct"""
        return InventoryProduct(self.sku_id, self.quantity)

@dataclass
class Coordinates:
    x: float
    y: float
    
class Location:
    """2D Cartesian coordinates"""
    def __init__(self, coords: Coordinates):
        self._coords = coords

    def get_distance(self, other_loc) -> float:
        """Get euclidean distace between two locations.
        
        Args:
            other_loc: the other Location to get the distance to.

        Returns:
            the float distance.
        """
        return ((self._coords.x - other_loc.coords.x) ** 2 + (self._coords.y - other_loc.coords.y) ** 2) ** 0.5

    @property
    def coords(self):
        return self._coords

class Node:
    """Superclass for inventory and demand nodes."""
    def __init__(self, inv_prods: list, loc: Location):
        """Initialize the node.
        
        Args:
            inv_prods: list of InventoryProducts for either an order or inventory.
            loc: Location of the node.
        """
        self._inv = Inventory(inv_prods)
        self._loc = loc

    @property
    def inv(self):
        return self._inv
    
    @property
    def loc(self):
        return self._loc

class DemandNode(Node):
    """Generalization of customer orders."""
    def __init__(self, inv_prods: list, loc: Location):
        """Initialize the DemandNode.
        
        Arg:
            inv_prods: list of InventoryProducts for the order.
            loc: Location of the customer order.
        """
        super().__init__(inv_prods, loc)

class Inventory:
    """Container for collection of SKUs."""
    def __init__(self, inv_prods: list = None):
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
        """Remove quantity of a particular SKU"""
        if inv_prod.sku_id not in self._inv_dict or self._inv_dict[inv_prod.sku_id] < inv_prod.quantity: 
            raise Exception("Tried to remove unavailable product.")
        
        self._inv_dict[inv_prod.sku_id] -= inv_prod.quantity
        self._inv_size -= inv_prod.quantity

        # Sanity check    
        assert self._inv_size >= 0
    
    
    def product_quantity(self, sku_id: int) -> int:
        """Get the quatity for a SKU."""
        if sku_id not in self._inv_dict:
            return 0
        else:
            return self._inv_dict[sku_id]

    @property
    def inv_size(self):
        """Get the total quantity."""
        return self._inv_size
    
    @property
    def sku_ids(self):
        return self._inv_dict.keys()
    
    def items(self):
        """Genrator for the non-item products the inventory."""
        inv_prods = [InventoryProduct(sku_id, quantity) for sku_id, quantity in self._inv_dict.items()]
        inv_prods = sorted(inv_prods, key=lambda p: p.sku_id)
        for inv_prod in inv_prods: 
            if inv_prod.quantity > 0:
                yield inv_prod


class InventoryNode(Node):
    """Inventory node for fulfillment nodes or stores."""
    def __init__(self, inv_prods: list, loc: Location, inv_node_id: int):
        """Initialize the DemandNode.
        
        Arg:
            inv_prods: list of InventoryProducts for the initial inventory.
            loc: Location of the node.
            inv_node_id: unique ID for this node.
        """
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
        """Get the total quantity of a SKU across all InventoryNodes.
        
        Args:
            sku_id: the unique ID for the SKU.
        """
        return self._inv.product_quantity(sku_id)

    def add_product(self, inv_node_id: int, inv_prod: InventoryProduct):
        """Add a product to an InventoryNode.
        
        Args:
            inv_node_id: the ID of the InventoryNode to add products to.
            inv_prod: the SKU and quantity to add.
        """
        self._inv_nodes_dict[inv_node_id].inv.add_product(inv_prod)
        self._inv.add_product(inv_prod)

    def remove_product(self, inv_node_id: int, inv_prod: InventoryProduct):
        """Add a product to an InventoryNode.
        
        Args:
            inv_node_id: the ID of the InventoryNode to remove products from.
            inv_prod: the SKU and quantity to remove.
        """
        self._inv_nodes_dict[inv_node_id].inv.remove_product(inv_prod)
        self._inv.remove_product(inv_prod)

    def empty(self):
        """Empty all the stock from all the invetory nodes."""
        for inv_node_id, inv_node in self._inv_nodes_dict.items():
            for inv_prod in inv_node.inv.items():
                self.remove_product(inv_node_id, inv_prod)
