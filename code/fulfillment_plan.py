from nodes import InventoryProduct, Inventory

class Fulfillment:
    """A fulfillment request for a single inventory node."""
    def __init__(self,
                inv_node_id: str,
                inv_prods: list[InventoryProduct] = None):

        self._inv_node_id = inv_node_id
        self._inv = Inventory(inv_prods)

    @property
    def inv(self):
        return self._inv

    @property
    def inv_node_id(self):
        return self._inv_node_id


class FulfillmentPlan:
    """Manages all the fulfillment requests for every item in order."""
    def __init__(self):
        self._fulfillments = {}

        # Keep up with all the items in the plan 
        self._inv = Inventory()

    def add_product(self, inv_node_id: str, inv_prod: InventoryProduct):
        """Add products to be fulfilled by a given inventory node.
        
        Args:
            inv_node_id: unqiue integer ID of inventory node.
            inv_prod: InventoryProduct that indentifies a SKU and the quantity to be fulfilled.
        """
        if inv_node_id is None:
            raise Exception("Invalid inventory node ID.")

        # Add to dict
        if inv_node_id not in self._fulfillments:
            self._fulfillments[inv_node_id] = Fulfillment(inv_node_id)
        
        # Add to existing list of fulfillments
        self._fulfillments[inv_node_id].inv.add_product(inv_prod)
        
        # Keep up with all items in an order
        self._inv.add_product(inv_prod)
        

    def remove_product(self, inv_node_id: str, inv_prod: InventoryProduct):
        """Remove products to be fulfilled by a given inventory node.
        
        Args:
            inv_node_id: unqiue integer ID of inventory node.
            inv_prod: InventoryProduct that indentifies a SKU and the quantity to be fulfilled.
        """
        if inv_node_id is None or inv_node_id not in self._fulfillments:
            raise Exception("Invalid inventory node ID.")
        
        # Remove from fulfillment
        self._fulfillments[inv_node_id].inv.remove_product(inv_prod)
        
        # Remove from total fulfillment inventory
        self._inv.remove_product(inv_prod)


    def fulfill_quantity(self, inv_node_id: str, sku_id: str) -> int:
        """Get the quantity of SKU currently allocated in the request to the specified inventory node."""
        if inv_node_id in self._fulfillments:
            return self._fulfillments[inv_node_id].inv.product_quantity(sku_id)
        else:
            return 0
    
    def get_fulfillment(self, inv_node_id: str):
        if inv_node_id in self._fulfillments:
            return self._fulfillments[inv_node_id]
        else:
            return None

    def fulfillments(self):
        for fulfillment in self._fulfillments.values():
            yield fulfillment

    def copy(self):
        """Create a copy of this fulfillment plan."""
        plan_copy = FulfillmentPlan()
        
        for fulfillment in self._fulfillments.values():

            # Add all the inventory allocated to this node
            for inv_prod in fulfillment.inv.items():
                plan_copy.add_product(
                    fulfillment.inv_node_id,
                    inv_prod
                )

        return plan_copy

    @property
    def inv(self):
        return self._inv