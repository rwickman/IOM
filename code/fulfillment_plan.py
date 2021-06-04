from simulator import InventoryProduct, Inventory

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

    def add_product(self, inv_node_id: str, inv_prod: InventoryProduct):
        if inv_node_id is None:
            raise Exception("Invalid inventory node ID.")

        if inv_node_id not in self._fulfillments:
            self._fulfillments[inv_node_id] = Fulfillment(inv_node_id)
        
        self._fulfillments[inv_node_id].inv.add_product(inv_prod)

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