import pandas as pd
import numpy as np

from nodes import Location, Coordinates, InventoryProduct, DemandNode

class DatasetSimulation:
    def __init__(self,
                loc_csv = "data/olist_data/location/demand_locs.csv",
                inv_node_loc_csv="data/olist_data/location/inv_node_locs.csv",
                orders_csv="data/olist_data/orders/train_orders.csv"):
        self._loc_df = pd.read_csv(loc_csv)
        self._inv_node_loc_df = pd.read_csv(inv_node_loc_csv)
        self._orders =  pd.read_csv(orders_csv)
        self._init_demand()

    def _init_demand(self):
        """Initialize properties of the demand."""

        # Compute average order demand
        self._order_line_lam = self._orders["order_id"].value_counts().mean()
        
        # Create the PID distribution
        pid_count = self._orders["product_id"].value_counts()
        self._pid_distr = pid_count / pid_count.sum()
        self._cur_sku_distr = self._pid_distr.copy()

        # Number of SKUs
        self.num_skus = len(self._pid_distr)

    def init_sku_distr(self, stock: list):
        """Reset the SKU distribution
        
        Args:
            stock: list of aggregate InventoryProducts in all the InventoryNodes. 
        """
        self._total_stock = 0
        self._cur_sku_distr = np.zeros_like(self._pid_distr.copy())
        
        # Get the probability for each item in the inventory
        for item in stock:
            assert item.quantity > 0
            self._cur_sku_distr[item.sku_id] = self._pid_distr[item.sku_id]
            self._total_stock += item.quantity
        
        # Compute the new probability based on normalizing over the available products
        self._normalize_sku_distr()

    
    def sample_loc(self) -> Location:
        """Sample a location from dataset."""
        # Sample a row
        loc_row_sample = self._loc_df.sample()
        
        # Create the sampled location
        loc_sample = Location(
            Coordinates(
                float(loc_row_sample["geolocation_lat"]),
                float(loc_row_sample["geolocation_lng"])))

        return loc_sample
    
    def _normalize_sku_distr(self):
        self._cur_sku_distr = self._cur_sku_distr / self._cur_sku_distr.sum()

    def gen_demand_node(self, inv_dict: dict) -> DemandNode:
        
        # Sample a location
        loc = self.sample_loc()

        # Sample the number of items wanted
        order_size = np.random.poisson(self._order_line_lam)

        # Clip to valid range
        order_size = np.clip(order_size, 1, self._total_stock)
        
        # Update the total stock to account for current order
        self._total_stock -= order_size

        # Keep up with what items and the quantity of items in order
        cur_demand_dict = {}
        demand_prods_dict = {}
        
        for i in range(order_size):
            # Sample an item
            item_idx = np.random.choice(len(self._cur_sku_distr), p=self._cur_sku_distr)
            
            # Verify it valid inventory
            assert self._cur_sku_distr[item_idx] > 0.0

            # Add to order
            if item_idx not in cur_demand_dict:
                cur_demand_dict[item_idx] = 1
            else:
                cur_demand_dict[item_idx] += 1
            
            if item_idx not in demand_prods_dict:
                 demand_prods_dict[item_idx] = InventoryProduct(item_idx, 1)
            else:
                demand_prods_dict[item_idx].quantity += 1
            
            # Verify not requesting too much of this item
            assert cur_demand_dict[item_idx] <= inv_dict[item_idx] 
            

            # If there isn't inventory left, set the probability to 0
            if cur_demand_dict[item_idx] == inv_dict[item_idx]:
                self._cur_sku_distr[item_idx] = 0.0
                self._normalize_sku_distr()
        print("GEN DEMAND: ", demand_prods_dict.values())
        return DemandNode(demand_prods_dict.values(), loc)