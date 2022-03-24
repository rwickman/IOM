import pandas as pd
import numpy as np
import random
import torch
from config import device
from scipy.stats import betabinom

from nodes import Location, Coordinates, InventoryProduct, DemandNode, InventoryNode, InventoryNodeManager

class DatasetSimulator:
    def __init__(self,
                args,
                loc_csv = "data/olist_data/location/demand_locs.csv",
                inv_node_loc_csv="data/olist_data/location/inv_node_locs.csv",
                orders_csv="data/olist_data/orders/train_orders.csv"):
        self.args = args
        if self.args.stratified:
            self._cur_sample_step = 0
        self._loc_df = pd.read_csv(loc_csv)
        self._inv_node_loc_df = pd.read_csv(inv_node_loc_csv)
        self._orders =  pd.read_csv(orders_csv)
        self._init_demand()
        c_1 = (self._loc_df["geolocation_lat"].min(), self._loc_df["geolocation_lng"].min()) 
        c_2 = (self._loc_df["geolocation_lat"].max(), self._loc_df["geolocation_lng"].max())
        self._max_dist = ((c_1[0] - c_2[0]) ** 2 + (c_1[1] - c_2[1]) ** 2) ** 0.5

        print("@@@@@@@@@@@@@@@@@@\nCOORD DIST: ", ((c_1[0] - c_2[0]) ** 2 + (c_1[1] - c_2[1]) ** 2)**0.5, "\n")

        self._coord_bounds = max(
            abs(self._loc_df["geolocation_lat"].max() - self._loc_df["geolocation_lat"].min()),
            abs(self._loc_df["geolocation_lng"].max() - self._loc_df["geolocation_lng"].min()))


    def _sample_max_stock(self):
        #return min(max(betabinom.rvs(self.args.ds_max_stock, 0.4, 2), 1), self.args.ds_max_stock)
        if self.args.stratified:
            step_size = self.args.order_max // self.args.num_inv_nodes
            min_stock = max(self._cur_sample_step * step_size, self.args.ds_min_stock)
            max_stock = min((self._cur_sample_step + 1) * (step_size), self.args.ds_max_stock)
            if max_stock >= self.args.ds_max_stock:
                self._cur_sample_step = 0
            else:
                self._cur_sample_step += 1
            print(f"\nSTOCK RANGE: [{min_stock}, {max_stock}]\n")
            return random.randint(min_stock, max_stock)
        else:
            return random.randint(1, self.args.ds_max_stock)

    def _init_demand(self):
        """Initialize properties of the demand."""

        # Compute average order demand
        self._order_line_lam = self._orders["order_id"].value_counts().mean()
        
        # Create the PID distribution
        pid_count = torch.tensor(
            self._orders["product_id"].value_counts().sort_index().tolist(), 
            dtype=torch.float64, device=device)
        
        print("pid_count", pid_count)
        #self._sku_distr = torch.tensor([1/8]).repeat(8).double() #torch.tensor(pid_count / pid_count.sum())
        #self._sku_distr = self._sku_distr / self._sku_distr.sum()
        
        # # TODO: DELETE THIS
        # pid_count = pid_count[:10]
        
        self._sku_distr = (pid_count / pid_count.sum())

        print("self._sku_distr", self._sku_distr, self._sku_distr.sum())
        self._cur_sku_distr = self._sku_distr.clone().cpu().detach()

        # Number of SKUs
        self.num_skus = len(self._sku_distr)

        self._cur_stock_max = self._sample_max_stock()
        print("self._cur_stock_max", self._cur_stock_max)
    

    def init_sku_distr(self, stock: list):
        """Reset the SKU distribution
        
        Args:
            stock: list of aggregate InventoryProducts in all the InventoryNodes. 
        """
        self._total_stock = 0
        self._cur_sku_distr = torch.zeros_like(self._sku_distr).detach().cpu()
        
        # Get the probability for each item in the inventory
        for item in stock:
            assert item.quantity > 0
            self._cur_sku_distr[item.sku_id] = 1/len(self._sku_distr)#self._sku_distr[item.sku_id]
            self._total_stock += item.quantity
        # print("self._total_stock", self._total_stock)
        # Compute the new probability based on normalizing over the available products
        self._normalize_sku_distr()


        self._cur_stock_max = self._sample_max_stock()#random.randint(self.args.ds_min_stock, self.args.ds_max_stock)
        # print("self._cur_stock_max", self._cur_stock_max)
    
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
        distr_sum = self._cur_sku_distr.sum()
        if float(distr_sum) > 0:
            self._cur_sku_distr = self._cur_sku_distr / distr_sum


    def gen_demand_node(self, inv_dict: dict) -> DemandNode:
        # TODO: Change how the cur_sku_distr is updated

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
            # Sample a product
            item_idx = np.random.choice(len(self._cur_sku_distr), p=self._cur_sku_distr.detach().cpu())
            
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

        return DemandNode(demand_prods_dict.values(), loc, self.num_skus)
    
    def gen_inv_node_stock(self):
        """Use SKU distribution to sample the stock for an inventory node."""
        num_stock = random.randint(min(self.args.ds_min_stock, self._cur_stock_max), self._cur_stock_max)
        
        stock_dict = {}
        for i in range(num_stock):
            # Sample a product
            #item_idx = np.random.choice(len(self._sku_distr), p=self._sku_distr)
            item_idx = np.random.choice(len(self._sku_distr), p=self._sku_distr.detach().cpu())

            if item_idx not in stock_dict:
                stock_dict[item_idx] = InventoryProduct(item_idx, 1)
            else:
                stock_dict[item_idx].quantity += 1
        
        # print("num_stock", num_stock, "len(stock_dict)", len(stock_dict))
        return stock_dict.values()
    
    @property
    def cur_sku_distr(self):
        return self._cur_sku_distr.float().to(device)

            

class TestDatasetSimulator(DatasetSimulator):
    def __init__(self,
                args,
                loc_csv = "data/olist_data/location/demand_locs.csv",
                inv_node_loc_csv="data/olist_data/location/inv_node_locs.json",
                orders_csv="data/olist_data/orders/val_orders.csv",
                inv_stock_json="data/olist_data/inv_stock/test_inv_node_stock.json"):

        self.args = args
        self._orders = pd.read_csv(orders_csv)
        self._inv_stock_json = inv_stock_json
        self._inv_node_man = self._gen_inv_nodes()
         
        
    def _load_locs(self, loc_json: str) -> list:
        """Load inventory node locations.
        
        Returns:
            a list of city 2D inventory node locations.
        """
        inv_locs = []
        with open(loc_json) as f:
            coords_list = json.load(f)
            for coords in coords_list:
                loc = Location(Coordinates(coords[0], coords[1]))
                inv_locs.append(loc)
        
        return inv_locs

    # def _load_locs(self, loc_json: str) -> list:
    #     """Load inventory node locations.
        
    #     Returns:
    #         a list of city 2D inventory node locations.
    #     """
    #     inv_locs = []
    #     with open(self.args.inv_loc) as f:
    #         self.inv_locs = json.load(f)


    def _gen_demand_node(self):
        pass

    def _gen_inv_nodes(self):
        # For generating location, will just have to get all seller_ids 
        # and map them to a inventory node
        # 
        # Generating inventory you will need to get all rows with seller ID
        #  Then, add the stock to the inventory node that corresponds to that seller
        
        inv_locs = self._load_locs()
        with open(self._inv_stock_json) as f:
            inv_stock_dict = json.load(self._inv_stock_json)

        inv_nodes = []
        for inv_node_id, stock_dict in inv_stock_dict.items():
            inv_prods = []
            for sku_id, quantity in stock_dict.items():
                inv_prods.append(
                    InventoryProduct(sku_id, quantity))
            
            inv_nodes.append(
                InventoryNode(inv_prods, inv_locs[inv_node_id], self.args.num_skus))
        self._inv_nodes = inv_nodes
        return InventoryNodeManager(self._inv_nodes, self.args.num_skus) 

# TestDatasetSimulator()