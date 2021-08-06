import random
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import math
from scipy.stats import beta

from nodes import *


class Simulator:
    """Simulator for simulating omni-channel order fulfillment. """
    def __init__(self, args, policy=None):
        """Initilize the simulator.
        
        Args:
            args: Namespace of CLI arguments.
            policy: Policy that represents the algorithm to make order fulfillment decisions.
        """
        self.args = args

        self._train_dict = {
            "policy_name" : self.args.policy,
            "ep_avg_rewards" : [],
            "policy_losses" : [],
            "inv_node_coords" : []
        }

        if self.args.city_loc:
            self._city_locs = self._load_locs(self.args.city_loc)

        if policy:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            self._policy = policy
            self._train_file = os.path.join(
                self.args.save_dir,
                "train_dict.json")

            if self.args.load:
                self._load()
        else:
            self._policy = None

        self._init_inv_nodes()

    def _init_inv_nodes(self):
        """initialize the inventory nodes."""
        
        # Check if you want specific location coordinates
        if self.args.inv_loc:
            inv_node_locs = self._load_locs(self.args.inv_loc)

        self._inv_nodes = []
        for i in range(self.args.num_inv_nodes):        
            if self.args.inv_loc:
                inv_node = self._gen_inv_node(inv_node_locs[i])
            elif self.args.load:
                # Create using saved location
                coords = Coordinates(
                    self._train_dict["inv_node_coords"][i][0],
                    self._train_dict["inv_node_coords"][i][1])
                loc = Location(coords)
                inv_node = self._gen_inv_node(loc)
            else:
                inv_node = self._gen_inv_node()

                # Save new location to restore later
                self._train_dict["inv_node_coords"].append(
                    [inv_node.loc.coords.x, inv_node.loc.coords.y])  

            self._inv_nodes.append(inv_node)

        self._inv_node_man = InventoryNodeManager(self._inv_nodes)

    def _reset(self):
        """Reset simulator for next episode"""
        # Check if inventory is still left
        if self._inv_node_man.inv.inv_size > 0 and not self.args.eval:
            # Indicate to the policy that 
            self._policy.early_stop_handler()
        
        # Empty the inventory
        self._inv_node_man.empty()

        # Restock the inventory in the inventory nodes
        self._restock_inv()

        # Reset policy for new episode
        if self._policy:
            self._policy.reset()

    def _gen_inv_node(self, loc: Location = None) -> InventoryNode:
        """Generate an inventory node.
        
        The generated inventory node will be initialized with random inventory and a
        random location if not given. 

        Args:
            loc: the optional location of where to generate the inventory node.
        
        Returns:
            the generated inventory node.
        """
        if loc is None:
            loc = self._rand_loc()

        inv_node_id = len(self._inv_nodes)

        # Generate the inventory for this node
        inv_prods = self._gen_inv_node_stock(self.args.max_inv_prod)

        return InventoryNode(inv_prods, loc, inv_node_id)

    def _gen_inv_node_stock(self, max_inv_prod: int, inv_sku_lam: float = None) -> list:
        """Generate inventory for an inventory node.
        
        Args:
            max_inv_prod: maximum inventory for every product 
                (e.g., every quantity for SKU will be between [self.args.min_inv_prod, max_inv_prod])
            inv_sku_lam: lambda value that parameterizes the beta distirbution that samples 
                how many SKUs will be sampled

        Returns:
            the list of InventoryProduct where each contains the quanity of each unique product SKU.
        """
        inv_prods = []
        
        sku_ids = list(range(self.args.num_skus))

        # Shuffle skus to allow choosing which SKU at least one and what SKUs get selected  
        random.shuffle(sku_ids)

        # Sample the number of skus
        if inv_sku_lam:
            num_skus = np.clip(np.random.poisson(inv_sku_lam), 1, self.args.num_skus)
            sku_ids = sku_ids[:num_skus]
        
        for i, sku_id in enumerate(sku_ids):
            if self.args.min_inv_prod == 0 and i == 0:
                # Choose a sku to have at least 1 unit
                min_inv_prod = 1
            else:
                min_inv_prod = self.args.min_inv_prod

            if self.args.ramp_max_prod:
                num_eps = len(self._train_dict["ep_avg_rewards"])
                max_prod_scale = num_eps/self.args.ramp_eps
                cur_max_inv_prod = int(max_prod_scale * max_inv_prod)
            else:
                cur_max_inv_prod = max_inv_prod

            cur_max_inv_prod = max(max_inv_prod, 1.0)
            
            # Generate a random quantity for this SKU
            rand_quant = random.randint(min_inv_prod, cur_max_inv_prod)

            # Make product 
            if rand_quant > 0:
                inv_prods.append(InventoryProduct(sku_id, rand_quant))

        return inv_prods


    def _restock_inv(self):
        """Restock (i.e., reinitialize) the inventory at all of the inventory nodes."""
        
        # Get the max_inv_prod
        if self.args.rand_max_prod:
            # Set the max inventory for all SKUs for every node
            max_inv_prod = random.randint(1, self.args.max_inv_prod)
        else:
            max_inv_prod = self.args.max_inv_prod

        # Get the inventory SKU lambda value
        if self.args.rand_inv_sku_lam:
            inv_sku_lam = random.uniform(1, self.args.num_skus)

        elif self.args.inv_sku_lam:
            inv_sku_lam = self.args.inv_sku_lam
        else:
            inv_sku_lam = None
        
        for i in range(self.args.num_inv_nodes):
            inv_prods = self._gen_inv_node_stock(max_inv_prod, inv_sku_lam)
            for inv_prod in inv_prods:
                self._inv_node_man.add_product(i, inv_prod)

    def _gen_demand_node(self, stock: list = None):
        """Generate a demand node.
        
        Args:
            stock: the optional list of InventoryProducts that verifies demand 
                generated can actually be fulfilled.

        Returns:
            the generated demand node
        """
        if not stock:
            stock = self._inv_node_man.stock

            # Get non-zero items
            stock = [item for item in stock if item.quantity > 0]
        
        # Create random location
        loc = self._gen_demand_loc()

        # Sample the number of order lines
        num_lines = np.random.poisson(self.args.order_line_lam)

        # Clip to valid range
        num_lines = np.clip(num_lines, 1, len(stock))

        # Sample the SKUs
        prods = np.random.choice(stock, size=num_lines, replace=False)

        # Choose a random sku_id to allow for at least one order
        random.shuffle(prods)

        inv_prods = []
        for i, item in enumerate(prods):
            num_demand = np.random.poisson(self.args.demand_lam)
            # Make the demand for the first item at least one
            if i == 0:
                num_demand = max(1, num_demand)

            # Clip demand at stock inventory limit
            num_demand = min(num_demand, item.quantity)

            inv_prods.append(InventoryProduct(item.sku_id, num_demand))
        
        return DemandNode(inv_prods, loc)


    def _sample_circle_point(self):
        """Generate a point on within a circle centered at (0,0).

        Returns:
            a tuple of x and y coordinates.
        """
        # Generate point on perimeter
        theta = random.random() * 2 * math.pi
        
        # Generate radius
        r = self.args.city_radius * beta.rvs(
            self.args.demand_beta_a, self.args.demand_beta_b) ** 0.5 
        
        # Get points based on polar coordinates
        x = r * math.cos(theta)
        y = r * math.sin(theta)

        return x, y

    def _rand_loc(self) -> Location:
        """Sample a random unifrom location from x, y between [-self.args.coord_bounds, self.args.coord_bounds].
        
        Returns:
            the sampled location.
        """
        rand_x = random.uniform(-self.args.coord_bounds, self.args.coord_bounds)
        rand_y = random.uniform(-self.args.coord_bounds, self.args.coord_bounds)
        coords = Coordinates(rand_x, rand_y)
        return Location(coords)

    def _gen_demand_loc(self) -> Location:
        """Generate a location for a demand node.
        
        How the location is generated depends on if cities were passed to CLI arguments or not.
        If so, it will use city location generation, else it will use random uniform generation.

        """
        if self.args.city_loc:
            # Perform rejection sampling to generate point that remains inside grid bounds
            while True:
                # Generate a point on a circle centered at (0,0)
                x, y = self._sample_circle_point()

                # Sample the city
                rand_city_loc = random.sample(self._city_locs, 1)[0]

                # Shift the coords to make it centered at sampled city center
                demand_coords = Coordinates(
                                        x + rand_city_loc.coords.x,
                                        y + rand_city_loc.coords.y)

                # Check if it is within the bounds of the given grid
                if demand_coords.x >= -self.args.coord_bounds  \
                    and demand_coords.x <= self.args.coord_bounds  \
                    and demand_coords.y >= -self.args.coord_bounds  \
                    and demand_coords.y <= self.args.coord_bounds:
                    return Location(demand_coords)
        else:
            return self._rand_loc()

    def _save(self):
        """Save training information."""
        with open(self._train_file, "w") as f:
            json.dump(self._train_dict, f)
    
    def _load(self):
        """Load training information."""
        if not os.path.exists(self._train_file):
            raise Exception(f"Cannot load because train file {self._train_file} does not exists.")
        with open(self._train_file) as f:
            self._train_dict = json.load(f)


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

    def plot_results(self):
        """Plot the training results."""
        def moving_average(x):
            return np.convolve(x, np.ones(self.args.reward_smooth_w), 'valid') / self.args.reward_smooth_w
        
        fig, axs = plt.subplots(2)
        y = moving_average(self._train_dict["ep_avg_rewards"])
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

    def remove_products(self, policy_results):
        """Remove products in the fulfillment decisions.
        
        Args:
            policy_results: contains the results of an episode of fulfillment decisions.
        """
        # Remove the products from the inventory nodes
        for fulfillment in policy_results.fulfill_plan.fulfillments():
            for inv_prod in fulfillment.inv.items():
                self._inv_node_man.remove_product(
                    fulfillment.inv_node_id,
                    inv_prod)


    def run(self):
        """Run the simulator for self.args.episodes episodes."""
        for e_i in range(self.args.episodes):
            rewards = []
            for t in range(self.args.order_max):
                if self._inv_node_man.inv.inv_size <= 0:
                    break

                demand_node = self._gen_demand_node()

                # Get the fulfillment plan
                policy_results = self._policy(self._inv_nodes, demand_node)

                self.remove_products(policy_results)
                
                rewards.extend(
                    [exp.reward for exp in policy_results.exps])

            # Reset the simulator for the next episode
            self._reset()
            
            if len(rewards) == 0:
                continue
            
            # Compute the average reward over the episode
            avg_reward = (sum(rewards) / len(rewards)) #* (2 * self.args.coord_bounds)
            print(e_i, "avg_reward", avg_reward, t)

            self._train_dict["ep_avg_rewards"].append(avg_reward)

            # Train if this is a trainable policy
            if self._policy.is_trainable and self._policy.is_train_ready():
                for i in range(self.args.train_iter):
                    loss = self._policy.train()

                self._train_dict["policy_losses"].append(loss)

        # Save the policy
        if self._policy.is_trainable:
            self._policy.save()

        # Save the training results
        self._save()
        if self.args.plot:
            self.plot_results()