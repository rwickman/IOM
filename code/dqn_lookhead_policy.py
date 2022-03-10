import torch

from reward_manager import RewardManager
from dqn_emb_policy import DQNEmb, DQNEmbTrainer
from config import device
from nodes import InventoryNode, DemandNode, InventoryProduct
from fulfillment_plan import FulfillmentPlan
from policy import PolicyResults, Policy, Experience
from reward_manager import RewardManager


class DQNLookaheadTrainer(DQNEmbTrainer):
    """Fulfills orders based on the predicted best order fulfillment for an order."""
    def __init__(self, args, reward_man: RewardManager):
        super().__init__(args, reward_man, "dqn_emb_lookahead_model.pt", "DQNEmbLookahead")
    
    def _fulfill_search(self,
                        inv_nodes,
                        demand_node,
                        inv,
                        inv_locs,
                        demand,
                        demand_loc,
                        cur_fulfill,
                        fulfill_plan,
                        sku_distr,
                        inv_prods,
                        inv_prod_idx,
                        cur_val,
                        exps):
        inv_prod = inv_prods[inv_prod_idx]
        print("\ncur_val", cur_val, " inv_prod_idx", inv_prod_idx, "len(inv_prods)", len(inv_prods), "quantity", inv_prods[inv_prod_idx].quantity, "CUR QUANTITY", fulfill_plan.inv.product_quantity(inv_prod.sku_id))
        if fulfill_plan.inv.product_quantity(inv_prod.sku_id) >= inv_prod.quantity:
            # Start fulfilling the next item in the order 
            inv_prod_idx += 1
            inv_prod = inv_prods[inv_prod_idx]
        

        # Create one hot encoded vector for the current item selection
        item_hot = torch.zeros(self.args.num_skus).to(device)
        item_hot[inv_prod.sku_id] = 1

        # iterate over children
        state = self._create_state(
                    inv,
                    inv_locs,
                    demand,
                    demand_loc,
                    cur_fulfill,
                    item_hot,
                    sku_distr)
        
        # Get indices of nodes that have nonzero inventory
        valid_idxs = self._get_valid_actions(state)
    
        # Create the current state
        state = self._create_state(
            inv,
            inv_locs,
            demand,
            demand_loc,
            cur_fulfill,
            item_hot,
            sku_distr)
            
        if len(exps) > 0:
            # Update the previous experience next state
            exps[-1].next_state = state


        best_plan = best_val = best_last_action = best_exps = None
        is_last_item = fulfill_plan.inv.product_quantity(inv_prod.sku_id) + 1 >= inv_prod.quantity and \
                inv_prod_idx + 1 >= len(inv_prods)
        
        print("is_last_item", is_last_item)
        print("valid_idxs", valid_idxs)
        # Iterate over every action
        for valid_idx in valid_idxs:
            action = int(valid_idx)
            print("action", action)

            # Get the reward for routing to this inventory node
            reward = self._reward_man.get_reward(
                            inv_nodes[int(valid_idx)],
                            demand_node,
                            fulfill_plan)


            if is_last_item:
                # Get prediction for future value of routing decision as this is the last item in order
                model_pred = self.predict(state)

                # Compute the value of this fulfillment plan
                ## (Subtract reward as model_pred contains estimate of current reward)
                child_val = cur_val + model_pred[action] - reward
                print("FINAL VALUE", child_val)

                # Update the best possible order fulfillment
                if best_val is None or child_val > best_val:
                    best_val = child_val
                    best_plan = fulfill_plan
                    best_last_action = action
                    best_last_exp = Experience(state, action, reward, is_expert=False)
            else:
                # Increase items fulfilled at this location
                cur_fulfill[action, inv_prod.sku_id] += 1
                print("BEFORE ADD: ", fulfill_plan.inv.product_quantity(inv_prod.sku_id))
                fulfill_plan.add_product(action, InventoryProduct(inv_prod.sku_id, 1))
                print("AFTER ADD: ", fulfill_plan.inv.product_quantity(inv_prod.sku_id))

                # Decrease inventory at node
                inv[action, inv_prod.sku_id] -= 1
                demand[inv_prod.sku_id] -= 1

                # Recursively search this fulfillment path
                print("GOING IN", action)
                child_plan, child_val, child_exps = self._fulfill_search(inv_nodes,
                            demand_node,
                            inv,
                            inv_locs,
                            demand,
                            demand_loc,
                            cur_fulfill,
                            fulfill_plan,
                            sku_distr,
                            inv_prods,
                            inv_prod_idx,
                            cur_val + reward,
                            exps + [Experience(state, action, reward, is_expert=False)])
                print("OUT BRUH", action)

                if best_val is None or child_val > best_val:
                    best_val = child_val
                    best_plan = child_plan
                    best_exps = child_exps

                # Backtrack
                cur_fulfill[action, inv_prod.sku_id] -= 1
              
                fulfill_plan.remove_product(action, InventoryProduct(inv_prod.sku_id, 1))
                inv[action, inv_prod.sku_id] += 1
                demand[inv_prod.sku_id] += 1


        if is_last_item:
            best_plan = best_plan.copy()
            best_plan.add_product(
                best_last_action,
                InventoryProduct(inv_prod.sku_id, 1))
            best_exps = exps.copy()
            best_exps.append(best_last_exp)
            
        return best_plan, best_val, best_exps

    
    def __call__(self,
                inv_nodes: list,
                demand_node: DemandNode,
                sku_distr: torch.Tensor,
                argmax=False) -> PolicyResults:
        """Create a fulfillment decision for the DemandNode using a RL based policy.
        
        Args:
            list of InventoryNodes.
            demand_node: the DeamndNode representing the current order.
        
        Returns:
            the fulfillment decision results.
        """
        inv_locs = torch.zeros(self.args.num_inv_nodes, 2).to(device)

        inv = torch.zeros(self.args.num_inv_nodes, self.args.num_skus).to(device)

        # Create current inventory vector
        for inv_node in inv_nodes:
            inv_locs[inv_node.inv_node_id, 0] = inv_node.loc.coords.x
            inv_locs[inv_node.inv_node_id, 1] = inv_node.loc.coords.y
            inv[inv_node.inv_node_id] = torch.tensor(inv_node.inv.inv_list).to(device)

        # Create demand vector
        demand_loc = torch.zeros(2).to(device)
        demand_loc[0] = demand_node.loc.coords.x
        demand_loc[1] = demand_node.loc.coords.y
        
        demand = torch.tensor(demand_node.inv.inv_list).to(device)

        # Make item fulfillment actions
        cur_fulfill = torch.zeros(self.args.num_inv_nodes, self.args.num_skus).to(device)
        
        inv_prods = list(demand_node.inv.items())

        fulfill_plan, best_val, exps = self._fulfill_search(
                        inv_nodes,
                        demand_node,
                        inv,
                        inv_locs,
                        demand,
                        demand_loc,
                        cur_fulfill,
                        FulfillmentPlan(),
                        sku_distr,
                        inv_prods,
                        0,
                        0,
                        [])
        
        print("best_val", best_val)
        print("exps", exps)
        # Create the results from the order
        results = PolicyResults(fulfill_plan, exps)

        # Set next state of the previous order to the first item in the next order
        # This is required since an episode consists of multiple orders
        if not self.args.eval:
            if len(self._exp_buffer) > 0:
                self._exp_buffer[-1].next_state = results.exps[0].state
            self._exp_buffer.extend(results.exps)



        return results