import torch, random, os
import torch.nn as nn
import torch.optim as optim

from reward_manager import RewardManager
from shared_models import InvEncoder
from transformer import Encoder
from config import *
from nodes import InventoryNode, DemandNode, InventoryProduct
from fulfillment_plan import FulfillmentPlan
from policy import PolicyResults, Policy, Experience
from reward_manager import RewardManager
from rl_policy import RLPolicy
from replay_memory import ReplayMemory, PrioritizedExpReplay

def huber_loss(inp, tgt, delta =1.0, weights=None):
    """Huber loss that accepts weights."""  
    diff = (inp - tgt).abs()
    loss = torch.where(diff < delta, 0.5 * (inp - tgt)**2, delta * (diff - 0.5 * delta))
    if weights is not None:
        loss = loss * weights

    return loss.mean()

class ValueLookaheadPolicy(Policy):
    """Fulfills orders based on the predicted best order fulfillment for an order."""
    def __init__(self, args, reward_man: RewardManager, model_file="val_emb_model.pt", model_name="ValueLookaheadPolicy"):
        super().__init__(args, True, load_args=True)

        self._reward_man = reward_man
        self._model_file = os.path.join(self.args.save_dir, model_file)
        self._model_name = model_name
        self._model_tgt_name = model_name + "_tgt"


        # Temporary storage space for the experiences b/c timesteps span multiple orders
        self._exp_buffer = []

        self._val_model = ValueEmb(self.args).to(device)
        self._val_model_tgt = ValueEmb(self.args).to(device)
        self._val_model_tgt.eval()

        self._optim = optim.Adam(self._val_model.parameters(), self.args.lr)
        if self.args.no_per:
            self._memory = ReplayMemory(self.args)
        else:
            self._memory = PrioritizedExpReplay(self.args)
        
        self._train_step = 0
        self._last_save_step = 0

        self._gam_arr = self._create_gamma_arr()

        if self.args.load:
            self.load()

    def _create_gamma_arr(self):
        """Create a gamma tensor for multi-step DQN."""
        gam_arr = torch.ones(self.args.dqn_steps)
        for i in range(1, self.args.dqn_steps):
            gam_arr[i] = self.args.gamma * gam_arr[i-1] 
        return gam_arr


    def _update_target(self):
        """Perform soft update of the target policy."""
        self._val_model_tgt.load_state_dict(self._val_model.state_dict())
        self._val_model_tgt.eval()

    @property
    def epsilon_threshold(self):
        """Return the current epsilon value used for epsilon-greedy exploration."""
        cur_epsilon = max((1 - self._train_step/self.args.decay_steps) * self.args.epsilon , self.args.min_epsilon)
        
        return cur_epsilon

    def _create_state(self,
                inv: torch.Tensor,
                inv_locs: torch.Tensor,
                sku_distr: torch.Tensor) -> torch.Tensor:
        inv_locs = inv_locs / inv_locs_scale_factor

        # Expected fulfillment
        expec_fulfill = (inv * sku_distr).sum(axis=1).to(device).unsqueeze(1)

        # Fulfillment variance 
        var_fulfill = (((inv - expec_fulfill) ** 2 * sku_distr).sum(axis=1)).to(device).unsqueeze(1)
        
        # Total stock at each inventory node
        total_stock = inv.sum(axis=1).to(device).unsqueeze(1) / inv_scale_factor

        state = torch.cat((inv_locs, expec_fulfill, var_fulfill, total_stock), 1).unsqueeze(0)

        return state

    def _add_stored_exps(self):
        """Add experinced stored in temporary buffer into replay memory.
        
        This method makes the assumption that self._exp_buffer only contains experiences
        from the same episode.
        """
        rewards = torch.zeros(self.args.dqn_steps)
        for i in reversed(range(len(self._exp_buffer))):
            rewards[0] = self._exp_buffer[i].reward
            cur_gamma = self.args.gamma
            # print("i", i)
            # if i + self.args.dqn_steps < len(self._exp_buffer):
            #     # Update the experince reward to be the n-step return
            #     # NOTE: n experiences at the end will use 1-step return
            #     self._exp_buffer[i].reward = rewards.dot(self._gam_arr)
            #     self._exp_buffer[i].next_state = self._exp_buffer[i + self.args.dqn_steps].state
            #     cur_gamma = cur_gamma ** self.args.dqn_steps
            #     print("i", i, cur_gamma)
            # elif i != len(self._exp_buffer) - 1:

            #     self._exp_buffer[i].reward = rewards.dot(self._gam_arr)
            #     cur_gamma = self.args.gamma ** (len(self._exp_buffer) - i - 1)
            #     self._exp_buffer[i].next_state = self._exp_buffer[-1].state
            #     print("i", i, "(len(self._exp_buffer) - i", "cur_gamma", cur_gamma)
                


            self._exp_buffer[i].gamma = cur_gamma

            if self.args.no_per:
                self._memory.add(self._exp_buffer[i])
            else:
                with torch.no_grad():
                    val_pred = self._val_model(self._exp_buffer[i].state)
                    
                    if self._exp_buffer[i].next_state is not None:
                        val_pred_next = self._val_model_tgt(self._exp_buffer[i].next_state)


                        td_target = self._exp_buffer[i].reward + self._exp_buffer[i].gamma * val_pred_next
                    else:
                        td_target = self._exp_buffer[i].reward

                td_error = td_target - val_pred
                self._memory.add(self._exp_buffer[i], td_error.item())

            # Shift the rewards down
            # print("rewards", rewards)
            # print("self._gam_arr", self._gam_arr)
            rewards = rewards.roll(1)


        # Clear the experiences from the buffer
        self._exp_buffer.clear()

    def _unwrap_exps(self, exps):
        """Extract the states, actions and rewards from the experiences."""
        states = torch.zeros(self.args.batch_size, self.args.num_inv_nodes, self._val_model.inp_size).to(device)
        rewards = torch.zeros(self.args.batch_size).to(device)
        next_states = torch.zeros(self.args.batch_size, self.args.num_inv_nodes, self._val_model.inp_size).to(device)
        next_state_mask = torch.zeros(self.args.batch_size).to(device)
        gammas = torch.zeros(self.args.batch_size).to(device)

        # Unwrap the experiences
        for i, exp in enumerate(exps):
            states[i] = exp.state
            rewards[i] = exp.reward
            gammas[i] = exp.gamma
            if exp.next_state is not None:
                next_states[i] = exp.next_state 
                next_state_mask[i] = 1
                # print("NEXT STATE IS NONE")


        return states, rewards, next_states, next_state_mask.nonzero().flatten(), gammas


    def compute_lower_bound(self, states, rewards):
        max_reward = -1
        return torch.clip((states[:, :, -1].sum(axis=1) * max_reward * inv_scale_factor),  min=-1/(1-self.args.gamma)) + rewards


    def train(self) -> float:
        """Train the model over a sampled batch of experiences.
        
        Returns:
            the loss for the batch
        """
        if self.args.no_per:
            exps = self._memory.sample(self.args.batch_size)
        else:
            is_ws, exps, indices = self._memory.sample(self.args.batch_size, self._train_step)
        
        
        states, rewards, next_states, next_state_mask, gammas = self._unwrap_exps(exps)

        td_targets = rewards.clone()
        with torch.no_grad():
            td_targets[next_state_mask] = td_targets[next_state_mask] + gammas[next_state_mask] * self._val_model_tgt(next_states[next_state_mask])

        
        # Force it to not assume state-value are greater than the reward (as rewards are strictly negative)
        reward_upper_bound = self.compute_lower_bound(next_states, rewards)
        
        td_targets = torch.clip(td_targets, max=rewards, min=reward_upper_bound)

        val_preds = self._val_model(states)

        self._optim.zero_grad()
        # Compute loss
        td_errors = td_targets.detach() - val_preds
        if self.args.no_per:
            loss = torch.mean(td_errors ** 2)
        else:
            loss = torch.mean(td_errors ** 2 * is_ws)
            #loss = huber_loss(val_preds, td_targets.detach(), weights=is_ws)
            self._memory.update_priorities(indices, td_errors.detach().abs())
        
        loss.backward()

        # Clip gradient
        nn.utils.clip_grad.clip_grad_norm_(
            self._val_model.parameters(),
            self.args.max_grad_norm)

        # Train model
        self._optim.step()

        self._train_step += 1
        
        if self._train_step % self.args.tgt_update_step == 0:
            print("\nUPDATING TARGET\n")
            self._update_target()

        
        if (self._train_step + 1) % 64 == 0:
            print("val_preds", val_preds)
            print("td_targets", td_targets)
            print("loss", loss)
            print("rewards", rewards)
            print("reward_upper_bound", reward_upper_bound)
            print("gammas", gammas)
            print("Epsilon threshold:", self.epsilon_threshold)
        
        return loss.item()

    def save(self):
        """Save the models, optimizer, and other related data."""
        super().save()
        if self._train_step - self._last_save_step >= 1024:
            self._last_save_step = self._train_step
            cur_model_file = self._model_file.split(".")[0]
            cur_model_file += f"_{self._train_step}.pt"
            print("cur_model_file", cur_model_file)
        else:
            cur_model_file = None
        
        
        model_dict = {
            self._model_name : self._val_model.state_dict(),
            self._model_tgt_name : self._val_model_tgt.state_dict(),
            "optimizer" : self._optim.state_dict(),
            "train_step" : self._train_step,
            "save_step" : self._last_save_step,
        }

        if cur_model_file is not None:
            torch.save(model_dict, cur_model_file)    
        
        torch.save(model_dict, self._model_file)

        # Save the experience replay
        self._memory.save()


    def load(self):
        """Load the models, optimizer, and other related data."""
        print("LOADING self._model_file", self._model_file, "\n")
        model_dict = torch.load(self._model_file, map_location=device)
        
        # Load the models
        self._val_model.load_state_dict(model_dict[self._model_name])
        self._val_model_tgt.load_state_dict(model_dict[self._model_tgt_name])
        self._optim.load_state_dict(model_dict["optimizer"])
        
        self._train_step = model_dict["train_step"]
        self._last_save_step = model_dict["save_step"]

        # Set the learning rate to the one currently given in the args
        self._optim.param_groups[0]["lr"] = self.args.lr

    def reset(self):
        """Reset for next episode."""
        # Add experiences to memory and empty exp buffer
        self._add_stored_exps()


    def is_train_ready(self) -> bool:
        """Check for if the model is ready to start training."""
        return self._memory.cur_cap() >= self.args.min_exps

    def _get_valid_actions(self, inv, item_hot):
        return (inv * item_hot).sum(axis=1).nonzero()     

    def _fulfill_search(self,
                        inv_nodes,
                        demand_node,
                        inv,
                        inv_locs,
                        fulfill_plan,
                        sku_distr,
                        inv_prods,
                        inv_prod_idx,
                        cur_val,
                        cur_depth=0):
        inv_prod = inv_prods[inv_prod_idx]
        if fulfill_plan.inv.product_quantity(inv_prod.sku_id) >= inv_prod.quantity:
            # Start fulfilling the next item in the order 
            inv_prod_idx += 1
            inv_prod = inv_prods[inv_prod_idx]
        

        # Create one hot encoded vector for the current item selection
        item_hot = torch.zeros(self.args.num_skus).to(device)
        item_hot[inv_prod.sku_id] = 1

        
        # Get indices of nodes that have nonzero inventory
        valid_idxs = self._get_valid_actions(inv, item_hot)
        best_plan = best_val = best_last_action = best_reward = best_last_state = None
        is_last_item = fulfill_plan.inv.product_quantity(inv_prod.sku_id) + 1 >= inv_prod.quantity and \
                inv_prod_idx + 1 >= len(inv_prods)

        # print("\nvalid_idxs", valid_idxs)
        # print("inv_prod_idx", inv_prod_idx, "is_last_item", is_last_item)
        # print("fulfill_plan.inv.product_quantity(inv_prod.sku_id)", fulfill_plan.inv.product_quantity(inv_prod.sku_id))


        # Iterate over every action
        for valid_idx in valid_idxs:
            action = int(valid_idx)

            # Get the reward for routing to this inventory node
            reward = self._reward_man.get_reward(
                            inv_nodes[int(valid_idx)],
                            demand_node,
                            fulfill_plan)


            if is_last_item:
                # Decrease inventory at node
                inv[action, inv_prod.sku_id] -= 1

                state = self._create_state(
                    inv,
                    inv_locs,
                    sku_distr)

                 
                # Set the next_state to None if the inventory is empty
                total_stock = state[:, :, -1].sum().item()
                assert total_stock >= 0

                if total_stock == 0:
                    state = None
                    val_pred = 0
                else:
                    # Get prediction for future value of routing decision as this is the last item in order
                    val_pred = self._val_model(state).item()
                    #print("val_pred", val_pred)
                    
                    


                # Backtrack inventory
                inv[action, inv_prod.sku_id] += 1

                # Compute the value of this fulfillment plan
                ## (Subtract reward as model_pred contains estimate of current reward)
                child_val = cur_val + reward  +  val_pred * self.args.gamma

                if not self.args.eval and self.epsilon_threshold >= random.random():
                    child_val += -1 * random.random()


                # Update the best possible order fulfillment
                if best_val is None or child_val > best_val:
                    best_val = child_val
                    best_plan = fulfill_plan
                    best_last_action = action
                    best_last_state = state
                    best_reward = cur_val + reward
            else:
                # Increase items fulfilled at this location
                fulfill_plan.add_product(action, InventoryProduct(inv_prod.sku_id, 1))

                # Decrease inventory at node
                inv[action, inv_prod.sku_id] -= 1

                # Recursively search this fulfillment path
                child_plan, child_val, last_state, total_reward = self._fulfill_search(
                    inv_nodes,
                    demand_node,
                    inv,
                    inv_locs,
                    fulfill_plan,
                    sku_distr,
                    inv_prods,
                    inv_prod_idx,
                    cur_val + reward,
                    cur_depth + 1)

                if best_val is None or (child_val > best_val or (not self.args.eval and self.epsilon_threshold >= random.random())):
                    best_val = child_val
                    best_plan = child_plan
                    best_last_state = last_state
                    best_reward = total_reward
                    

                # Backtrack
                fulfill_plan.remove_product(action, InventoryProduct(inv_prod.sku_id, 1))
                inv[action, inv_prod.sku_id] += 1


        if is_last_item:
            best_plan = best_plan.copy()
            best_plan.add_product(
                best_last_action,
                InventoryProduct(inv_prod.sku_id, 1))
            

        return best_plan, best_val, best_last_state, best_reward

    
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
        
        demand = torch.tensor(demand_node.inv.inv_list).to(device)
        
        inv_prods = list(demand_node.inv.items())

        state = self._create_state(inv, inv_locs, sku_distr)

        fulfill_plan, _, next_state, reward = self._fulfill_search(
                        inv_nodes,
                        demand_node,
                        inv,
                        inv_locs,
                        FulfillmentPlan(),
                        sku_distr,
                        inv_prods,
                        0,
                        0)
        

        
        # Create the results from the order
        exp = Experience(state, None, reward, next_state)
        results = PolicyResults(fulfill_plan, [exp])

        if not self.args.eval:
            self._exp_buffer.append(exp)

        return results

class ValueEmb(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.inp_size = 5
        self.inv_encoder = InvEncoder(self.args, inv_inp_size=self.inp_size)

        self.state_enc = Encoder(self.args, self.args.num_enc_layers)
        self.val_pred_emb = nn.Parameter(torch.randn(1, 1, self.args.emb_size))

        self._val_out_1 = nn.Linear(self.args.emb_size, self.args.emb_size)
        self._val_act_1 = nn.GELU()
        self._val_out_2 = nn.Linear(self.args.emb_size, 1)


    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        val_embs = self.val_pred_emb.repeat(batch_size, 1, 1)
        
        # Get the inventory node embeddings
        inv_embs = self.inv_encoder(state)

        # Run through transformer encoder
        embs = torch.cat((val_embs, inv_embs), 1)
        embs = self.state_enc(embs)

        # Get value prediction based on the value prediction embedding
        val_out = self._val_act_1(
            self._val_out_1(embs[:, 0]))
        
        val_out = self._val_out_2(val_out)
        #print("val_out", val_out)
        if batch_size == 1:
            return val_out.view(-1)
        else:
            return val_out.view(-1)





