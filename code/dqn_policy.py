import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import math

from simulator import InventoryNode, DemandNode, InventoryProduct
from reward_manager import RewardManager
from fulfillment_plan import FulfillmentPlan
from policy import PolicyResults, Policy, Experience
from replay_memory import ReplayMemory, PrioritizedExpReplay

class DQNTrainer(Policy):
    def __init__(self, args, reward_man: RewardManager):
        super().__init__(True)
        self.args = args
        self._reward_man = reward_man
        self._model_file = os.path.join(self.args.save_dir, "dqn_model.pt")
        self._dqn = DQN(self.args)
        self._dqn_target = DQN(self.args)
        self._optim = optim.Adam(self._dqn.parameters(), self.args.lr)
        self._train_step = 0
        self._lr_scheduler = optim.lr_scheduler.MultiplicativeLR(
            self._optim,
            lr_lambda=lambda e: self.args.lr_gamma)
        
        # Replay memory to store experiences
        if self.args.no_per:
            self._memory = ReplayMemory(self.args)
        else:
            self._memory = PrioritizedExpReplay(self.args)
        
        # Temporary storage space for the experiences b/c timesteps span multiple orders
        self._exp_buffer = []

        self._loss_fn = nn.SmoothL1Loss()
        
        if self.args.load:
            self.load()

    def _update_target(self):
        self._dqn_target.load_state_dict(self._dqn.state_dict())

    @property
    def epsilon_threshold(self):
        return self.args.min_epsilon + (self.args.epsilon - self.args.min_epsilon) * \
            math.exp(-1. * self._train_step / self.args.epsilon_decay)

    def _get_valid_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Get valid actions for the current state vector."""
        #Derive the inventory and SKU ID from the state vector 
        inv = state[:self.args.num_inv_nodes * self.args.num_skus].reshape(self.args.num_inv_nodes, self.args.num_skus)
        sku_id = int(state[-self.args.num_skus:].nonzero())

        valid_actions = (inv[:, sku_id] > 0).nonzero().flatten()
        return valid_actions 
    

    def _add_stored_exps(self):
        for exp in self._exp_buffer:
            if self.args.no_per:
                self._memory.add(exp)
            else:
                with torch.no_grad():
                    q_value = self._dqn(exp.state)[exp.action]
                    if exp.next_state is not None:
                        # Get the valid action for next state the maximizes the q-value
                        valid_actions = self._get_valid_actions(exp.next_state)
                        q_next = self._dqn(exp.next_state)
                        next_action = self.sample_action(q_next[valid_actions], argmax=True)
                        next_action = valid_actions[next_action] 

                        # Compute TD target based on target function q-value for next state
                        q_next_target = self._dqn_target(exp.next_state)[next_action]
                        td_target = exp.reward + self.args.gamma *  q_next_target
                    else:
                        td_target = exp.reward

                td_error = td_target - q_value
                self._memory.add(exp, td_error)
        
        self._exp_buffer.clear()

    def is_train_ready(self) -> bool:
        return self._memory.cur_cap() >= self.args.batch_size

    def save(self):
        model_dict = {
            "DQN" : self._dqn.state_dict(),
            "optimizer" : self._optim.state_dict(),
            "lr_scheduler" : self._lr_scheduler.state_dict(),
            "train_step" : self._train_step
        }

        torch.save(model_dict, self._model_file)
    
    def load(self):
        model_dict = torch.load(self._model_file)
        self._dqn.load_state_dict(model_dict["DQN"])
        self._dqn_target.load_state_dict(model_dict["DQN"])
        self._optim.load_state_dict(model_dict["optimizer"])
        self._lr_scheduler.load_state_dict(model_dict["lr_scheduler"])
        self._train_step = model_dict["train_step"]

    def reset(self):
        # Add experiences to memory and empty exp buffer
        self._add_stored_exps()

    def sample_action(self, q_values: torch.Tensor, argmax=False) -> int:
        if not argmax and self.epsilon_threshold >= np.random.rand():
            # Perform random action
            action = np.random.randint(q_values.shape[0])
        else:
            with torch.no_grad():
                # Perform action that maximizes expected return
                action =  q_values.max(0)[1]
        action = int(action)

        return action

    def train(self) -> float:
        if self.args.no_per:
            exps = self._memory.sample(self.args.batch_size)
        else:
            is_ws, exps, indices = self._memory.sample(self.args.batch_size)
        td_targets = torch.zeros(self.args.batch_size)
        states = torch.zeros(self.args.batch_size, self._dqn.inp_size)
        next_states = torch.zeros(self.args.batch_size, self._dqn.inp_size)
        rewards = torch.zeros(self.args.batch_size)
        next_state_mask = torch.zeros(self.args.batch_size)
        actions = []

        # Create state-action values
        for i, exp in enumerate(exps):
            states[i] = exp.state
            actions.append(exp.action)
            rewards[i] = exp.reward
            if exp.next_state is not None:
                next_states[i] = exp.next_state 
                next_state_mask[i] = 1

        # Select the q-value for every state
        actions = torch.tensor(actions, dtype=torch.int64)

        q_values = self._dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_next  = self._dqn(next_states)
        q_next_target = self._dqn_target(next_states).detach()
        for i in range(self.args.batch_size):
            if not next_state_mask[i]:
                td_targets[i] = rewards[i]
            else: 
                # Get the argmax next action for DQN
                valid_actions = self._get_valid_actions(next_states[i])
                action = self.sample_action(q_next[i][valid_actions], True)
                action = int(valid_actions[action])

                # Set TD Target using the q-value of the target network
                # This is the Double-DQN target
                td_targets[i] = rewards[i] + self.args.gamma * q_next_target[i, action]

        self._optim.zero_grad()

        # Compute loss
        td_errors = q_values - td_targets
        if self.args.no_per:
            loss = torch.mean(td_errors ** 2)
        else:
            loss = torch.mean(td_errors ** 2  *  is_ws)
            self._memory.update_priorities(indices, td_errors.detach().abs())
        
        loss.backward()
        
        # Clip gradient
        nn.utils.clip_grad.clip_grad_norm_(
            self._dqn.parameters(),
            self.args.max_grad_norm)
        
        # Train model
        self._optim.step()

        # Check if using decay and min lr not reached
        if not self.args.no_lr_decay and self._optim.param_groups[0]["lr"] > self.args.min_lr:
            # If so, decay learning rate
            self._lr_scheduler.step()

        self._train_step += 1

        if self._train_step % self.args.tgt_update_step:
            self._update_target()

        return float(loss.detach())

    def __call__(self, 
                inv_nodes: list[InventoryNode],
                demand_node: list[DemandNode],
                argmax=False) -> FulfillmentPlan:

        inv = torch.zeros(self.args.num_inv_nodes, self.args.num_skus)

        # Create current inventory vector
        for inv_node in inv_nodes:
            for sku_id in range(self.args.num_skus):
                inv[inv_node.inv_node_id, sku_id] = inv_node.inv.product_quantity(sku_id)

        # Create demand vector
        demand = torch.zeros(self.args.num_skus)
        for sku_id in range(self.args.num_skus):
            demand[sku_id] = demand_node.inv.product_quantity(sku_id)

        # Keep up with fulfillment requests for every inventory node
        fulfill_plan = FulfillmentPlan()

        # Make item fulfillment actions
        cur_fulfill = torch.zeros(self.args.num_inv_nodes, self.args.num_skus)
        exps  = []
        for i, inv_prod in enumerate(demand_node.inv.items()):
            for j in range(inv_prod.quantity):
                # Create one hot encoded vector for the current item selection
                item_hot = torch.zeros(self.args.num_skus)
                item_hot[inv_prod.sku_id] = 1

                # Create the current state
                state = torch.cat((inv.flatten(), demand, cur_fulfill.flatten(), item_hot))

                if i != 0 or j != 0:
                    # Update the previous experience next state
                    exps[-1].next_state = state

                # Run through policy
                q_values = self._dqn(state)

                # Get indicies of nodes that have nonzero inventory
                valid_idxs = (inv[:, inv_prod.sku_id] > 0).nonzero().flatten()

                # Select an inventory node
                action = self.sample_action(q_values[valid_idxs], argmax)
                action = int(valid_idxs[action])

                reward = self._reward_man.get_reward(
                                inv_nodes[action],
                                demand_node,
                                fulfill_plan)

                # Add experience
                exps.append(
                    Experience(state.clone(), action, reward))

                # Increase items fulfilled at this location
                cur_fulfill[action, inv_prod.sku_id] += 1
                fulfill_plan.add_product(action, InventoryProduct(inv_prod.sku_id, 1))
                
                # Decrease inventory at node
                inv[action, inv_prod.sku_id] -= 1
                demand[inv_prod.sku_id] -= 1

        # Create the results from the order
        results = PolicyResults(fulfill_plan, exps)
        
        # Set next state of the previous order to the first item in the next order
        if len(self._exp_buffer) > 0:
            self._exp_buffer[-1].next_state = results.exps[0].state
        self._exp_buffer.extend(results.exps)
        return results

class DQN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Inventory nodes + demand left + currently allocated items + one-hot encoding for item to allocate
        self.inp_size = self.args.num_inv_nodes * self.args.num_skus + \
                    self.args.num_skus + \
                    self.args.num_inv_nodes * self.args.num_skus + \
                    self.args.num_skus

        self.action_dim = self.args.num_inv_nodes

        self._fc_1 = nn.Linear(self.inp_size, self.args.hidden_size)
        self._fc_2 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self._q_out = nn.Linear(self.args.hidden_size, self.action_dim)

    def __call__(self, 
                state) -> torch.Tensor:
    
        x = F.relu(self._fc_1(state))
        x = F.relu(self._fc_2(x))
        return self._q_out(x)