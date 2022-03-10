import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import math


from nodes import InventoryNode, DemandNode
from reward_manager import RewardManager
from policy import PolicyResults
from rl_policy import RLPolicy
from replay_memory import ReplayMemory, PrioritizedExpReplay
from primal_dual_policy import PrimalDual

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def huber_loss(inp, tgt, delta =1.0, weights=None):
    """Huber loss that accepts weights."""  
    diff = (inp - tgt).abs()
    loss = torch.where(diff < delta, 0.5 * (inp - tgt)**2, delta * (diff - 0.5 * delta))
    if weights is not None:
        loss = loss * weights

    return loss.mean()


class DQNTrainer(RLPolicy):
    """Trainer and policy for DQN RL approach."""
    def __init__(self, args, reward_man: RewardManager, model_file="dqn_model.pt", model_name="DQN"):
        super().__init__(args, reward_man)

        self._model_file = os.path.join(self.args.save_dir, model_file)
        self._model_name = model_name
        self._model_tgt_name = model_name + "_tgt"

        self._dqn = self._create_model()
        self._dqn_target = self._create_model().eval()
        
        self._optim = optim.Adam(self._dqn.parameters(), self.args.lr, weight_decay=self.args.weight_decay)
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

        self._loss_fn = nn.HuberLoss()
        
        if self.args.load:
            self.load()

        if self._train_step < self.args.expert_pretrain:
            self._expert_policy = PrimalDual(self.args, self._reward_man, is_expert=True)
        
        self._gam_arr = self._create_gamma_arr()
        self._last_save_step = 0 
        
    def _create_gamma_arr(self):
        """Create a gamma tensor for multi-step DQN."""
        gam_arr = torch.ones(self.args.dqn_steps)
        for i in range(1, self.args.dqn_steps):
            gam_arr[i] = self.args.gamma * gam_arr[i-1] 
        return gam_arr

    def _create_model(self) -> nn.Module:
        """Create the PyTroch model."""
        return DQN(self.args).to(device)

    def _update_target(self):
        """Perform soft update of the target policy."""
        for tgt_dqn_param, dqn_param in zip(self._dqn_target.parameters(), self._dqn.parameters()):
            tgt_dqn_param.data.copy_(
                self.args.tgt_tau * dqn_param.data + (1.0-self.args.tgt_tau) * tgt_dqn_param.data)
        #self._dqn_target.load_state_dict(self._dqn.state_dict())

    @property
    def epsilon_threshold(self):
        """Return the current epsilon value used for epsilon-greedy exploration."""
        cur_step = max(self._train_step - self.args.expert_pretrain, 0)
        cur_epsilon = max((1 - cur_step/self.args.decay_steps) * self.args.epsilon , self.args.min_epsilon)
        
        return cur_epsilon
        # return self.args.min_epsilon + (self.args.epsilon - self.args.min_epsilon) * \
        #     math.exp(-1. * cur_step / self.args.epsilon_decay)
    


    def _add_stored_exps(self):
        """Add experinced stored in temporary buffer into replay memory.
        
        This method makes the assumption that self._exp_buffer only contains experiences
        from the same episode.
        """
        rewards = torch.zeros(self.args.dqn_steps)
        for i in reversed(range(len(self._exp_buffer))):
            rewards[0] = self._exp_buffer[i].reward
            cur_gamma = self.args.gamma
            if i + self.args.dqn_steps < len(self._exp_buffer):
                # Update the experince reward to be the n-step return
                # NOTE: n experiences at the end will use 1-step return
                self._exp_buffer[i].reward = rewards.dot(self._gam_arr)
                self._exp_buffer[i].next_state = self._exp_buffer[i + self.args.dqn_steps].state
                cur_gamma = cur_gamma ** self.args.dqn_steps
            # elif i + 1 < len(self._exp_buffer):
            #     self._exp_buffer[i].reward = rewards.dot(self._gam_arr)
            #     self._exp_buffer[i].next_state = self._exp_buffer[-1].next_state
            #     cur_gamma = self.args.gamma ** (len(self._exp_buffer) - i)

            self._exp_buffer[i].gamma = cur_gamma

            if self.args.no_per:
                self._memory.add(self._exp_buffer[i])
            else:
                with torch.no_grad():
                    q_value = self._dqn(self._exp_buffer[i].state)[self._exp_buffer[i].action]
                    if self._exp_buffer[i].next_state is not None:
                        # Get the valid action for next state the maximizes the q-value

                        
                        valid_actions = self._get_valid_actions(self._exp_buffer[i].next_state)
                        q_next = self._dqn(self._exp_buffer[i].next_state)

                        next_action = self.sample_action(q_next[valid_actions], argmax=True)
                        next_action = valid_actions[next_action]


                        # Compute TD target based on target function q-value for next state
                        q_next_target = self._dqn_target(self._exp_buffer[i].next_state)[next_action]

                        td_target = self._exp_buffer[i].reward + self._exp_buffer[i].gamma *  q_next_target

                
                    else:
                        td_target = self._exp_buffer[i].reward

                td_error = td_target - q_value
                self._memory.add(self._exp_buffer[i], td_error)      

            # Shift the rewards down
            rewards = rewards.roll(1)

        # Clear the experiences from the experince buffer
        self._exp_buffer.clear()


    def is_train_ready(self) -> bool:
        """Check for if the model is ready to start training."""
        return self._memory.cur_cap() >= self.args.min_exps

    def save(self):
        """Save the models, optimizer, and other related data."""
        super().save()
        if self._train_step - self._last_save_step >= 256:
            self._last_save_step = self._train_step
            cur_model_file = self._model_file.split(".")[0]
            cur_model_file += f"_{self._train_step}.pt"
            print("cur_model_file", cur_model_file)
        else:
            cur_model_file = None
        
        
        model_dict = {
            self._model_name : self._dqn.state_dict(),
            self._model_tgt_name : self._dqn_target.state_dict(),
            "optimizer" : self._optim.state_dict(),
            "lr_scheduler" : self._lr_scheduler.state_dict(),
            "train_step" : self._train_step,
            "save_step" : self._last_save_step,
            "hyper_dict" : self._hyper_dict
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
        self._dqn.load_state_dict(model_dict[self._model_name])
        if self._model_tgt_name in model_dict:
            self._dqn_target.load_state_dict(model_dict[self._model_tgt_name])
        else:
            self._dqn_target.load_state_dict(model_dict[self._model_name])

        self._optim.load_state_dict(model_dict["optimizer"])
        self._lr_scheduler.load_state_dict(model_dict["lr_scheduler"])
        self._train_step = model_dict["train_step"]
        self._last_save_step = model_dict["save_step"]
        self._hyper_dict = model_dict["hyper_dict"]

        self._optim.param_groups[0]["lr"] = self.args.lr

    def reset(self):
        """Reset for next episode."""
        # Add experiences to memory and empty exp buffer
        self._add_stored_exps()

    def early_stop_handler(self):
        """Handle early termination of this episode."""
        # Pop the last state as the next_state is None
        self._exp_buffer.pop()

    def sample_action(self, q_values: torch.Tensor, argmax=False) -> int:
        """Sample an action from the given q-values."""
        if not argmax and self.epsilon_threshold >= np.random.rand():
            # Perform random action
            action = np.random.randint(q_values.shape[0])
        else:
            with torch.no_grad():
                # Perform action that maximizes expected return
                action =  q_values.max(0)[1]
        action = int(action)

        return action

    def _unwrap_exps(self, exps):
        """Extract the states, actions and rewards from the experiences."""
        states = torch.zeros(self.args.batch_size, self._dqn.inp_size).to(device)
        actions = []
        rewards = torch.zeros(self.args.batch_size).to(device)
        next_states = torch.zeros(self.args.batch_size, self._dqn.inp_size).to(device)
        next_state_mask = torch.zeros(self.args.batch_size).to(device)
        is_experts = torch.zeros(self.args.batch_size, dtype=torch.bool).to(device)
        gammas = torch.zeros(self.args.batch_size).to(device)
        valid_actions = torch.zeros(self.args.batch_size, self.args.num_inv_nodes).to(device)

        # Unwrap the experiences
        for i, exp in enumerate(exps):
            states[i] = exp.state
            actions.append(exp.action)
            rewards[i] = exp.reward
            is_experts[i] = exp.is_expert
            gammas[i] = exp.gamma
            if exp.next_state is not None:
                next_states[i] = exp.next_state 
                next_state_mask[i] = 1
                valid_actions[i, self._get_valid_actions(exp.next_state)] = 1

                assert valid_actions[i].sum().item() > 0
        return states, actions, rewards, next_states, next_state_mask, is_experts, gammas, valid_actions

    def train(self) -> float:
        """Train the model over a sampled batch of experiences.
        
        Returns:
            the loss for the batch
        """
        if self.args.no_per:
            exps = self._memory.sample(self.args.batch_size)
        else:
            is_ws, exps, indices = self._memory.sample(self.args.batch_size, self._train_step)
        
        td_targets = torch.zeros(self.args.batch_size).to(device)
        
        states, actions, rewards, next_states, next_state_mask, is_experts, gammas, valid_actions = self._unwrap_exps(exps)

        # Select the q-value for every state
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        
        q_values_matrix = self._dqn(states)
        q_values = q_values_matrix.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Run policy on nonempty states
        nonzero_next_states = next_state_mask.nonzero().flatten()

        # Get next q-values
        q_next = self._dqn(next_states).detach() 

        # Add by factor to ignore invalid q-values for argmax
        q_next += 1e4
        
        # Mask out invalid actions
        q_next = q_next * valid_actions
        next_actions = q_next.argmax(1)

        # Get target q_values
        q_next_target = self._dqn_target(next_states).detach()
        target_vals = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_vals = target_vals * next_state_mask
        td_targets = rewards + gammas * target_vals


        # index used for getting the next nonempty next state
        # q_next_idx = 0
        
        expert_margin_loss = 0

        # for i in range(self.args.batch_size):
        #     # Compute expert margin classification loss (i.e., imitation loss)
        #     if is_experts[i]:
        #         margin_mask = torch.ones(self.args.num_inv_nodes).to(device)
        #         # Mask out the expert action    
        #         margin_mask[actions[i]] = 0
                
        #         # Set to margin value
        #         margin_mask = margin_mask * self.args.expert_margin
                
        #         # Compute the expert imitation loss
        #         expert_margin_loss += torch.max(q_values_matrix[i] + margin_mask) - q_values[i]

            
        #     if not next_state_mask[i]:
        #         td_targets[i] = rewards[i]
        #     else:
        #         # Get the argmax next action for DQN
        #         valid_actions = self._get_valid_actions(next_states[i])

        #         action = self.sample_action(
        #             q_next[q_next_idx][valid_actions], True)
        #         action = int(valid_actions[action])

        #         # Set TD Target using the q-value of the target network
        #         # This is the Double-DQN target
        #         td_targets[i] = rewards[i] + gammas[i] * q_next_target[q_next_idx, action]
        #         q_next_idx += 1
 
        self._optim.zero_grad()

        # Compute loss
        td_errors = td_targets.detach() - q_values
        loss_fac = 100
        if self.args.no_per:
            loss = torch.mean(td_errors ** 2)
        else:
            #loss =self._loss_fn(q_values * is_ws, td_targets * is_ws)
            loss = loss_fac * huber_loss(q_values,td_targets.detach(), weights=is_ws)
            self._memory.update_priorities(indices, td_errors.detach().abs(), is_experts)

        #loss += expert_margin_loss * self.args.expert_lam
        loss.backward()
        
        # Clip gradient
        nn.utils.clip_grad.clip_grad_norm_(
            self._dqn.parameters(),
            self.args.max_grad_norm)


        # Train model
        self._optim.step()

        # Check if using decay and min lr not reached
        if not self.args.no_lr_decay:
            if self._optim.param_groups[0]["lr"] > self.args.min_lr:
                # If so, decay learning rate
                self._lr_scheduler.step()
            else:
                self._optim.param_groups[0]["lr"] = self.args.min_lr

        self._train_step += 1

        if self._train_step % self.args.tgt_update_step == 0:
            #print("\nUPDATING TARGET\n")
            self._update_target()
        
        # Print out q_values and td_targets for debugging/progress updates
        if (self._train_step + 1) % 64 == 0:
            print("self.epsilon_threshold", self.epsilon_threshold)
            print("LR", self._optim.param_groups[0]["lr"])
            print("q_values", q_values)
            print("td_targets", td_targets)
            print("loss", loss)
            print("self._dqn._q_adv_1.weight.grad", self._dqn._q_adv_1.weight.grad.mean(), self._dqn._q_adv_1.weight.grad.min(), self._dqn._q_adv_1.weight.grad.max())
            print("self._dqn.inv_encoder._inv_emb_fc_1.weight.grad", self._dqn.inv_encoder._inv_emb_fc_1.weight.grad.mean(), self._dqn.inv_encoder._inv_emb_fc_1.weight.grad.min(), self._dqn.inv_encoder._inv_emb_fc_1.weight.grad.max())
            print("is_ws:", is_ws, "\n")
            # print("td_error", td_errors)
            # print("target_vals", target_vals)
            # print("td_error ** 2", td_errors ** 2)
            # print("nonzero_next_states", nonzero_next_states)
            # print("next_state_mask", next_state_mask)
            # print("is_ws", is_ws)
            # print("loss", loss)

        return float(loss.detach() / loss_fac)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Make a prediction on the q_values, used for superclsas __call__."""
        q_values = self._dqn(state)
        return q_values

    def __call__(self, 
                inv_nodes: list,
                demand_node: DemandNode,
                sku_distr: torch.Tensor,
                argmax=False) -> PolicyResults:
        
        # Call super class to get PolicyResults        
        results = super().__call__(inv_nodes, demand_node, sku_distr, argmax)
        
        # Set next state of the previous order to the first item in the next order
        # This is required since an episode consists of multiple orders
        if not self.args.eval:
            if len(self._exp_buffer) > 0:
                self._exp_buffer[-1].next_state = results.exps[0].state
            self._exp_buffer.extend(results.exps)

        return results

    def compute_return(self,
        rewards: list) -> torch.Tensor:
        """Compute the return for an episode."""
        with torch.no_grad():
            returns = torch.zeros_like(rewards, dtype=torch.float32).to(device)
            T = len(rewards)
            for t in reversed(range(T)):
                # Check if it is a terminal state
                if t == T - 1:
                    returns[t] = rewards[t]
                else:
                    returns[t] = rewards[t] + self.args.gamma * returns[t+1]

            return returns

class DQN(nn.Module):
    """PyTorch module for the FFN DQN policy."""
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Inventory nodes + demand left + currently allocated items + one-hot encoding for item to allocate + locations coordinates
        self.inp_size = self.args.num_inv_nodes * self.args.num_skus + \
                    self.args.num_skus + \
                    self.args.num_inv_nodes * self.args.num_skus + \
                    self.args.num_skus + \
                    self.args.num_inv_nodes * 2  + 2

        self.action_dim = self.args.num_inv_nodes

        self._fc_1 = nn.Linear(self.inp_size, self.args.hidden_size)

        # Create hidden fcs
        self.hidden_fcs =  nn.ModuleList([
            nn.Linear(self.args.hidden_size, self.args.hidden_size) for _ in range(self.args.num_hidden - 1)])
        self._q_out = nn.Linear(self.args.hidden_size, self.action_dim)

    def forward(self, 
                state) -> torch.Tensor:
        """Run the state through the network. """
        x = F.gelu(self._fc_1(state))
        for hidden_fc in self.hidden_fcs:
            x = F.gelu(hidden_fc(x))
        return self._q_out(x)