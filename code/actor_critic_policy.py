import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.distributions import Categorical

from simulator import InventoryNode, DemandNode, InventoryProduct
from reward_manager import RewardManager
from fulfillment_plan import FulfillmentPlan
from policy import PolicyResults, Policy, Experience
from rl_policy import RLPolicy
from transformer import Encoder, Decoder
from shared_models import DemandEncoder, InvEncoder 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExpBuffer:
    def __init__(self, action_dim):
        self.ep_reset = True
        self.action_dim = action_dim
        self.clear()

    def add_exp(self, exp: Experience):
        #print(exp)
        self.states.append(exp.state)
        self.actions.append(exp.action)
        self.rewards.append(exp.reward)
        self.next_states.append(exp.next_state)

    def add_val_pred(self, val_pred):
        self.val_preds.append(val_pred)

    def add_actor_pred(self, actor_pred):
        self.actor_preds.append(actor_pred.tolist())

    def convert_to_tensor(self):
        """Convert all saved lists to tensors."""
        #print(len(self.states), len(self.rewards), len(self.actions), len(self.actor_preds), len(self.val_preds))
        #print("self.states", self.states)
        self.states = torch.stack(self.states)
        #print("AFTER", self.states)
        #self.next_states = torch.stack(self.next_states)
        self.rewards = torch.tensor(self.rewards).to(device)
        
        # Create one-hot action vectors
        # one_hot_actions = torch.zeros(len(self.actions), self.action_dim)
        # self.actions = torch.tensor(self.actions).unsqueeze(1)
        # self.actions = one_hot_actions.scatter_(1, self.actions, 1)
        
        self.actions = torch.tensor(self.actions).unsqueeze(1).to(device)

        self.actor_preds = torch.tensor(self.actor_preds).to(device)
        self.val_preds = torch.tensor(self.val_preds).to(device)
        # print(self.states, self.rewards, self.actions, self.actor_preds, self.val_preds)

    def pop(self):
        """Pop the last state."""
        self.states.pop()
        self.actions.pop()
        self.rewards.pop()
        self.next_states.pop()
        self.val_preds.pop()
        self.actor_preds.pop()

    def clear(self):
        self.ep_reset = True
        self.states = []
        self.next_states = []
        self.rewards = []
        self.actions = []
        self.val_preds = []
        self.actor_preds = []
        self.num_ep = 0

class ActorCriticPolicy(RLPolicy):
    def __init__(self, args, reward_man: RewardManager):
        super().__init__(args, reward_man)

        self._model_file = os.path.join(
            self.args.save_dir, "actor_critic_model.pt")
        self._actor_critic = ActorCritic(self.args).to(device)
        self._optim = optim.Adam(
            self._actor_critic.parameters(), self.args.lr)
        self._lr_scheduler = optim.lr_scheduler.MultiplicativeLR(
            self._optim,
            lr_lambda=lambda e: self.args.lr_gamma)

        # Critic Huber loss function
        self.critic_loss_fn = nn.SmoothL1Loss()

        self._exp_buffer = ExpBuffer(self.args.num_inv_nodes)

        # Load model saved model parameters
        if self.args.load:
            self.load()

    def sample_action(self, actor_pred: torch.Tensor, argmax=False) -> int:
        if len(actor_pred) == 1:
            return 0
        
        if argmax:
            return actor_pred.argmax()
        else:
            action_distr = Categorical(actor_pred)
            action = action_distr.sample()

        return int(action)

    def predict(self, state):
        with torch.no_grad():
            actor_pred, val_pred = self._actor_critic(state)
            self._exp_buffer.add_val_pred(val_pred)
            self._exp_buffer.add_actor_pred(actor_pred)

        return actor_pred

    def is_train_ready(self) -> bool:
        return len(self._exp_buffer.states) > self.args.min_exps

    def save(self):
        super().save()
        model_dict = {
            "actor_critic" : self._actor_critic.state_dict(),
            "optimizer" : self._optim.state_dict(),
            "lr_scheduler" : self._lr_scheduler.state_dict()
        }

        torch.save(model_dict, self._model_file)

    def load(self):
        model_dict = torch.load(self._model_file, map_location=device)
        self._actor_critic.load_state_dict(model_dict["actor_critic"])
        self._optim.load_state_dict(model_dict["optimizer"])
        self._lr_scheduler.load_state_dict(model_dict["lr_scheduler"])

    def train(self) -> float:
        # Convert saved experiences to tensors
        self._exp_buffer.convert_to_tensor()

        num_exs = len(self._exp_buffer.rewards)
        print("num_exs", num_exs)
        num_minibatches = max(num_exs // self.args.batch_size, 1)

        # Compute return, advantages, and TD lambda return for saved experiences
        returns, advs, tdlamret = self.advs_and_returns(
            self._exp_buffer.rewards,
            self._exp_buffer.val_preds,
            self._exp_buffer.next_states)

        total_loss = 0

        print("rewards", self._exp_buffer.rewards[:10])    
        # Iterate over several epochs
        for i in range(self.args.ac_epochs):
            batch_ids = torch.randperm(num_exs)
            
            # Iterate over several batches
            for j in range(num_minibatches):
                start = j * (num_exs // num_minibatches)
                end = (j + 1) * (num_exs // num_minibatches) 
                m_batch_idxs = batch_ids[start:end]

                # Get current actor and critic prediction
                actor_pred, val_pred = self._actor_critic(
                    self._exp_buffer.states[m_batch_idxs])
                if j == 0:
                    print("val_pred.view(-1)", val_pred.view(-1)[:10])
                    print("tdlamret[m_batch_idxs]", tdlamret[m_batch_idxs][:10])
                    print("actor_pred", actor_pred[:5])

                if self.args.vpg:
                    # Compute vanilla actor-critic objective
                    actor_loss, critic_loss = self.vanilla_pg_loss(
                        actor_pred,
                        val_pred,
                        tdlamret[m_batch_idxs].unsqueeze(1),
                        self._exp_buffer.actions[m_batch_idxs],
                        advs[m_batch_idxs],
                        self._exp_buffer.num_ep)
                else:
                    actor_loss, critic_loss = self.ppo_loss(
                        self._exp_buffer.actor_preds[m_batch_idxs],
                        actor_pred,
                        val_pred,
                        tdlamret[m_batch_idxs].unsqueeze(1),
                        self._exp_buffer.actions[m_batch_idxs],
                        advs[m_batch_idxs])

                
                print("critic_loss", critic_loss)
                print("actor_loss", actor_loss)
                loss = actor_loss + critic_loss

                self._update_params(loss)
                total_loss += float(critic_loss.detach())
        
        # Remove experience from buffer
        self._exp_buffer.clear()

        return total_loss 

    def _update_params(self, loss):
        self._optim.zero_grad()
        # Compute gradients
        loss.backward()
        
        # Clip norm of gradient
        nn.utils.clip_grad_norm_(
            self._actor_critic.parameters(),
            self.args.max_grad_norm)

        # Update parameters
        self._optim.step()

        # Check if using decay and min lr not reached
        # print("LR", self._optim.param_groups[0]["lr"])
        # 
        if not self.args.no_lr_decay: 
            if self._optim.param_groups[0]["lr"] > self.args.min_lr:
                # If so, decay learning rate
                self._lr_scheduler.step()
            else:
                self._optim.param_groups[0]["lr"] = self.args.min_lr

    def vanilla_pg_loss(self,
                    actor_pred: torch.Tensor,
                    val_pred: torch.Tensor,
                    val_true: torch.Tensor,
                    actions: torch.Tensor,
                    advs: torch.Tensor,
                    num_ep: int) -> torch.Tensor:
        """Compute vanilla policy gradient actor critic loss."""
        
        # Get negative log probablity
        logprob = torch.log(
            actor_pred.gather(1, actions).view(-1) + self.args.eps) 

        # Verify shapes match up
        assert logprob.shape == advs.shape
        assert val_pred.shape == val_true.shape
        
        # Compute actor loss
        entropy = 0.01 * -torch.sum(actor_pred.gather(1, actions) * torch.log(actor_pred.gather(1, actions) + self.args.eps))
        actor_loss = -(1/num_ep) * torch.sum(logprob * advs) #+ entropy

        # Compute critic loss
        critic_loss = self.args.critic_lam * self.critic_loss_fn(val_pred, val_true)

        return actor_loss, critic_loss

    def ppo_loss(self,
                old_actor_pred: torch.Tensor,
                new_actor_pred: torch.Tensor,
                val_pred: torch.Tensor,
                val_true: torch.Tensor,
                actions: torch.Tensor,
                advs: torch.Tensor) -> torch.Tensor:
        # Create the distrs
        old_actor_distr = Categorical(old_actor_pred)
        new_actor_distr = Categorical(new_actor_pred)
        
        # Get new and old log probability 
        old_log_prob = old_actor_distr.log_prob(actions.flatten())

        new_log_prob = new_actor_distr.log_prob(actions.flatten())

        # Compute ratio
        ratio = torch.exp(new_log_prob - old_log_prob.detach())  # new_prob/(old_prob + self.args.eps)

        # Compute the first surrogate objective
        surr_1 = ratio * advs

        # Compute the second surrogate objective that is clipped
        surr_2 = torch.clip(
            ratio, 1.0 - self.args.ppo_clip, 1.0 + self.args.ppo_clip) * advs
        
        entropy = 0.01 * -torch.sum(
            new_actor_pred.gather(1, actions) * torch.log(new_actor_pred.gather(1, actions) + self.args.eps))
        
        actor_loss = -torch.mean(torch.minimum(surr_1, surr_2)) #+ entropy

        critic_loss = self.args.critic_lam * self.critic_loss_fn(val_pred, val_true)

        return actor_loss, critic_loss

    def advs_and_returns(self,
        rewards: list[float],
        val_preds: list[float],
        next_states: list[torch.Tensor],
        scale_advs=True):
        """Compute the return for an episode ."""
        with torch.no_grad():
            returns = torch.zeros_like(rewards, dtype=torch.float32).to(device)
            advs = torch.zeros_like(rewards, dtype=torch.float32).to(device)
            T = len(rewards)
            for t in reversed(range(T)):
                # Check if it is a terminal state
                if t == T - 1 or next_states[t] is None:
                    returns[t] = rewards[t]
                    advs[t] = returns[t] -  val_preds[t]
                else:
                    returns[t] = rewards[t] + self.args.gamma * returns[t+1]

                    # TD Residual
                    delta = rewards[t] + self.args.gamma * val_preds[t+1] - val_preds[t]

                    # GAE
                    advs[t] = delta + self.args.gamma * self.args.gae_lam * advs[t+1]
            
            tdlamret = advs + val_preds

            # Standardize advantages
            if scale_advs:
                advs = (advs - advs.mean()) / (advs.std() + self.args.eps)

            return returns, advs, tdlamret

    def early_stop_handler(self):
        """Handle early termination of this episode."""
        self._exp_buffer.pop()

    def reset(self):
        """Reset policy for next episode."""
        self._exp_buffer.ep_reset = True
        self._exp_buffer.num_ep += 1

    def __call__(self, 
                inv_nodes: list[InventoryNode],
                demand_node: DemandNode,
                argmax=False) -> PolicyResults:
        results = super().__call__(
            inv_nodes, demand_node, argmax)
        
        if len(self._exp_buffer.states) > 0 and not self._exp_buffer.ep_reset:

            self._exp_buffer.next_states[-1] = results.exps[0].state
        
        # Set that it is not new episode anymore
        self._exp_buffer.ep_reset = False
        
        for exp in results.exps:
            self._exp_buffer.add_exp(exp)

        return results


class ActorCritic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.action_dim = self.args.num_inv_nodes
 
        self.inv_encoder = InvEncoder(self.args)
        self.demand_encoder = DemandEncoder(self.args)
        
        # Inv embs + cur Item demand + total demand + loc
        self.inp_size = self.args.emb_size * (self.args.num_inv_nodes + 1)


        self.state_enc = Encoder(self.args, self.args.num_enc_layers)

        # Demand will be cur item quantity, total quantity left
        self._fc_1 = nn.Linear(self.args.emb_size, self.args.hidden_size)

        # Create hidden fcs
        self.hidden_fcs =  nn.ModuleList([
            nn.Linear(self.args.hidden_size, self.args.hidden_size) for _ in range(self.args.num_hidden - 1)])
        
        self.critic_dec = Decoder(self.args)#nn.Linear(self.args.hidden_size, 1)
        self.critic_out = nn.Linear(self.args.emb_size, 1) 
        self.actor_out = nn.Linear(self.args.hidden_size, 1)

    def _extract_state(self, state: torch.Tensor):
        """Extract the individual parts of the state.
        
            Did it this way to prevent having to rewrite a lot of other functions.
        """
        batch_size = state.shape[0]
        
        # Split into parts
        inv_end = self.args.num_inv_nodes * self.args.num_skus
        inv = state[:, :inv_end].reshape(
            batch_size, self.args.num_inv_nodes, self.args.num_skus)

        inv_locs_end = inv_end + self.args.num_inv_nodes*2
        inv_locs = state[:, inv_end:inv_locs_end].reshape(
            batch_size, self.args.num_inv_nodes, 2)

        demand_end = inv_locs_end + self.args.num_skus 
        demand = state[:, inv_locs_end:demand_end].reshape(
            batch_size, self.args.num_skus)

        demand_loc_end = demand_end + 2
        demand_loc = state[:, demand_end:demand_loc_end].reshape(
            batch_size, 2)

        fulfill_end = demand_loc_end + self.args.num_inv_nodes * self.args.num_skus
        cur_fulfill = state[:, demand_loc_end:fulfill_end].reshape(
            batch_size, self.args.num_inv_nodes, self.args.num_skus)

        item_hot = state[:, fulfill_end:].reshape(
            batch_size, self.args.num_skus)

        return inv, inv_locs, demand, demand_loc, cur_fulfill, item_hot

    def forward(self, 
                state, inv_mask: torch.tensor = None) -> torch.Tensor:
        # Add batch dimension if it is not present
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]
        inv, inv_locs, demand, demand_loc, cur_fulfill, item_hot = self._extract_state(state)
        #print("inv, inv_locs, demand, demand_loc, cur_fulfill, item_hot", inv, inv_locs, demand, demand_loc, cur_fulfill, item_hot)
        
        # Get inventory mask
        non_zero_items = item_hot.nonzero(as_tuple=True)
        inv_mask = 1 - (inv[non_zero_items[0], :, non_zero_items[1]] > 0).int()
        
        item_hot = item_hot.unsqueeze(1)
        demand = demand.unsqueeze(1)

        inv_embs = self.inv_encoder(inv, inv_locs, demand, cur_fulfill, item_hot)

        demand_embs = self.demand_encoder(demand, demand_loc, item_hot)
        
        fc_inp = torch.cat((inv_embs, demand_embs), 1)

        # Encode the state
        state_embs = self.state_enc(fc_inp)

        critic_pred, _ = self.critic_dec(state_embs[:, inv_embs.shape[1]:], state_embs[:, :inv_embs.shape[1]])
        critic_pred = self.critic_out(critic_pred).view(batch_size, -1)
        
        x = F.relu(self._fc_1(state_embs[:, :inv_embs.shape[1]]))        
        for hidden_fc in self.hidden_fcs:
            x = F.relu(hidden_fc(x))
        
        dk = torch.tensor(self.args.emb_size, dtype=torch.float32)
        actor_pred = self.actor_out(x).view(batch_size, -1) / torch.sqrt(dk)

        # Mask out invalid actions
        actor_pred += inv_mask * -1e9
        
        # Get action distribution
        actor_pred = F.softmax(actor_pred, dim=-1)

        if batch_size == 1:
            # print("item_hot.unsqueeze(1)", item_hot.unsqueeze(1))
            # print("inv", inv)
            # print("actor_pred", actor_pred)
            actor_pred = actor_pred.view(-1)
        return actor_pred, critic_pred

