import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import argparse
from dataclasses import dataclass
import os
from torch.distributions import Categorical
import matplotlib.pyplot as plt

@dataclass
class Experience:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor = None

class ExpBuffer:
    def __init__(self, action_dim):
        self.ep_reset = True
        self.action_dim = action_dim
        self.clear()

    def add_exp(self, exp):
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
        self.states = torch.stack(self.states)
        #self.next_states = torch.stack(self.next_states)
        self.rewards = torch.tensor(self.rewards)
        
        # Create one-hot action vectors
        # one_hot_actions = torch.zeros(len(self.actions), self.action_dim)
        # self.actions = torch.tensor(self.actions).unsqueeze(1)
        # self.actions = one_hot_actions.scatter_(1, self.actions, 1)
        
        self.actions = torch.tensor(self.actions).unsqueeze(1)

        self.actor_preds = torch.tensor(self.actor_preds)
        self.val_preds = torch.tensor(self.val_preds)
        # print(self.states, self.rewards, self.actions, self.actor_preds, self.val_preds)
        # print("self.next_states", self.next_states)

    def clear(self):
        self.ep_reset = True
        self.states = []
        self.next_states = []
        self.rewards = []
        self.actions = []
        self.val_preds = []
        self.actor_preds = []
        self.num_ep = 0

class ActorCriticPolicy:
    def __init__(self, args):
        self.args = args
        self._model_file = os.path.join(
            self.args.save_dir, "actor_critic_model.pt")
        self._actor_critic = ActorCritic(
            self.args)
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

    def save(self):
        model_dict = {
            "actor_critic" : self._actor_critic.state_dict(),
            "optimizer" : self._optim.state_dict(),
            "lr_scheduler" : self._lr_scheduler.state_dict()
        }

        torch.save(model_dict, self._model_file)

    def load(self):
        model_dict = torch.load(self._model_file)
        self._actor_critic.load_state_dict(model_dict["actor_critic"])
        self._optim.load_state_dict(model_dict["optimizer"])
        self._lr_scheduler.load_state_dict(model_dict["lr_scheduler"])


    def advs_and_returns(self,
        rewards: list[float],
        val_preds: list[float],
        next_states: list[torch.Tensor],
        scale_advs=True):
        """Compute the return for an episode ."""
        with torch.no_grad():
            returns = torch.zeros_like(rewards, dtype=torch.float32)
            advs = torch.zeros_like(rewards, dtype=torch.float32)
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
        # print("neg_logprob", neg_logprob)
        # print("neg_logprob * advs", neg_logprob * advs)
        entropy = 0.01 * -torch.sum(actor_pred.gather(1, actions) * torch.log(actor_pred.gather(1, actions) + self.args.eps))
        actor_loss = -(1/num_ep) * torch.sum(logprob * advs) + entropy
        
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
        # Get new and old log probability
        old_prob = old_actor_pred.gather(1, actions).view(-1)

        new_prob = new_actor_pred.gather(1, actions).view(-1)

        # Compute ratio
        ratio = new_prob/(old_prob + self.args.eps)
        print("ratio", ratio)
        print("advs", advs)

        # Compute the first surrogate objective
        surr_1 = ratio * advs

        # Compute the second surrogate objective that is clipped
        surr_2 = torch.clip(
            ratio, 1.0 - self.args.ppo_clip, 1.0 + self.args.ppo_clip) * advs

        #entropy = 0.01 * -torch.sum(new_prob * torch.log(new_prob + self.args.eps))
        
        actor_loss = -torch.mean(torch.min(surr_1, surr_2))# + entropy

        critic_loss = self.args.critic_lam * self.critic_loss_fn(val_pred, val_true)
        print("actor_loss", actor_loss, "critic_loss", critic_loss)
        return actor_loss, critic_loss
 
 
    def train(self) -> float:
        # Convert saved experiences to tensors
        self._exp_buffer.convert_to_tensor()

        num_exs = len(self._exp_buffer.rewards)
        num_minibatches = max(num_exs // self.args.batch_size, 1)

        # Compute return, advantages, and TD lambda return for saved experiences
        returns, advs, tdlamret = self.advs_and_returns(
            self._exp_buffer.rewards,
            self._exp_buffer.val_preds,
            self._exp_buffer.next_states)
        print("ADVS", advs)
        print("returns", returns)
        print("rewards", self._exp_buffer.rewards)
        # print("returns", returns)
        # print("advs", advs)
        # print("tdlamret", tdlamret)
        total_loss = 0
        print(self._exp_buffer.states.shape)
        # Iterate over several epochs
        for i in range(self.args.noptepochs):
            batch_ids = torch.randperm(num_exs)
            
            # Iterate over several batches
            for j in range(num_minibatches):
                start = j * (num_exs // num_minibatches)
                end = (j + 1) * (num_exs // num_minibatches) 
                m_batch_idxs = batch_ids[start:end]

                # Get current actor and critic prediction
                actor_pred, val_pred = self._actor_critic(
                    self._exp_buffer.states[m_batch_idxs])
                print(actor_pred.shape)
                print("actor_pred[:5]", actor_pred[:5], "self._exp_buffer.actor_preds[m_batch_idxs]", self._exp_buffer.actor_preds[m_batch_idxs][:5])
                
                print("val_pred", val_pred[:5].squeeze(), tdlamret[m_batch_idxs][:5])
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

                print("actor_loss", actor_loss)
                print("critic_loss", critic_loss)
                loss = actor_loss + critic_loss

                self._update_params(loss)
                total_loss += float(loss.detach())
        
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
        if not self.args.no_lr_decay and self._optim.param_groups[0]["lr"] > self.args.min_lr:
            # If so, decay learning rate
            self._lr_scheduler.step()

    def sample_action(self, actor_pred: torch.Tensor, argmax=False) -> int:
        if len(actor_pred) == 1:
            return 0
        
        if argmax:
            return actor_pred.argmax()
        else:
            action_distr = Categorical(actor_pred)
            action = action_distr.sample()

        return int(action)
    
    def reset(self):
        """Reset policy for next episode."""
        self._exp_buffer.ep_reset = True
        self._exp_buffer.num_ep += 1
    
    def __call__(self, x):
        actor_pred, val_pred = self._actor_critic(x)
        action = self.sample_action(actor_pred)
        self._exp_buffer.add_val_pred(val_pred)
        self._exp_buffer.add_actor_pred(actor_pred)
        return action, val_pred

class ActorCritic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.inp_size = 4
        self.action_dim = 2
        
        # Model parameters
        self._fc_1 = nn.Linear(self.inp_size, self.args.hidden_size)
        self._fc_2 = nn.Linear(self.args.hidden_size, self.args.hidden_size)

        self._actor_out = nn.Linear(self.args.hidden_size, self.action_dim)
        self._critic_out = nn.Linear(self.args.hidden_size, 1)
    
    def __call__(self, state):
        x = F.relu(self._fc_1(state))
        x = F.relu(self._fc_2(x))
        actor_pred = F.softmax(self._actor_out(x), dim=-1)
        val_pred = self._critic_out(x)

        return actor_pred, val_pred

def run(policy, args):
    env = gym.make("CartPole-v0")
    total_rewards = []
    for e_i in range(args.episodes):
        ob = env.reset()
        # ob = np.hstack((ob, [np.log(1)])) 
        done = False
        total_reward = 0
        # state = cur_frame.repeat(1,self._args.n_frames,1,1)
        state = torch.tensor(ob, dtype=torch.float32)
        while not done:
            env.render()
            # Get current action
            action, _ = policy(state)
            
          
            # Perform action in environment
            ob, reward, done, _ = env.step(action)
            total_reward += reward

            # Add memory step
            if done:
                next_state = None
                reward = -reward
            else:
                next_state = torch.tensor(ob, dtype=torch.float32)
            
            if next_state is None:
                e_t = Experience(state.clone(), action, reward, next_state)
            else:
                e_t = Experience(state.clone(), action, reward, next_state.clone())
            policy._exp_buffer.add_exp(e_t)
            state = next_state
        policy.reset()
        total_rewards.append(total_reward)
        print(total_reward)
        if (e_i + 1) % 4 == 0:
            policy.train()
            policy.save()
    
    plt.plot(total_rewards)
    plt.show()
            

def main(args):
    policy = ActorCriticPolicy(args)
    run(policy, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a fulfillment simulation.')
    parser.add_argument('--coord_bounds', type=int, default=10,
                    help='Max bounds for coordinates.')
    parser.add_argument("--num_skus", type=int, default=2,
                    help="Number of unique products SKUs.")
    parser.add_argument("--max_inv_prod", type=int, default=20,
                    help="Max inventory for each product across all inventory nodes.")
    parser.add_argument("--min_inv_prod", type=int, default=1,
                    help="Min inventory for each product across all inventory nodes.")
    parser.add_argument("--num_inv_nodes", type=int, default=2,
                    help="Number of inventory nodes.")
    parser.add_argument("--demand_lam", type=float, default=1.0,
                    help="Lambda parameter for sampling demand from demand poisson distribution.")
    parser.add_argument("--T_max", type=int, default=128,
                    help="Max number of orders")
    parser.add_argument("--reward_alpha", type=float, default=0.5,
                    help="Reward item discount.")
    parser.add_argument("--emb_size", type=int, default=256,
                    help="Embedding size.")
    parser.add_argument("--hidden_size", type=int, default=256,
                    help="Number of hidden units used for NN policy.")
    parser.add_argument("--epsilon", type=float, default=0.9,
                    help="Initial epsilon used for epsilon-greedy in DQN.")
    parser.add_argument("--min_epsilon", type=float, default=0.05,
                    help="Minimum epsilon value used for epsilon-greedy in DQN.")
    parser.add_argument("--epsilon_decay", type=int, default=1024,
                    help="Epsilon decay step.")
    parser.add_argument("--lr", type=float, default=6e-4,
                    help="Learning rate used for DRL models.")
    parser.add_argument("--lr_gamma", type=float, default=0.999,
                    help="Learning rate decay factor.")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                    help="Minimum learning rate.")
    parser.add_argument("--no_lr_decay", action="store_true",
                    help="Don't use lr decau.")
    parser.add_argument("--max_grad_norm", type=float, default=2.0,
                    help="Maximum gradient norm.")
    parser.add_argument("--batch_size", type=int, default=128,
                    help="Batch size used for training.")
    parser.add_argument("--mem_cap", type=int, default=100000,
                    help="Replay memory capacity.")
    parser.add_argument("--gamma", type=float, default=0.99,
                    help="Gamma value for discounting reward.")
    parser.add_argument("--gae_lam", type=float, default=0.95,
                    help="GAE lambda.")
    parser.add_argument("--episodes", type=int, default=1024,
                    help="Number of episodes.")
    parser.add_argument("--save_dir", default="models",
                    help="Directory to save the models.")
    parser.add_argument("--load", action="store_true",
                    help="Load saved models.")
    parser.add_argument("--no_per", action="store_true",
                    help="Don't use Prioritized Experience Replay (PER) for DQN model.")
    parser.add_argument("--per_beta", type=float, default=0.4,
                    help="Beta used for proportional priority.")
    parser.add_argument("--eps", type=float, default=1e-16,
                    help="Epsilon used for proportional priority.")
    parser.add_argument("--per_alpha", type=float, default=0.6,
                    help="Alpha used for proportional priority.")
    parser.add_argument("--tgt_update_step", type=int, default=5,
                    help="Number of training batches before target is updated.")
    parser.add_argument("--plot", action="store_true",
                    help="Plot training results.")
    parser.add_argument("--reward_smooth_w", type=int, default=32,
                    help="Window size for reward smoothing plot.")
    parser.add_argument("--policy", default="naive",
                    help="Policy to use (e.g., naive, dqn, primal) .")
    parser.add_argument("--loc_json", default=None,
                    help="JSON containing location coordinates for every inventory node.")

    pd_args = parser.add_argument_group("Primal-Dual")
    pd_args.add_argument("--kappa", type=float, default=2,
                    help="Kappa value used in Primal-Dual Urban algorithm.")

    eval_args = parser.add_argument_group("Evaluation")
    eval_args.add_argument("--eval", action="store_true",
                    help="Evaluate the policies.")
    eval_args.add_argument("--policy_dir", default="../policies",
                    help="Directory containing the model policies to load during evaluation.")
    eval_args.add_argument("--eval_episodes", type=int, default=512,
                    help="Number of evaluation episodes.")
    eval_args.add_argument("--num_bar_ep", type=int, default=5,
                    help="Number of episodes to plot on average reward episode figure.")


    vis_args = parser.add_argument_group("Visual")
    vis_args.add_argument("--vis", action="store_true",
                    help="Visualize the policies.")
    vis_args.add_argument("--screen_size", type=int, default=1024,
                    help="Screen size of the Pygame window.")
    vis_args.add_argument("--screen_padding", type=int, default=128,
                    help="Padding around fulfillment grid for information to fit.")
    vis_args.add_argument("--font_size", type=int, default=24,
                    help="Padding around fulfillment grid for information to fit.")
    
    ac_args = parser.add_argument_group("Actor Critic")
    ac_args.add_argument("--noptepochs", type=int, default=3,
                    help="Number of epochs to train batch of episodes")
    ac_args.add_argument("--critic_lam", type=float, default=0.5,
                    help="Critic loss weighting")
    ac_args.add_argument("--min_exps", type=int, default=512,
                    help="The minimum number of timesteps to run before training over stored experience.")
    ac_args.add_argument("--ppo_clip", type=float, default=0.2,
                    help="PPO surrogate loss clipping.")
    ac_args.add_argument("--vpg", action="store_true",
                    help="Use vanilla policy gradient loss for actor-critic policy.")
    ac_args.add_argument("--reward_scale_factor", type=float, default=0.01,
                    help="Reward scaling factor that may helps the policy learn faster.")
    args = parser.parse_args()
    main(args)