import os, sys, unittest 
import torch

# Add code path to sys path
file_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(file_path, "../code")
sys.path.append(code_path)


from dqn_policy import DQNTrainer
from reward_manager import RewardManager
from fake_args import FakeArgs
from fake_ep import FakeEpisode
from actor_critic_policy import ActorCriticPolicy

class TestACPolicy(unittest.TestCase):
    def setUp(self):
        self.args = FakeArgs()
        self.reward_man = RewardManager(self.args)
        self.ac_policy = ActorCriticPolicy(self.args, self.reward_man)
    
    def _create_fake_results(self):
        # Create fake predict function to return pred based on calls
        call_count = 0
        def fake_predict(state):
            nonlocal call_count
            if call_count <= 1:
                actor_pred = torch.tensor([1000000.0, 0.0])
            elif call_count == 2:
                actor_pred = torch.tensor([0.0, 1000000.0])

            else:
                self.assertTrue(False)
            call_count += 1
            return actor_pred
        
        # Change predict function for test
        self.ac_policy.predict = fake_predict
        self.fake_ep = FakeEpisode()
        results = self.ac_policy(
            self.fake_ep.inv_nodes,
            self.fake_ep.demand_node,
            argmax=True)
    
        return results

    def test_sample_action(self):
        expected_action = 2
        actor_pred = torch.tensor([0.0, 0.0, 10000000.0])
        action = self.ac_policy.sample_action(actor_pred)
        self.assertEqual(action, expected_action)

    def test_sample_action_argmax(self):
        expected_action = 1
        actor_pred = torch.tensor([1.0, 2.5, 0.64])
        action = self.ac_policy.sample_action(actor_pred, argmax=True)
        self.assertEqual(action, expected_action)
    
    def test_vanilla_pg_loss(self):        
        actor_distr = torch.tensor([
            [0.15, 0.5, 0.35],
            [0.8, 0.15, 0.05]])
        val_pred = torch.tensor([0.25, 0.1])
        val_true = torch.tensor([0.5, 0.3])
        actions = torch.tensor([[2], [0]])
        advs = torch.tensor([0.2, 1.0])

        neglogprob = -torch.log(
            torch.tensor([actor_distr[0,2], actor_distr[1,0]]))
        expected_actor_loss = (neglogprob[0] * advs[0] + neglogprob[1] * advs[1])

        # Huber Loss
        expected_critic_loss = self.args.critic_lam * \
            (0.5 * (val_pred[0] - val_true[0]) ** 2 + \
            0.5 * (val_pred[1] - val_true[1]) ** 2) / len(val_true)

        actor_loss, critic_loss = self.ac_policy.vanilla_pg_loss(
            actor_distr,
            val_pred,
            val_true,
            actions,
            advs, 1)

        # Verify critic loss is correct
        self.assertEqual(critic_loss, expected_critic_loss)

        # Verify actor loss is correct
        self.assertAlmostEqual(float(actor_loss), float(expected_actor_loss), 4)

    def test_compute_return(self):
        rewards = torch.tensor([1.0, 1.0, 2.0, 3.0, 1.0])
        val_preds = torch.tensor([0.5, 1.0, 1.0, 2.0, 1.5])
        next_states = [0, 0, None, 0, 0]
        
        expected_returns = [
            rewards[0] + rewards[1] * self.args.gamma + rewards[2] * self.args.gamma**2,
            rewards[1] + rewards[2] * self.args.gamma,
            rewards[2],
            rewards[3] + rewards[4] * self.args.gamma,
            rewards[4],
        ]
        
        # Create the expected advanatages
        expected_advs = [
            rewards[4] - val_preds[4]
        ]
        expected_advs.insert(
            0, (rewards[3] + self.args.gamma * val_preds[4] - val_preds[3]) + 
            self.args.gamma *  self.args.gae_lam * expected_advs[0])
        expected_advs.insert(
            0, rewards[2] - val_preds[2])
        expected_advs.insert(
            0, (rewards[1] + self.args.gamma * val_preds[2] - val_preds[1]) + 
            self.args.gamma *  self.args.gae_lam * expected_advs[0])
        expected_advs.insert(
            0, (rewards[0] + self.args.gamma * val_preds[1] - val_preds[0]) + 
            self.args.gamma *  self.args.gae_lam * expected_advs[0])

        # Create the expected TD-lambda return
        expected_tdlamret = torch.tensor(expected_advs) + val_preds

        returns, advs, tdlamret = self.ac_policy.advs_and_returns(
            rewards,
            val_preds,
            next_states,
            scale_advs=False)

        self.assertListEqual(returns.tolist(), expected_returns)
        self.assertListEqual(advs.tolist(), expected_advs)
        self.assertListEqual(tdlamret.tolist(), expected_tdlamret.tolist())
        
        # Check if standardization is corrext
        _, advs, _ = self.ac_policy.advs_and_returns(
            rewards,
            val_preds,
            next_states,
            scale_advs=True)

        expected_advs = torch.tensor(expected_advs)
        expected_advs = (expected_advs - expected_advs.mean()) / (expected_advs.std() + self.args.eps)
        self.assertListEqual(advs.tolist(), expected_advs.tolist())
        
    # def test_call_states(self):
    #     # Verify it is initialized correctly
    #     self.assertTrue(self.ac_policy._exp_buffer.ep_reset)

    #     results = self._create_fake_results()

    #     # Check states and next states
    #     for i in range(3):
    #         self.assertListEqual(results.exps[i].state.tolist(), self.fake_ep.states[i].tolist())    
    #         self.assertListEqual(self.ac_policy._exp_buffer.states[i].tolist(), self.fake_ep.states[i].tolist()) 
    #         if i > 0:
    #             self.assertListEqual(results.exps[i-1].next_state.tolist(), self.fake_ep.states[i].tolist())
    #             self.assertListEqual(self.ac_policy._exp_buffer.next_states[i-1].tolist(), self.fake_ep.states[i].tolist())
                
    #     # Check the next state after the episode is None
    #     self.assertEqual(results.exps[2].next_state, None)
    #     self.assertEqual(self.ac_policy._exp_buffer.next_states[2], None)

    #     # Verify the flag is False
    #     self.assertFalse(self.ac_policy._exp_buffer.ep_reset)
        
    #     # Run a order to verify experience buffer is getting updated correctly
    #     results = self._create_fake_results()
       
    #    # Check states and next states
    #     for i in range(3):
    #         self.assertListEqual(results.exps[i].state.tolist(), self.fake_ep.states[i].tolist())    
    #         self.assertListEqual(self.ac_policy._exp_buffer.states[i].tolist(), self.fake_ep.states[i].tolist()) 
    #         if i > 0:
    #             self.assertListEqual(results.exps[i-1].next_state.tolist(), self.fake_ep.states[i].tolist())
    #             self.assertListEqual(self.ac_policy._exp_buffer.next_states[3+i-1].tolist(), self.fake_ep.states[i].tolist())

    #     # Check the next state after the episode is None
    #     self.assertEqual(results.exps[2].next_state, None)
    #     self.assertEqual(self.ac_policy._exp_buffer.next_states[5], None)
        
        
    #     # Verify the flag is still False
    #     self.assertFalse(self.ac_policy._exp_buffer.ep_reset)
        

    def test_call_rewards(self):
        results = self._create_fake_results()
        # Verify same length
        self.assertEqual(len(results.exps), len(self.ac_policy._exp_buffer.rewards))
        
        # Verify all rewards are equal
        for i in range(len(results.exps)):
            self.assertEqual(results.exps[i].reward, self.ac_policy._exp_buffer.rewards[i])
    

    def test_exp_buffer_convert_to_tensor(self):
        results = self._create_fake_results()
        
        self.ac_policy._exp_buffer.convert_to_tensor()
        
        expected_states = []
        expected_rewards = []
        expected_actions = []
        for exp in results.exps:
            expected_states.append(exp.state.tolist())
            expected_rewards.append(round(exp.reward, 5))
            expected_actions.append([exp.action])

        # Verify states
        self.assertListEqual(self.ac_policy._exp_buffer.states.tolist(), expected_states)

        # Verify rewards
        rewards = [round(reward, 5) for reward in self.ac_policy._exp_buffer.rewards.tolist()]
        self.assertListEqual(rewards, expected_rewards)

        # Verify actions
        self.assertListEqual(expected_actions, [[a] for a in self.fake_ep.actions])
        self.assertListEqual(self.ac_policy._exp_buffer.actions.tolist(), expected_actions)

        
        