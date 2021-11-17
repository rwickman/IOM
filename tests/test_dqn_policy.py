import os, sys, unittest 
import torch

# Add code path to sys path
file_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(file_path, "../code")
sys.path.append(code_path)

from dqn_policy import DQNTrainer
from reward_manager import RewardManager
from fake_args import FakeArgs
from fake_ep import FakeEpisode, FakeEpisode_2

class FakeDQN:
    def __init__(self):
        self._call_count = 0
    
    def __call__(self, x):
        if self._call_count == 0:
            q_vals = torch.tensor([-0.4, 0.2])
        elif self._call_count == 1:
            q_vals = torch.tensor([2.1, 1])
        else:
            q_vals = torch.tensor([3.0, 1.5])
        
        self._call_count += 1
        return q_vals

class FakeDQN_2:
    def __init__(self):
        self._call_count = 0
    
    def __call__(self, x):
        if self._call_count == 0:
            q_vals = torch.tensor([-0.4, 0.2, 0.25, 0.1])
        elif self._call_count == 1:
            q_vals = torch.tensor([-0.4, 0.2, 0.25, 0.1])
        else:
            q_vals = torch.tensor([[1.5, 0.2, 0.25, 0.1]])
        
        self._call_count += 1
        return q_vals



class FakeMemory:
    def __init__(self):
        self._memory = []
    
    def add(self, exp, priority=None):
        if priority is None:
            self._memory.append(exp)
        else:
            self._memory.append((exp, priority))

class TestDQNPolicy(unittest.TestCase):
    def setUp(self):
        self.args = FakeArgs()
        self.reward_man = RewardManager(self.args)
        self.dqn_trainer = DQNTrainer(self.args, self.reward_man)

    def _create_fake_results(self, fake_ep_num=0):
        if fake_ep_num == 0:
            self.fake_ep = FakeEpisode()
            self.dqn_trainer._dqn = FakeDQN()
        else:
            self.fake_ep = FakeEpisode_2()
            self.dqn_trainer._dqn = FakeDQN_2()

        results = self.dqn_trainer(
            self.fake_ep.inv_nodes,
            self.fake_ep.demand_node,
            argmax=True)
    
        return results

    def test_sample_action(self):
        q_vals = torch.tensor([0.4, 1.5, 0.1])
        expected_action = 1
        action = self.dqn_trainer.sample_action(q_vals, True)
        self.assertEqual(action, expected_action)

    def test_call_dqn_actions(self):
        results = self._create_fake_results()
        # Test expected actions are equal 
        self.assertEqual(len(results.exps), 3)
        for i in range(len(results.exps)):
            self.assertEqual(results.exps[i].action, self.fake_ep.actions[i])

    def test_call_dqn_states(self):
        results = self._create_fake_results()
        self.fake_ep.norm_states()
        self.assertListEqual(results.exps[0].state.tolist(), self.fake_ep.states[0].tolist()) 

        # Check second states are equal
        self.assertListEqual(results.exps[0].next_state.tolist(), self.fake_ep.states[1].tolist())
        self.assertListEqual(results.exps[1].state.tolist(), self.fake_ep.states[1].tolist())

        # Check last states are equal
        self.assertListEqual(results.exps[1].next_state.tolist(), self.fake_ep.states[2].tolist())
        self.assertListEqual(results.exps[2].state.tolist(), self.fake_ep.states[2].tolist())

        # Check the next state after the episode is None
        self.assertEqual(results.exps[2].next_state, None)


    def test_call_dqn_rewards(self):
        results = self._create_fake_results()
        
        # Test the reward values are correctly set
        for i in range(len(results.exps)):
            self.assertAlmostEqual(results.exps[i].reward, self.fake_ep.rewards[i])

    def test_add_stored_exps_no_per(self):
        results = self._create_fake_results()
        self.dqn_trainer._memory = FakeMemory()
        self.dqn_trainer._add_stored_exps()

        # Make sure all three experiences were added
        self.assertEqual(len(self.dqn_trainer._memory._memory), 3)

        # Check buffer was emptied
        self.assertEqual(len(self.dqn_trainer._exp_buffer), 0)


    def test_add_stored_exps_per(self):
        self.dqn_trainer.args.no_per = False
        results = self._create_fake_results()

        self.dqn_trainer._memory = FakeMemory()
        
        self.dqn_trainer._dqn_target = FakeDQN()
        
        # Make it act like predicting on the next_state
        self.dqn_trainer._dqn_target._call_count += 1
        
        # Reset call count
        self.dqn_trainer._dqn._call_count = 0

        self.dqn_trainer._add_stored_exps()

        # Make sure all three experiences were added
        self.assertEqual(len(self.dqn_trainer._memory._memory), 3)

        # Check buffer was emptied
        self.assertEqual(len(self.dqn_trainer._exp_buffer), 0)

        q_vals = [0.2, 1, 3]
        td_targets = [
            -self.fake_ep.dist_2 * self.fake_ep.reward_scale_factor,
            -self.fake_ep.dist_1 * self.args.reward_alpha * self.fake_ep.reward_scale_factor+ q_vals[1] * self.args.gamma,
            -self.fake_ep.dist_1 * self.fake_ep.reward_scale_factor + q_vals[2] * self.args.gamma
        ]

        td_errors = [
            td_targets[0] - q_vals[0],
            td_targets[1] - 2.1,
            td_targets[2] - q_vals[2]
        ]

        # Check the td_errors are correct        
        for i in reversed(range(len(self.dqn_trainer._memory._memory))):
            el = self.dqn_trainer._memory._memory[i]
            self.assertAlmostEqual(float(el[1]), td_errors[i], 5)


    def test_create_state(self):
        self.fake_ep_2 = FakeEpisode_2()
        fake_item_hot = torch.tensor([0,0,1])
        state = self.dqn_trainer._create_state(
            self.fake_ep_2.inv,
            self.fake_ep_2.locs,
            self.fake_ep_2.demand,
            self.fake_ep_2.demand_loc_coords,
            torch.zeros(self.fake_ep_2.num_inv_nodes, self.fake_ep_2.num_skus),
            fake_item_hot)

        expected_state = torch.tensor(
            [0,1,3,0,2,4,1,0,1,4,1,2,
            0, 0, 1.0, 1.0, -1.0, -1.0, 0.2, 0.2,
            2,0,1,
            0.2, 0.1,
            0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,1]
        )

        self.assertListEqual(state.tolist(), expected_state.tolist())

    def test_get_valid_actions(self):
        self.fake_ep_2 = FakeEpisode_2()
        self.dqn_trainer.args.num_inv_nodes = 4
        self.dqn_trainer.args.num_skus = 3

        expected_valid_actions = [[2,3], [0,1,3], [0,1,2,3]]
        for i in range(self.dqn_trainer.args.num_skus):
            fake_item_hot = torch.zeros(self.dqn_trainer.args.num_skus)
            fake_item_hot[i] = 1
            state = self.dqn_trainer._create_state(
                self.fake_ep_2.inv,
                self.fake_ep_2.locs,
                self.fake_ep_2.demand,
                self.fake_ep_2.demand_loc_coords,
                torch.zeros(self.fake_ep_2.num_inv_nodes, self.fake_ep_2.num_skus),
                fake_item_hot)

            valid_actions = self.dqn_trainer._get_valid_actions(state)
            self.assertListEqual(valid_actions.tolist(), expected_valid_actions[i])

    def test_results_ep_2(self):
        self.args.num_inv_nodes = 4
        self.args.num_skus = 3
        self.dqn_trainer = DQNTrainer(self.args, self.reward_man)
        results = self._create_fake_results(1)

        # Check all the correct keys are in the dict
        self.assertTrue(2 in results.fulfill_plan._fulfillments)
        self.assertTrue(3 in results.fulfill_plan._fulfillments)
        self.assertTrue(0 in results.fulfill_plan._fulfillments)

        # Check the fufillment requests are correct
        self.assertEqual(results.fulfill_plan._fulfillments[2]._inv._inv_dict[0], 1)
        self.assertEqual(results.fulfill_plan._fulfillments[3]._inv._inv_dict[0], 1)
        self.assertEqual(results.fulfill_plan._fulfillments[0]._inv._inv_dict[2], 1)

        # Check the Experiences are correct
        expected_state_1 = torch.tensor(
            [0,1,3,0,2,4,1,0,1,4,1,2,
            0, 0, 1.0, 1.0, -1.0, -1.0, 0.2, 0.2,
            2,0,1,
            0.2, 0.1,
            0,0,0,0,0,0,0,0,0,0,0,0,
            1,0,0]
        )
        expected_state_2 = torch.tensor(
            [0,1,3,0,2,4,0,0,1,4,1,2,
            0, 0, 1.0, 1.0, -1.0, -1.0, 0.2, 0.2,
            1,0,1,
            0.2, 0.1,
            0,0,0,0,0,0,1,0,0,0,0,0,
            1,0,0]
        )
        expected_state_3 = torch.tensor(
            [0,1,3,0,2,4,0,0,1,3,1,2,
            0, 0, 1.0, 1.0, -1.0, -1.0, 0.2, 0.2,
            0,0,1,
            0.2, 0.1,
            0,0,0,0,0,0,1,0,0,1,0,0,
            0,0,1]
        )

        # Check the states and next_states are correct
        self.assertListEqual(results.exps[0].state.tolist(), expected_state_1.tolist())
        self.assertListEqual(results.exps[0].next_state.tolist(), expected_state_2.tolist())

        self.assertListEqual(results.exps[1].state.tolist(), expected_state_2.tolist())
        self.assertListEqual(results.exps[1].next_state.tolist(), expected_state_3.tolist())

        self.assertListEqual(results.exps[2].state.tolist(), expected_state_3.tolist())
        self.assertEqual(results.exps[2].next_state, None)

        # Check the rewards are correct
        expected_rewards = [ 
            -16.278820596099706 * self.fake_ep.reward_scale_factor,
            -1.0 * self.fake_ep.reward_scale_factor,
            -2.23606797749979 * self.fake_ep.reward_scale_factor
        ] 
        self.assertEqual(results.exps[0].reward, expected_rewards[0])
        self.assertEqual(results.exps[1].reward, expected_rewards[1])
        self.assertEqual(results.exps[2].reward, expected_rewards[2])

        # Check the actions are correct
        expected_actions = [2,3,0]
        self.assertEqual(results.exps[0].action, expected_actions[0])
        self.assertEqual(results.exps[1].action, expected_actions[1])
        self.assertEqual(results.exps[2].action, expected_actions[2])

        
        
