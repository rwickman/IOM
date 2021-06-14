import os, sys, unittest 
import torch

# Add code path to sys path
file_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(file_path, "../code")
sys.path.append(code_path)

from dqn_policy import DQNTrainer
from reward_manager import RewardManager
from simulator import Cordinates, Location, InventoryProduct, InventoryNode, DemandNode
from fake_args import FakeArgs
from fake_ep import FakeEpisode

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

    def _create_fake_results(self):
        self.fake_ep = FakeEpisode()
        self.dqn_trainer._dqn = FakeDQN()
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
        self.args.no_per = False
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

        q_vals = [-0.4, 2.1, 1.5]
        td_targets = [
            -self.fake_ep.dist_1 + q_vals[1] * self.args.gamma,
            -self.fake_ep.dist_1 * self.args.reward_alpha + q_vals[2] * self.args.gamma,
            -self.fake_ep.dist_2
        ]

        td_errors = [
            td_targets[0] - q_vals[0],
            td_targets[1] - 3.0,
            td_targets[2] - q_vals[2],
        ]

        # Check the td_errors are correct
        for i, el in enumerate(self.dqn_trainer._memory._memory):
            self.assertAlmostEqual(float(el[1]), td_errors[i], 5)