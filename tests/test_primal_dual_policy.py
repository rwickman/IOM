import os, sys, unittest 

# Add code path to sys path
file_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(file_path, "../code")
sys.path.append(code_path)

import numpy as np

from policy import PolicyResults
from primal_dual_policy import PrimalDual
from reward_manager import RewardManager
from simulator import *
from fake_args import FakeArgs
from fake_ep import FakeEpisode

class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.args = FakeArgs()
        self.reward_man = RewardManager(self.args)
        self.policy = PrimalDual(self.args, self.reward_man)
    
    def test_call_exps_first_ep(self):
        """Test with dual variables all equal to 0."""
        fake_ep = FakeEpisode()
        results = self.policy(fake_ep.inv_nodes, fake_ep.demand_node)
        
        for i in range(len(results.exps)):
            self.assertEqual(results.exps[i].state, fake_ep.sku_ids[i])
            self.assertEqual(results.exps[i].action, fake_ep.actions[i])
            self.assertEqual(results.exps[i].reward, fake_ep.rewards[i])

    def test_call_exps_nonzero_dual(self):
        fake_ep = FakeEpisode()
        expected_actions = [0, 1, 1]
        expected_rewards = [
            -fake_ep.dist_1,
            -fake_ep.dist_2,
            -fake_ep.dist_2 * self.args.reward_alpha,
        ]

        # Set nonzero dual variables
        self.policy._dual_lams = np.array([
            [50.0, 100.0],
            [0.5, 1.5]])

        results = self.policy(fake_ep.inv_nodes, fake_ep.demand_node)
        for i in range(len(results.exps)):
            self.assertEqual(results.exps[i].state, fake_ep.sku_ids[i])
            self.assertEqual(results.exps[i].action, expected_actions[i])
            self.assertEqual(results.exps[i].reward, expected_rewards[i])



        





        
