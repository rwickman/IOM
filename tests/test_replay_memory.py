import os, sys, unittest 
import torch

# Add code path to sys path
file_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(file_path, "../code")
sys.path.append(code_path)

from replay_memory import PrioritizedExpReplay 
from fake_args import FakeArgs
from fake_ep import FakeEpisode

class TestReplayMemory(unittest.TestCase):

    def setUp(self):
        self.args = FakeArgs()
        self.per_mem = PrioritizedExpReplay(self.args)
        self.fake_ep = FakeEpisode()
        self.td_errors = torch.tensor([-1, 10, 2])

    def assertExpEqual(self, e_1, e_2):
        """Assert that the experiences are equal."""
        self.assertListEqual(e_1.state.tolist(), e_2.state.tolist())
        if e_1.next_state is not None and e_2.next_state is not None:
            self.assertListEqual(e_1.next_state.tolist(), e_2.next_state.tolist())
        elif (e_1.next_state is None and e_2.next_state is not None) or (e_1.next_state is not None and e_2.next_state is None):
            self.assertEqual(e_1.next_state, e_2.next_state) 
            
        self.assertEqual(e_1.reward, e_2.reward)
        self.assertEqual(e_1.action, e_2.action)
        


    def test_per_compute_priortity(self):
        expected_priority = 1.5157170212253226
        p = self.per_mem._compute_priority(self.td_errors[2])
        self.assertAlmostEqual(p, expected_priority, 5)

    def test_per_add(self):
        self.per_mem.add(self.fake_ep.exps[0], self.td_errors[0])
        is_ws, exp_sample, indices = self.per_mem.sample(1, 0)
        
        self.assertListEqual(exp_sample[0].state.tolist(), self.fake_ep.exps[0].state.tolist())
        self.assertListEqual(exp_sample[0].next_state.tolist(), self.fake_ep.exps[0].next_state.tolist())
        self.assertEqual(exp_sample[0].action, self.fake_ep.exps[0].action)
        self.assertEqual(exp_sample[0].reward, self.fake_ep.exps[0].reward)
    

    def test_sum_tree_get_single_item(self):
        for i in range(1):
            self.per_mem.add(self.fake_ep.exps[i], self.td_errors[i])
        
        p_1, exp_1, tree_idx_1 = self.per_mem._sum_tree.get(0)
        p_2, exp_2, tree_idx_2 = self.per_mem._sum_tree.get(self.per_mem._sum_tree.total())
        
        # Make sure these return the same elements
        self.assertEqual(p_1, p_2)
        self.assertEqual(tree_idx_1, tree_idx_2)
        self.assertListEqual(exp_1.state.tolist(), exp_2.state.tolist())
        self.assertListEqual(exp_1.next_state.tolist(), exp_2.next_state.tolist())
        self.assertEqual(exp_1.action, exp_2.action)
        self.assertEqual(exp_1.reward, exp_2.reward)

    def test_sum_tree_get_multi_item(self):
        for i in range(3):
            self.per_mem.add(self.fake_ep.exps[i], self.td_errors[i])
        
        expected_tree_idxs = [3,4,5]
        get_list = [
            self.per_mem._sum_tree.get(0),
            self.per_mem._sum_tree.get(self.per_mem._compute_priority(torch.tensor(self.td_errors[2]))),
            self.per_mem._sum_tree.get(self.per_mem._sum_tree.total())
        ] 

        for i in range(3):
            # Verify the tree idxs are equal
            self.assertEqual(get_list[i][2], expected_tree_idxs[i]) 
            
            # Verify the experiences are equal
            self.assertExpEqual(get_list[i][1], self.fake_ep.exps[i])

    def test_sum_tree_get_overflow(self):
        for i in range(3):
            self.per_mem.add(self.fake_ep.exps[i], self.td_errors[i])

        # Makes the memory have to drop the firt item
        for i in range(2):
            self.per_mem.add(self.fake_ep.exps[-1], self.td_errors[-1])

        expected_tree_idxs = [3,4,6]

        get_list = [
            self.per_mem._sum_tree.get(0),
            self.per_mem._sum_tree.get(self.per_mem._compute_priority(self.td_errors[2])+1),
            self.per_mem._sum_tree.get(self.per_mem._sum_tree.total())
        ] 

        # Verify values as first leaf in tree should now contain last experience
        # Verify the tree idxs are equal
        self.assertEqual(get_list[0][2], expected_tree_idxs[0]) 
        
        # Verify the experiences are equal
        self.assertExpEqual(get_list[0][1], self.fake_ep.exps[2])

        
        for i in range(1, 3):
            self.assertEqual(get_list[i][2], expected_tree_idxs[i]) 
            
            # Verify the experiences are equal
            self.assertExpEqual(get_list[i][1], self.fake_ep.exps[i])
        
        

