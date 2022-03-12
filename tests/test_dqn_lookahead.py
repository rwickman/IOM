import torch
import os, sys, unittest


# Add code path to sys path
file_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(file_path, "../code")
sys.path.append(code_path)

from fake_args import FakeArgs
from fake_ep import FakeEpisode, FakeEpisode_2
from dqn_lookhead_policy import DQNLookaheadTrainer
from reward_manager import RewardManager

class FakeDQNPolicy:
    def __init__(self):
        self.num_calls = 0
    
    def __call__(self, x):
        return [0.0, 0.0]


class TestDQNLookahead(unittest.TestCase):
    def setUp(self):
        self.args = FakeArgs()
        self.fake_ep = FakeEpisode()
        self.reward_man = RewardManager(self.args)
        #self.args.num_inv_nodes = 3
        #self.args.num_skus = 4
        self.trainer = DQNLookaheadTrainer(self.args, self.reward_man)
        self.trainer._dqn = FakeDQNPolicy()
        

    
    def test_call(self):
        self.trainer.args.eval = True
        sku_distr = torch.zeros(self.args.num_skus).cuda()
        print("CALLED TRAINER")
        results_1 = self.trainer(self.fake_ep.inv_nodes, self.fake_ep.demand_node, sku_distr)
        #results_2 = self.trainer(self.fake_ep_2.inv_nodes, self.fake_ep_2.demand_node, sku_distr)
        
        assert True
        assert len(results_1.exps) == 3
        for i, exp in enumerate(results_1.exps):
            assert exp.reward == self.fake_ep.rewards[i]
        
        for i in range(len(results_1.exps) - 1):
            self.assertListEqual(results_1.exps[i].next_state.tolist(), results_1.exps[i+1].state.tolist())
            self.assertGreater((results_1.exps[i].state != results_1.exps[i+1].state).sum().item(), 0)  
        
        for exp in results_1.exps:
            print(exp)
        
        self.assertEqual(results_1.fulfill_plan.fulfill_quantity(0, 0), 1)
        self.assertEqual(results_1.fulfill_plan.fulfill_quantity(0, 1), 1)
        self.assertEqual(results_1.fulfill_plan.fulfill_quantity(1, 0), 0)
        self.assertEqual(results_1.fulfill_plan.fulfill_quantity(1, 1), 1)
        for i in range(2):
            print(f"\n NODE {i}")
            for inv_prod in results_1.fulfill_plan.get_fulfillment(i).inv.items():
                print(inv_prod)
        


        


        

       # print(results_2.exps)



# if __name__ == '__main__':
#     unittest.main()