from dataclasses import asdict
import os, sys, unittest 
import torch
import torch.nn as nn

# Add code path to sys path
file_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(file_path, "../code")
sys.path.append(code_path)

from dqn_att_policy import DQNAtt, DQNAttTrainer, DQNAttState
from reward_manager import RewardManager
from fake_args import FakeArgs
from fake_ep import FakeEpisode

class FakeTransformer(nn.Module):
    def __init__(self, args):
        self.args = args

    def __call__(self, enc_inp, dec_inp, enc_pad_mask, dec_pad_mask):
        self.enc_inp = enc_inp
        self.dec_inp = dec_inp
        self.enc_pad_mask = enc_pad_mask
        self.dec_pad_mask = dec_pad_mask

        return torch.zeros(enc_inp.shape[0], self.dec_inp.shape[0], self.args.hidden_size), None
        


class TestDQNAttPolicy(unittest.TestCase):
    def setUp(self):
        self.args = FakeArgs()
        self.fake_ep = FakeEpisode()
        self.reward_man = RewardManager(self.args)
        #self.args.num_inv_nodes = 3
        #self.args.num_skus = 4
        self.trainer = DQNAttTrainer(self.args, self.reward_man)

    def test_extract_state(self):
        expected_inv = [[2, 1], [0, 2]]
        expected_inv_loc = [[0,0], [10,10]]
        expected_demand = [1, 2]
        expected_demand_loc = [2,1]
        expected_cur_fulfill = [[0,0],[0,0]]
        expected_item_hot = [1,0]

        # Convert flattened state to its individual elements
        inv, inv_locs, demand, demand_loc, cur_fulfill, item_hot = self.trainer._dqn._extract_state(
            self.fake_ep.states[0].unsqueeze(0)) 
        
        self.assertListEqual(inv[0].tolist(), expected_inv)
        self.assertListEqual(inv_locs[0].tolist(), expected_inv_loc)
        self.assertListEqual(demand[0].tolist(), expected_demand)
        self.assertListEqual(demand_loc[0].tolist(), expected_demand_loc)
        self.assertListEqual(cur_fulfill[0].tolist(), expected_cur_fulfill)
        self.assertListEqual(item_hot[0].tolist(), expected_item_hot)

    def test_dqn_att_call_single_state(self):
        self.trainer._dqn(self.fake_ep.states[0].unsqueeze(0))
    
    def test_dqn_att_call_multi_state(self):
        states = torch.stack(self.fake_ep.states)
        self.trainer._dqn(states)

    def test_dqn_create_padded(self):
        inv = torch.tensor([
            [[0, 1, 2],
            [0,0,0]],
            [[0,0,1],
            [0,9,3]]
        ])

        locs = torch.tensor([
            [[0,0], [10,10]],
            [[0,0], [10,10]]
        ])

        # The expeted padded results
        expected_quant_pad = [
            [[1],[2],[0]],
            [[1],[9],[3]]
        ]
        expected_sku_ids_pad = [
            [1,2,0],
            [2,1,2]
        ]
        expected_node_ids_pad = [
            [0,0,0],
            [0,1,1]
        ]
        expected_locs_pad = [
            [[0,0],[0,0], [0,0]],
            [[0,0],[10,10], [10,10]]
        ]
        expected_padded_mask = [
            [[[0,0,1]]],
            [[[0,0,0]]]
        ]


        # Get padded values
        inv_sku_quantities_pad, sku_ids_pad, node_ids_pad, locs_pad, padding_mask = self.trainer._dqn._create_padded(inv, locs)
        
        # Verify values were padded correctly
        self.assertListEqual(inv_sku_quantities_pad.tolist(), expected_quant_pad)
        self.assertListEqual(sku_ids_pad.tolist(), expected_sku_ids_pad)
        self.assertListEqual(node_ids_pad.tolist(), expected_node_ids_pad)
        self.assertListEqual(locs_pad.tolist(), expected_locs_pad)
        self.assertListEqual(padding_mask.tolist(), expected_padded_mask)


    def test_call_padding_single_state(self):
        fake_transformer = FakeTransformer(self.args)

        self.trainer._dqn._transformer = fake_transformer
        self.trainer._dqn(self.fake_ep.states[0].unsqueeze(0))

        expected_enc_pad_mask = [[[[0,0,0,0,0,0]]]]

        expected_dec_pad_mask = [[[[0]]]]
        self.assertListEqual(fake_transformer.enc_pad_mask.tolist(), expected_enc_pad_mask)
        self.assertListEqual(fake_transformer.dec_pad_mask.tolist(), expected_dec_pad_mask)

        self.trainer._dqn(self.fake_ep.states[2].unsqueeze(0))

        expected_enc_pad_mask = [[[[0,0,0,0]]]]

        expected_dec_pad_mask = [[[[0,0,0]]]]
        self.assertListEqual(fake_transformer.enc_pad_mask.tolist(), expected_enc_pad_mask)
        self.assertListEqual(fake_transformer.dec_pad_mask.tolist(), expected_dec_pad_mask)

    

    def test_call_padding_multi_state(self):
        """Verify the padding masks that get called to the transformer model are correct"""
        fake_transformer = FakeTransformer(self.args)

        self.trainer._dqn._transformer = fake_transformer
        states = torch.stack(self.fake_ep.states)
        self.trainer._dqn(states)

        expected_enc_pad_mask = [
            [[[0,0,0,0,0,0]]],
            [[[0,0,0,0,1,0]]],
            [[[0,0,1,0,1,0]]]]

        expected_dec_pad_mask = [
            [[[0,1,1]]],
            [[[0,0,1]]],
            [[[0,0,0]]]]

        self.assertListEqual(fake_transformer.enc_pad_mask.tolist(), expected_enc_pad_mask)
        self.assertListEqual(fake_transformer.dec_pad_mask.tolist(), expected_dec_pad_mask)






    # def test_dqn_att_call(self):
    #     inv = torch.tensor([
    #         [[0, 2, 4, 0],
    #         [1, 2, 0, 1],
    #         [1, 0, 0, 2]],
    #         [[0, 2, 4, 0],
    #         [1, 2, 0, 0],
    #         [1, 0, 0, 0]]
    #     ])
    #     inv_locs = torch.tensor([
    #         [0.5, 0.5],
    #         [1.0, 1.0],
    #         [0.0, 0.0]
    #     ])

    #     cur_fulfill = torch.tensor([
    #         [[0, 2, 4, 0],
    #         [0, 0, 0, 0],
    #         [0, 1, 0, 0]],
    #         [[0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0]]
    #     ])

    #     demand = torch.tensor([
    #         [0,2, 3, 0],
    #         [5,0, 3, 2]])
    #     demand_locs = torch.tensor([0.5,0.5])
    #     item_hot = torch.tensor([ [0,1,0,0],[0,0,1,0]])

    #     q_vals = self.trainer._dqn(
    #         DQNAttState(inv, inv_locs, demand, demand_locs, cur_fulfill, item_hot))

    # def test_get_valid_actions(self):
    #     expected_valid_actions_1 = [0,1]
    #     expected_valid_actions_2 = [2]

    #     # Create the fake state
    #     inv = torch.tensor([
    #         [0, 2, 0, 0],
    #         [0,1,0,0],
    #         [0,0,8,0]])
    #     inv_locs = torch.tensor([
    #         [0.5, 0.5],
    #         [1.0, 1.0],
    #         [0.0, 0.0]
    #     ])
    #     cur_fulfill = torch.tensor([
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0]])
        
    #     demand = torch.tensor([0,2, 3, 0])
    #     demand_locs = torch.tensor([0.5,0.5])
    #     item_hot = torch.tensor([0,1,0,0])
    #     state = self.trainer._create_state(
    #         inv,
    #         inv_locs,
    #         demand,
    #         demand_locs,
    #         cur_fulfill,
    #         item_hot
    #     )
        
    #     # Check if valid actions are correct
    #     valid_actions = self.trainer._get_valid_actions(state)
    #     self.assertListEqual(valid_actions.tolist(), expected_valid_actions_1)

    #     # Change the item selection
    #     item_hot = torch.tensor([0,0,1,0])
    #     state = self.trainer._create_state(
    #         inv,
    #         inv_locs,
    #         demand,
    #         demand_locs,
    #         cur_fulfill,
    #         item_hot
    #     )
        
    #     # Check if state is valid again
    #     valid_actions = self.trainer._get_valid_actions(state)
    #     self.assertListEqual(valid_actions.tolist(), expected_valid_actions_2)