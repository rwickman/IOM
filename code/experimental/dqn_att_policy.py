"""
Original DQNEmb implementation that uses explict inventory and SKU embeddings
"""

import torch
from torch.functional import cartesian_prod
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
import torch.optim as optim
import numpy as np
import os
import math
from collections import Counter
from dataclasses import dataclass

from simulator import InventoryNode, DemandNode, InventoryProduct
from reward_manager import RewardManager
from fulfillment_plan import FulfillmentPlan
from policy import PolicyResults, Experience
from rl_policy import RLPolicy
from dqn_policy import DQNTrainer
from replay_memory import ReplayMemory, PrioritizedExpReplay
from transformer import Transformer, MultiHeadAttention, create_padding_mask, PointWiseFFN



# Index for the embedding representing the current item to fulfill
CUR_PROD_ID = torch.tensor([1])

# Index for the embedding representing no products have been fulfulled yet
START_ID = torch.tensor([2])

@dataclass
class DQNAttState:
    inv: torch.Tensor
    inv_locs: torch.Tensor
    demand: torch.Tensor
    demand_loc: torch.Tensor
    cur_fulfill: torch.Tensor
    item_hot: torch.Tensor

class DQNAttTrainer(DQNTrainer):
    def __init__(self, args, reward_man: RewardManager):
        super().__init__(args, reward_man, "dqn_att_model.pt", "DQNAtt")

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        q_values = self._dqn(state)
        return q_values

    def _create_model(self) -> nn.Module:
        return DQNAtt(self.args)

class DQNAtt(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self._sku_embs = nn.Embedding(self.args.num_skus, self.args.hidden_size)
        self._inv_node_embs = nn.Embedding(self.args.num_inv_nodes, self.args.hidden_size)
        self._demand_embs = nn.Embedding(3, self.args.hidden_size)

        self._node_sku_encoder = NodeSkuEncoder(self.args)

        self._prod_encoder = ProductEncoder(self.args)

        #self._demand_encoder = NodeSkuEncoder(self.args)
        self._transformer = Transformer(self.args)

        self.inv_q_out = InvNodeQOut(self.args)

        #self._q_out_1 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        #self._q_out_2 = nn.Linear(self.args.hidden_size, self.args.num_inv_nodes)

    def _create_padded(self, inv: torch.tensor, locs: torch.tensor) -> tuple:
        """Pad the inventory, locations, sku_ids, node_ids, and get a padding mask."""
        batch_size = inv.shape[0]
        # print("inv", inv)
        # Get nonzero elements
        inv_nonzero = inv.nonzero(as_tuple=True)
        batch_ids, node_ids, sku_ids = inv_nonzero
        # print("inv", inv)
        # print("batch_ids", batch_ids)
        # print("node_ids", node_ids)
        # print("sku_ids", sku_ids)

        # Check if there are zero nonzero elements
        if len(batch_ids) == 0:
            return None, None
        
        # As all examples in batch have to be the same size, will now have to pad

        # Get the maximum size of embs over all batches
        max_size = max(Counter(batch_ids.tolist()).values())
        
        inv_sku_quantities_pad = torch.zeros(batch_size, max_size, 1)
        sku_ids_pad = torch.zeros(batch_size, max_size, dtype=torch.int64)
        node_ids_pad = torch.zeros(batch_size, max_size, dtype=torch.int64)
        locs_pad = torch.zeros(batch_size, max_size, 2)
        
        # Create the padded matrices, this may need to be optimized later on
        row_idx = 0
        prev_b_idx = batch_ids[0]        
        
        for i, b_idx in enumerate(batch_ids):
            if b_idx != prev_b_idx:
                prev_b_idx = b_idx
                row_idx = 0

            inv_sku_quantities_pad[b_idx, row_idx] = inv[b_idx, node_ids[i], sku_ids[i]]
            
            sku_ids_pad[b_idx, row_idx] = sku_ids[i]
            node_ids_pad[b_idx, row_idx] = node_ids[i]
            locs_pad[b_idx, row_idx] = locs[b_idx, node_ids[i]]
    
            row_idx += 1
      
        padding_mask = create_padding_mask(inv_sku_quantities_pad)

        return inv_sku_quantities_pad, sku_ids_pad, node_ids_pad, locs_pad, padding_mask

    def _create_comb(self, inv: torch.tensor, locs: torch.tensor, is_demand=False) -> tuple:
        
        # Get the padded values
        padded_tuple = self._create_padded(inv, locs)
        if padded_tuple[0] is None:
            return  padded_tuple

        inv_sku_quantities_pad, sku_ids_pad, node_ids_pad, locs_pad, padding_mask = padded_tuple

        # Get the embs of the skus in inventory
        inv_sku_embs = self._sku_embs(sku_ids_pad)

        # Get the embs of the inv nodes
        if is_demand:
            node_embs = self._demand_embs(node_ids_pad)
        else:
            node_embs = self._inv_node_embs(node_ids_pad)

        # Embed the inventory of the node
        inv_sku_comb = self._node_sku_encoder(
            node_embs,
            inv_sku_embs,
            inv_sku_quantities_pad,
            locs_pad)

        return inv_sku_comb, padding_mask

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

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Combine Inventory"""
        # Add batch dimension if it is not present
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]
        inv, inv_locs, demand, demand_loc, cur_fulfill, item_hot = self._extract_state(state)

        # Inventory node embeddings
        inv_sku_comb, inv_padding_mask = self._create_comb(inv, inv_locs)

        # print("inv_padding_mask", inv_padding_mask, "inv_padding_mask.shape", inv_padding_mask)
        # Demand embeddings
        demand_sku_comb, demand_padding_mask = self._create_comb(
            demand.unsqueeze(1), demand_loc, is_demand=True)
        #print("demand_sku_comb", demand_sku_comb)
        #print("demand_padding_mask", demand_padding_mask)
        #print("------------------------------------\n\n")

        # Current Fulfillment Embeddings
        fulfill_sku_comb, fulfill_padding_mask = self._create_comb(
            cur_fulfill, inv_locs)

        # Get start embedding        
        # start_ids = START_ID.repeat(batch_size).unsqueeze(1)
        # start_padding_mask = create_padding_mask(start_ids)
        # start_embs = self._demand_embs(start_ids)

        # Create the decoder input, which is just the current fulfillment embedding 
        # if fulfill_sku_comb is not None:
        #     fulfill_padding_mask = torch.cat((start_padding_mask, fulfill_padding_mask), -1)
        #     fulfill_sku_comb = torch.cat((start_embs, fulfill_sku_comb), 1)
        # else:
        #     fulfill_sku_comb = start_embs
        #     fulfill_padding_mask = start_padding_mask

        # Create the embedding representing the current product to be fulfilled
        cur_prod_idx = item_hot.nonzero(as_tuple=True)[1]
        # print("\n\nitem_hot", item_hot)
        # print("cur_prod_idx", cur_prod_idx)
        prod_padding_mask = create_padding_mask(torch.ones(batch_size, 1))

        cur_prod_embs = self._prod_encoder(
            self._sku_embs(cur_prod_idx.unsqueeze(1)),
            self._demand_embs(CUR_PROD_ID.repeat(batch_size).unsqueeze(1)))

        if fulfill_sku_comb is not None:
            enc_state = torch.cat(
                (inv_sku_comb, demand_sku_comb, cur_prod_embs, fulfill_sku_comb), 1)
            enc_padding_mask = torch.cat(
                (inv_padding_mask, demand_padding_mask, prod_padding_mask, fulfill_padding_mask), -1)
        else:
            enc_state = torch.cat(
                (inv_sku_comb, demand_sku_comb, cur_prod_embs), 1)
            enc_padding_mask = torch.cat(
                (inv_padding_mask, demand_padding_mask, prod_padding_mask), -1)

        inv_node_embs = self._inv_node_embs(
            torch.arange(self.args.num_inv_nodes).repeat(batch_size, 1))
        
        # Run embedded state through the transformer
        dec_output, _ = self._transformer(
            enc_state,
            inv_node_embs,
            enc_padding_mask)

        
        # # Compute final Q-values
        # q_vals = F.relu(self._q_out_1(dec_out_comb.reshape(-1,dec_out_comb.shape[-1])))
        # q_vals = self._q_out_2(q_vals)

        
        # Get the Q-values
        q_vals = self.inv_q_out(dec_output)
        # q_vals = F.relu(self._q_out_1(dec_output[:, 0]))
        # q_vals = self._q_out_2(q_vals)

        if batch_size == 1:
            return q_vals.view(-1)
        else:
            return q_vals.view(q_vals.shape[0], q_vals.shape[1])

class NodeSkuEncoder(nn.Module):
    """Embed the node with the inventory."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Combine the quantity with the SKU emb, in the future this can be extended to add other
        # ## node/sku information if required
        # self._sku_fc = Linear(self.args.hidden_size + 1, self.args.hidden_size)
        
        # # Combine locations with inventory nodes, can be extended in future to add other inv node information
        # self._inv_node_fc = Linear(self.args.hidden_size + 2, self.args.hidden_size)

        self.fc1 = Linear(self.args.hidden_size * 2 + 3, self.args.hidden_size)
        self.fc2 = Linear(self.args.hidden_size, self.args.hidden_size)

    def forward(self, node_embs, sku_embs, sku_quantities, locs):
        # (batch_size, # items, hidden_size)

        # sku_embs = torch.cat((sku_embs, sku_quantities), -1)

        # node_embs = torch.cat((node_embs, locs), -1)

        # Combine the inventory node embedding with the sku embedding
        sku_comb = F.relu(
            self.fc1(torch.cat((sku_embs, sku_quantities, node_embs, locs), -1)))
        sku_comb = self.fc2(sku_comb)
        # sku_comb = self.fc3(sku_comb)
        return sku_comb


class ProductEncoder(nn.Module):
    """Embed the product you currently want to be fulfilled with the demand node element."""
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.fc1 = Linear(self.args.hidden_size * 2, self.args.hidden_size)
        self.fc2 = Linear(self.args.hidden_size, self.args.hidden_size)
    
    def forward(self, demand_sku_emb, cur_prod_emb):
        # Combine the demand sku embedding with the selection embedding
        sku_comb = F.relu(
            self.fc1(torch.cat((demand_sku_emb, cur_prod_emb), -1)))
        sku_comb = self.fc2(sku_comb)
        return sku_comb

class InvNodeQOut(nn.Module):
    """Uses attention to combine decoder output with inv node embs"""
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.q_out_1 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        
        # Create hidden fcs
        self.hidden_fcs =  nn.ModuleList([
            nn.Linear(self.args.hidden_size, self.args.hidden_size) for _ in range(self.args.num_hidden - 1)])
        self.q_out_2 = nn.Linear(self.args.hidden_size, 1)
        

    def forward(self, dec_output):
        # Combine decoder output with inv_node_embs, and get q outputs
        q_out = F.relu(self.q_out_1(dec_output))
        for hidden_fc in self.hidden_fcs:
            q_out = F.relu(hidden_fc(q_out))
        q_out = self.q_out_2(q_out)

        return q_out