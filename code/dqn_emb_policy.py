import torch
import torch.nn as nn
import torch.nn.functional as F

from reward_manager import RewardManager
from dqn_policy import DQNTrainer
from transformer import Encoder, Decoder
from shared_models import DemandEncoder, InvEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNEmbTrainer(DQNTrainer):
    def __init__(self, args, reward_man: RewardManager):
        super().__init__(args, reward_man, "dqn_emb_model.pt", "DQNEmb")

    def _create_model(self) -> nn.Module:
        return DQNEmb(self.args).to(device)


class DQNEmb(nn.Module):
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
        self._q_out = nn.Linear(self.args.hidden_size, 1)

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
                state) -> torch.Tensor:
        # Add batch dimension if it is not present
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]
        inv, inv_locs, demand, demand_loc, cur_fulfill, item_hot = self._extract_state(state)
        item_hot = item_hot.unsqueeze(1)
        demand = demand.unsqueeze(1)

        # Get inventory node embeddings
        inv_embs = self.inv_encoder(inv, inv_locs, demand, cur_fulfill, item_hot)
        #inv_embs = inv_embs.view(batch_size, -1)

        # Get demand node embedding
        demand_embs = self.demand_encoder(demand, demand_loc, item_hot)
        
        state_inp = torch.cat((inv_embs, demand_embs), 1)
        
        # Get updated invetnory node embeddings
        state_embs = self.state_enc(state_inp)
        
        x = F.relu(self._fc_1(state_embs[:, :inv_embs.shape[1]]))

        for hidden_fc in self.hidden_fcs:
            x = F.relu(hidden_fc(x))
        q_vals = self._q_out(x)

        if batch_size == 1:
            return q_vals.view(-1)
        else:
            return q_vals.view(q_vals.shape[0], q_vals.shape[1])