import torch
import torch.nn as nn
import torch.nn.functional as F

from reward_manager import RewardManager
from dqn_policy import DQNTrainer
from transformer import Encoder, Decoder
from shared_models import DemandEncoder, InvEncoder
from nodes import DemandNode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNEmbTrainer(DQNTrainer):
    def __init__(self, args, reward_man: RewardManager):
        super().__init__(args, reward_man, "dqn_emb_model.pt", "DQNEmb")

    def _create_model(self) -> nn.Module:
        return DQNEmb(self.args).to(device)


    def _get_valid_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Get valid actions for the current state vector."""
        # Get the cur_item_quantity from the state vector
        cur_item_quantity = state[self.args.num_inv_nodes:self.args.num_inv_nodes*2]

        # Get the ones that are nonzero
        valid_actions = (cur_item_quantity > 0).nonzero().flatten()

        return valid_actions


    def _create_state(self,
                inv: torch.Tensor,
                inv_locs: torch.Tensor,
                demand: torch.Tensor,
                demand_loc: torch.Tensor,
                cur_fulfill: torch.Tensor,
                item_hot: torch.Tensor,
                sku_distr: torch.Tensor) -> torch.Tensor:
        """Create state for input into model"""

        # Scale features
        inv = inv / 100#self._hyper_dict["max_inv_prod"]
        inv_locs = inv_locs.flatten() / self._hyper_dict["coord_bounds"]
        demand = demand / 100#self._hyper_dict["max_inv_prod"]
        demand_loc = demand_loc.flatten() / self._hyper_dict["coord_bounds"]
        cur_fulfill = cur_fulfill / 100#self._hyper_dict["max_inv_prod"]

        # Normalize SKU distribution based on inventory
        sku_distr[(inv.sum(axis=0) == 0).nonzero().flatten()] = 0
        distr_sum = sku_distr.sum() 
        if float(distr_sum) > 0:
            sku_distr = sku_distr / distr_sum


        sku_distr = torch.log(sku_distr + 1e-16)


        # Total amount of inventory
        inv_totals = inv.sum(axis = -1)


        # Get total number of products currently getting fulfilled in the order
        cur_fulfill_totals = cur_fulfill.sum(axis=-1)

        # Get quantity of current product
        cur_item_quantity = (item_hot * inv).sum(axis=-1)

        # Get number of products each inventory node could fulfill
        num_potential_fulfill = torch.min(
            inv, demand.repeat(inv.shape[0], 1)).sum(axis=-1)

        # Get the total sum of demand that is requested
        demand_totals = demand.sum(axis=-1).unsqueeze(0)

        # Get demand for current item only
        demand_quantity = (item_hot * demand).sum(axis=-1).unsqueeze(0)

        # Get the probability of this item getting ordered
        demand_prob = (item_hot * sku_distr).sum(axis=-1).unsqueeze(0)

        # Concatenate together
        return torch.cat((
            cur_fulfill_totals,
            cur_item_quantity,
            inv_totals,
            inv_locs,
            num_potential_fulfill,
            demand_totals,
            demand_quantity,
            demand_loc, 
            demand_prob))

class DQNEmb(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.action_dim = self.args.num_inv_nodes
 
        self.inv_encoder = InvEncoder(self.args)
        self.demand_encoder = DemandEncoder(self.args)
        
        self.inp_size = 5 + 6 * self.args.num_inv_nodes

        self.state_enc = Encoder(self.args, self.args.num_enc_layers)

        # Demand will be cur item quantity, total quantity left
        # self._fc_1 = nn.Linear(self.args.emb_size, self.args.hidden_size)

        # # Create hidden fcs
        # self.hidden_fcs =  nn.ModuleList([
        #     nn.Linear(self.args.hidden_size, self.args.hidden_size) for _ in range(self.args.num_hidden - 1)])
        
        # self._q_out = nn.Linear(self.args.hidden_size, 1)
        self._q_adv_1 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self._q_adv_2 = nn.Linear(self.args.hidden_size, 1)
        
        self._q_val_1 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self._q_val_2 = nn.Linear(self.args.hidden_size, 1)

         

    def _extract_state(self, state: torch.Tensor):
        """Extract the individual parts of the state.
        
            Did it this way to prevent having to rewrite a lot of other functions.
        """
        batch_size = state.shape[0]
        
        # Unwrap the state
        cur_end_idx = 0
        cur_fulfill_totals = state[:, :self.args.num_inv_nodes].unsqueeze(2)
        cur_end_idx += self.args.num_inv_nodes
        
        cur_item_quantity = state[:, cur_end_idx:cur_end_idx+self.args.num_inv_nodes].unsqueeze(2)
        cur_end_idx += self.args.num_inv_nodes
        
        inv_totals = state[:, cur_end_idx:cur_end_idx+self.args.num_inv_nodes].unsqueeze(2)
        cur_end_idx += self.args.num_inv_nodes

        inv_locs = state[:, cur_end_idx:cur_end_idx+self.args.num_inv_nodes*2].reshape(
            batch_size, self.args.num_inv_nodes, 2)
        cur_end_idx += self.args.num_inv_nodes * 2

        num_potential_fulfill = state[:, cur_end_idx:cur_end_idx+self.args.num_inv_nodes].unsqueeze(2)
        cur_end_idx += self.args.num_inv_nodes
    

        demand_totals = state[:, cur_end_idx:cur_end_idx+1]
        cur_end_idx += 1

        demand_quantity = state[:, cur_end_idx:cur_end_idx+1]
        cur_end_idx += 1
 
        demand_loc = state[:, cur_end_idx:cur_end_idx+2]
        cur_end_idx += 2

        demand_prob = state[:, cur_end_idx:]

        # Create the input for the inventory node encoder
        inv_enc_inp = torch.cat((
            cur_fulfill_totals,
            cur_item_quantity,
            inv_totals,
            inv_locs,
            num_potential_fulfill
        ), axis = -1)


        # Create the input for the demand node encoder
        demand_enc_inp = torch.cat((
            demand_totals,
            demand_quantity,
            demand_loc,
            demand_prob
        ), axis = 1)
        
        return inv_enc_inp, demand_enc_inp

    def forward(self, 
                state) -> torch.Tensor:
        # Add batch dimension if it is not present
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        batch_size = state.shape[0]
        inv_enc_inp, demand_enc_inp = self._extract_state(state)
        #print("inv_enc_inp", inv_enc_inp)
        # print("demand_enc_inp", demand_enc_inp)
        # Get inventory node embeddings
        inv_embs = self.inv_encoder(inv_enc_inp)

        # Get demand node embedding
        demand_embs = self.demand_encoder(demand_enc_inp).unsqueeze(1)
        
        state_inp = torch.cat((inv_embs, demand_embs), 1)
        
        # Get updated invetnory node embeddings
        state_embs = self.state_enc(state_inp)
        
        #x = F.relu(self._fc_1(state_embs[:, :inv_embs.shape[1]]))

        # for hidden_fc in self.hidden_fcs:
        #     x = F.relu(hidden_fc(x))
        
        adv = F.relu(self._q_adv_1(state_embs[:, :inv_embs.shape[1]]))
        adv = self._q_adv_2(state_embs[:, :inv_embs.shape[1]])

        val = F.relu(self._q_val_1(state_embs[:, 0]))
        val = self._q_val_2(val)
        #print("adv.shape")
        # print("adv.shape", adv.shape)
        # print("val.shape", val.shape)
        # print("adv.mean(1)", adv.mean(1))
        # print("adv.mean(1):", adv.mean(1))
        # print("adv.squeeze(-1)", adv.squeeze(-1))
        # print("adv.squeeze(-1) - adv.mean(1)", adv.squeeze(-1) - adv.mean(1), "\n")

        q_vals = val + adv.squeeze(-1) - adv.mean(1)
        #print("q_vals.shape", q_vals.shape)
        #q_vals = self._q_out(x)

        if batch_size == 1:
            #print("q_vals.view(-1)", q_vals.view(-1))
            return q_vals.view(-1)

        else:
            return q_vals.view(q_vals.shape[0], q_vals.shape[1])