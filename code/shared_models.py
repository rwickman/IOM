import torch
import torch.nn as nn
import torch.nn.functional as F

class DemandEncoder(nn.Module):
    """Encodes demand node state as an embedding."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Cur item demand + total demand + loc
        self.demand_inp_size = 1 + 1 + 2
        self._demand_emb_fc_1 = nn.Linear(self.demand_inp_size , self.args.emb_size)
        self._demand_emb_fc_2 = nn.Linear(self.args.emb_size, self.args.emb_size)
    
    def forward(self,
                demand: torch.tensor,
                demand_loc: torch.tensor,
                item_hot: torch.tensor):
        """Get demand embedding from node info."""
        # Get the total sum of demand that is requested
        demand_totals = demand.sum(axis=-1)

        # Get demand for current item only
        demand_quantity = (item_hot * demand).sum(axis=-1)
        demand_inp = torch.cat((demand_totals, demand_quantity, demand_loc), -1).unsqueeze(1)
        demand_embs = F.relu(
            self._demand_emb_fc_1(demand_inp))
        demand_embs = self._demand_emb_fc_2(demand_embs)

        return demand_embs

class InvEncoder(nn.Module):
    """Encodes inventory node state as an embedding."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Total Fulfill + cur item quantity + total stock + loc
        self.inv_inp_size = 1 + 1 + 1 + 1 + 2

        # self.inv_inp_size + cur item quantity + total demand quantity left  
        
        self._inv_emb_fc_1 = nn.Linear(self.inv_inp_size, self.args.emb_size)
        self._inv_emb_fc_2 = nn.Linear(self.args.emb_size, self.args.emb_size)

    def forward(self, 
                inv: torch.Tensor,
                inv_locs: torch.Tensor,
                demand: torch.Tensor,
                cur_fulfill: torch.Tensor,
                item_hot:torch.Tensor):
        """Get inventory embedding from node info."""

        # Get total amount of inventory
        inv_totals = inv.sum(axis = -1).unsqueeze(2)

        # Get total number of items currently getting fulfilled in the order
        cur_fulfill_totals = cur_fulfill.sum(axis=-1).unsqueeze(2)
        
        # Get quantity of current item
        cur_item_quantity = (item_hot * inv).sum(axis=-1).unsqueeze(2)

        # Get number of items similar
        num_potential_fulfill = torch.min(
            inv, demand.repeat(1, inv.shape[1], 1))

        num_potential_fulfill = num_potential_fulfill.sum(axis=-1).unsqueeze(2)

        # Get inventory node embedding
        inv_inp = torch.cat((cur_fulfill_totals, cur_item_quantity, inv_totals, inv_locs, num_potential_fulfill), axis=-1)

        inv_embs = F.relu(self._inv_emb_fc_1(inv_inp))
        inv_embs = self._inv_emb_fc_2(inv_embs)
        
        return inv_embs