import torch
import torch.nn as nn
import torch.nn.functional as F

class DemandEncoder(nn.Module):
    """Encodes demand node state as an embedding."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Cur item demand + total demand + loc + demand prob
        self.demand_inp_size = 1 + 1 + 2 + 1
        self._demand_emb_fc_1 = nn.Linear(self.demand_inp_size , self.args.emb_size)
        self._demand_emb_fc_2 = nn.Linear(self.args.emb_size, self.args.emb_size)
    
    def forward(self, demand_enc_inp: torch.Tensor):
        """Get demand embedding from node info."""
        
        demand_embs = F.gelu(
            self._demand_emb_fc_1(demand_enc_inp))
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

    def forward(self, inv_enc_inp:torch.Tensor):
        """Get inventory embedding from node info."""
        
        inv_embs = F.gelu(self._inv_emb_fc_1(inv_enc_inp))
        inv_embs = self._inv_emb_fc_2(inv_embs)
        
        return inv_embs