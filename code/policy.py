import torch
from dataclasses import dataclass

from fulfillment_plan import FulfillmentPlan

@dataclass
class Experience:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor = None

@dataclass
class PolicyResults:
    fulfill_plan: FulfillmentPlan
    exps: list[Experience]

class Policy:
    def __init__(self, args, is_trainable=True):
        self.args = args
        self.is_trainable = is_trainable        
