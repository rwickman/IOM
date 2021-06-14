from dataclasses import dataclass
import torch

from fulfillment_plan import FulfillmentPlan

@dataclass
class Experience:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor = None

@dataclass
class PolicyResults:
    fulfill_plan: list[Experience]
    exps: list[float]

class Policy:
    def __init__(self, is_trainable=True):
        self.is_trainable = is_trainable
        

    def reset(self):
        pass