import json
import os
import torch
from dataclasses import dataclass
from argparse import Namespace

from fulfillment_plan import FulfillmentPlan

@dataclass
class Experience:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor = None
    is_expert: bool = False
    gamma: float = 0.99

@dataclass
class PolicyResults:
    fulfill_plan: FulfillmentPlan
    exps: list

class Policy:
    """Superclass for all policies."""
    def __init__(self, args, is_trainable=True, load_args=False):
        self.args = Namespace(**vars(args))
        self.is_trainable = is_trainable
        if load_args:
            self._args_file = os.path.join(self.args.save_dir, "args.json")
            if self.args.load:
                self.load_args()

    def save(self):
        with open(self._args_file, "w") as f:
            json.dump(vars(self.args), f)

    def load_args(self):
        """Load previous CLI arguments."""
        print("\n@@@@@@@@@@@@@@Loading args@@@@@@@@@@@@\n")
        with open(self._args_file) as f:
            args_dict = json.load(f)

        cur_args_dict = vars(self.args)
        for key, val in args_dict.items():
            if key in self.args.load_arg_keys:
                cur_args_dict[key] = val
                setattr(self.args, key, val)
                print(f"{key}: {val}")


    def reset(self):
        pass

    def early_stop_handler(self):
        pass        
