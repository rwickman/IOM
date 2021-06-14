import numpy as np
import torch
from dataclasses import dataclass
import random

from policy import Experience

class ReplayMemory:
    def __init__(self, args):
        self.args = args
        self._memory = []
        # Pointer to end of memory
        self._cur_pos = 0

    def add(self, exp: Experience):
        """Add an experience."""
        if len(self._memory) >= self.args.mem_cap:
            self._memory[self._cur_pos] = exp
        else:
            self._memory.append(exp)

        # Update end of memory
        self._cur_pos = (self._cur_pos + 1) %  self.args.mem_cap 

    def sample(self, batch_size):
        """Sample batch size experience replay."""
        return np.random.choice(self._memory, size=batch_size, replace=False)

    def cur_cap(self):
        return len(self._memory)

class PrioritizedExpReplay:
    def __init__(self, args):
        self.args = args
        self._sum_tree = SumTree(self.args)
    
    def add(self, exp: Experience, error: float):
        """Append experience."""
        priority = self._compute_priority(error)
        self._sum_tree.add(exp, priority)

    def sample(self, batch_size: int):
        """Sample batch size experience replay."""
        segment = self._sum_tree.total() / batch_size
        
        priorities = torch.zeros(batch_size)
        exps = []
        indices = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            mass = random.uniform(a, b)
            p, e_i, tree_idx = self._sum_tree.get(mass)
            priorities[i] = p
            exps.append(e_i)
            indices.append(tree_idx)

        # Compute importance sampling weights
        sample_ps = priorities / self._sum_tree.total()
        is_ws = (sample_ps  * self.cur_cap()) ** -self.args.per_beta

        # Normalize to scale the updates downwards
        is_ws  = is_ws / is_ws.max()
        return is_ws, exps, indices

        #return np.random.choice(self._memory, size=self.args.batch_size, replace=False)

    def cur_cap(self):
        return self._sum_tree.cur_cap()

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = self._compute_priority(error)
            self._sum_tree.update(idx, priority)

    def _compute_priority(self, td_error):
        return (abs(td_error) + self.args.eps) ** self.args.per_alpha 

class SumTree:
    def __init__(self, args):
        self.args = args
        # sum tree 
        self.tree = torch.zeros(2 * self.args.mem_cap - 1)
        self.memory = []
        # Pointer to end of memory
        self._end_pos = 0
    
    def add(self, exp, priority):
        """Add experience to sum tree."""
        
        # Add experience to memory
        if len(self.memory) < self.args.mem_cap:
            self.memory.append(exp)
        else:
            self.memory[self._end_pos] = exp
        
        idx = self.args.mem_cap + self._end_pos - 1
    
        # Update memorysum tree
        self.update(idx, priority)
        
        # Update end pointer
        self._end_pos = (self._end_pos + 1) % self.args.mem_cap

    def update(self, idx, priority):
        """Update priority of element and propagate through tree."""
        # Compute priority difference
        diff = priority - self.tree[idx]

        # Propagate update through tree
        while idx >= 0:
            self.tree[idx] += diff
            # Update to parent idx
            idx = (idx - 1) // 2

    def total(self):
        return self.tree[0]

    def get(self, val):
        """Sample from sum tree based on the sampled value."""
        tree_idx = self._retrieve(val)
        data_idx = tree_idx - self.args.mem_cap + 1
        data = self.memory[data_idx]
        
        return self.tree[tree_idx], data, tree_idx

    def _retrieve(self, val):
        idx = 0
        left = 2 * idx + 1
        right = 2 * idx + 2
        while left < len(self.tree):
            if val <= self.tree[left]:
                idx = left
            else:
                idx = right
                val -= self.tree[left]

            left = 2 * idx + 1
            right= 2 * idx + 2

        return idx

    def cur_cap(self):
        return len(self.memory)