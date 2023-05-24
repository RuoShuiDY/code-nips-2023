import itertools

import numpy as np
import pdb
from typing import List, Dict, Tuple


EXP_MAX = 300.0


def np_exp(x):
    return np.exp(min(max(x, -EXP_MAX), EXP_MAX))


class ExpWeights(object):
    
    def __init__(self, 
                 arms: List,
                 dims: Tuple,
                 notes: Tuple,
                 lr: float = 0.2,
                 window: int = 5,
                 decay: float = 0.9,
                 init: float = 0.0,
                 use_std: bool = False,
                 eps: float = 1e-4,
                 ) -> None:
        self.dims = dims
        self.notes = notes

        self.arms = arms
        self.w = {i: init for i in range(len(self.arms))}
        self.arm = 0
        self.value = self.arms[self.arm]
        self.error_buffer = []
        self.window = window
        self.lr = lr
        self.decay = decay
        self.use_std = use_std
        self.eps = eps
        
        self.choices = [self.arm]
        self.data = []
        
    def sample(self, arm=None) -> float:
        self.arm = np.random.choice(range(0, len(self.arms)), p=self.get_probs()) if arm is None else arm
        self.value = self.arms[self.arm]
        self.choices.append(self.arm)
        return self.value

    def get_probs(self) -> List:
        p = np.array([np_exp(x) for x in self.w.values()])
        p /= np.sum(p)  # normalize to make it a distribution
        return list(p)

    def get_log(self) -> Dict:
        bandit_log = {}
        joint_probs = np.array(self.get_probs()).reshape(self.dims)
        for axis in range(len(self.dims)):
            sum_axis = list(range(len(self.dims)))
            sum_axis.remove(axis)
            marginal_probs = list(np.sum(joint_probs, axis=tuple(sum_axis)))
            for idx, marginal_prob in enumerate(marginal_probs):
                bandit_log.update(
                    {f'marginal_{self.notes[axis]}_arm{idx}': marginal_prob}
                )
        return bandit_log

    def update_dists(self, feedback: float, norm: float = 1.0) -> None:
        # Since this is non-stationary, subtract mean of previous self.window errors. 
        self.error_buffer.append(feedback)
        self.error_buffer = self.error_buffer[-self.window:]
        
        # normalize
        feedback -= np.mean(self.error_buffer)
        norm = np.std(self.error_buffer) if self.use_std and len(self.error_buffer) > 1 else norm
        feedback /= (norm + self.eps)
        
        # update arm weights
        self.w[self.arm] *= self.decay
        self.w[self.arm] += self.lr * (feedback/max(np_exp(self.w[self.arm]), self.eps))
        
        self.data.append(feedback)
