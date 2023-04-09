from copy import deepcopy

import numpy as np
import nn.nn as nn


class ExponentialScheduler(object):
    def __init__(self, initial_lr=0.001, stage_length=500, staircase=False, decay=0.5):
        """
            learning_rate = initial_lr * decay ** curr_stage, where
            curr_stage = step / stage_length          if staircase = False
            curr_stage = floor(step / stage_length)   if staircase = True
        """
        super(ExponentialScheduler, self).__init__()
        self.decay = decay
        self.staircase = staircase
        self.initial_lr = initial_lr
        self.stage_length = stage_length

    def __call__(self, step=None):
        return self.learning_rate(step=step)

    def copy(self):
        return deepcopy(self)

    def learning_rate(self, step):
        cur_stage = step / self.stage_length
        if self.staircase:
            cur_stage = np.floor(cur_stage)
        return self.initial_lr * self.decay ** cur_stage


class SGD(object):
    """
    w_{t+1} = (1- decay * lr) * w_{t} - lr * grad
    """
    def __init__(self, net: nn.Module, lr=0.001, lr_scheduler=ExponentialScheduler, decay=None):
        self.lr = lr
        self.lr_scheduler = lr_scheduler(initial_lr=lr)
        self.decay = decay
        self.net = net
        self.cur_step = 0

    def step(self):
        self.lr = self.lr_scheduler(self.cur_step)
        for layer in self.net.layers:
            if layer.requires_grad:
                # update weight
                grad = layer.weight_grad.copy()
                if self.decay:
                    layer.weight -= layer.weight * self.decay * self.lr
                layer.weight -= self.lr * grad

                # update bias
                grad = layer.bias_grad.copy().reshape(1, -1)
                if self.decay:
                    layer.bias -= layer.bias * self.decay * self.lr
                layer.bias -= self.lr * grad
        self.cur_step += 1