import torch
import torch.nn as nn
import torch.nn.functional as F

class QuadrupletNet(nn.Module):
    def __init__(self, base_network):
        super(QuadrupletNet, self).__init__()
        self.base_network = base_network


    def forward(self, inputs):
        anchor, positive, weak_positive, negative, _, _ = inputs
        embedded_anchor = self.base_network(anchor)
        embedded_positive = self.base_network(positive)
        if min(weak_positive.shape) > 0:
            embedded_weak_positive = self.base_network(weak_positive)
        else:
            embedded_weak_positive = torch.tensor([])
        embedded_negative = self.base_network(negative)
        return embedded_anchor, embedded_positive, embedded_weak_positive, embedded_negative