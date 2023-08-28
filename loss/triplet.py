import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist

class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, weak_margin=1, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.weak_margin = weak_margin
        self.reduction = reduction

    def forward(self, query, positive_keys, negative_keys):

        if query.shape[-1] != positive_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

        positive_dist = F.pairwise_distance(query, positive_keys)
        negative_dist = F.pairwise_distance(query, negative_keys)

        output = torch.clamp(positive_dist - negative_dist + self.margin, min=0.0)


        if self.reduction == 'mean':
            loss = torch.mean(output, dim=0)
        else:
            loss = torch.sum(output, dim=0)
        
        return loss


class QuadrupletLoss(nn.Module):
    def __init__(self, distance_function, margin=0.1, weak_margin=1, reduction='mean'):
        super().__init__()
        self.distance_function = distance_function if distance_function is not None else F.pairwise_distance
        self.margin = margin
        self.weak_margin = weak_margin
        self.reduction = reduction

    def forward(self, query, positive_keys, weak_positive_keys=None, negative_keys=None):

        if query.shape[-1] != positive_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

        positive_dist = self.distance_function(query, positive_keys)
        positive_dist_mean = torch.mean(positive_dist, dim=-1)
        # print(f'positive mean: {positive_dist_mean.shape}')
        negative_dist = self.distance_function(query, negative_keys)
        negative_dist_mean = torch.mean(negative_dist, dim=-1)
        # print(f'negative mean: {negative_dist_mean.shape}')
    
        if min(weak_positive_keys.shape) > 0:
            weak_positive_dist = self.distance_function(query, weak_positive_keys)
            weak_positive_dist_mean = torch.mean(weak_positive_dist, dim=-1)
            # print(f'weak_positive mean: {weak_positive_dist_mean.shape}')
            output = torch.clamp(positive_dist_mean - negative_dist_mean + weak_positive_dist_mean - negative_dist_mean + self.margin, min=0.0)
        else:
            output = torch.clamp(positive_dist_mean - negative_dist_mean + self.margin, min=0.0)
        # print(f'output: {output.shape}')

        if self.reduction == 'mean':
            loss = torch.mean(output, dim=0)
        else:
            loss = torch.sum(output, dim=0)
        
        return loss
