import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the negative_keys are handled.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys: (M, D) Tensor with negative samples (e.g. embeddings of other inputs)
    Returns:
         Value of the InfoNCE Loss.
     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_keys, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, query, positive_keys, negative_keys):
        # Embedding vectors should have same number of components.
        if query.shape[-1] != positive_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')
            
        # Normalize to unit vectors
        # query, positive_keys, weak_positive_keys, negative_keys = normalize(query, positive_keys, weak_positive_keys, negative_keys)
        
        # Cosine between positive pairs
        # positive_logits = torch.sum(query @ transpose(positive_keys), dim=-1)
        positive_logits = F.cosine_similarity(query, positive_keys, dim=-1)
        # negative_logits = torch.sum(query @ transpose(negative_keys), dim=-1)
        negative_logits = F.cosine_similarity(query, negative_keys, dim=-1)
        # print(positive_logits.shape, negative_logits.shape)

        logits = torch.stack([positive_logits, negative_logits], dim=-1)

        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        
        return F.cross_entropy(logits / self.temperature, labels, reduction=self.reduction)


class QuadraInfoNCE(nn.Module):

    def __init__(self, temperature=0.1, weak_temperature=1, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.weak_temperature = weak_temperature
        self.reduction = reduction

    def forward(self, query, positive_keys, weak_positive_keys, negative_keys):
        # print(query.shape, positive_keys.shape, weak_positive_keys.shape, negative_keys.shape)
        
        # Embedding vectors should have same number of components.
        if query.shape[-1] != positive_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')
            
        # Normalize to unit vectors
        # query, positive_keys, weak_positive_keys, negative_keys = normalize(query, positive_keys, weak_positive_keys, negative_keys)    # query: [10, 1, 768 * 2]


        # Cosine between positive pairs
        positive_logits = F.cosine_similarity(query, positive_keys, dim=-1) # [10, 1]
        # print(f"positive_logits: {positive_logits.shape}")

        # Cosine between all query-negative combinations
        negative_logits = F.cosine_similarity(query, negative_keys, dim=-1) # [10, 1]
        # print(f"negative_logits: {negative_logits.shape}")

        # First index in last dimension are the positive samples
        positive_exp = torch.exp(positive_logits / self.temperature)
        positive_exp_sum = torch.sum(positive_exp, dim=-1)
        # print(f'positive exp: {torch.mean(positive_exp, dim=0)}')
        negative_exp = torch.exp(negative_logits / self.temperature)
        negative_exp_sum = torch.sum(negative_exp, dim=-1)
        # print(f'negative exp: {torch.mean(negative_exp, dim=0)}')

        # weak_positive_logits = query @ transpose(weak_positive_keys)
        if min(weak_positive_keys.shape) > 0:
            weak_positive_logits = F.cosine_similarity(query, weak_positive_keys, dim=-1)
            weak_positive_logits /= self.weak_temperature
            # print(f"weak_positive_logits: {weak_positive_logits.shape}")
            # weak_positive_exp = torch.mean(torch.exp(weak_positive_logits / self.temperature), dim=-1, keepdim=False).view(-1)
            weak_positive_exp = torch.exp(weak_positive_logits / self.temperature)
            weak_positive_exp_sum = torch.sum(weak_positive_exp, dim=-1)
            output = -torch.log((positive_exp_sum + weak_positive_exp_sum) / (positive_exp_sum + weak_positive_exp_sum + negative_exp_sum))
        else:
            output = -torch.log(torch.div(positive_exp_sum, (positive_exp_sum + negative_exp_sum)))
        if self.reduction == 'mean':
            loss = torch.mean(output, dim=0)
        elif self.reduction == 'sum':
            loss = torch.sum(output, dim=0)
        else:
            loss = torch.mean(output, dim=0)
        return loss
    

def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]
