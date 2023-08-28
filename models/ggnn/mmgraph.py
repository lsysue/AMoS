import os
import argparse

import torch
import torch.nn as nn


class MultiModelGraph(nn.Module):
    def __init__(self, n_nodes, n_edge_types, d_annotation, device='cuda'):
        super(MultiModelGraph, self).__init__()

        self.device = device
        
        self.n_nodes = n_nodes
        self.n_edge_types = n_edge_types
        self.d_annotation = d_annotation

        self.adj_matrix = None
        self.annotation = None


    def forward(self, data, text_dist, geo_dist):
        batch_size, n_samples = data.shape[0:2]
        text_type_id = 0
        geo_type_id = 1
        assert n_samples == self.n_nodes
        self.adj_matrix = torch.zeros([batch_size, self.n_nodes, self.n_nodes * self.n_edge_types * 2], device=self.device)
        self.annotation = torch.zeros([batch_size, self.n_nodes, self.d_annotation], device=self.device)

        for i in range(self.n_nodes):
            self.annotation[:, i] = data[:, i].clone()
            
            for j in range(i+1, self.n_nodes):
                d_geo = geo_dist[:, i, j]
                self.adj_matrix[:, j, geo_type_id * self.n_nodes + i] = 1 - d_geo
                self.adj_matrix[:, i, (geo_type_id + self.n_edge_types) * self.n_nodes + j] = 1 - d_geo

                d_text = text_dist[:, i, j]
                self.adj_matrix[:, j, text_type_id * self.n_nodes + i] = 1 - d_text
                self.adj_matrix[:, i, (text_type_id + self.n_edge_types) * self.n_nodes + j] = 1 - d_text
        return self.adj_matrix, self.annotation

    
    def fix_weights(self, model):
        for k, v in model.named_parameters():
            v.requires_grad = False

    def init_weights(self, module):
        """ Initialize the weights """
        for m in module:
            if isinstance(m, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                m.weight.data.normal_(mean=0.0, std=1.0)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
    


