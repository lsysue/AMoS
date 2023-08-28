import os
import sys
import argparse

import torch
import torch.nn as nn
from torch.utils.data import _utils

from .ggnn.mmgraph import MultiModelGraph
from .ggnn.ggnn import GGNN


class GraphNet_with_Classifier(nn.Module):
    def __init__(self, ggnn_n_nodes, ggnn_n_edge_types, ggnn_d_annotation, ggnn_d_state, ggnn_n_steps, init_weight = True):
        super(GraphNet_with_Classifier, self).__init__()

        self.base_network = BaseGraphNet_with_Classifier(ggnn_n_nodes, ggnn_n_edge_types, ggnn_d_annotation, ggnn_d_state, ggnn_n_steps)

        if init_weight:
            self._initialize_weights()

    def forward(self, inputs):
        if len(inputs) == 6:
            anchor, positive, weak_positive, negative, text_dist, geo_dist = inputs
            n_anchor, n_positive, n_weak_positive, n_negative = anchor.shape[1], positive.shape[1], weak_positive.shape[1], negative.shape[1]
            if n_weak_positive > 0:
                data = torch.cat((anchor, positive, weak_positive, negative), dim=1) # data: [batch_size, n_nodes, d_state]
            else:
                data = torch.cat((anchor, positive, negative), dim=1)
        elif len(inputs) == 3:
            data, text_dist, geo_dist = inputs
        
        outputs = self.base_network((data, text_dist, geo_dist))

        return outputs[:, 1:, :]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weights, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class BaseGraphNet_with_Classifier(nn.Module):
    def __init__(self, ggnn_n_nodes, ggnn_n_edge_types, ggnn_d_annotation, ggnn_d_state, ggnn_n_steps, init_weight = True):
        super(BaseGraphNet_with_Classifier, self).__init__()
        self.ggnn_n_nodes = ggnn_n_nodes
        self.ggnn_n_edge_types = ggnn_n_edge_types
        self.ggnn_d_annotation = ggnn_d_annotation
        self.ggnn_d_state = ggnn_d_state
        self.ggnn_n_steps = ggnn_n_steps

        self.ggnn = GGNN(ggnn_d_state, ggnn_d_annotation, \
                            ggnn_n_nodes, ggnn_n_edge_types, ggnn_n_steps)
        self.mmgraph = MultiModelGraph(ggnn_n_nodes, ggnn_n_edge_types, ggnn_d_annotation)
        self.fc = nn.Linear(ggnn_d_state, 2)
        self.classifier = nn.Softmax(dim=-1)

        if init_weight:
            self._initialize_weights()
        

    def forward(self, inputs):
        x, text_dist, geo_dist = inputs
        adj_matrix, annotation = self.mmgraph(x, text_dist, geo_dist)
        batch_size, n_nodes = adj_matrix.shape[0:2]
        n_edge_types = adj_matrix.shape[-1] / (n_nodes * 2)

        assert n_nodes == self.ggnn_n_nodes and n_edge_types == self.ggnn_n_edge_types

        padding = torch.zeros((batch_size, self.ggnn_n_nodes, self.ggnn_d_state - self.ggnn_d_annotation), device=annotation.device)
        init_input = torch.cat((annotation, padding), 2)
        ggnn_output = self.ggnn(init_input, annotation, adj_matrix)  # ggnn_output: [batch_size, ggnn_n_nodes, ggnn_d_annotations]
        
        anchor_state = ggnn_output[:, 0:1, :].repeat(1, self.ggnn_n_nodes, 1)    # anchor_state: [batch_size, 1, ggnn_d_annotations]
        # print(ggnn_output[:, 0, :].shape)
        # print('okk:', anchor_state.shape, ggnn_output.shape)

        match_states = ggnn_output - anchor_state    # match_state: [batch_size, ggnn_n_nodes, ggnn_d_annotations]
        x = self.fc(match_states)
        outputs = self.classifier(x)
        # outputs = self.fc(ggnn_output)

        return outputs
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weights, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class GraphNet(nn.Module):
    def __init__(self, ggnn_n_nodes, ggnn_n_edge_types, ggnn_d_annotation, ggnn_d_state, ggnn_n_steps, init_weight = True):
        super(GraphNet, self).__init__()

        self.base_network = BaseGraphNet(ggnn_n_nodes, ggnn_n_edge_types, ggnn_d_annotation, ggnn_d_state, ggnn_n_steps)

        if init_weight:
            self._initialize_weights()

    def forward(self, inputs):
        if len(inputs) == 6:
            anchor, positive, weak_positive, negative, text_dist, geo_dist = inputs
            n_anchor, n_positive, n_weak_positive, n_negative = anchor.shape[1], positive.shape[1], weak_positive.shape[1], negative.shape[1]
            if n_weak_positive > 0:
                data = torch.cat((anchor, positive, weak_positive, negative), dim=1) # data: [batch_size, n_nodes, d_state]
            else:
                data = torch.cat((anchor, positive, negative), dim=1)
        elif len(inputs) == 3:
            data, text_dist, geo_dist = inputs
        
        outputs = self.base_network((data, text_dist, geo_dist))
        anchor_output, positive_output, weak_positive_output, negative_output = torch.split(outputs, [n_anchor, n_positive, n_weak_positive, n_negative], dim=1)
        
        return anchor_output, positive_output, weak_positive_output, negative_output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weights, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class BaseGraphNet(nn.Module):
    def __init__(self, ggnn_n_nodes, ggnn_n_edge_types, ggnn_d_annotation, ggnn_d_state, ggnn_n_steps, init_weight = True):
        super(BaseGraphNet, self).__init__()
        self.ggnn_n_nodes = ggnn_n_nodes
        self.ggnn_n_edge_types = ggnn_n_edge_types
        self.ggnn_d_annotation = ggnn_d_annotation
        self.ggnn_d_state = ggnn_d_state
        self.ggnn_n_steps = ggnn_n_steps

        self.ggnn = GGNN(ggnn_d_state, ggnn_d_annotation, \
                            ggnn_n_nodes, ggnn_n_edge_types, ggnn_n_steps)
        self.mmgraph = MultiModelGraph(ggnn_n_nodes, ggnn_n_edge_types, ggnn_d_annotation)
        self.fc = nn.Linear(ggnn_d_state, ggnn_d_state)

        if init_weight:
            self._initialize_weights()
        

    def forward(self, inputs):
        x, text_dist, geo_dist = inputs
        adj_matrix, annotation = self.mmgraph(x, text_dist, geo_dist)
        batch_size, n_nodes = adj_matrix.shape[0:2]
        n_edge_types = adj_matrix.shape[-1] / (n_nodes * 2)

        assert n_nodes == self.ggnn_n_nodes and n_edge_types == self.ggnn_n_edge_types

        padding = torch.zeros((batch_size, self.ggnn_n_nodes, self.ggnn_d_state - self.ggnn_d_annotation), device=annotation.device)
        init_input = torch.cat((annotation, padding), 2)
        ggnn_output = self.ggnn(init_input, annotation, adj_matrix)  # ggnn_output: [batch_size, ggnn_n_nodes, ggnn_d_annotations]
        
        outputs = self.fc(ggnn_output)

        return outputs
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weights, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
