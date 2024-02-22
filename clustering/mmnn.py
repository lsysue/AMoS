import torch
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform

try:
    from pynndescent import NNDescent

    pynndescent_available = True
except Exception as e:
    warnings.warn('pynndescent not installed: {}'.format(e))
    pynndescent_available = False
    pass

class NearestNeighborClustering(object):
    def __init__(self, initial_rank=None, metric='cosine', mode='strong', verbose=True):
        self.initial_rank = initial_rank
        self.metric = metric
        self.verbose = verbose
        self.mode = mode
        self.use_ann_above_samples = 70000
        

    def forward(self, data, adj_matrix=None, text_dist=None, geo_dist=None, wifi_dist=None):
        n_data = data.shape[0]
        # if len(data.shape) == 3:
        #     data = data.view(n_data, -1)
        min_sim = None
        adj_matrix, original_distance = self.clust_rank(data, adj_matrix, text_dist, geo_dist, wifi_dist)
        initial_rank = None
        num_clusters, cluster_groups = self.get_cluster(adj_matrix, torch.Tensor([]), min_sim)
        cluster_groups = torch.Tensor(cluster_groups, device=data.device)

        if self.verbose:
            print(f'Partition 0: {num_clusters} clusters')
        return cluster_groups, num_clusters, adj_matrix
    
    def clust_rank(self, data, adj_matrix, text_dist=None, geo_dist=None, wifi_dist=None):
        s = data.shape[0]
        initial_rank = self.initial_rank
        if initial_rank is not None:
            orig_dist = torch.empty(size=(1, 1))
        elif s <= self.use_ann_above_samples:
            orig_dist = torch.Tensor(squareform(pdist(data, metric=self.metric)))
            orig_dist.fill_diagonal_(1e12)
            
            orig_edge_weights, orig_indices, orig_indptr = self.get_adj_matrix(orig_dist)
            if text_dist is not None:
                text_edge_weights, text_indices, text_indptr = self.get_adj_matrix(text_dist)
            if geo_dist is not None:
                geo_edge_weights, geo_indices, geo_indptr = self.get_adj_matrix(geo_dist)
            if wifi_dist is not None:
                wifi_edge_weights, wifi_indices, wifi_indptr = self.get_adj_matrix(wifi_dist)
                # wifi_edge_weights = torch.where(wifi_edge_weights > 0, 1.0, 0.0)

        else:
            if not pynndescent_available:
                raise MemoryError("You should use pynndescent for inputs larger than {} samples.".format(self.use_ann_above_samples))
            if self.verbose:
                print('Using PyNNDescent to compute 1st-neighbours at this step ...')

            knn_index = NNDescent(
                data,
                n_neighbors=2,
                metric=self.metric,
            )

            result, orig_dist = knn_index.neighbor_graph
            initial_rank = result[:, 1]
            orig_dist[:, 0] = 1e12
            print('Step PyNNDescent done ...')

        # The Clustering Equation
        
        orig_A = sp.csr_matrix((orig_edge_weights, orig_indices, orig_indptr), shape=(s, s))
        if text_dist is not None:
            text_A = sp.csr_matrix((text_edge_weights, text_indices, text_indptr), shape=(s, s))
            semantic_A = text_A
        if geo_dist is not None:
            geo_A = sp.csr_matrix((geo_edge_weights, geo_indices, geo_indptr), shape=(s, s))
            spatial_A = geo_A
        if wifi_dist is not None:
            wifi_A = sp.csr_matrix((wifi_edge_weights, wifi_indices, wifi_indptr), shape=(s, s))
            spatial_A = geo_A.maximum(wifi_A)   # |
        
        # A = orig_A
        # if semantic_A != None and spatial_A != None:
        #     # A = A.multiply(semantic_A.multiply(spatial_A))  # &
        #     A = A.maximum(semantic_A.multiply(spatial_A))  # &
        # if adj_matrix != None:
        #     # A = A.multiply(adj_matrix)
        #     A = A.maximum(adj_matrix)
        if adj_matrix is not None:
            A = orig_A
        else:
            A = wifi_A.maximum(text_A.multiply(geo_A))
    
        A = A + sp.eye(s, format='csr')
        A = A.tolil()
        A.setdiag(0)

        return A, orig_dist

    def get_adj_matrix(self, dist):
        min_dist, initial_rank = torch.min(dist, axis=1)
        edge_weights = torch.tensor([])
        indices = torch.tensor([])
        indptr = torch.tensor([0])
        for row_i, m in enumerate(min_dist):
            # if m == 1:
            #     continue
            this_row = dist[row_i, :]
            this_max_dist = torch.max(this_row)

            if this_max_dist == m:
                edge_weights = torch.cat((edge_weights, torch.tensor([0])), dim=-1)
                indices = torch.cat((indices, torch.tensor([0])), dim=-1)
                indptr = torch.cat((indptr, torch.tensor([indptr[-1] + 1])), dim=-1)
                continue
            
            this_row_mask = torch.where(this_row == m, 1.0, 0.0)
            this_row_idx = torch.nonzero(this_row_mask).view(-1)
            
            this_row_cnt = (torch.sum(this_row_mask) + indptr[-1]).unsqueeze(0)
            # the indice for the edge of each row
            
            edge_weights = torch.cat((edge_weights, torch.ones(this_row_idx.shape) * (1-m)), dim=-1)
            indices = torch.cat((indices, this_row_idx), dim=-1)
            indptr = torch.cat((indptr, this_row_cnt), dim=-1)
        return edge_weights, indices, indptr

    def get_cluster(self, adj_matrix, original_distance, min_sim):
        if min_sim is not None:
            outside_indices = torch.where((original_distance * adj_matrix.toarray()) > min_sim)
            for i in outside_indices:
                adj_matrix[i] = 0

        num_clusters, cluster_labels = sp.csgraph.connected_components(csgraph=adj_matrix, directed=True, connection=self.mode, return_labels=True)
        # cluster_label = torch.tensor(cluster_label, dtype=torch.float32)
        # transposed_adj_matrix = adj_matrix.transpose()
        # outliers = np.where(transposed_adj_matrix.sum(axis=1) == 0)[0]
        # cluster_label[outliers] = torch.arange(num_clust, num_clust + len(outliers), dtype=torch.float32)
        # num_clust += len(outliers)
        return num_clusters, cluster_labels
