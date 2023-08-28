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

class StrongNearestNeighborClustering(object):
    def __init__(self, initial_rank=None, metric='cosine', verbose=True, use_ann_above_samples=70000):
        self.initial_rank = initial_rank
        self.metric = metric
        self.verbose = verbose
        self.use_ann_above_samples = use_ann_above_samples
        

    def forward(self, data, adj_matrix=None, text_dist=None, geo_dist=None, wifi_dist=None):
        n_data = data.shape[0]
        # if len(data.shape) == 3:
        #     data = data.view(n_data, -1)
        min_sim = None
        adj_matrix, original_distance = self.clust_rank(data, adj_matrix, text_dist, geo_dist, wifi_dist)
        initial_rank = None
        group, n_cluster = self.get_cluster(adj_matrix, torch.Tensor([]), min_sim)
        c, mat = self.get_merge(torch.Tensor([]), group, data)

        if self.verbose:
            print(f'Partition 0: {n_cluster} clusters')

        n_cluster = [n_cluster]
        return c.unsqueeze(-1), n_cluster, adj_matrix
    
    def clust_rank(self, mat, adj_matrix, text_dist=None, geo_dist=None, wifi_dist=None):
        s = mat.shape[0]
        initial_rank = self.initial_rank
        if initial_rank is not None:
            orig_dist = torch.empty(size=(1, 1))
        elif s <= self.use_ann_above_samples:
            orig_dist = torch.Tensor(squareform(pdist(mat, metric=self.metric)))
            orig_dist.fill_diagonal_(1e12)
            
            orig_edge_weights, orig_indices, orig_indptr = self.get_adj_matrix(orig_dist)
            if text_dist != None:
                text_edge_weights, text_indices, text_indptr = self.get_adj_matrix(text_dist)
            if geo_dist != None:
                geo_edge_weights, geo_indices, geo_indptr = self.get_adj_matrix(geo_dist)
            if wifi_dist != None:
                wifi_edge_weights, wifi_indices, wifi_indptr = self.get_adj_matrix(wifi_dist)
                wifi_edge_weights = torch.where(wifi_edge_weights > 0, 1.0, 0.0)

            
        else:
            if not pynndescent_available:
                raise MemoryError("You should use pynndescent for inputs larger than {} samples.".format(self.use_ann_above_samples))
            if self.verbose:
                print('Using PyNNDescent to compute 1st-neighbours at this step ...')

            knn_index = NNDescent(
                mat,
                n_neighbors=2,
                metric=self.metric,
            )

            result, orig_dist = knn_index.neighbor_graph
            initial_rank = result[:, 1]
            orig_dist[:, 0] = 1e12
            print('Step PyNNDescent done ...')

        # The Clustering Equation
        # orig_A = sp.csr_matrix((orig_edge_weights, orig_indices, orig_indptr), shape=(s, s))
        
        orig_A = sp.csr_matrix((torch.ones(orig_indices.shape), orig_indices, orig_indptr), shape=(s, s))
        semantic_A, spatial_A = None, None
        if text_dist != None:
            text_A = sp.csr_matrix((torch.ones(text_indices.shape), text_indices, text_indptr), shape=(s, s))
            semantic_A = text_A
        if geo_dist != None:
            geo_A = sp.csr_matrix((torch.ones(geo_indices.shape), geo_indices, geo_indptr), shape=(s, s))
            spatial_A = geo_A
        if wifi_dist != None:
            wifi_A = sp.csr_matrix((wifi_edge_weights, wifi_indices, wifi_indptr), shape=(s, s))
            spatial_A = geo_A.maximum(wifi_A)   # |
        
        A = orig_A
        if semantic_A != None and spatial_A != None:
            A = A.multiply(semantic_A.multiply(spatial_A))  # &
        if adj_matrix != None:
            A = A.multiply(adj_matrix)

        # A = semantic_A.multiply(spatial_A)
    
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
            this_row = dist[row_i, :]
            
            this_row_mask = torch.where(this_row == m, 1.0, 0.0)
            this_row_idx = torch.nonzero(this_row_mask).view(-1)
            # if m < 1e-7:
            #     this_row[this_row_idx] = 1e12
            #     second_m = torch.min(this_row)
            #     this_row_secmask = torch.where(this_row == second_m, 1.0, 0.0)
            #     this_row_mask = torch.add(this_row_mask, this_row_secmask)
            #     this_row_idx = torch.nonzero(this_row_mask).view(-1)
            
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

        num_clust, u = sp.csgraph.connected_components(csgraph=adj_matrix, directed=True, connection='strong', return_labels=True)
        u = torch.Tensor(u)
        return u, num_clust

    def get_merge(self, c, u, data):
        if len(c) != 0:
            _, ig = torch.unique(c, return_inverse=True)
            c = u[ig]
        else:
            c = u

        mat = self.cool_mean(data, c)
        return c, mat
    
    def cool_mean(self, M, u):
        s = M.shape[0]
        un, nf = torch.unique(u, return_counts=True)
        umat = sp.csr_matrix((torch.ones(s, dtype=torch.float32), (torch.arange(0, s), u)), shape=(s, len(un)))
        return (umat.T @ M) / nf.unsqueeze(1)
    

class WeakNearestNeighborClustering(object):
    def __init__(self, initial_rank=None, metric='cosine', verbose=True, use_ann_above_samples=70000):
        self.initial_rank = initial_rank
        self.metric = metric
        self.verbose = verbose
        self.use_ann_above_samples = use_ann_above_samples
        

    def forward(self, data, text_dist=None, geo_dist=None, wifi_dist=None):
        n_data = data.shape[0]
        # if len(data.shape) == 3:
        #     data = data.view(n_data, -1)
        min_sim = None
        adj_matrix, original_distance = self.clust_rank(data, text_dist, geo_dist, wifi_dist)
        initial_rank = None
        group, n_cluster = self.get_cluster(adj_matrix, torch.Tensor([]), min_sim)
        c, mat = self.get_merge(torch.Tensor([]), group, data)

        if self.verbose:
            print(f'Partition 0: {n_cluster} clusters')

        n_cluster = [n_cluster]
        return c.unsqueeze(-1), n_cluster, adj_matrix
    
    def clust_rank(self, mat, text_dist=None, geo_dist=None, wifi_dist=None):
        s = mat.shape[0]
        initial_rank = self.initial_rank
        if initial_rank is not None:
            orig_dist = torch.empty(size=(1, 1))
        elif s <= self.use_ann_above_samples:
            orig_dist = torch.Tensor(squareform(pdist(mat, metric=self.metric)))
            orig_dist.fill_diagonal_(1e12)
            
            orig_edge_weights, orig_indices, orig_indptr = self.get_adj_matrix(orig_dist)
            if text_dist != None:
                text_edge_weights, text_indices, text_indptr = self.get_adj_matrix(text_dist)
            if geo_dist != None:
                geo_edge_weights, geo_indices, geo_indptr = self.get_adj_matrix(geo_dist)
            if wifi_dist != None:
                wifi_edge_weights, wifi_indices, wifi_indptr = self.get_adj_matrix(wifi_dist)
                wifi_edge_weights = torch.where(wifi_edge_weights > 0, 1.0, 0.0)

            
        else:
            if not pynndescent_available:
                raise MemoryError("You should use pynndescent for inputs larger than {} samples.".format(self.use_ann_above_samples))
            if self.verbose:
                print('Using PyNNDescent to compute 1st-neighbours at this step ...')

            knn_index = NNDescent(
                mat,
                n_neighbors=2,
                metric=self.metric,
            )

            result, orig_dist = knn_index.neighbor_graph
            initial_rank = result[:, 1]
            orig_dist[:, 0] = 1e12
            print('Step PyNNDescent done ...')

        # The Clustering Equation
        # orig_A = sp.csr_matrix((orig_edge_weights, orig_indices, orig_indptr), shape=(s, s))
        
        orig_A = sp.csr_matrix((torch.ones(orig_indices.shape), orig_indices, orig_indptr), shape=(s, s))
        semantic_A, spatial_A = None, None
        if text_dist != None:
            text_A = sp.csr_matrix((torch.ones(text_indices.shape), text_indices, text_indptr), shape=(s, s))
            semantic_A = text_A
        if geo_dist != None:
            geo_A = sp.csr_matrix((torch.ones(geo_indices.shape), geo_indices, geo_indptr), shape=(s, s))
            spatial_A = geo_A
        if wifi_dist != None:
            wifi_A = sp.csr_matrix((wifi_edge_weights, wifi_indices, wifi_indptr), shape=(s, s))
            spatial_A = geo_A.maximum(wifi_A)   # |
        
        A = orig_A
        if semantic_A != None and spatial_A != None:
            A = A.maximum(semantic_A.multiply(spatial_A))  # &

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
            this_row = dist[row_i, :]
            
            this_row_mask = torch.where(this_row == m, 1.0, 0.0)
            this_row_idx = torch.nonzero(this_row_mask).view(-1)
            # if m < 1e-7:
            #     this_row[this_row_idx] = 1e12
            #     second_m = torch.min(this_row)
            #     this_row_secmask = torch.where(this_row == second_m, 1.0, 0.0)
            #     this_row_mask = torch.add(this_row_mask, this_row_secmask)
            #     this_row_idx = torch.nonzero(this_row_mask).view(-1)
            
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

        num_clust, cluster_label = sp.csgraph.connected_components(csgraph=adj_matrix, directed=True, connection='weak', return_labels=True)

        cluster_label = torch.tensor(cluster_label, dtype=torch.float32)
        transposed_adj_matrix = adj_matrix.transpose()
        outliers = np.where(transposed_adj_matrix.sum(axis=1) == 0)[0]
        cluster_label[outliers] = torch.arange(num_clust, num_clust + len(outliers), dtype=torch.float32)
        num_clust += len(outliers)
        return cluster_label, num_clust

    def get_merge(self, c, u, data):
        if len(c) != 0:
            _, ig = torch.unique(c, return_inverse=True)
            c = u[ig]
        else:
            c = u

        mat = self.cool_mean(data, c)
        return c, mat
    
    def cool_mean(self, M, u):
        s = M.shape[0]
        un, nf = torch.unique(u, return_counts=True)
        umat = sp.csr_matrix((torch.ones(s, dtype=torch.float32), (torch.arange(0, s), u)), shape=(s, len(un)))
        return (umat.T @ M) / nf.unsqueeze(1)