import torch
import warnings
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform

try:
    from pynndescent import NNDescent

    pynndescent_available = True
except Exception as e:
    warnings.warn('pynndescent not installed: {}'.format(e))
    pynndescent_available = False
    pass


class FINCH(object):
    def __init__(self, initial_rank=None, required_n_cluster=None, exit_n_cluster=2, metric='cosine', 
                 ensure_early_exit=True, verbose=True, use_ann_above_samples=70000):
        self.initial_rank = initial_rank
        self.required_n_cluster = required_n_cluster  # requested number of clusters
        self.exit_n_cluster = exit_n_cluster            # the final merged number of clusters
        self.metric = metric
        self.verbose = verbose
        self.ensure_early_exit = ensure_early_exit
        self.use_ann_above_samples = use_ann_above_samples
        

    def forward(self, data, distance=None):
        exit_n_cluster = self.exit_n_cluster
        n_data = data.shape[0]
        # if len(data.shape) == 3:
        #     data = data.view(n_data, -1)
        min_sim = None
        adj_matrix, original_distance = self.clust_rank(data, distance)
        initial_rank = None
        group, n_cluster = self.get_cluster(adj_matrix, torch.Tensor([]), min_sim)
        c, mat = self.get_merge(torch.Tensor([]), group, data)

        if self.verbose:
            print(f'Partition 0: {n_cluster} clusters')

        if self.ensure_early_exit:
            if original_distance.shape[-1] > 2:
                min_sim = torch.max(original_distance * adj_matrix.toarray())

        c_ = c
        k = 1
        n_cluster = [n_cluster]

        while exit_n_cluster > 1:
            adj_matrix, original_distance = self.clust_rank(mat)
            u, curr_n_cluster = self.get_cluster(adj_matrix, original_distance, min_sim)
            c_, mat = self.get_merge(c_, u, data)

            n_cluster.append(curr_n_cluster)
            c = torch.column_stack((c, c_))
            exit_n_cluster = n_cluster[-2] - curr_n_cluster

            if curr_n_cluster == 1 or exit_n_cluster < 1:
                n_cluster = n_cluster[:-1]
                c = c[:, :-1]
                break

            if self.verbose:
                print('Partition {}: {} clusters'.format(k, n_cluster[k]))
            k += 1

        if self.required_n_cluster is not None:
            if self.required_n_cluster not in n_cluster:
                ind = [i for i, v in enumerate(n_cluster) if v >= self.required_n_cluster]
                required_cluster_labels = self.required_k_clustering(c[:, ind[-1]], data)
            else:
                required_cluster_labels = c[:, n_cluster.index(self.required_n_cluster)]
        else:
            required_cluster_labels = None

        return c, n_cluster, required_cluster_labels
    
    
    def clust_rank(self, mat, dist=None):
        s = mat.shape[0]
        initial_rank = self.initial_rank
        if initial_rank is not None:
            orig_dist = torch.empty(size=(1, 1))
        elif s <= self.use_ann_above_samples:
            orig_dist = torch.Tensor(squareform(pdist(mat, metric=self.metric)))
            if dist != None:
                final_dist = orig_dist + dist
            else:
                final_dist = orig_dist.clone()
            final_dist.fill_diagonal_(1e12)
            initial_rank = torch.argmin(final_dist, axis=1)
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

            result, final_dist = knn_index.neighbor_graph
            initial_rank = result[:, 1]
            final_dist[:, 0] = 1e12
            print('Step PyNNDescent done ...')

        # The Clustering Equation
        A = sp.csr_matrix((torch.ones(size=initial_rank.shape, dtype=torch.float32), (torch.arange(0, s), initial_rank)), shape=(s, s))
        A = A + sp.eye(s, format='csr')
        A = A @ A.T

        A = A.tolil()
        A.setdiag(0)
        return A, final_dist

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

    def update_adj(self, adj, d):
        # Update adj, keep one merge at a time
        idx = adj.nonzero()
        v = torch.argsort(d[idx])
        v = v[:2]
        x = [idx[0][v[0]], idx[0][v[1]]]
        y = [idx[1][v[0]], idx[1][v[1]]]
        a = sp.lil_matrix(adj.get_shape())
        a[x, y] = 1
        return a
    
    def required_k_clustering(self, c, data):
        iter_ = len(torch.unique(c)) - self.required_n_cluster
        c_, mat = self.get_merge([], c, data)
        for i in range(iter_):
            adj, orig_dist = self.clust_rank(mat)
            adj = self.update_adj(adj, orig_dist)
            u, _ = self.get_clust(adj, [], min_sim=None)
            c_, mat = self.get_merge(c_, u, data)
        return c_


class AdvFINCH(object):
    def __init__(self, initial_rank=None, required_n_cluster=None, exit_n_cluster=2, metric='cosine', enable_hierarchy = True, ensure_early_exit=True, verbose=True, use_ann_above_samples=70000):
        self.initial_rank = initial_rank
        self.required_n_cluster = required_n_cluster  # requested number of clusters
        self.exit_n_cluster = exit_n_cluster            # the final merged number of clusters
        self.metric = metric
        self.enable_hierarchy = enable_hierarchy
        self.verbose = verbose
        self.ensure_early_exit = ensure_early_exit
        self.use_ann_above_samples = use_ann_above_samples
        

    def forward(self, data, distance=None):
        exit_n_cluster = self.exit_n_cluster
        n_data = data.shape[0]
        # if len(data.shape) == 3:
        #     data = data.view(n_data, -1)
        min_sim = None
        adj_matrix, original_distance = self.clust_rank(data, distance)
        initial_rank = None
        group, n_cluster = self.get_cluster(adj_matrix, torch.Tensor([]), min_sim)
        c, mat = self.get_merge(torch.Tensor([]), group, data)

        if self.verbose:
            print(f'Partition 0: {n_cluster} clusters')

        if self.ensure_early_exit:
            if original_distance.shape[-1] > 2:
                min_sim = torch.max(original_distance * adj_matrix.toarray())

        n_cluster = [n_cluster]
        if not self.enable_hierarchy:
            return c.unsqueeze(-1), n_cluster, adj_matrix

        c_ = c
        k = 1

        while exit_n_cluster > 1:
            adj_matrix, original_distance = self.clust_rank(mat)
            u, curr_n_cluster = self.get_cluster(adj_matrix, original_distance, min_sim)
            c_, mat = self.get_merge(c_, u, data)

            n_cluster.append(curr_n_cluster)
            c = torch.column_stack((c, c_))
            exit_n_cluster = n_cluster[-2] - curr_n_cluster

            if curr_n_cluster == 1 or exit_n_cluster < 1:
                n_cluster = n_cluster[:-1]
                c = c[:, :-1]
                break

            if self.verbose:
                print('Partition {}: {} clusters'.format(k, n_cluster[k]))
            k += 1

        if self.required_n_cluster is not None:
            if self.required_n_cluster not in n_cluster:
                ind = [i for i, v in enumerate(n_cluster) if v >= self.required_n_cluster]
                required_cluster_labels = self.required_k_clustering(c[:, ind[-1]], data)
            else:
                required_cluster_labels = c[:, n_cluster.index(self.required_n_cluster)]
        else:
            required_cluster_labels = None

        return c, n_cluster, required_cluster_labels
    
    
    def clust_rank(self, mat, dist=None):
        s = mat.shape[0]
        initial_rank = self.initial_rank
        if initial_rank is not None:
            orig_dist = torch.empty(size=(1, 1))
        elif s <= self.use_ann_above_samples:
            orig_dist = torch.Tensor(squareform(pdist(mat, metric=self.metric)))
            if dist != None:
                final_dist = orig_dist + dist
            else:
                final_dist = orig_dist.clone()
            final_dist.fill_diagonal_(1e12)
            
            min_dist, initial_rank = torch.min(final_dist, axis=1)
            edge_weights = torch.tensor([])
            indices = torch.tensor([])
            indptr = torch.tensor([0])
            for row_i, m in enumerate(min_dist):
                this_row = final_dist[row_i, :]
                
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

            result, final_dist = knn_index.neighbor_graph
            initial_rank = result[:, 1]
            final_dist[:, 0] = 1e12
            print('Step PyNNDescent done ...')

        # The Clustering Equation
        # A = sp.csr_matrix((torch.ones(size=indices.shape, dtype=torch.float32), indices, indptr), shape=(s, s))
        A = sp.csr_matrix((edge_weights, indices, indptr), shape=(s, s))
        # A = sp.csr_matrix((), indices, indptr), shape=(s, s))
        A = A + sp.eye(s, format='csr')

        A = A.tolil()
        A.setdiag(0)
        return A, final_dist

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

    def update_adj(self, adj, d):
        # Update adj, keep one merge at a time
        idx = adj.nonzero()
        v = torch.argsort(d[idx])
        v = v[:2]
        x = [idx[0][v[0]], idx[0][v[1]]]
        y = [idx[1][v[0]], idx[1][v[1]]]
        a = sp.lil_matrix(adj.get_shape())
        a[x, y] = 1
        return a
    
    def required_k_clustering(self, c, data):
        iter_ = len(torch.unique(c)) - self.required_n_cluster
        c_, mat = self.get_merge([], c, data)
        for i in range(iter_):
            adj, orig_dist = self.clust_rank(mat)
            adj = self.update_adj(adj, orig_dist)
            u, _ = self.get_clust(adj, [], min_sim=None)
            c_, mat = self.get_merge(c_, u, data)
        return c_
    

class NearestNeighborClustering(object):
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