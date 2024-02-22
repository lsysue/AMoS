import random
import torch
import torch.utils.data as data
import numpy as np
from .datautils import pairwise_distance, top_k, multimodal_distance
from scipy.spatial.distance import pdist, cdist, squareform

class Quadruplets(data.Dataset):
    def __init__(self, data,
                text_distance = None,
                geo_distance = None, 
                multimodal_distance = None,
                distance_metric = 'euclidean',
                cluster_labels = None,
                true_labels = None,
                random_weak_positive_sampling = True,
                random_positive_sampling = True, 
                random_negative_sampling = True,
                positive_sampling = 0.005, 
                negative_sampling = 0.01,
                weak_positive_sampling = 0.001, 
                device = 'cuda:0'
                ):
        
        self.device = device

        self.random_positive_sampling = random_positive_sampling
        self.random_weak_positive_sampling = random_weak_positive_sampling
        self.random_negative_sampling = random_negative_sampling

        self.positive_sampling = positive_sampling
        self.negative_sampling = negative_sampling
        self.weak_positive_sampling = weak_positive_sampling

        self.text_distance = text_distance.to(self.device)
        self.geo_distance = geo_distance.to(self.device)
        self.multimodal_distance = multimodal_distance.to(self.device)
        self.data = data.to(self.device)
        self.cluster_labels = cluster_labels.to(self.device)
        self.true_labels = true_labels.to(self.device)

        self.n_data = data.shape[0]
        self.feat_distmat = pairwise_distance(data, data, metric=distance_metric)   # size=(N, N, 2)
        self.fused_distmat = self.feat_distmat.to(self.device)
        # if multimodal_distance == None:
        #     self.fused_distmat = self.feat_distmat
        # else:
        #     self.fused_distmat = multimodal_distance + self.feat_distmat
        self.fused_distmat = self.fused_distmat.to(self.device)


    def __getitem__(self, anchor_index: int):

        anchor = self.data[anchor_index].unsqueeze(0)
        anchor_cluster_label = self.cluster_labels[anchor_index].unsqueeze(0)
        anchor_true_label = self.true_labels[anchor_index].unsqueeze(0)

        labels = torch.where(self.cluster_labels == anchor_cluster_label, 1.0, 0.0)
        tlabels = torch.where(self.true_labels == anchor_true_label, 1.0, 0.0)

        # weak_distlist = self.multimodal_distance[anchor_index]
        geo_distlist = self.geo_distance[anchor_index]
        text_distlist = self.text_distance[anchor_index]
        # geo_distmax = torch.max(geo_distlist)
        # text_distmax = torch.max(text_distlist)
        geo_distmax = torch.max(self.geo_distance)
        text_distmax = torch.max(self.text_distance)
        self.geo_positive_probs = 1 - geo_distlist / geo_distmax
        self.text_positive_probs = 1 - text_distlist / text_distmax

        # positive sampling
        if self.random_positive_sampling:
            n_positive, positive_indices = self.random_positive_sampler(anchor_index, labels)
        else:
            n_positive, positive_indices = self.positive_sampler(anchor_index, labels)

        if min(positive_indices.shape) and len(positive_indices) >= n_positive:
            positive_candidates = self.data[positive_indices]
        elif min(positive_indices.shape) and len(positive_indices) < n_positive:
            n_padding = n_positive - len(positive_indices)
            positive_candidates = self.data[positive_indices]
            positive_padding = anchor.clone().repeat(n_padding, 1)
            positive_padding_noise = torch.randn(positive_padding.shape, device=self.device) * 0.01
            positive_padding += positive_padding_noise
            positive_candidates = torch.cat((positive_candidates, positive_padding), dim=0)
            positive_indices = torch.cat((positive_indices.to(torch.long), torch.tensor([anchor_index], device=self.device, dtype=torch.long).repeat(n_padding)), dim=0)
        else:
            n_padding = n_positive
            positive_padding = anchor.clone().repeat(n_padding, 1)
            positive_padding_noise = torch.randn(positive_padding.shape, device=self.device) * 0.01
            positive_candidates = positive_padding + positive_padding_noise
            positive_indices = torch.tensor([anchor_index], device=self.device, dtype=torch.long).repeat(n_padding)
        positive_labels = labels[positive_indices]
        positive_tlabels = tlabels[positive_indices]

        # positive_candidates = positive_candidates + torch.randn(positive_candidates.shape, device=self.device)
        # weak positive sampling

        if self.random_weak_positive_sampling:
            n_weak_positive, weak_positive_indices = self.random_weak_positive_sampler(labels)
        else:
            n_weak_positive, weak_positive_indices = self.weak_positive_sampler(anchor_index, labels)
        
        if min(weak_positive_indices.shape) and len(weak_positive_indices) >= n_weak_positive:
            weak_positive_candidates = self.data[weak_positive_indices]
            # weak_positive_labels = torch.ones(size=weak_positive_indices.shape, device=self.device)
            weak_positive_labels = labels[weak_positive_indices]
            
            # weak_positive_labels = torch.where(self.geo_positive_probs[weak_positive_indices] > 0.995, 1.0, 0.0)
            weak_positive_labels = torch.where(torch.logical_or(self.geo_positive_probs[weak_positive_indices] >= 0.998, self.text_positive_probs[weak_positive_indices] >= 0.998), 1.0, 0.0)
            weak_positive_tlabels = tlabels[weak_positive_indices]
        elif min(weak_positive_indices.shape) and len(weak_positive_indices) < n_weak_positive:
            n_padding = n_weak_positive - len(weak_positive_indices)
            weak_positive_candidates = self.data[weak_positive_indices]
            weak_positive_padding = anchor.clone().repeat(n_padding, 1)
            weak_positive_padding_noise = torch.randn(weak_positive_padding.shape, device=self.device) * 0.00001
            weak_positive_padding += weak_positive_padding_noise
            weak_positive_candidates = torch.cat((weak_positive_candidates, weak_positive_padding), dim=0)
            # weak_positive_labels = torch.ones(size=weak_positive_indices.shape, device=self.device)
            weak_positive_labels = labels[weak_positive_indices]
            # weak_positive_labels = torch.where(self.geo_positive_probs[weak_positive_indices] > 0.995, 1.0, 0.0)
            weak_positive_labels = torch.where(torch.logical_or(self.geo_positive_probs[weak_positive_indices] >= 0.998, self.text_positive_probs[weak_positive_indices] >= 0.998), 1.0, 0.0)
            weak_positive_indices = torch.cat((weak_positive_indices.to(torch.long), torch.tensor([anchor_index], device=self.device, dtype=torch.long).repeat(n_padding)), dim=0)
            weak_positive_labels = torch.cat((weak_positive_labels.to(torch.long), torch.tensor([1], device=self.device, dtype=torch.long).repeat(n_padding)), dim=0)
            weak_positive_tlabels = tlabels[weak_positive_indices]
        else:
            n_padding = n_weak_positive
            weak_positive_padding = anchor.clone().repeat(n_padding, 1)
            weak_positive_padding_noise = torch.randn(weak_positive_padding.shape, device=self.device) * 0.00001
            weak_positive_candidates = weak_positive_padding + weak_positive_padding_noise
            weak_positive_indices = torch.tensor([anchor_index], device=self.device, dtype=torch.long).repeat(n_padding)
            weak_positive_labels = labels[weak_positive_indices]
            # weak_positive_labels = torch.ones(size=weak_positive_indices.shape, device=self.device)
            weak_positive_tlabels = tlabels[weak_positive_indices]


        # negative sampling
        if self.random_negative_sampling:
            n_negative, negative_indices = self.random_negative_sampler(labels)
        else:
            n_negative, negative_indices = self.negative_sampler(anchor_index, labels)

        negative_candidates = self.data[negative_indices]
        # negative_labels = self.cluster_labels[negative_indices]
        negative_labels = labels[negative_indices]
        negative_tlabels = tlabels[negative_indices]

        candidate_indices = torch.cat((positive_indices.to(torch.long), weak_positive_indices.to(torch.long), negative_indices.to(torch.long)), dim=0)
        # random_order = torch.randperm(candidate_indices.size(0))
        # shuffled_candidate_indices = candidate_indices[random_order]
        # sample_indices = torch.cat((torch.tensor([anchor_index], dtype=torch.long, device=self.device), shuffled_candidate_indices), dim=0)
        sample_indices = torch.cat((torch.tensor([anchor_index], dtype=torch.long, device=self.device), candidate_indices), dim=0)
        # text_distance = self.text_distance - torch.diag(torch.diag(self.text_distance))
        # geo_distance = self.geo_distance - torch.diag(torch.diag(self.geo_distance))
        # text_distance = text_distance[sample_indices, :][:, sample_indices] + torch.eye(len(sample_indices), device=self.device)
        # geo_distance = geo_distance[sample_indices, :][:, sample_indices] + torch.eye(len(sample_indices), device=self.device)
        # text_distance = text_distance[sample_indices, :][:, sample_indices]
        # geo_distance = geo_distance[sample_indices, :][:, sample_indices]
        text_distance = self.text_distance[sample_indices, :][:, sample_indices]
        geo_distance = self.geo_distance[sample_indices, :][:, sample_indices]
        # sample_labels = labels[shuffled_candidate_indices]
        # sample_tlabels = labels[shuffled_candidate_indices]
        sample_labels = labels[candidate_indices]
        sample_tlabels = labels[candidate_indices]

        # weak_positive_texts = self.text_positive_probs[weak_positive_indices]
        # weak_positive_geos = self.geo_positive_probs[weak_positive_indices]
        # wfp_indices = torch.argwhere(labels - tlabels == 1).view(-1)
        # wfp_geo_probs = self.geo_positive_probs[wfp_indices]
        # wfp_text_probs = self.text_positive_probs[wfp_indices]
        # # fp_probs = torch.index_select(self.geo_positive_probs, 0, fp_indices)
        # wfn_indices = torch.argwhere(tlabels - labels == 1).view(-1)
        # wfn_geo_probs = self.geo_positive_probs[wfn_indices]
        # wfn_text_probs = self.text_positive_probs[wfn_indices]

        return (anchor, positive_candidates, weak_positive_candidates, negative_candidates, text_distance, geo_distance), \
            sample_labels, sample_tlabels, sample_indices
            # (anchor_cluster_label, positive_labels, weak_positive_labels, negative_labels), \
            # (anchor_true_label, positive_tlabels, weak_positive_tlabels, negative_tlabels), sample_indices


    def positive_sampler(self, anchor_index: int, labels: torch.Tensor):
        # choose the closest samples from the same cluster
        # determine the number of positive samples
        if self.positive_sampling >= 1:
            n_positive = round(self.positive_sampling)
        else:
            n_positive = round(self.n_data * self.positive_sampling)
        # calculate the distance between the anchor and all candidates
        fused_feat_distlist = self.fused_distmat[anchor_index, :]
        # set distances of all samples in different clusters to maximum
        all_positive_indices = torch.argwhere(labels == 1).view(-1)
        if n_positive >= len(all_positive_indices):
            return n_positive, all_positive_indices
        
        inside_feat_distlist = torch.where(labels == 1, fused_feat_distlist, 1e12)
        inside_feat_distlist[anchor_index] = 1e12

        positive_indices, positive_distances = top_k(inside_feat_distlist, n_positive, find_maximum=False)
        if positive_distances[0] >= 1e12:
            return torch.tensor([], device=self.device)

        random_select = torch.tensor(random.sample(range(len(positive_indices)), 1), device=self.device)
        positive_indices = torch.index_select(positive_indices, 0, random_select)
        return n_positive, positive_indices
    
    def random_positive_sampler(self, anchor_index, labels):
        # choose the closest samples from the different cluster
        # determine the number of negative samples
        if self.positive_sampling >= 1:
            n_positive = round(self.positive_sampling)
        else:
            n_positive = round(self.n_data * self.positive_sampling)
        
        all_positive_indices = torch.argwhere(labels == 1).view(-1)
        all_positive_indices = torch.tensor(
            list(set(all_positive_indices.cpu().numpy()).difference(set([anchor_index]))),
            device=self.device)
        if n_positive >= len(all_positive_indices):
            return n_positive, all_positive_indices
        
        random_select = torch.tensor(random.sample(range(len(all_positive_indices)), n_positive), device=self.device)
        positive_indices = torch.index_select(all_positive_indices, 0, random_select)

        return n_positive, positive_indices

    def negative_sampler(self, anchor_index: int, labels: torch.Tensor):
        # choose the closest samples from the different cluster
        # determine the number of negative samples
        if self.negative_sampling >= 1:
            n_negative = round(self.negative_sampling)
        else:
            n_negative = round(self.n_data * self.negative_sampling)
        # calculate the distance between the anchor and all candidates
        fused_feat_distlist = self.fused_distmat[anchor_index, :]
        inside_feat_distlist = torch.where(labels == 0, fused_feat_distlist, 1e12)

        clusters = torch.unique(self.cluster_labels)
        negative_indices = torch.LongTensor([0])
        # for each cluster, find the sample with the closest distance as a negative sample
        for c in clusters:
            if self.cluster_labels[anchor_index] == c:
                continue
            else:
                inside_feat_distlist = fused_feat_distlist.clone().detach()
                outside_indices = torch.argwhere(self.cluster_labels != c)
                inside_feat_distlist[outside_indices] = 1e12
                inside_indices, _ = top_k(inside_feat_distlist, n_negative, find_maximum=False)
                negative_indices = torch.cat((negative_indices, inside_indices), dim=0)
        negative_indices = torch.unique(negative_indices[1:])              

        return n_negative, negative_indices
        
    def random_negative_sampler(self, labels):
        # choose the closest samples from the different cluster
        # determine the number of negative samples
        if self.negative_sampling >= 1:
            n_negative = round(self.negative_sampling)
        else:
            n_negative = round(self.n_data * self.negative_sampling)
        
        # all_indices = torch.arange(self.n_data, device=self.device)
        all_negative_indices = torch.argwhere(labels == 0).view(-1)
        if n_negative >= len(all_negative_indices):
            return n_negative, all_negative_indices
        
        negative_probs = torch.where(labels == 0, 1 - self.geo_positive_probs, 0.0)
        # random_select = torch.tensor(random.sample(range(len(all_negative_indices)), n_negative), device=self.device)
        negative_indices = torch.multinomial(negative_probs, n_negative, replacement=False).to(self.device)
        # negative_indices = torch.index_select(all_indices, 0, random_select)
        # print(negative_indices)

        return n_negative, negative_indices

    def weak_positive_sampler(self, anchor_index, labels):
        # randomly choose some samples from the same cluster whose geographical distances are bounded
        if self.weak_positive_sampling >= 1:
            n_weak_positive = round(self.weak_positive_sampling)
        else:
            n_weak_positive = round(self.n_data * self.weak_positive_sampling)

        if n_weak_positive < 1:
            return n_weak_positive, torch.tensor([], device=self.device)
        # geo_distlist = self.fused_distmat[anchor_index, :, 1] # use geo coordinates to measure geo distance
        # geo_feat_distlist = self.fused_distmat[anchor_index, :, 1]  # use geo embedding/feature to measure geo distance
        
        all_negative_indices = torch.argwhere(labels == 0).view(-1)
        geo_positive_probs = torch.where(labels == 0, self.geo_positive_probs, 0.0)

        weak_positive_indices, top_k_weak_prob = top_k(geo_positive_probs, n_weak_positive, find_maximum=True)
        if top_k_weak_prob[0] < 0.85:
            return n_weak_positive, torch.tensor([], device=self.device)

        if n_weak_positive >= len(weak_positive_indices):
            return n_weak_positive, weak_positive_indices
        else:
            random_select = torch.tensor(random.sample(range(len(weak_positive_indices)), n_weak_positive), device=self.device)
            weak_positive_indices = torch.index_select(weak_positive_indices, 0, random_select)

        return n_weak_positive, weak_positive_indices
    
    def random_weak_positive_sampler(self, labels):
        if self.weak_positive_sampling >= 1:
            n_weak_positive = round(self.weak_positive_sampling)
        else:
            n_weak_positive = round(self.n_data * self.weak_positive_sampling)
        
        if n_weak_positive < 1:
            return n_weak_positive, torch.tensor([])

        # all_indices = torch.arange(self.n_data, device=self.device)
        all_negative_indices = torch.argwhere(labels == 0).view(-1)
        assert n_weak_positive <= len(all_negative_indices)

        geo_positive_probs = torch.where(labels == 0, self.geo_positive_probs, 0.0)
        weak_positive_indices = torch.multinomial(geo_positive_probs, n_weak_positive, replacement=False).to(self.device)
        # weak_positive_indices = torch.index_select(all_indices, 0, random_select)
        return n_weak_positive, weak_positive_indices

    def __len__(self):
        return self.n_data


class Graphset(data.Dataset):
    def __init__(self, data,
                text_distance = None,
                geo_distance = None,
                n_candidates = 20,
                cluster_labels = None,
                true_labels = None,
                device = 'cuda:0'
                ):
        
        self.device = device

        self.data = data.to(self.device)
        self.text_distance = text_distance
        self.geo_distance = geo_distance
        self.n_candidates = n_candidates
        self.cluster_labels = cluster_labels.to(self.device)
        self.true_labels = true_labels.to(self.device)

        self.n_data = data.shape[0]


    def __getitem__(self, anchor_index: int):

        # anchor = self.data[anchor_index].unsqueeze(0)
        anchor_cluster_label = self.cluster_labels[anchor_index]
        # all_positive_indices = torch.argwhere(self.cluster_labels == anchor_cluster_label).view(-1)
        # all_positive_indices = torch.tensor(
        #     list(set(all_positive_indices.cpu().numpy()).difference(set([anchor_index]))),
        #     device=self.device)
        # all_negative_indices = torch.argwhere(self.cluster_labels != anchor_cluster_label).view(-1)
        # if len(all_positive_indices) >= self.n_candidates:
        #     n_positive = self.n_candidates // 2
        # else:
        #     n_positive = all_positive_indices.shape[0]

        # n_negative = self.n_candidates - n_positive

        # positive_indices = all_positive_indices[torch.randperm(all_positive_indices.size(0), device=self.device)[:n_positive]]
        # negative_indices = all_negative_indices[torch.randperm(all_negative_indices.size(0), device=self.device)[:n_negative]]

        # candidate_indices = torch.cat((positive_indices.to(torch.long), negative_indices.to(torch.long)), dim=0)
        select_probs = torch.where(torch.arange(self.n_data) != anchor_index, 1.0 / (self.n_data - 1), 0.0)
        candidate_indices = torch.multinomial(select_probs, self.n_candidates, replacement=False)
        sample_indices = torch.cat((torch.tensor([anchor_index], dtype=torch.long, device=self.device), candidate_indices.to(self.device)), dim=0)
        samples = self.data[sample_indices]
        text_dist = self.text_distance[sample_indices, :][:, sample_indices]
        geo_dist = self.geo_distance[sample_indices, :][:, sample_indices]

        anchor_true_label = self.true_labels[anchor_index]
        candidate_cluster_labels = torch.where(self.cluster_labels[candidate_indices] == anchor_cluster_label, 1, 0)
        candidate_true_labels = torch.where(self.true_labels[candidate_indices] == anchor_true_label, 1, 0)

        return (samples, text_dist, geo_dist), candidate_cluster_labels, candidate_true_labels, candidate_indices


    def __len__(self):
        return self.n_data