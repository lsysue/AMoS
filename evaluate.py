import os
import time
import random
import pickle
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics import (pair_confusion_matrix,
                            rand_score, adjusted_rand_score,
                            normalized_mutual_info_score, 
                            adjusted_mutual_info_score)
from sklearn.metrics import (precision_score, recall_score, f1_score)

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import minimize

from geopy.distance import geodesic

from tqdm import tqdm
from ast import literal_eval
from datetime import datetime

from datasets.eleme import Eleme
from datasets.dataset import Triplets
from datasets.datautils import normalize, top_k
from clustering.finch import FINCH, AdvFINCH
from models.modelutils import (load_pretrained_model, save_checkpoint, load_checkpoint, create_output_dirs)
from models.basenet import BaseFC, ThreeLayerFC, FourLayerFC
from models.tripletnet import TripletNet
from models.transformer.bert import Bert
import misc.distributed_helper as du_helper
from config.param_parser import load_config, arg_parser

np.set_printoptions(threshold=np.inf)

# Select the appropriate model with the specified cfg parameters
def base_model(cfg):
    assert cfg.MODEL.ARCH in ['basenet', 'basenet-3', 'basenet-4', 'transformer']
    if cfg.MODEL.ARCH == 'basenet':
        model = BaseFC(device=cfg.GLOBAL.DEVICE, d_inputs=cfg.DATA.D_TEXT + cfg.DATA.D_GEO, dropout=cfg.MODEL.DROPOUT)
    elif cfg.MODEL.ARCH == 'basenet-3':
        model = ThreeLayerFC(device=cfg.GLOBAL.DEVICE, d_inputs=cfg.DATA.D_TEXT + cfg.DATA.D_GEO, dropout=cfg.MODEL.DROPOUT)
    elif cfg.MODEL.ARCH == 'basenet-4':
        model = FourLayerFC(device=cfg.GLOBAL.DEVICE, d_inputs=cfg.DATA.D_TEXT + cfg.DATA.D_GEO, dropout=cfg.MODEL.DROPOUT)
    elif cfg.MODEL.ARCH == 'transformer':
        model = Bert(hidden=cfg.DATA.D_TEXT, n_layers=2, attn_heads=2)
    return model

def cluster_algorithm(cfg):
    assert cfg.ITERCLUSTER.METHOD in ['finch', 'advfinch', 'kmeans', 'dbscan', 'optics']
    if cfg.ITERCLUSTER.METHOD == 'finch':
        model = FINCH(initial_rank=None, required_n_cluster=None, exit_n_cluster=2, metric=cfg.ITERCLUSTER.DIST_METRIC, ensure_early_exit=True, verbose=True, use_ann_above_samples=70000)
    elif cfg.ITERCLUSTER.METHOD == 'advfinch':
        model = AdvFINCH(initial_rank=None, required_n_cluster=None, exit_n_cluster=2, metric=cfg.ITERCLUSTER.DIST_METRIC, enable_hierarchy=False, ensure_early_exit=True, verbose=True, use_ann_above_samples=70000)
    elif cfg.ITERCLUSTER.METHOD == 'kmeans':
        model = KMeans(n_clusters=cfg.ITERCLUSTER.K)
    elif cfg.ITERCLUSTER.METHOD == 'dbscan':
        model = DBSCAN(eps=cfg.ITERCLUSTER.EPS, min_samples=cfg.ITERCLUSTER.MIN_SAMPLES, metric=cfg.ITERCLUSTER.DIST_METRIC)
    elif cfg.ITERCLUSTER.METHOD == 'optics':
        model = OPTICS(min_samples=cfg.ITERCLUSTER.MIN_SAMPLES, metric=cfg.ITERCLUSTER.DIST_METRIC)
    return model


def get_rand_index_and_f_measure(labels_true, labels_pred, beta=1.):
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    ri = (tp + tn) / (tp + tn + fp + fn)
    ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    p, r = tp / (tp + fp), tp / (tp + fn)
    f_beta = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
    return p, r, f_beta

def evaluate_clustering(true_labels, cluster_labels):
    labeled_index = torch.argwhere(true_labels >= 1).view(-1)
    def have_anno(x):
        if x in labeled_index:
            return 1
        else:
            return 0
        # return x not in labeled_index
    
    anno_mask = torch.tensor(list(map(have_anno, list(range(cluster_labels.shape[0])))))
    anno_indices = torch.nonzero(anno_mask)
    pred_labels = cluster_labels[anno_indices].view(-1)
    true_labels = true_labels[anno_indices].view(-1)

    p, r, f_beta = get_rand_index_and_f_measure(true_labels, pred_labels)
    ri = rand_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ami = adjusted_mutual_info_score(true_labels, pred_labels)
    return ri, ari, nmi, ami

def evaluate_top_k(true_labels, pred_embedding, cfg):
    pred_dist = torch.tensor(squareform(pdist(pred_embedding, metric=cfg.DATASET.DIST_METRIC)))
    pred_dist.fill_diagonal_(2)
    true_labels = true_labels.unsqueeze(0).repeat((true_labels.shape[-1], 1))
    true_labels = torch.where(true_labels == torch.diag(true_labels)[:, None], 1, 0)
    true_labels = true_labels - torch.diag(torch.diag(true_labels))

    true_labels = squareform(true_labels)

    max_matrics = None
    max_thdhold = None

    # for thd_hold in [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09]:
    for thd_hold in [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.3,
                     1.5, 1.7, 1.9]:

        cur_thd_hold = thd_hold

        # logging.info(f'embedding distance: {pred_dist}')
        if cfg.EVAL.THRES_OR_TOPK == 0:
            # pred_labels = torch.where(pred_dist < cfg.EVAL.THRESHOLD, 1, 0)
            pred_labels = torch.where(pred_dist < cur_thd_hold, 1, 0)
        else:
            topk_indices, _ = top_k(pred_dist, k=cfg.EVAL.TOPK, find_maximum=False)
            pred_labels = torch.zeros(true_labels.shape)
            pred_labels[topk_indices[:, 0], topk_indices[:, 1]] = 1

        pred_labels = squareform(pred_labels)

        p = precision_score(true_labels, pred_labels, average='binary')
        r = recall_score(true_labels, pred_labels, average='binary')
        f_beta = f1_score(true_labels, pred_labels, average='binary')

        if max_thdhold is None:
            max_thdhold = cur_thd_hold
            max_matrics = {'p':p, 'r':r, 'f_beta':f_beta}
        elif max_matrics['f_beta'] < f_beta:
            max_thdhold = cur_thd_hold
            max_matrics = {'p': p, 'r': r, 'f_beta': f_beta}

    print('max_thdhold:', max_thdhold)
    p = max_matrics['p']
    r = max_matrics['r']
    f_beta = max_matrics['f_beta']

    return p, r, f_beta, max_thdhold

def objective_function(x, data_points):
    total_distance = 0
    for data_point in data_points:
        total_distance += geodesic((data_point[1], data_point[0]), (x[1], x[0])).km
    return total_distance

def exclude_outliers(adj_matrix, indices):
    sub_adj_matrix = adj_matrix[indices, :][:, indices]
    in_degrees = np.array(sub_adj_matrix.sum(axis=0)).flatten()
    highest_in_degree_index = np.argwhere(in_degrees)
    neighbors = set(np.where(sub_adj_matrix[:, highest_in_degree_index])[0]) | set(np.where(sub_adj_matrix[highest_in_degree_index, :])[1])
    print(f'neighbors: {neighbors}')
    indices = list(neighbors | set(highest_in_degree_index))
    print(indices)
    return indices


def text_standardization(text_candidates):
    text_candidates = text_candidates.drop(columns=['user_text', 'O'])
    text_candidates = text_candidates.dropna(axis=1, how='all')
    modes = text_candidates.mode()
    column_with_max_mode = modes.apply(lambda x: x.mode().max(), axis=0)
    # print(column_with_max_mode)
    print("elements:", column_with_max_mode.to_dict().values())
    all_elements = list(column_with_max_mode.to_dict().values())
    for elem in list(column_with_max_mode.to_dict().values()):
        elem_list = elem.split(', ')
        all_elements += elem_list
        all_elements.remove(elem)
    print('new elements:', all_elements)
    sorted_elements = sorted([elem for elem in all_elements if not pd.isna(elem)],
                             key=lambda x: int(x.split(':')[0]))
    print("sorted elements:", sorted_elements)
    sorted_elements = list(map(lambda x: x.split(':')[-1], sorted_elements))
    print("new sorted elements:", sorted_elements)
    print('-' * 12)
    # elements = sorted(column_with_max_mode.to_dict().values())
    # # print(elements)
    # for i in range(len(elements)):
    #     elements[i] = elements[i].split(':')[-1]
    # print(elements)
    standardized_text = ''.join(sorted_elements)
    return standardized_text

def geo_standardization(geo_candidates):
    init_point = np.mean(geo_candidates, axis=0)
    optimized_geo = minimize(objective_function, init_point, args=(geo_candidates),method='Nelder-Mead')
    return optimized_geo.x


def evaluate(args, cfg):
    start_epoch = 0
    cudnn.benchmark = True
    
    # Check if this is the master process (true if not distributed)
    is_master_proc = du_helper.is_master_proc(cfg.NUM_GPUS)

    # Cuda and current device
    cuda = torch.cuda.is_available()
    device = torch.cuda.current_device()

    if is_master_proc:
        create_output_dirs(cfg)

    t = datetime.now()
    output_path = os.path.join(cfg.OUTPUT_PATH, f'{args.iterative_cluster}_{t.month}-{t.day}-{t.hour}-{t.minute}/')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    logging.basicConfig(filename=os.path.join(output_path, "evaluate.log"), filemode="w", \
                        format="%(asctime)s %(name)s:%(levelname)s:%(message)s", \
                        datefmt="%Y-%M-%d %H:%M:%S", level=logging.DEBUG)

    # ======================== Similarity Network Setup ========================

    # base model for contrastive learning   
    base_net = base_model(cfg)
    triplet_net = TripletNet(base_net)
    cluster = cluster_algorithm(cfg)

    ## SyncBatchNorm
    if cfg.SYNC_BATCH_NORM:
        print('Converting BatchNorm*D to SyncBatchNorm!')
        triplet_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(triplet_net)

    n_parameters = sum([p.data.nelement() for p in triplet_net.parameters()])
    if(is_master_proc):
        print('Number of params: {}'.format(n_parameters))
    

    def DDP(model):
        # Transfer model to DDP
        model = model.cuda(device=device)
        if torch.cuda.device_count() > 1:
            #model = nn.DataParallel(model)
            model = torch.nn.parallel.DistributedDataParallel(module=model,
                    device_ids=[device],
                    #broadcast_buffers=False)
                    )
        return model

    # Load checkpoint if path exists

    # load_path = os.path.join(output_path, f"{cfg.MODEL.ARCH}/last_checkpoint.pth.tar")
    # print(load_path)
    # if args.checkpoint_path is not None and os.path.exists(load_path):
    if args.checkpoint_path is not None:
        chosen_epoch, triplet_net = load_checkpoint(triplet_net, args.checkpoint_path, is_master_proc)

    # if cuda:  # TODO:distributed training
    #     triplet_net = DDP(triplet_net)
    triplet_net = triplet_net.cuda(device=device)

    # Load pretrained backbone if path exists
    if args.pretrain_path is not None:
        triplet_net = load_pretrained_model(triplet_net, args.pretrain_path, is_master_proc)

    # ============================== Data Loaders ==============================

    if not os.path.exists(os.path.join(cfg.GLOBAL.DATA_DIR, args.data_path)):
        dataset_loader = Eleme(cfg, text_encoder='chinesebert', geo_encoder='gpsbert', save_path=None)
        dataset = dataset_loader.__make_dataset__()
    else:
        with open(os.path.join(cfg.GLOBAL.DATA_DIR, args.data_path), 'rb') as f:
            dataset = pickle.load(f)

    from copy import deepcopy
    text_description = dataset['text_description']
    text_embedding = torch.Tensor(dataset['text_embedding'])
    if isinstance(dataset['geo_description'][0], str):
        geo_description = list(map(literal_eval, list(dataset['geo_description'])))
    else:
        geo_description = list(dataset['geo_description'])
        # dataset['geo_description'] = geo_description
    geo_description = np.array(geo_description)
    if cfg.DATA.ENCODE_GEO:
        geo_embedding = torch.Tensor(dataset['geo_embedding'])
    else:
        geo_embedding = torch.Tensor(geo_description)
        geo_embedding = deepcopy(geo_embedding)
        geo_embedding[:, 0] = geo_embedding[:, 0] - 120
        geo_embedding[:, 1] = geo_embedding[:, 1] - 30
        geo_embedding = geo_embedding.repeat(1, cfg.DATA.D_GEO // geo_embedding.shape[-1])
    # text_embedding = normalize(text_embedding)
    # geo_embedding = normalize(geo_embedding)
    text_distance_matrix = torch.Tensor(dataset['text_distance_matrix'])
    print(f"====== text distance matrix ======")
    text_distance_matrix = normalize(text_distance_matrix)
    print(text_distance_matrix)

    geo_distance_matrix = torch.Tensor(dataset['geo_distance_matrix'])
    print(f"====== geo distance matrix ======")
    geo_distance_matrix = normalize(geo_distance_matrix)
    print(geo_distance_matrix)

    wifi_distance_matrix = torch.Tensor(dataset['wifi_distance_matrix'])
    print(f"====== wifi distance matrix ======")
    wifi_distance_matrix = torch.where(torch.isinf(wifi_distance_matrix), 1, wifi_distance_matrix)
    wifi_distance_matrix = normalize(wifi_distance_matrix)
    print(wifi_distance_matrix)

    true_labels = torch.Tensor(dataset['true_cluster_label'])

    data = torch.cat((text_embedding, geo_embedding), dim=-1)
    
    n_data = data.shape[0]
    print(data.shape)

    if cfg.DATASET.INIT_DIST_FUSION == 't+g+w+e':
        init_multimodal_distance = text_distance_matrix + geo_distance_matrix + wifi_distance_matrix
    elif cfg.DATASET.INIT_DIST_FUSION == 't+g*w+e':
        init_multimodal_distance = cfg.DATASET.FUSION_TEXT_W * text_distance_matrix + cfg.DATASET.FUSION_GEO_W * torch.mul(wifi_distance_matrix, geo_distance_matrix)
    else:
        logging.info(f'INIT_DIST_FUSION is defaultly set to: e')
        init_multimodal_distance = None
    
    
    # p, r, f_beta, threshold = evaluate_top_k(true_labels, data, cfg)
    # print(f'==> Metrics of retrieval:{p}, {r}, {f_beta}')
    # with open(os.path.join(output_path, 'retrieval_metrics.txt'), "a") as f:
    #     f.write(f'initial, {threshold}, {p}, {r}, {f_beta}\n')

    cluster_output_path = os.path.join(output_path, 'cluster_results.csv')
    result_path = os.path.join(output_path, 'result.csv')


    if cfg.ITERCLUSTER.METHOD == 'finch' or cfg.ITERCLUSTER.METHOD == 'advfinch':
        init_cluster_labels, n_init_cluster, _ = cluster.forward(data, distance=init_multimodal_distance)
        init_cluster_labels = init_cluster_labels[:, cfg.ITERCLUSTER.FINCH_PARTITION]
    elif cfg.ITERCLUSTER.METHOD in ['kmeans', 'dbscan', 'optics']:
        init_cluster_labels = cluster.fit(data).labels_
        init_cluster_labels = torch.tensor(init_cluster_labels)
        # print(init_cluster_labels)
        
    cluster_labels = pd.DataFrame({'initial': init_cluster_labels})
    ri, ari, nmi, ami = evaluate_clustering(true_labels, init_cluster_labels)
    print(f'==> Metrics of clustering:{ri}, {ari}, {nmi}, {ami}')
    with open(os.path.join(output_path, 'cluster_metrics.txt'), "a") as f:
        f.write(f'initial:, {ri}, {ari}, {nmi}, {ami}\n')


    triplet_net.eval()
    with torch.no_grad():
        data = data.to(device)
        updated_data = triplet_net.base_network.forward(data).detach().cpu()
    # p, r, f_beta, threshold = evaluate_top_k(true_labels, updated_data, cfg)
    # print(f'==> Metrics of retrieval:{p}, {r}, {f_beta}')
    # with open(os.path.join(output_path, 'retrieval_metrics.txt'), "a") as f:
    #     f.write(f'updated, {threshold}, {p}, {r}, {f_beta}\n')

    print('\n=> Clustering')
    start_time = time.time()

    if cfg.ITERCLUSTER.METHOD == 'finch' or cfg.ITERCLUSTER.METHOD == 'advfinch':
        updated_cluster_labels, n_updated_cluster, adj_matrix = cluster.forward(updated_data, distance=init_multimodal_distance)
        updated_cluster_labels = updated_cluster_labels[:, cfg.ITERCLUSTER.FINCH_PARTITION]
    elif cfg.ITERCLUSTER.METHOD in ['kmeans', 'dbscan', 'optics']:
        updated_cluster_labels = cluster.fit(updated_data).labels_
        updated_cluster_labels = torch.tensor(updated_cluster_labels)
    print('Time to cluster: {:.2f}s'.format(time.time()-start_time))

    cluster_labels.insert(cluster_labels.shape[1], f'updated', updated_cluster_labels)
    with open(cluster_output_path, "wb") as f:
        pickle.dump(cluster_labels, f)

    ri, ari, nmi, ami = evaluate_clustering(true_labels, updated_cluster_labels)
    print(f'==> Metrics of clustering:{ri}, {ari}, {nmi}, {ami}')
    with open(os.path.join(output_path, 'cluster_metrics.txt'), "a") as f:
        f.write(f'updated, {ri}, {ari}, {nmi}, {ami}\n')

    # Save cluster assignments corresponding to unshuffled order of dataset
    print('Saved cluster labels to', cluster_output_path)

    with open(os.path.join(cfg.GLOBAL.DATA_DIR, args.data_path).replace('.dat', '_text.dat'), 'rb') as f:
        text_elements = pickle.load(f)

    
    # print(adj_matrix)
    standardized_results = pd.DataFrame(columns=['raw_text', 'raw_geo', 'new_text', 'new_geo'])
    standardized_results['raw_text'] = text_description
    standardized_results['raw_geo'] = list(geo_description)
    for anchor_index in range(n_data):
        anchor_cluster_labels = updated_cluster_labels[anchor_index]
        labels = np.where(updated_cluster_labels == anchor_cluster_labels, 1.0, 0.0)
        same_cluster_indices = np.argwhere(labels == 1.0).flatten()
        n_candidates = same_cluster_indices.shape[0]
        # print(same_cluster_indices.shape[0], same_cluster_indices)
        if n_candidates > 5:
            same_cluster_indices = exclude_outlier(adj_matrix, same_cluster_indices)
            
        #     text_candidates = text_description[same_cluster_indices]
        #     geo_candidates = geo_description[same_cluster_indices]
        #     standardized_results.at[anchor_index, 'new_text'] = text_candidates
        #     standardized_results.at[anchor_index, 'new_geo'] = geo_candidates
        # else:
        geo_candidates = geo_description[same_cluster_indices]
        parsed_text_candidates = text_elements.iloc[same_cluster_indices]
        standardized_text = text_standardization(parsed_text_candidates)
        standardized_geo = geo_standardization(geo_candidates)
        # print(standardized_results.at[anchor_index, 'raw_text'], standardized_results.at[anchor_index, 'raw_geo'], standardized_text, standardized_geo)
        # print(parsed_text_candidates)
        standardized_results.at[anchor_index, 'new_text'] = standardized_text
        standardized_results.at[anchor_index, 'new_geo'] = standardized_geo
    
    with open(result_path, 'wb') as f:
        pickle.dump(standardized_results, f)
    print(standardized_results)


if __name__ == '__main__':

    random.seed(7)
    torch.manual_seed(7)
    np.random.seed(7)
    torch.cuda.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    print ('\n==> Parsing parameters:')
    args = arg_parser().parse_args()
    cfg = load_config(args)


    # Set shard_id to $SLURM_NODEID if running on compute canada
    shard_id = args.shard_id

    # Print node information
    print ('Total nodes:', args.num_shards)
    print ('Node id:', shard_id)

    # Set visible GPU devices and print gpu information
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    if torch.cuda.is_available():
        cfg.NUM_GPUS = torch.cuda.device_count()
        print("Using {} GPU(s) per node".format(cfg.NUM_GPUS))

    print(f'BATCH_SIZE is set to: {cfg.TRAIN.BATCH_SIZE}')
    print(f'NUM_WORKERS is set to: {cfg.TRAIN.NUM_DATA_WORKERS}')
    # Launch processes for all gpus
    print('\n==> Launching gpu processes...')
    evaluate(args, cfg)
    # du_helper.launch_processes(args, cfg, func=train, shard_id=shard_id,
    #     NUM_SHARDS=args.num_shards, ip_address_port=args.ip_address_port)