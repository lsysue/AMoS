import os
import time
import random
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn import TripletMarginWithDistanceLoss
from torch.utils.data import DataLoader

from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import (pair_confusion_matrix,
                            rand_score, adjusted_rand_score,
                            normalized_mutual_info_score, 
                            adjusted_mutual_info_score)
from sklearn.metrics import (precision_score, recall_score, f1_score)

from scipy.spatial.distance import pdist, cdist, squareform

from tqdm import tqdm
from ast import literal_eval
from datetime import datetime

from datasets.eleme import Eleme
from datasets.quadruplet import Quadruplets, Graphset
from datasets.datautils import normalize, top_k
from clustering.finch import FINCH, AdvFINCH, NearestNeighborClustering
from models.modelutils import (load_pretrained_model, save_checkpoint, load_checkpoint, create_output_dirs)
from models.modelutils import AverageMeter
from models.basenet import BaseFC, ThreeLayerFC, FourLayerFC
from models.transformer.bert import Bert
from models.quadrupletnet import QuadrupletNet
from models.graphnet import GraphNet
from loss.infonce import InfoNCE, QuadraInfoNCE, TEST
from loss.triplet import TripletLoss, QuadrupletLoss
import misc.distributed_helper as du_helper
from config.param_parser import load_config, arg_parser

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")
warnings.filterwarnings('ignore')

# Select the appropriate model with the specified cfg parameters
def cluster_algorithm(cfg):
    assert cfg.ITERCLUSTER.METHOD in ['finch', 'advfinch', 'nncluster', 'kmeans', 'dbscan', 'optics']
    if cfg.ITERCLUSTER.METHOD == 'finch':
        model = FINCH(initial_rank=None, required_n_cluster=None, exit_n_cluster=2, metric=cfg.ITERCLUSTER.DIST_METRIC, ensure_early_exit=True, verbose=True, use_ann_above_samples=70000)
    elif cfg.ITERCLUSTER.METHOD == 'advfinch':
        model = AdvFINCH(initial_rank=None, required_n_cluster=None, exit_n_cluster=2, metric=cfg.ITERCLUSTER.DIST_METRIC, enable_hierarchy=False, ensure_early_exit=True, verbose=True, use_ann_above_samples=70000)
    elif cfg.ITERCLUSTER.METHOD == 'nncluster':
        model = NearestNeighborClustering(initial_rank=None, metric=cfg.ITERCLUSTER.DIST_METRIC, verbose=True, use_ann_above_samples=70000)
    elif cfg.ITERCLUSTER.METHOD == 'kmeans':
        model = KMeans(n_clusters=cfg.ITERCLUSTER.K)
    elif cfg.ITERCLUSTER.METHOD == 'dbscan':
        model = DBSCAN(eps=cfg.ITERCLUSTER.EPS, min_samples=cfg.ITERCLUSTER.MIN_SAMPLES, metric=cfg.ITERCLUSTER.DIST_METRIC)
    elif cfg.ITERCLUSTER.METHOD == 'optics':
        model = OPTICS(min_samples=cfg.ITERCLUSTER.MIN_SAMPLES, metric=cfg.ITERCLUSTER.DIST_METRIC)
    return model

def base_model(cfg):
    assert cfg.MODEL.BASE_ARCH in ['basenet', 'basenet-3', 'basenet-4', 'transformer']
    if cfg.MODEL.BASE_ARCH == 'basenet':
        model = BaseFC(device=cfg.GLOBAL.DEVICE, d_inputs=cfg.DATA.D_TEXT + cfg.DATA.D_GEO, dropout=cfg.MODEL.DROPOUT)
    elif cfg.MODEL.BASE_ARCH == 'basenet-3':
        model = ThreeLayerFC(device=cfg.GLOBAL.DEVICE, d_inputs=cfg.DATA.D_TEXT + cfg.DATA.D_GEO, dropout=cfg.MODEL.DROPOUT)
    elif cfg.MODEL.BASE_ARCH == 'basenet-4':
        model = FourLayerFC(device=cfg.GLOBAL.DEVICE, d_inputs=cfg.DATA.D_TEXT + cfg.DATA.D_GEO, dropout=cfg.MODEL.DROPOUT)
    elif cfg.MODEL.BASE_ARCH == 'transformer':
        model = Bert(hidden=cfg.DATA.D_TEXT + cfg.DATA.D_GEO, n_layers=2, attn_heads=2)
    return model


def train_epoch(train_loader, model, 
                        criterion, optimizer, 
                        epoch, cfg, device, output_path, is_master_proc=True):
    loss_meter = AverageMeter()
    world_size = du_helper.get_world_size()
    # switching to training mode
    model.train()

    # Training loop
    start = time.time()
    conf_mat_sum = np.zeros([4])
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        # anchor_input, positive_inputs, weak_positive_inputs, negative_inputs = inputs
        anchor_target, positive_targets, weak_positive_targets, negative_targets = targets
        # anchor_index, positive_indices, weak_positive_indices, negative_indices = indices
        batch_size = torch.tensor(inputs[0].size(0)).to(device)

        targets = torch.cat((positive_targets, negative_targets), dim=1).to(torch.long)
        # anchor_output, positive_outputs, weak_positive_outputs, negative_outputs = model(inputs)
        outputs = model(inputs)
        # print(outputs.shape, targets.shape)
        outputs = outputs.reshape(-1, outputs.shape[-1])
        targets = targets.reshape(-1).to(torch.long)
        # print(outputs.shape, targets.shape)

        loss = criterion(outputs, targets)

        # Compute gradient and perform optimization step
        loss.backward()
        optimizer.step()

        pred_labels = torch.argmax(outputs, dim=-1).cpu()
        targets = targets.cpu()
        conf_mat = confusion_matrix(targets, pred_labels, labels=range(2)).flatten()
        conf_mat_sum += conf_mat

        # Average loss across all gpu processes # TODO: distributed training
        # if cfg.NUM_GPUS > 1:
        #     [loss] = du_helper.all_reduce([loss], avg=True)
        #     [batch_size_world] = du_helper.all_reduce([batch_size], avg=False)
        # else:
        #     batch_size_world = batch_size
        batch_size_world = batch_size

        batch_size_world = batch_size_world.item()

        # Update running loss
        loss_meter.update(loss.item(), batch_size_world)

        # Log
        if is_master_proc and ((batch_idx + 1) * world_size) % cfg.TRAIN.LOG_INTERVAL == 0:
            logging.critical(f"Train Epoch: {epoch} [{loss_meter.count}/{len(train_loader.dataset)} | {100. * (loss_meter.count / len(train_loader.dataset)):.1f}%]\t"
                      f"Loss: {loss_meter.val} ({loss_meter.avg}) \t")

    tn, fp, fn, tp = conf_mat_sum.tolist()
    if tp == 0:
        prec, recall = 0.0, 0.0
        f1 = 0.0
    else:
        prec = tp / (tp + fp) 
        recall = tp / (tp + fn)
        f1 = 2 * prec * recall / (prec + recall)

    if (is_master_proc):
        logging.critical(f"\nTrain set: Average loss: {loss_meter.avg}\n")
        logging.critical(f"epoch:{epoch} runtime:{(time.time()-start)/3600}")
        with open(os.path.join(output_path, 'train_loss.txt'), 'a') as f:
            f.write(f"epoch:{epoch}, runtime:{round((time.time()-start)/3600,2)}, {loss_meter.avg}\n")
        with open(os.path.join(output_path, 'graphnet_trainresult.txt'), 'a') as f:
            f.write(f"{epoch}, {prec}, {recall}, {f1}\n")


def val_epoch(val_loader, model, 
                criterion, epoch, cfg, device, output_path, is_master_proc=True):
    loss_meter = AverageMeter()
    world_size = du_helper.get_world_size()
    # switching to training mode
    model.eval()

    # Training loop
    start = time.time()
    conf_mat_sum = np.zeros([4])
    for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader)):
        # anchor_input, positive_inputs, weak_positive_inputs, negative_inputs = inputs
        anchor_target, positive_targets, weak_positive_targets, negative_targets = targets
        # anchor_index, positive_indices, weak_positive_indices, negative_indices = indices
        batch_size = torch.tensor(inputs[0].size(0)).to(device)

        targets = torch.cat((positive_targets, negative_targets), dim=1).to(torch.long)
        # anchor_output, positive_outputs, weak_positive_outputs, negative_outputs = model(inputs)
        with torch.no_grad():
            outputs = model(inputs)
        # print(outputs.shape, targets.shape)
        outputs = outputs.reshape(-1, outputs.shape[-1])
        targets = targets.reshape(-1).to(torch.long).cpu()
        # print(outputs.shape, targets.shape)

        pred_labels = torch.argmax(outputs, dim=-1).cpu()
        conf_mat = confusion_matrix(targets, pred_labels, labels=range(2)).flatten()
        conf_mat_sum += conf_mat

    tn, fp, fn, tp = conf_mat_sum.tolist()
    if tp == 0:
        prec, recall = 0.0, 0.0
        f1 = 0.0
    else:
        prec = tp / (tp + fp) 
        recall = tp / (tp + fn)
        f1 = 2 * prec * recall / (prec + recall)

    if (is_master_proc):
        with open(os.path.join(output_path, 'graphnet_valresult.txt'), 'a') as f:
            f.write(f"{epoch}, {prec}, {recall}, {f1}\n")


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
    return ri, ari, nmi, ami, p, r, f_beta

def evaluate_top_k(true_labels, pred_embedding, cfg):
    print(pred_embedding.shape)
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
    


def train(args, cfg):
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
    
    logging.basicConfig(filename=os.path.join(output_path, "train.log"), filemode="w", \
                        format="%(asctime)s %(name)s:%(levelname)s:%(message)s", \
                        datefmt="%Y-%M-%d %H:%M:%S", level=logging.DEBUG)


    # Print training parameters
    if is_master_proc:
        logging.critical(f'OUTPUT_PATH is set to: {cfg.OUTPUT_PATH}')
        logging.critical(f'Training on dataset: {os.path.join(cfg.GLOBAL.DATA_DIR, args.data_path)}')

        logging.critical(f'Is geo data encoded?: {cfg.DATA.ENCODE_GEO}')

        logging.critical(f'Label source of sampling: {cfg.DATASET.LABEL_SOURCE}')
        logging.critical(f'DIST_METRIC of sampling: {cfg.DATASET.DIST_METRIC}')
        logging.critical(f'DIST_METRIC of clustering: {cfg.ITERCLUSTER.DIST_METRIC}')
        logging.critical(f'INIT_DIST_FUSION of sampling: {cfg.DATASET.INIT_DIST_FUSION}')
        logging.critical(f'ITER_DIST_FUSION of sampling: {cfg.DATASET.ITER_DIST_FUSION}')
        logging.critical(f'Random positive sampling? {cfg.DATASET.RANDOM_POSITIVE_SAMPLING}')
        logging.critical(f'Random weak positive sampling? {cfg.DATASET.RANDOM_WEAK_POSITIVE_SAMPLING}')
        logging.critical(f'Random negative sampling? {cfg.DATASET.RANDOM_NEGATIVE_SAMPLING}')
        logging.critical(f'Number/Prob of POSITIVE_SAMPLING: {cfg.DATASET.POSITIVE_SAMPLING}')
        logging.critical(f'Number/Prob of WEAK_POSITIVE_SAMPLING: {cfg.DATASET.WEAK_POSITIVE_SAMPLING}')
        logging.critical(f'Number/Prob of NEGATIVE_SAMPLING: {cfg.DATASET.NEGATIVE_SAMPLING}')
        
        logging.critical(f'Base model for contrastive learning: {cfg.MODEL.MAIN_ARCH}-{cfg.MODEL.BASE_ARCH}')

        logging.critical(f'CLUSTER METHOD is set to: {cfg.ITERCLUSTER.METHOD}')
        logging.critical(f'Iteratively clustering?: {args.iterative_cluster}')
        if args.iterative_cluster:
            logging.critical(f'ITERCLUSTER.INTERVAL is set to: {cfg.ITERCLUSTER.INTERVAL}')

        logging.critical(f'Criterion is set to: {cfg.LOSS.TYPE}')
        logging.critical(f'OPTIMIZER is set to: {cfg.OPTIM.OPTIMIZER}')
        logging.critical(f'Learning rate is set to {cfg.OPTIM.LR}')

        if cfg.OPTIM.OPTIMIZER == 'adam':
            logging.info(f'Weight decay = {cfg.OPTIM.WD}')
        elif cfg.OPTIM.OPTIMIZER == 'sgd':
            logging.info(f'Momentum = {cfg.OPTIM.MOMENTUM}')

    # ======================== Network Setup ========================

    # base model for contrastive learning   
    cluster = cluster_algorithm(cfg)
    base_net = base_model(cfg)
    if cfg.MODEL.MAIN_ARCH == 'quadrupletnet':
        main_net = QuadrupletNet(base_net)
    elif cfg.MODEL.MAIN_ARCH == 'graphnet':
        main_net = GraphNet(ggnn_n_nodes=cfg.MODEL.GGNN_N_NODES, ggnn_n_edge_types=cfg.MODEL.GGNN_N_EDGE_TYPES, ggnn_d_annotation=cfg.DATA.D_TEXT + cfg.DATA.D_GEO, ggnn_d_state=cfg.MODEL.GGNN_D_STATE, ggnn_n_steps=cfg.MODEL.GGNN_N_STEPS, init_weight = True)
    ## SyncBatchNorm
    if cfg.SYNC_BATCH_NORM:
        print('Converting BatchNorm*D to SyncBatchNorm!')
        main_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(main_net)

    n_parameters = sum([p.data.nelement() for p in main_net.parameters()])
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
        start_epoch, main_net = load_checkpoint(main_net, args.checkpoint_path, is_master_proc)

    # if cuda:  # TODO:distributed training
    #     main_net = DDP(main_net)
    main_net = main_net.cuda(device=device)

    # Load pretrained backbone if path exists
    if args.pretrain_path is not None:
        main_net = load_pretrained_model(main_net, args.pretrain_path, is_master_proc)

    # ======================== Loss and Optimizer Setup ========================
    if(is_master_proc):
        print('\n==> Setting criterion...')

    if cfg.LOSS.TYPE == 'infonce':
        criterion = InfoNCE(temperature=cfg.LOSS.TEMPERATURE).to(device)
    elif cfg.LOSS.TYPE == 'quadrainfonce':
        criterion = QuadraInfoNCE(temperature=cfg.LOSS.TEMPERATURE, weak_temperature=cfg.LOSS.WEAK_TEMPERATURE)
    elif cfg.LOSS.TYPE == 'triplet':
        criterion = TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(), margin=cfg.LOSS.MARGIN)
        # criterion = TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=cfg.LOSS.MARGIN)
    elif cfg.LOSS.TYPE == 'quadruplet':
        # criterion = QuadrupletLoss(distance_function=nn.PairwiseDistance(), margin=cfg.LOSS.MARGIN)
        criterion = QuadrupletLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=cfg.LOSS.MARGIN)
    elif cfg.LOSS.TYPE == 'ce':
        # criterion = TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(), margin=cfg.LOSS.MARGIN)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        # criterion = MSELoss()
    else:
        assert False, 'Loss Type:{} not recognized'.format(cfg.LOSS.TYPE)

    if cfg.OPTIM.OPTIMIZER == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, main_net.parameters()), lr=cfg.OPTIM.LR, weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, main_net.parameters()), lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.MOMENTUM, weight_decay=cfg.OPTIM.WD)
    else:
        print(f"{cfg.OPTIM.OPTIMIZER} optimizer not supported")


    # ============================== Data Loaders ==============================

    if not os.path.exists(os.path.join(cfg.GLOBAL.DATA_DIR, args.data_path)):
        dataset_loader = Eleme(cfg, text_encoder='chinesebert', geo_encoder='gpsbert', save_path=None)
        dataset = dataset_loader.__make_dataset__()
    else:
        with open(os.path.join(cfg.GLOBAL.DATA_DIR, args.data_path), 'rb') as f:
            dataset = pickle.load(f)

    from copy import deepcopy

    text_embedding = torch.Tensor(dataset['parsed_text_embedding'])
    if isinstance(dataset['geo_description'][0], str):
        geo_description = list(map(literal_eval, list(dataset['geo_description'])))
    else:
        geo_description = list(dataset['geo_description'])
        # dataset['geo_description'] = geo_description
    if cfg.DATA.ENCODE_GEO:
        geo_embedding = torch.Tensor(dataset['geo_embedding'])
    else:
        geo_embedding = torch.Tensor(geo_description)
        geo_embedding = deepcopy(geo_embedding)
        geo_embedding[:, 0] = geo_embedding[:,0] - 120
        geo_embedding[:, 1] = geo_embedding[:, 1] - 30
        geo_embedding = geo_embedding.repeat(1, cfg.DATA.D_GEO // geo_embedding.shape[-1])
    # text_embedding = normalize(text_embedding)
    # geo_embedding = normalize(geo_embedding)
    text_distance_matrix = torch.Tensor(dataset['text_distance_matrix'])
    print(f"====== text distance matrix ======")
    text_distance_matrix = normalize(text_distance_matrix)
    # print(text_distance_matrix)

    geo_distance_matrix = torch.Tensor(dataset['geo_distance_matrix'])
    print(f"====== geo distance matrix ======")
    geo_distance_matrix = normalize(geo_distance_matrix)
    # print(geo_distance_matrix)

    wifi_distance_matrix = torch.Tensor(dataset['wifi_distance_matrix'])
    print(f"====== wifi distance matrix ======")
    wifi_distance_matrix = torch.where(torch.isinf(wifi_distance_matrix), 1, wifi_distance_matrix)
    wifi_distance_matrix = normalize(wifi_distance_matrix)
    # print(wifi_distance_matrix)

    true_labels = torch.Tensor(dataset['true_cluster_label'])

    init_data = torch.cat((text_embedding, geo_embedding), dim=-1)

    if cfg.DATASET.INIT_DIST_FUSION == 't+g+w+e':
        init_multimodal_distance = text_distance_matrix + geo_distance_matrix + wifi_distance_matrix
    elif cfg.DATASET.INIT_DIST_FUSION == 't+g*w+e':
        init_multimodal_distance = cfg.DATASET.FUSION_TEXT_W * text_distance_matrix + cfg.DATASET.FUSION_GEO_W * torch.mul(wifi_distance_matrix, geo_distance_matrix)
    else:
        logging.info(f'INIT_DIST_FUSION is defaultly set to: e')
        init_multimodal_distance = None
    
    if cfg.DATASET.ITER_DIST_FUSION == 't+g+w+e':
        iter_multimodal_distance = text_distance_matrix + geo_distance_matrix + wifi_distance_matrix
    elif cfg.DATASET.ITER_DIST_FUSION == 't+g*w+e':
        iter_multimodal_distance = cfg.DATASET.FUSION_TEXT_W * text_distance_matrix + cfg.DATASET.FUSION_GEO_W * torch.mul(wifi_distance_matrix, geo_distance_matrix)
    else:
        logging.info(f'ITER_DIST_FUSION is defaultly set to: e')
        iter_multimodal_distance = None
    
    
    # p, r, f_beta, threshold = evaluate_top_k(true_labels, init_data, cfg)
    # print(f'==> Metrics of retrieval:{p}, {r}, {f_beta}')
    # with open(os.path.join(output_path, 'retrieval_metrics.txt'), "a") as f:
    #     f.write(f'initial, {threshold}, {p}, {r}, {f_beta}\n')

    if args.start_epoch != None:
        start_epoch = args.start_epoch
    
    cluster_output_path = os.path.join(output_path, 'cluster_results.csv')


    if cfg.DATASET.LABEL_SOURCE == 'cluster':
        if cfg.ITERCLUSTER.METHOD in ['finch', 'advfinch']:
            init_cluster_labels, n_init_cluster, _ = cluster.forward(init_data, distance=init_multimodal_distance)
            init_cluster_labels = init_cluster_labels[:, cfg.ITERCLUSTER.FINCH_PARTITION]
        elif cfg.ITERCLUSTER.METHOD == 'nncluster':
            init_cluster_labels, n_init_cluster, _ = cluster.forward(init_data, text_dist=text_distance_matrix, geo_dist=geo_distance_matrix, wifi_dist=wifi_distance_matrix)
            init_cluster_labels = init_cluster_labels[:, cfg.ITERCLUSTER.FINCH_PARTITION]
        elif cfg.ITERCLUSTER.METHOD in ['kmeans', 'dbscan', 'optics']:
            init_cluster_labels = cluster.fit(init_data).labels_
            init_cluster_labels = torch.tensor(init_cluster_labels)
            # print(init_cluster_labels)
        
        cluster_labels = pd.DataFrame({'initial': init_cluster_labels})
        with open(cluster_output_path, "wb") as f:
            pickle.dump(cluster_labels, f)
        ri, ari, nmi, ami, p, r, f_beta = evaluate_clustering(true_labels, init_cluster_labels)
        print(f'==> Metrics of clustering:{ri}, {ari}, {nmi}, {ami}\n==> {p}, {r}, {f_beta}')
        with open(os.path.join(output_path, 'cluster_metrics.txt'), "a") as f:
            f.write(f'initial:, {ri}, {ari}, {nmi}, {ami}, {p}, {r}, {f_beta}\n')
    
        print('Saved cluster labels to', cluster_output_path)

        train_dataset = Quadruplets(data=init_data, text_distance=text_distance_matrix, geo_distance=geo_distance_matrix, multimodal_distance=init_multimodal_distance, distance_metric=cfg.DATASET.DIST_METRIC, cluster_labels=init_cluster_labels, true_labels=true_labels, random_positive_sampling=cfg.DATASET.RANDOM_POSITIVE_SAMPLING, random_weak_positive_sampling=cfg.DATASET.RANDOM_WEAK_POSITIVE_SAMPLING, random_negative_sampling=cfg.DATASET.RANDOM_NEGATIVE_SAMPLING, positive_sampling=cfg.DATASET.POSITIVE_SAMPLING, negative_sampling=cfg.DATASET.NEGATIVE_SAMPLING, weak_positive_sampling=cfg.DATASET.WEAK_POSITIVE_SAMPLING, device=cfg.GLOBAL.DEVICE)
        # train_dataset = Quadruplets(data=init_data, text_distance=text_distance_matrix, geo_distance=geo_distance_matrix, multimodal_distance=iter_multimodal_distance, distance_metric=cfg.DATASET.DIST_METRIC, cluster_labels=init_cluster_labels, true_labels=true_labels, random_positive_sampling=cfg.DATASET.RANDOM_POSITIVE_SAMPLING, random_weak_positive_sampling=cfg.DATASET.RANDOM_WEAK_POSITIVE_SAMPLING, random_negative_sampling=cfg.DATASET.RANDOM_NEGATIVE_SAMPLING, positive_sampling=cfg.DATASET.POSITIVE_SAMPLING, negative_sampling=cfg.DATASET.NEGATIVE_SAMPLING, weak_positive_sampling=cfg.DATASET.WEAK_POSITIVE_SAMPLING, device=cfg.GLOBAL.DEVICE)
        # train_dataset = Graphset(data=init_data, text_distance=text_distance_matrix, geo_distance=geo_distance_matrix, n_candidates = cfg.MODEL.GGNN_N_NODES - 1, cluster_labels=init_cluster_labels, true_labels=true_labels, device=cfg.GLOBAL.DEVICE)

    elif cfg.DATASET.LABEL_SOURCE == 'true':
        train_dataset = Quadruplets(data=init_data, text_distance=text_distance_matrix, geo_distance=geo_distance_matrix, multimodal_distance=init_multimodal_distance, distance_metric=cfg.DATASET.DIST_METRIC, cluster_labels=true_labels, true_labels=true_labels, random_positive_sampling=cfg.DATASET.RANDOM_POSITIVE_SAMPLING, random_weak_positive_sampling=cfg.DATASET.RANDOM_WEAK_POSITIVE_SAMPLING, random_negative_sampling=cfg.DATASET.RANDOM_NEGATIVE_SAMPLING, positive_sampling=cfg.DATASET.POSITIVE_SAMPLING, negative_sampling=cfg.DATASET.NEGATIVE_SAMPLING, weak_positive_sampling=cfg.DATASET.WEAK_POSITIVE_SAMPLING, device=cfg.GLOBAL.DEVICE)
        # train_dataset = Quadruplets(data=init_data, text_distance=text_distance_matrix, geo_distance=geo_distance_matrix, multimodal_distance=iter_multimodal_distance, distance_metric=cfg.DATASET.DIST_METRIC, cluster_labels=true_labels, true_labels=true_labels, random_positive_sampling=cfg.DATASET.RANDOM_POSITIVE_SAMPLING, random_weak_positive_sampling=cfg.DATASET.RANDOM_WEAK_POSITIVE_SAMPLING, random_negative_sampling=cfg.DATASET.RANDOM_NEGATIVE_SAMPLING, positive_sampling=cfg.DATASET.POSITIVE_SAMPLING, negative_sampling=cfg.DATASET.NEGATIVE_SAMPLING, weak_positive_sampling=cfg.DATASET.WEAK_POSITIVE_SAMPLING, device=cfg.GLOBAL.DEVICE)
        # train_dataset = Graphset(data=init_data, text_distance=text_distance_matrix, geo_distance=geo_distance_matrix, n_candidates = cfg.MODEL.GGNN_N_NODES - 1, cluster_labels=true_labels, true_labels=true_labels, device=cfg.GLOBAL.DEVICE)
    val_dataset = Quadruplets(data=init_data, text_distance=text_distance_matrix, geo_distance=geo_distance_matrix, multimodal_distance=init_multimodal_distance, distance_metric=cfg.DATASET.DIST_METRIC, cluster_labels=true_labels, true_labels=true_labels, random_positive_sampling=cfg.DATASET.RANDOM_POSITIVE_SAMPLING, random_weak_positive_sampling=cfg.DATASET.RANDOM_WEAK_POSITIVE_SAMPLING, random_negative_sampling=cfg.DATASET.RANDOM_NEGATIVE_SAMPLING, positive_sampling=cfg.DATASET.POSITIVE_SAMPLING, negative_sampling=cfg.DATASET.NEGATIVE_SAMPLING, weak_positive_sampling=cfg.DATASET.WEAK_POSITIVE_SAMPLING, device=cfg.GLOBAL.DEVICE)
        
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)
     # Setting is_master_proc to false when loading single video data loaders 
    # deliberately to not re-print data loader information

    # ============================= Training loop ==============================

    for epoch in range(start_epoch + 1, cfg.TRAIN.EPOCHS + 1):
        if (is_master_proc):
            print (f'\nEpoch {epoch}/{cfg.TRAIN.EPOCHS}')

        # ====================== Training for this epoch =======================

        # Train
        if (is_master_proc):
            print(f'==> training with Triplet Loss with criterion:{criterion}')
            train_epoch(train_dataloader, main_net, criterion, optimizer, epoch, cfg, device, output_path, is_master_proc)

        # ============================= Evaluation =============================

        if is_master_proc:
            # Cluster

            # Update embeding
            iter_data = val_epoch(val_dataloader, main_net, criterion, epoch, cfg, device, output_path, is_master_proc)

            # iter_data = init_data.clone()
            # iter_data = iter_data.detach().view(-1, (cfg.DATA.D_TEXT + cfg.DATA.D_GEO)).to(cfg.GLOBAL.DEVICE)
            # # iter_data_text, iter_data_geo = torch.chunk(iter_data, 2, dim=-1)

            # logging.debug(f"EPOCH:{epoch} | curr data before model: {iter_data[0]}")
            # main_net.eval()
            # with torch.no_grad():
            #     # iter_data_text = model.forward(iter_data_text).detach().cpu()
            #     # iter_data_geo = model.forward(iter_data_geo).detach().cpu()
            #     iter_data = main_net.base_network.forward(iter_data).detach().cpu()
            #     # iter_data = normalize(iter_data)

            # p, r, f_beta, threshold = evaluate_top_k(true_labels, iter_data, cfg)
            # print(f'==> Metrics of retrieval:{p}, {r}, {f_beta}')
            # with open(os.path.join(output_path, 'retrieval_metrics.txt'), "a") as f:
            #     f.write(f'epoch:{epoch}, {threshold}, {p}, {r}, {f_beta}\n')

            # print('\n=> Clustering')
            # start_time = time.time()

            # if cfg.ITERCLUSTER.METHOD in ['finch', 'advfinch']:
            #     iter_cluster_labels, n_iter_cluster, _ = cluster.forward(iter_data, distance=iter_multimodal_distance)
            #     iter_cluster_labels = iter_cluster_labels[:, cfg.ITERCLUSTER.FINCH_PARTITION]
            # elif cfg.ITERCLUSTER.METHOD == 'nncluster':
            #     iter_cluster_labels, n_iter_cluster, _ = cluster.forward(iter_data)
            #     iter_cluster_labels = iter_cluster_labels[:, cfg.ITERCLUSTER.FINCH_PARTITION]
            # elif cfg.ITERCLUSTER.METHOD in ['kmeans', 'dbscan', 'optics']:
            #     iter_cluster_labels = cluster.fit(iter_data).labels_
            #     iter_cluster_labels = torch.tensor(iter_cluster_labels)
            # print('Time to cluster: {:.2f}s'.format(time.time()-start_time))

            # if os.path.exists(cluster_output_path):
            #     with open(cluster_output_path, "rb") as f:
            #         cluster_labels = pickle.load(f)
            #     cluster_labels.insert(cluster_labels.shape[1], f'epoch: {epoch}', iter_cluster_labels)
            # else:
            #     cluster_labels = pd.DataFrame({f'epoch: {epoch}': iter_cluster_labels})
            # with open(cluster_output_path, "wb") as f:
            #     pickle.dump(cluster_labels, f)

            # ri, ari, nmi, ami, p, r, f_beta = evaluate_clustering(true_labels, iter_cluster_labels)
            # print(f'==> Metrics of clustering:{ri}, {ari}, {nmi}, {ami}\n==> {p}, {r}, {f_beta}')
            # with open(os.path.join(output_path, 'cluster_metrics.txt'), "a") as f:
            #     f.write(f'epoch:{epoch}, {ri}, {ari}, {nmi}, {ami}, {p}, {r}, {f_beta}\n')

            # # Save cluster assignments corresponding to unshuffled order of dataset
            # print('Saved cluster labels to', cluster_output_path)
            
            # Make processes wait for master process to finish with clustering
            # if cfg.NUM_GPUS > 1:
            #     torch.distributed.barrier()


        # Save checkpoint # TODO:
        # if torch.cuda.device_count() > 1:
        #     state_dict = main_net.module.state_dict()
        # else:
        #     state_dict = main_net.state_dict()
        state_dict = main_net.state_dict()

        if epoch == cfg.TRAIN.EPOCHS:
            filename = f'last_checkpoint.pth.tar'
            save_checkpoint({
                'epoch': epoch,
                'state_dict': state_dict,
                # 'best_prec1': best_acc,
            }, args.iterative_cluster, cfg.MODEL.MAIN_ARCH, output_path, is_master_proc, filename)

        if epoch % 10 == 0:
            filename = f'checkpoint_{epoch}.pth.tar'
            save_checkpoint({
                'epoch': epoch,
                'state_dict': state_dict,
                # 'best_prec1': best_acc,
            }, args.iterative_cluster, cfg.MODEL.MAIN_ARCH, output_path, is_master_proc, filename)

        
                # =================== Compute embeddings and cluster ===================

        if args.iterative_cluster and epoch % cfg.ITERCLUSTER.INTERVAL == 0:
            # Rebuild train_loader with new cluster assignments as pseudolabels
            if(is_master_proc):
                print('\n==> Rebuilding training data loader (quadrupletnet)...')
            train_dataset = Quadruplets(init_data, text_distance=text_distance_matrix, geo_distance=geo_distance_matrix, multimodal_distance=init_multimodal_distance, distance_metric=cfg.DATASET.DIST_METRIC, cluster_labels=iter_cluster_labels, true_labels=true_labels, random_positive_sampling=cfg.DATASET.RANDOM_POSITIVE_SAMPLING, random_weak_positive_sampling=cfg.DATASET.RANDOM_WEAK_POSITIVE_SAMPLING, random_negative_sampling=cfg.DATASET.RANDOM_NEGATIVE_SAMPLING, positive_sampling=cfg.DATASET.POSITIVE_SAMPLING, negative_sampling=cfg.DATASET.NEGATIVE_SAMPLING, weak_positive_sampling=cfg.DATASET.WEAK_POSITIVE_SAMPLING, device=cfg.GLOBAL.DEVICE)
            
            train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)


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
    train(args, cfg)
    # du_helper.launch_processes(args, cfg, func=train, shard_id=shard_id,
    #     NUM_SHARDS=args.num_shards, ip_address_port=args.ip_address_port)