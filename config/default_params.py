"""Configs."""
import os
from fvcore.common.config import CfgNode

# Config definition
_C = CfgNode()

# -----------------------------------------------------------------------------
# Global options
# -----------------------------------------------------------------------------
_C.GLOBAL = CfgNode()
_C.GLOBAL.ROOT_DIR = '/datapool/workspace/3002lsy/repos/Eleme'
_C.GLOBAL.DATA_DIR = '/datapool/workspace/3002lsy/repos/Eleme/data'
_C.GLOBAL.DEVICE = 'cuda:0'
# -----------------------------------------------------------------------------
# Training options
# -----------------------------------------------------------------------------
_C.TRAIN = CfgNode()
_C.TRAIN.EPOCHS = 100
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.NUM_DATA_WORKERS = 4
_C.TRAIN.LOG_INTERVAL = 5 #for print statements
_C.TRAIN.OUTPUT_PATH = os.path.join(_C.GLOBAL.ROOT_DIR, 'checkpoints/')
_C.TRAIN.EVAL_BATCH_SIZE = False

# -----------------------------------------------------------------------------
# Evaluation options
# -----------------------------------------------------------------------------
_C.EVAL = CfgNode()
_C.EVAL.THRES_OR_TOPK = 0
_C.EVAL.THRESHOLD = 1
_C.EVAL.TOPK = 1
# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()
_C.MODEL.TEXT_ENCODER_MODEL = os.path.join(_C.GLOBAL.ROOT_DIR, 'models/chinesebert/chinbert-base/')
_C.MODEL.GEO_ENCODER_MODEL = os.path.join(_C.GLOBAL.ROOT_DIR, 'models/gpsbert/gpsbert-base/')
_C.MODEL.MGEO_CONFIG = os.path.join(_C.GLOBAL.ROOT_DIR, 'models/mgeo/gis_config.json')
_C.MODEL.MGEO_MODEL = os.path.join(_C.GLOBAL.ROOT_DIR, 'models/mgeo/checkpoint_09.pth')
# _C.MODEL.ENCODER = { 'bert': 'bert-base-uncased', 
#                 'distilbert': 'distilbert-base-uncased', 
#                 'roberta': 'roberta-base',
#                 'chinesebert': os.path.join(_C.GLOBAL.CURR_DIR, 'models/chinesebert/chinbert-base/'),
#                 'gpsbert': os.path.join(_C.GLOBAL.CURR_DIR, 'models/gpsbert/gpsbert-base/')}
_C.MODEL.BASE_ARCH = "basenet"
# _C.MODEL.ARCH = "transformers"
_C.MODEL.MAIN_ARCH = "graphnet"
_C.MODEL.GGNN_N_NODES = 20
_C.MODEL.GGNN_N_EDGE_TYPES = 2
_C.MODEL.GGNN_D_STATE = 1024
_C.MODEL.GGNN_N_STEPS = 2
_C.MODEL.DROPOUT = 0.2

# -----------------------------------------------------------------------------
# Dataset options
# -----------------------------------------------------------------------------
_C.DATASET = CfgNode()
_C.DATASET.PATH = os.path.join(_C.GLOBAL.DATA_DIR, '37841_1d_3336.dat')
_C.DATASET.VERSION = 10

_C.DATASET.LABEL_SOURCE = 'cluster'
_C.DATASET.DIST_METRIC = 'cosine'
_C.DATASET.INIT_DIST_FUSION = 't+g*w+e'
_C.DATASET.ITER_DIST_FUSION = 'e'
_C.DATASET.FUSION_TEXT_W = 1.0
_C.DATASET.FUSION_GEO_W = 1.0   

_C.DATASET.RANDOM_POSITIVE_SAMPLING = False
_C.DATASET.RANDOM_WEAK_POSITIVE_SAMPLING = True
_C.DATASET.RANDOM_NEGATIVE_SAMPLING = True
_C.DATASET.POSITIVE_SAMPLING = 1.0
_C.DATASET.WEAK_POSITIVE_SAMPLING = 0.01
_C.DATASET.NEGATIVE_SAMPLING = 1.0


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()
_C.DATA.INPUT_CHANNEL_NUM = 2
_C.DATA.GEO_PRECISION = 4
_C.DATA.D_TEXT = 768
_C.DATA.D_GEO = 768
_C.DATA.ENCODE_GEO = True


# -----------------------------------------------------------------------------
# Loss Options
# -----------------------------------------------------------------------------
_C.LOSS = CfgNode()
_C.LOSS.TYPE = 'infonce'
# for infonce
_C.LOSS.TEMPERATURE = 0.1
_C.LOSS.WEAK_TEMPERATURE = 1
# for tripletloss
_C.LOSS.MARGIN = 1.0

# -----------------------------------------------------------------------------
# Optimizer options
# -----------------------------------------------------------------------------
_C.OPTIM = CfgNode()
_C.OPTIM.OPTIMIZER = 'Adam'
_C.OPTIM.WD = 0.00001
_C.OPTIM.LR = 0.001
_C.OPTIM.MOMENTUM = 0.5
_C.OPTIM.SCHEDULE = []

# -----------------------------------------------------------------------------
# Iterative clustering options
# -----------------------------------------------------------------------------
_C.ITERCLUSTER = CfgNode()
#_C.ITERCLUSTER.METHOD = 'spherical_kmeans'
# _C.ITERCLUSTER.METHOD = 'kmeans'
_C.ITERCLUSTER.METHOD = 'finch'
_C.ITERCLUSTER.DIST_METRIC = 'cosine'
_C.ITERCLUSTER.INTERVAL = 5
# FINCH
_C.ITERCLUSTER.FINCH_PARTITION = 0
# KMEANS
_C.ITERCLUSTER.K = 1000
# DBSCAN
_C.ITERCLUSTER.EPS = 0.5
_C.ITERCLUSTER.MIN_SAMPLES = 2
# OPTICS
_C.ITERCLUSTER.MAX_EPS = 0.5
# -----------------------------------------------------------------------------
# Misc options
# -----------------------------------------------------------------------------
_C.NUM_GPUS = 1
_C.OUTPUT_PATH = "./checkpoints"
_C.SYNC_BATCH_NORM = False


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


# Check assertions for the cfg parameters and return the cfg
def _assert_and_infer_cfg(cfg):
    #assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0
    #assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0

    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
