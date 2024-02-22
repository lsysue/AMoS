import json
import os
import random

from torch.utils.data import Dataset, IterableDataset
import torch.distributed as dist
import torch
from tqdm import tqdm

from .dataset_utils import pre_caption

import dill

class GisUtt:
    def __init__(self, pad_token_id, cls_token_id, device):
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.input_ids = None
        self.attention_mask = None
        self.token_type_ids = None
        self.rel_type_ids = None
        self.absolute_position_ids = None
        self.relative_position_ids = None
        self.device = device
        self.max_length = 32

    def update(self, gis_input_ids, gis_token_type_ids, gis_rel_type_ids, gis_absolute_position_ids, gis_relative_position_ids):
        gis_input_ids = [[self.cls_token_id] + json.loads(f) for f in gis_input_ids]
        gis_token_type_ids = [[self.pad_token_id] + json.loads(f) for f in gis_token_type_ids]
        gis_rel_type_ids = [[self.pad_token_id] + json.loads(f) for f in gis_rel_type_ids]
        gis_absolute_position_ids = [[[self.pad_token_id] * 4] + json.loads(f) for f in gis_absolute_position_ids]
        gis_relative_position_ids = [[[self.pad_token_id] * 4] + json.loads(f) for f in gis_relative_position_ids]

        gis_input_ids = [f[:self.max_length] for f in gis_input_ids]
        gis_token_type_ids = [f[:self.max_length] for f in gis_token_type_ids]
        gis_rel_type_ids = [f[:self.max_length] for f in gis_rel_type_ids]
        gis_absolute_position_ids = [f[:self.max_length] for f in gis_absolute_position_ids]
        gis_relative_position_ids = [f[:self.max_length] for f in gis_relative_position_ids]

        max_length = max([len(item) for item in gis_input_ids])
        self.input_ids = torch.tensor(
                    [f + [self.pad_token_id] * (max_length - len(f)) for f in gis_input_ids], dtype=torch.long
                            ).to(self.device)
        self.attention_mask = torch.tensor(
                    [[1] * len(f) + [0] * (max_length - len(f)) for f in gis_input_ids], dtype=torch.long
                            ).to(self.device)
        self.token_type_ids = torch.tensor(
                    [f + [self.pad_token_id] * (max_length - len(f)) for f in gis_token_type_ids], dtype=torch.long
                            ).to(self.device)
        self.rel_type_ids = torch.tensor(
                    [f + [self.pad_token_id] * (max_length - len(f)) for f in gis_rel_type_ids], dtype=torch.long
                            ).to(self.device)

        self.absolute_position_ids = torch.tensor(
                    [f + [[self.pad_token_id] * 4] * (max_length - len(f)) for f in gis_absolute_position_ids], dtype=torch.long
                            ).to(self.device)
        self.relative_position_ids = torch.tensor(
                    [f + [[self.pad_token_id] * 4] * (max_length - len(f)) for f in gis_relative_position_ids], dtype=torch.long
                            ).to(self.device)