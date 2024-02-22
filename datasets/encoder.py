import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import BertTokenizer, RobertaTokenizer, DistilBertTokenizer
from transformers import BertModel, RobertaModel, DistilBertModel

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from models.chinesebert.tokenizer import ChinBertTokenizer
from models.chinesebert.modeling_glycebert import GlyceBertModel
from models.mgeo.mgeo_dataset import GisUtt
from models.mgeo.tokenization_bert import BertTokenizer as MGeoBertTokenizer
from models.mgeo.model_mm import MGeo
from config.param_parser import arg_parser, load_config

from typing import Optional, Union, List

class MGeoEncoder(object):
    def __init__(self, cfg):
        super().__init__()

        self.device = cfg.GLOBAL.DEVICE

        self.tokenizer = MGeoBertTokenizer.from_pretrained('/datapool/workspace/3002lsy/huggingface_models/bert-base-chinese')
        self.model = MGeo(config={'gis_bert_config': cfg.MODEL.MGEO_CONFIG}, text_encoder='/datapool/workspace/3002lsy/huggingface_models/bert-base-chinese', tokenizer=self.tokenizer)

        checkpoint = torch.load(cfg.MODEL.MGEO_MODEL, map_location='cpu')
        state_dict = checkpoint['model']

        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.', '')
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]
            if 'doc_proj' in key:
                del state_dict[key]
        msg = self.model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % cfg.MODEL.MGEO_MODEL)
        print(msg)  

        self.model = self.model.to(self.device)

    def forward(self, text, gis):
        text_input = self.tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(self.device)
        gis_input_ids, gis_token_type_ids, gis_rel_type_ids, gis_absolute_position_ids, gis_relative_position_ids = (['[]'], ['[]'], ['[]'], ['[]'], ['[]'])
        gis_input = GisUtt(0, 1, self.device)
        gis_input.update(gis_input_ids, gis_token_type_ids, gis_rel_type_ids, gis_absolute_position_ids, gis_relative_position_ids)
        text_embedding, gis_embedding = self.model(text_input, gis_input)
        return text_embedding, gis_embedding

class TextEncoder(object):
    def __init__(self, cfg,
                 finetuning=True,
                 encoder='bert'):
        super().__init__()

        self.device = cfg.GLOBAL.DEVICE
        # print(self.device)

        self.finetuning = finetuning
        if encoder == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
        elif encoder == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.model = RobertaModel.from_pretrained('roberta-base')
        elif encoder == 'distilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        elif encoder == 'chinesebert':
            self.tokenizer = ChinBertTokenizer(cfg.MODEL.TEXT_ENCODER_MODEL)
            self.model = GlyceBertModel.from_pretrained(cfg.MODEL.TEXT_ENCODER_MODEL)
            self.is_pinyin = True
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
        
        self.model = self.model.to(self.device)

    def forward(self, x: Union[str, List[str]]) -> Tensor:
        
        if self.is_pinyin:
            input_ids, pinyin_ids = self.tokenizer.encode(x)
            input_ids = torch.LongTensor(input_ids).view(1, -1).to(self.device)
            pinyin_ids = pinyin_ids.view(1, -1, 8).to(self.device)
        else:
            input_ids = self.tokenizer.encode(x)
            input_ids = input_ids.view(1, -1).to(self.device)
            pinyin_ids = None
        

        if self.finetuning:
            self.model.train()
            self.train()
            if self.is_pinyin:
                output = self.model.forward(input_ids, pinyin_ids)
                pooled_output = output[0].mean(dim=1)
            else:
                attention_mask = torch.tensor(torch.where(input_ids != 0, 1, 0)).to(self.device)
                output = self.model(input_ids, attention_mask=attention_mask)
                pooled_output = output[0][:, :, :]
            

        else:
            self.model.eval()
            with torch.no_grad():
                if self.is_pinyin:
                    output = self.model.forward(input_ids, pinyin_ids)
                    pooled_output = output[0].mean(dim=1)
                else:
                    output = self.model(input_ids, attention_mask=attention_mask)
                    # pooled_output = output[0][:, :, :]
                    pooled_output = output[0].mean(dim=1)

        text_embedding = pooled_output
        # print(f"text_embedding: {text_embedding.shape}")

        return text_embedding

class GeoEncoder(object):
    def __init__(self, cfg,
                 finetuning=True,
                 encoder='bert'):
        super().__init__()

        self.device = cfg.GLOBAL.DEVICE

        self.finetuning = finetuning
        if encoder == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
        elif encoder == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.model = RobertaModel.from_pretrained('roberta-base')
        elif encoder == 'distilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        elif encoder == 'gpsbert':
            self.tokenizer = BertTokenizer(vocab_file=os.path.join(cfg.MODEL.GEO_ENCODER_MODEL, 'vocab.txt'), do_basic_tokenize=False)
            self.model = BertModel.from_pretrained(cfg.MODEL.GEO_ENCODER_MODEL)
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model = self.model.to(self.device)

    def forward(self, x: Union[str, List[str]]) -> Tensor:

        input_ids = self.tokenizer.encode(x)
        input_ids = torch.LongTensor(input_ids).view(1, -1).to(self.device)
        attention_mask = torch.tensor(torch.where(input_ids != 0, 1, 0)).to(self.device)

        if self.finetuning:
            self.model.train()
            self.train()
            output = self.model(input_ids, attention_mask=attention_mask)
            # pooled_output = output[0][:, :, :]
            pooled_output = output[0].mean(dim=1)

        else:
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_ids, attention_mask=attention_mask)
                # pooled_output = output[0][:, :, :]
                pooled_output = output[0].mean(dim=1)

        geo_embedding = pooled_output
        # print(f"geo_embedding: {geo_embedding.shape}")

        return geo_embedding


class LSTMFE(nn.Module):
    def __init__(self,
                 device='cpu',
                 input_size=100,
                 hidden_size=64,
                 bidirectional=True,
                 max_seq_len=16,
                 dropout=0.1):
        super().__init__()

        self.device = device
        self.hidden_size = hidden_size

        if bidirectional:
            self.b = 2
        else:
            self.b = 1

        self.drop = nn.Dropout(dropout)
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional)
        self.linear1 = nn.Linear(self.b*self.hidden_size * max_seq_len, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, 2)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.to(self.device)

        output, (_, _) = self.lstm(x)

        output = self.linear2(self.drop(self.relu(self.linear1(output.reshape(output.shape[0], -1)))))

        return F.log_softmax(output, dim=1)


if __name__ == '__main__':
    sent1 = '西子国际步行街7幢1-59'
    gps1 = '121.43286|29.33115'
    sent2 = '西子步行街8幢159'
    gps2 = '121.43275|29.33120'
    sent3 = '西子步行街7幢1-39'
    gps3 = '121.43279|29.33123'
    args = arg_parser().parse_args()
    cfg = load_config(args)
    mgeo_encoder = MGeoEncoder(cfg=cfg)
    # text_encoder = TextEncoder(cfg=cfg, device='cpu', finetuning=False, encoder='chinesebert')
    # gps_encoder = GeoEncoder(cfg=cfg, device='cpu', finetuning=False, encoder='chinesebert')
    # text_embedding = text_encoder.forward(sentence)
    # geo_embedding = gps_encoder.forward(sentence)
    txt_embed1, gis_embed1= mgeo_encoder.forward(sent1, gps1)
    txt_embed2, gis_embed2 = mgeo_encoder.forward(sent2, gps2)
    txt_embed3, gis_embed3 = mgeo_encoder.forward(sent3, gps3)
    from scipy.spatial.distance import cdist, pdist
    txt_embed1 = txt_embed1.detach().cpu().squeeze(0).numpy()
    txt_embed2 = txt_embed2.detach().cpu().squeeze(0).numpy()
    txt_embed3 = txt_embed3.detach().cpu().squeeze(0).numpy()
    gis_embed1 = gis_embed1.detach().cpu().squeeze(0).numpy()
    gis_embed2 = gis_embed2.detach().cpu().squeeze(0).numpy()
    gis_embed3 = gis_embed3.detach().cpu().squeeze(0).numpy()

    print(txt_embed1.shape, txt_embed2.shape, gis_embed1.shape, gis_embed2.shape)
    txt_dist = cdist(txt_embed1, txt_embed2, metric='cosine')
    gis_dist = cdist(gis_embed1, gis_embed2, metric='cosine')
    print(txt_dist.shape, gis_dist.shape)
    print(np.mean(txt_dist), np.max(txt_dist))
    txt_dist = cdist(txt_embed1, txt_embed3, metric='cosine')
    gis_dist = cdist(gis_embed1, gis_embed3, metric='cosine')
    print(txt_dist.shape, gis_dist.shape)
    print(np.mean(txt_dist), np.max(txt_dist))
