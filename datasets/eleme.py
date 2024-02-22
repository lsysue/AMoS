import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import pandas as pd
import pickle
import heapq

from typing import List

import torch
# from torch.utils.data import Dataset
from ast import literal_eval

from encoder import TextEncoder, GeoEncoder, MGeoEncoder
from datautils import data_process
from geopy.distance import geodesic
from datautils import data_process, levenshtein

from config.param_parser import arg_parser, load_config

args = arg_parser().parse_args()
cfg = load_config(args)


class Eleme(object):
    def __init__(self, cfg,
                 text_encoder = 'chinesebert', geo_encoder = 'gpsbert', 
                 data_path= None, save_path = None):
        
        self.device = cfg.GLOBAL.DEVICE
        self.save_path = save_path
        print(data_path, save_path)

        with open(data_path, 'rb') as f:
            raw_dataset = pickle.load(f)
        
        self.raw_dataset = raw_dataset.reset_index(drop=True)

        self.text_encoder = TextEncoder(cfg, finetuning=False, encoder=text_encoder)
        self.geo_encoder = GeoEncoder(cfg, finetuning=False, encoder=geo_encoder)
        self.multimodal_encoder = MGeoEncoder(cfg)


    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        anchor = self.raw_dataset.iloc[index]

    def __make_dataset__(self, eps=1e-6):
        # 分析重用已有dataset(.dat)
        old_dataset = None
        if os.path.exists(save_path):
            with open(self.save_path, 'rb') as f:
                old_dataset = pickle.load(f)
            print(old_dataset.keys())

        self.dataset = self.raw_dataset.copy()
        user_text_embedding_chinesebert_list = list()
        key_text_embedding_chinesebert_list = list()
        parsed_text_embedding_chinesebert_list = list()
        user_geo_embedding_gpsbert_list = list()
        key_geo_embedding_gpsbert_list = list()
        parsed_geo_embedding_gpsbert_list = list()

        user_text_embedding_mgeo_list = list()
        key_text_embedding_mgeo_list = list()
        parsed_text_embedding_mgeo_list = list()
        user_geo_embedding_mgeo_list = list()
        key_geo_embedding_mgeo_list = list()
        parsed_geo_embedding_mgeo_list = list()

        # user_text_description = list()
        # key_text_description = list()
        # parsed_text_description = list()
        # user_geo_description = list()
        # key_geo_description = list()
        # parsed_geo_description = list()

        user_text_distance_matrix = np.full(shape=(self.__len__(), self.__len__()), fill_value=np.inf)
        key_text_distance_matrix = np.full(shape=(self.__len__(), self.__len__()), fill_value=np.inf)
        parsed_text_distance_matrix = np.full(shape=(self.__len__(), self.__len__()), fill_value=np.inf)
        
        user_geo_distance_matrix = np.full(shape=(self.__len__(), self.__len__()), fill_value=np.inf)
        key_geo_distance_matrix = np.full(shape=(self.__len__(), self.__len__()), fill_value=np.inf)
        parsed_geo_distance_matrix = np.full(shape=(self.__len__(), self.__len__()), fill_value=np.inf)

        # wifi_distance_matrix = np.full(shape=(self.__len__(), self.__len__()), fill_value=np.inf)

        if old_dataset and 'text_embedding' in old_dataset.keys():
            user_text_embedding_chinesebert_list = old_dataset['text_embedding']
        if old_dataset and 'parsed_text_embedding' in old_dataset.keys():
            parsed_text_embedding_chinesebert_list = old_dataset['parsed_text_embedding']

        true_labels = np.array(list(self.raw_dataset['poi_id']))
        true_labels = np.where(np.isnan(true_labels), -1, true_labels)

        for i, this_data in self.raw_dataset.iterrows():
            
            if isinstance(this_data['parsed_geo'], str):
                this_data.at['parsed_geo'] = literal_eval(this_data['parsed_geo'])
            if isinstance(this_data['user_geo'], str):
                this_data.at['user_geo'] = literal_eval(this_data['user_geo'])
            if isinstance(this_data['key_geo'], str):
                this_data.at['key_geo',] =literal_eval(this_data['key_geo'])
            
            user_geo_word = str(round(this_data['user_geo'][0], cfg.DATA.GEO_PRECISION)).ljust(4 + cfg.DATA.GEO_PRECISION, '0') + '|' + str(round(this_data['user_geo'][1], cfg.DATA.GEO_PRECISION)).ljust(3 + cfg.DATA.GEO_PRECISION, '0')
            key_geo_word = str(round(this_data['key_geo'][0], cfg.DATA.GEO_PRECISION)).ljust(4 + cfg.DATA.GEO_PRECISION, '0') + '|' + str(round(this_data['key_geo'][1], cfg.DATA.GEO_PRECISION)).ljust(3 + cfg.DATA.GEO_PRECISION, '0')
            parsed_geo_word = str(round(this_data['parsed_geo'][0], cfg.DATA.GEO_PRECISION)).ljust(4 + cfg.DATA.GEO_PRECISION, '0') + '|' + str(round(this_data['parsed_geo'][1], cfg.DATA.GEO_PRECISION)).ljust(3 + cfg.DATA.GEO_PRECISION, '0')
            i_user_lon, i_user_lat = this_data['user_geo']
            i_key_lon, i_key_lat = this_data['key_geo']
            i_parsed_lon, i_parsed_lat = this_data['parsed_geo']

            if not old_dataset:
                user_text_embedding_chinesebert_list.append(self.text_encoder.forward(this_data['user_text']).detach().cpu().numpy())
                parsed_text_embedding_chinesebert_list.append(self.text_encoder.forward(this_data['parsed_text']).detach().cpu().numpy())
            
            key_text_embedding_chinesebert_list.append(self.text_encoder.forward(this_data['key_text']).detach().cpu().numpy())

            user_geo_embedding_gpsbert_list.append(self.geo_encoder.forward(user_geo_word).detach().cpu().numpy())
            key_geo_embedding_gpsbert_list.append(self.geo_encoder.forward(key_geo_word).detach().cpu().numpy())
            parsed_geo_embedding_gpsbert_list.append(self.geo_encoder.forward(parsed_geo_word).detach().cpu().numpy())

            user_text_embedding_mgeo, user_geo_embedding_mgeo = self.multimodal_encoder.forward(this_data['user_text'], this_data['user_geo'])
            key_text_embedding_mgeo, key_geo_embedding_mgeo = self.multimodal_encoder.forward(this_data['key_text'], this_data['key_geo'])
            parsed_text_embedding_mgeo, parsed_geo_embedding_mgeo = self.multimodal_encoder.forward(this_data['parsed_text'], this_data['parsed_geo'])

            user_text_embedding_mgeo_list.append(user_text_embedding_mgeo.detach().cpu().numpy())
            user_geo_embedding_mgeo_list.append(user_geo_embedding_mgeo.detach().cpu().numpy())
            key_text_embedding_mgeo_list.append(key_text_embedding_mgeo.detach().cpu().numpy())
            key_geo_embedding_mgeo_list.append(key_geo_embedding_mgeo.detach().cpu().numpy())
            parsed_text_embedding_mgeo_list.append(parsed_text_embedding_mgeo.detach().cpu().numpy())
            parsed_geo_embedding_mgeo_list.append(parsed_geo_embedding_mgeo.detach().cpu().numpy())


            # this_wifi = '|'.join(this_data['wifi_fp'].split('|')[0:2])
            # this_wifi_scanlist = ['|'.join(wifi.split('|')[0:2]) for wifi in this_data['wifi_scanlist']]
            
            for j, j_data in self.raw_dataset[i:].iterrows():
                if j == i:
                    continue
                
                if isinstance(j_data['user_geo'], str):
                    j_data.at['user_geo'] = literal_eval(j_data['user_geo'])
                if isinstance(j_data['key_geo'], str):
                    j_data.at['key_geo'] = literal_eval(j_data['key_geo'])
                if isinstance(j_data['parsed_geo'], str):
                    j_data.at['parsed_geo'] = literal_eval(j_data['parsed_geo'])
                j_user_lon, j_user_lat = j_data['user_geo']
                j_key_lon, j_key_lat = j_data['key_geo']
                j_parsed_lon, j_parsed_lat = j_data['parsed_geo']

                # text_distance = levenshtein(this_data['user_text'], j_data['user_text'])
                user_text_distance = levenshtein(this_data['user_text'], j_data['user_text'])
                user_text_distance_matrix[j, i] = user_text_distance_matrix[i, j] = user_text_distance
                key_text_distance = levenshtein(this_data['key_text'], j_data['key_text'])
                key_text_distance_matrix[j, i] = key_text_distance_matrix[i, j] = key_text_distance
                parsed_text_distance = levenshtein(this_data['parsed_text'], j_data['parsed_text'])
                parsed_text_distance_matrix[j, i] = parsed_text_distance_matrix[i, j] = parsed_text_distance
                

                user_geo_distance = geodesic([i_user_lat, i_user_lon], [j_user_lat, j_user_lon]).km
                key_geo_distance = geodesic([i_key_lat, i_key_lon], [j_key_lat, j_key_lon]).km
                parsed_geo_distance = geodesic([i_parsed_lat, i_parsed_lon], [j_parsed_lat, j_parsed_lon]).km
                user_geo_distance_matrix[j, i] = user_geo_distance_matrix[i, j] = user_geo_distance
                key_geo_distance_matrix[j, i] = key_geo_distance_matrix[i, j] = key_geo_distance
                parsed_geo_distance_matrix[j, i] = parsed_geo_distance_matrix[i, j] = parsed_geo_distance


                # j_wifi = '|'.join(j_data['wifi_fp'].split('|')[0:2])
                # j_wifi_scanlist = ['|'.join(wifi.split('|')[0:2]) for wifi in j_data['wifi_scanlist']]

                # if j_wifi in this_wifi_scanlist:
                #     if wifi_distance_matrix[i, j] != np.inf:
                #         print("i, j:", wifi_distance_matrix[i, j])
                #     wifi_distance_matrix[i, j] = 1
                # if this_wifi in j_wifi_scanlist:
                #     if wifi_distance_matrix[j, i] != np.inf:
                #         print("j, i:", wifi_distance_matrix[j, i])
                #     wifi_distance_matrix[j, i] = 1
                # n_intersect_wifi = len(set(this_wifi_scanlist).intersection(set(j_wifi_scanlist)))
                # n_union_wifi = len(set(this_wifi_scanlist).union(set(j_wifi_scanlist)))
                # wifi_distance_matrix[i, j] = wifi_distance_matrix[i, j] - float(n_intersect_wifi) / (n_union_wifi + eps)
                # wifi_distance_matrix[j, i] = wifi_distance_matrix[j, i] - float(n_intersect_wifi) / (n_union_wifi + eps)


        # print('-'*12)
        # print(text_distance_matrix)
        # print('-'*12)
        # print(geo_distance_matrix)
        # print('-'*12)
        # print(wifi_distance_matrix)
        print('='*24)
        self.dataset['user_text_embedding_chinesebert'] = user_text_embedding_chinesebert_list
        self.dataset['key_text_embedding_chinesebert'] = key_text_embedding_chinesebert_list
        self.dataset['parsed_text_embedding_chinesebert'] = parsed_text_embedding_chinesebert_list

        self.dataset['user_geo_embedding_gpsbert'] = user_geo_embedding_gpsbert_list
        self.dataset['key_geo_embedding_gpsbert'] = key_geo_embedding_gpsbert_list
        self.dataset['parsed_geo_embedding_gpsbert'] = parsed_geo_embedding_gpsbert_list

        self.dataset['user_text_embedding_mgeo'] = user_text_embedding_mgeo_list
        self.dataset['key_text_embedding_mgeo'] = key_text_embedding_mgeo_list
        self.dataset['parsed_text_embedding_mgeo'] = parsed_text_embedding_mgeo_list

        self.dataset['user_geo_embedding_mgeo'] = user_geo_embedding_mgeo_list
        self.dataset['key_geo_embedding_mgeo'] = key_geo_embedding_mgeo_list
        self.dataset['parsed_geo_embedding_mgeo'] = parsed_geo_embedding_mgeo_list
        

        if self.save_path:
            with open(self.save_path, 'wb') as f:
                pickle.dump(self.dataset, f)

        return self.dataset

if __name__== "__main__" :
    args = arg_parser().parse_args()
    cfg = load_config(args)
    data_path = os.path.join(cfg.GLOBAL.DATA_DIR, 'eleme_raw/37841_1d_0216_1590.csv')
    save_path = data_path.replace('.csv', '.dat')
    print(save_path)
    dataset_loader = Eleme(cfg, data_path=data_path, save_path=save_path)
    dataset = dataset_loader.__make_dataset__()
        # print(data.__getitem__(i))