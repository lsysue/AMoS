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

from encoder import TextEncoder, GeoEncoder
from datautils import data_process
from geopy.distance import geodesic
from datautils import data_process, levenshtein

from config.param_parser import arg_parser, load_config

args = arg_parser().parse_args()
cfg = load_config(args)


class Eleme(object):
    def __init__(self, cfg, 
                 text_encoder = 'chinesebert', geo_encoder = 'gpsbert', save_path = None):
        
        self.device = cfg.GLOBAL.DEVICE
        data_path = cfg.DATASET.PATH
        self.save_path = save_path

        if 'database' in os.path.dirname(data_path):
            raw_dataset = data_process(cfg, data_path)
        elif 'data' in os.path.dirname(data_path):
            # self.labeled, self.unlabeled = self.load_data(data_path)
            # raw_dataset = pd.read_csv(data_path)
            with open(data_path, 'rb') as f:
                raw_dataset = pickle.load(f)
        
        self.raw_dataset = raw_dataset.reset_index(drop=True)

        self.text_encoder = TextEncoder(cfg, finetuning=False, encoder=text_encoder)
        self.geo_encoder = GeoEncoder(cfg, finetuning=False, encoder=geo_encoder)


    def __len__(self):
        return len(self.raw_dataset)
    
    def __ulen__(self):
        return len(self.raw_unique_dataset)

    def __getitem__(self, index):
        anchor = self.raw_dataset.iloc[index]

    def __make_dataset__(self, eps=1e-6):
        # if os.path.exists(self.save_path):
        #     with open(self.save_path, 'rb') as f:
        #         self.dataset = pickle.load(f)
        #     parsed_text_embedding = list()
        #     for i, this_data in self.raw_dataset.iterrows():
        #         parsed_text_embedding.append(self.text_encoder.forward(this_data['parsed_text']).detach().cpu())
        #     self.dataset['parsed_text_description'] = np.array(list(self.raw_dataset['parsed_text']))
        #     self.dataset['parsed_text_embedding'] = np.array(torch.cat(parsed_text_embedding, 0))
        #     print(self.dataset['parsed_text_embedding'].shape)
        #     with open(self.save_path, 'wb') as f:
        #         pickle.dump(self.dataset, f)
        #     return self.dataset

        self.dataset = dict()
        text_embedding = list()
        parsed_text_embedding = list()
        geo_embedding = list()
        geo_description = list()
        cnt_off_site_orders = 0

        text_distmat = np.full(shape=(self.__len__(), self.__len__()), fill_value=np.inf)
        geo_distmat = np.full(shape=(self.__len__(), self.__len__()), fill_value=np.inf)
        wifi_distmat = np.full(shape=(self.__len__(), self.__len__()), fill_value=np.inf)

        true_labels = np.array(list(self.raw_dataset['poi_id']))
        true_labels = np.where(np.isnan(true_labels), -1, true_labels)

        for i, this_data in self.raw_dataset.iterrows():
            text_embedding.append(self.text_encoder.forward(this_data['user_text']).detach().cpu())
            parsed_text_embedding.append(self.text_encoder.forward(this_data['parsed_text']).detach().cpu())
            
            if isinstance(this_data['user_geo'], str):
                this_data.at['user_geo'] = literal_eval(this_data['user_geo'])
                # self.raw_dataset.loc[i, 'user_geo'] = literal_eval(this_data['user_geo'])
            if isinstance(this_data['key_geo'], str):
                this_data.at['key_geo',] =literal_eval(this_data['key_geo'])
                # self.raw_dataset.loc[i, 'key_geo'] = literal_eval(this_data['key_geo'])
            
            if this_data['is_off_site'] == 0:
                geo_description.append(this_data['user_geo'])
                this_geo_word = str(round(this_data['user_geo'][0], cfg.DATA.GEO_PRECISION)).ljust(4 + cfg.DATA.GEO_PRECISION, '0') + '|' + str(round(this_data['user_geo'][1], cfg.DATA.GEO_PRECISION)).ljust(3 + cfg.DATA.GEO_PRECISION, '0')
                ilon, ilat = this_data['user_geo']
            else:
                # print(f'off site orders')
                cnt_off_site_orders += 1
                geo_description.append(this_data['key_geo'])
                this_geo_word = str(round(this_data['key_geo'][0], cfg.DATA.GEO_PRECISION)).ljust(4 + cfg.DATA.GEO_PRECISION, '0') + '|' + str(round(this_data['key_geo'][1], cfg.DATA.GEO_PRECISION)).ljust(3 + cfg.DATA.GEO_PRECISION, '0')
                ilon, ilat = this_data['key_geo']

            geo_embedding.append(self.geo_encoder.forward(this_geo_word).detach().cpu())
            
            this_wifi = '|'.join(this_data['wifi_fp'].split('|')[0:2])
            this_wifi_scanlist = ['|'.join(wifi.split('|')[0:2]) for wifi in this_data['wifi_scanlist']]
            
            for j, j_data in self.raw_dataset[i:].iterrows():
                if j == i:
                    continue
                
                if isinstance(j_data['user_geo'], str):
                    j_data.at['user_geo'] = literal_eval(j_data['user_geo'])
                    # self.raw_dataset.loc[j, 'user_geo'] = literal_eval(j_data['user_geo'])
                if isinstance(j_data['key_geo'], str):
                    j_data.at['key_geo',] =literal_eval(j_data['key_geo'])
                    # self.raw_dataset.loc[j, 'key_geo'] = literal_eval(j_data['key_geo'])
                
                # text_distance = levenshtein(this_data['user_text'], j_data['user_text'])
                text_distance = levenshtein(this_data['parsed_text'], j_data['parsed_text'])
                text_distmat[j, i] = text_distmat[i, j] = text_distance
                
                if j_data['is_off_site'] == 0:
                    jlon, jlat = j_data['user_geo']
                else:
                    jlon, jlat = j_data['key_geo']
                geo_distance = geodesic([ilat, ilon], [jlat, jlon]).km
                geo_distmat[j, i] = geo_distmat[i, j] = geo_distance


                j_wifi = '|'.join(j_data['wifi_fp'].split('|')[0:2])
                j_wifi_scanlist = ['|'.join(wifi.split('|')[0:2]) for wifi in j_data['wifi_scanlist']]

                if j_wifi in this_wifi_scanlist:
                    if wifi_distmat[i, j] != np.inf:
                        print("i, j:", wifi_distmat[i, j])
                    wifi_distmat[i, j] = 1
                if this_wifi in j_wifi_scanlist:
                    if wifi_distmat[j, i] != np.inf:
                        print("j, i:", wifi_distmat[j, i])
                    wifi_distmat[j, i] = 1
                n_intersect_wifi = len(set(this_wifi_scanlist).intersection(set(j_wifi_scanlist)))
                n_union_wifi = len(set(this_wifi_scanlist).union(set(j_wifi_scanlist)))
                wifi_distmat[i, j] = wifi_distmat[i, j] - float(n_intersect_wifi) / (n_union_wifi + eps)
                wifi_distmat[j, i] = wifi_distmat[j, i] - float(n_intersect_wifi) / (n_union_wifi + eps)


        # print('-'*12)
        # print(text_distmat)
        # print('-'*12)
        # print(geo_distmat)
        # print('-'*12)
        # print(wifi_distmat)
        print(f"Number of off site orders: {cnt_off_site_orders}")
        print('='*24)
        text_embedding = torch.cat(text_embedding, 0)
        parsed_text_embedding = torch.cat(parsed_text_embedding, 0)
        geo_embedding = torch.cat(geo_embedding, 0)
        print(text_embedding.shape)
        print(parsed_text_embedding)
        print(geo_embedding.shape)

        self.dataset['text_description'] = np.array(list(self.raw_dataset['user_text']))
        self.dataset['parsed_text_description'] = np.array(list(self.raw_dataset['parsed_text']))
        self.dataset['geo_description'] = np.array(geo_description)

        self.dataset['text_embedding'] = np.array(text_embedding)
        self.dataset['parsed_text_embedding'] = np.array(parsed_text_embedding)
        self.dataset['geo_embedding'] = np.array(geo_embedding)

        self.dataset['text_distance_matrix'] = text_distmat
        self.dataset['geo_distance_matrix'] = geo_distmat
        self.dataset['wifi_distance_matrix'] = wifi_distmat
        self.dataset['true_cluster_label'] = true_labels

        if self.save_path:
            with open(self.save_path, 'wb') as f:
                pickle.dump(self.dataset, f)

        return self.dataset
                
    
    def __make_unique_dataset__(self, eps=1e-6):

        self.raw_unique_dataset = self.raw_dataset.copy()
        self.unique_dataset = dict()
        text_embedding = list()
        parsed_text_embedding = list()
        geo_embedding = list()
        geo_description = list()
        cnt_off_site_orders = 0

        for i, this_data in self.raw_dataset.iterrows():
            
            if this_data['is_off_site'] == 0:
                geo_description.append(str(this_data['user_geo']))
            else:
                cnt_off_site_orders += 1
                geo_description.append(str(this_data['key_geo']))

        self.raw_unique_dataset.insert(self.raw_unique_dataset.columns.get_loc('user_text'), 'geo', geo_description)
        # print(self.raw_unique_dataset.columns)
        self.raw_unique_dataset = self.raw_unique_dataset.drop_duplicates(subset=['geo', 'user_text'], keep='first')
        self.raw_unique_dataset = self.raw_unique_dataset.reset_index(drop=True)


        # if os.path.exists(self.save_path):
        #     with open(self.save_path, 'rb') as f:
        #         self.dataset = pickle.load(f)
        #     parsed_text_embedding = list()
        #     for i, this_data in self.raw_unique_dataset.iterrows():
        #         parsed_text_embedding.append(self.text_encoder.forward(this_data['parsed_text']).detach().cpu())
        #     self.dataset['parsed_text_description'] = np.array(list(self.raw_unique_dataset['parsed_text']))
        #     self.dataset['parsed_text_embedding'] = np.array(torch.cat(parsed_text_embedding, 0))
        #     print(self.dataset['parsed_text_embedding'].shape)
        #     with open(self.save_path, 'wb') as f:
        #         pickle.dump(self.dataset, f)
        #     return self.dataset

        true_labels = np.array(list(self.raw_unique_dataset['poi_id']))
        true_labels = np.where(np.isnan(true_labels), -1, true_labels)

        text_distmat = np.full(shape=(self.__ulen__(), self.__ulen__()), fill_value=np.inf)
        geo_distmat = np.full(shape=(self.__ulen__(), self.__ulen__()), fill_value=np.inf)
        wifi_distmat = np.full(shape=(self.__ulen__(), self.__ulen__()), fill_value=np.inf)

        # print('-'*12)
        # print(text_distmat.shape)
        # print('-'*12)
        # print(geo_distmat.shape)
        # print('-'*12)
        # print(wifi_distmat.shape)

        for i, this_data in self.raw_unique_dataset.iterrows():
            text_embedding.append(self.text_encoder.forward(this_data['user_text']).detach().cpu())
            parsed_text_embedding.append(self.text_encoder.forward(this_data['parsed_text']).detach().cpu())
            
            if isinstance(this_data['geo'], str):
                this_data.at['geo'] = literal_eval(this_data['geo'])
            
            this_geo_word = str(round(this_data['geo'][0], cfg.DATA.GEO_PRECISION)).ljust(4 + cfg.DATA.GEO_PRECISION, '0') + '|' + str(round(this_data['geo'][1], cfg.DATA.GEO_PRECISION)).ljust(3 + cfg.DATA.GEO_PRECISION, '0')
            ilon, ilat = this_data['geo']

            geo_embedding.append(self.geo_encoder.forward(this_geo_word).detach().cpu())
            
            this_wifi = '|'.join(this_data['wifi_fp'].split('|')[0:2])
            this_wifi_scanlist = ['|'.join(wifi.split('|')[0:2]) for wifi in this_data['wifi_scanlist']]
            
            for j, j_data in self.raw_unique_dataset[i:].iterrows():
                if j == i:
                    continue
                
                if isinstance(j_data['geo'], str):
                    j_data.at['geo'] = literal_eval(j_data['geo'])
                
                # text_distance = levenshtein(this_data['user_text'], j_data['user_text'])
                text_distance = levenshtein(this_data['parsed_text'], j_data['parsed_text'])
                text_distmat[j, i] = text_distmat[i, j] = text_distance
                
                jlon, jlat = j_data['geo']
                geo_distance = geodesic([ilat, ilon], [jlat, jlon]).km
                geo_distmat[j, i] = geo_distmat[i, j] = geo_distance


                j_wifi = '|'.join(j_data['wifi_fp'].split('|')[0:2])
                j_wifi_scanlist = ['|'.join(wifi.split('|')[0:2]) for wifi in j_data['wifi_scanlist']]

                if j_wifi in this_wifi_scanlist:
                    wifi_distmat[i, j] = 1
                if this_wifi in j_wifi_scanlist:
                    wifi_distmat[j, i] = 1
                n_intersect_wifi = len(set(this_wifi_scanlist).intersection(set(j_wifi_scanlist)))
                n_union_wifi = len(set(this_wifi_scanlist).union(set(j_wifi_scanlist)))
                wifi_distmat[i, j] = wifi_distmat[i, j] - float(n_intersect_wifi) / (n_union_wifi + eps)
                wifi_distmat[j, i] = wifi_distmat[j, i] - float(n_intersect_wifi) / (n_union_wifi + eps)
        
        print(f"Number of off site orders: {cnt_off_site_orders}")
        print('='*24)
        text_embedding = torch.cat(text_embedding, 0)
        parsed_text_embedding = torch.cat(parsed_text_embedding, 0)
        geo_embedding = torch.cat(geo_embedding, 0)
        print(text_embedding.shape)
        print(geo_embedding.shape)

        self.unique_dataset['text_description'] = np.array(list(self.raw_unique_dataset['user_text']))
        print(f"text descriptions: {self.unique_dataset['text_description'].shape}")
        self.unique_dataset['geo_description'] = np.array(list(self.raw_unique_dataset['geo']))
        self.unique_dataset['parsed_text_description'] = np.array(list(self.raw_unique_dataset['parsed_text']))
        print(f"geo descriptions: {self.unique_dataset['geo_description'].shape}")
        
        self.unique_dataset['text_embedding'] = np.array(text_embedding)
        self.unique_dataset['parsed_text_embedding'] = np.array(parsed_text_embedding)
        self.unique_dataset['geo_embedding'] = np.array(geo_embedding)

        self.unique_dataset['text_distance_matrix'] = text_distmat
        self.unique_dataset['geo_distance_matrix'] = geo_distmat
        self.unique_dataset['wifi_distance_matrix'] = wifi_distmat
        self.unique_dataset['true_cluster_label'] = true_labels

        if self.save_path:
            with open(self.save_path, 'wb') as f:
                pickle.dump(self.unique_dataset, f)

        return self.unique_dataset
    # def get_candidate(self, index):
    #     candi_index = list(range(len(self.labeled)))
    #     candi_index.remove(index)
    #     sample_index = candi_index[:cfg.ggnn_n_nodes-1]
    #     return sample_index

if __name__== "__main__" :
    args = arg_parser().parse_args()
    cfg = load_config(args)
    data_path = args.data_path
    save_path = data_path.replace('.csv', 'u.dat')
    # save_path = data_path.replace('.csv', '.dat')
    print(save_path)
    dataset_loader = Eleme(cfg, save_path=save_path)
    dataset = dataset_loader.__make_unique_dataset__()
    # dataset = dataset_loader.__make_dataset__()
        # print(data.__getitem__(i))