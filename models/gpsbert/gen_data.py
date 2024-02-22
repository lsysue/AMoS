import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

PRECISION = 4
DATASET_ID = 37841

# root_dir = '/mnt/c/Users/ALSC/Documents/POI/'
root_dir = '/datapool/workspace/3002lsy/repos/Eleme'
data_dir = os.path.join(root_dir, 'data')
# curr_dir = os.path.join(root_dir, 'models/gpsbert/')
curr_dir = os.getcwd()
print(curr_dir)

# dataset = pd.read_csv(os.path.join(data_dir, '37841_16d.csv'))
with open(os.path.join(data_dir, f'{DATASET_ID}_16d.csv'), 'rb') as f:
    dataset = pickle.load(f)
print(dataset.columns)
# dataset = pd.read_csv(os.path.join(data_dir, '37841_1d_3336.csv'))
# addr_vec = np.load(os.path.join(data_dir, '37841_1d_addr.npy'))

# data_clean = dataset.dropna(subset=['user_cwifi_lat', 'user_system_lat', 'user_poi_lat'], how='all')
# print(data_clean.shape)
# def not_have_geo(x):
#     return x not in data_clean.index
# geo_mask = list(map(not_have_geo, list(dataset.index)))
# addr_vec = np.delete(addr_vec, np.where(geo_mask), axis=0)
# print(addr_vec.shape)

special_tokens = ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]']
print(os.path.exists(f'./geo_vocab_{PRECISION}.txt'))
if not os.path.exists(f'./geo_vocab_{PRECISION}.txt'):
    vocab = set()
else:
    vocab = set()
    with open(f'./geo_vocab_{PRECISION}.txt', 'r') as f:
        for line in f.readlines():
            line = line.split('\n')[0]
            if line not in special_tokens:
                vocab.add(line)

if not os.path.exists(f'./traj_corpus_{PRECISION}.txt'):
    corpus = list()
else:
    corpus = list()
    with open(f'./traj_corpus_{PRECISION}.txt', 'r') as f:
        for line in f.readlines():
            line = line.split('\n')[0]
            corpus.append(line)
print(f'vocab size: {len(vocab)}')
print(f'corpus size: {len(corpus)}')

# data = data_clean.copy()
data = dataset.copy()
for i, r in data.iterrows():
    if pd.isna(r['user_cwifi_lat']) == False:
        user_coor_word = str(round(r['user_cwifi_lon'], PRECISION)).ljust(4 + PRECISION, '0') + '|' + str(round(r['user_cwifi_lat'], PRECISION)).ljust(3 + PRECISION, '0')
    elif pd.isna(r['user_system_lat']) == False:
        user_coor_word = str(round(r['user_system_lon'], PRECISION)).ljust(4 + PRECISION, '0') + '|' + str(round(r['user_system_lat'], PRECISION)).ljust(3 + PRECISION, '0')
    elif 'user_amap_lat' in data.columns and pd.isna(r['user_amap_lat']) == False:
        user_coor_word = str(round(r['user_amap_lon'], PRECISION)).ljust(4 + PRECISION, '0') + '|' + str(round(r['user_amap_lat'], PRECISION)).ljust(3 + PRECISION, '0')
    elif pd.isna(r['user_poi_lat']) == False:
        user_coor_word = str(round(r['user_poi_lon'], PRECISION)).ljust(4 + PRECISION, '0') + '|' + str(round(r['user_poi_lat'], PRECISION)).ljust(3 + PRECISION, '0')
    key_coor_word = str(round(r['key_lon'], PRECISION)).ljust(4 + PRECISION, '0') + '|' + str(round(r['key_lat'], PRECISION)).ljust(3 + PRECISION, '0')
    vocab.add(user_coor_word)
    vocab.add(key_coor_word)

    if pd.isna(r['cour_trace']) == False:
        this_traj_sent = list()
        cour_trace = str(r['cour_trace']).split(';')
        for cour_coor in cour_trace:
            cour_lon = float(cour_coor.split(',')[0])
            cour_lat = float(cour_coor.split(',')[1])
            cour_coor_word = str(round(cour_lon, PRECISION)).ljust(4 + PRECISION, '0') + '|' + str(round(cour_lat, PRECISION)).ljust(3 + PRECISION, '0')
            vocab.add(cour_coor_word)
            this_traj_sent.append(cour_coor_word)
        this_traj_sent = " ".join(this_traj_sent)
        corpus.append(this_traj_sent)

vocab = special_tokens + list(vocab)
print(f'-*16')
print(f'vocab size: {len(vocab)}')
print(f'corpus size: {len(corpus)}')

with open(f'geo_vocab_{PRECISION}.txt', 'w') as f:
    for word in vocab:
        f.write(word + '\n')
with open(f'traj_corpus_{PRECISION}.txt', 'w') as f:
    for sent in corpus:
        f.write(sent + '\n')