import os
import sys
import pickle
import pandas as pd
import torch
import torch.nn.functional as F
from typing import List
from ast import literal_eval
from geopy.distance import geodesic
from scipy.spatial.distance import cdist, pdist, squareform
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.param_parser import load_config, arg_parser

def data_process(cfg, raw_data_path: str):
    # clean data with keyword='nan'
    with open(raw_data_path, 'rb') as f:
        raw_data = pickle.load(f)
    # raw_data = pd.read_csv(raw_data_path).dropna(subset=['keyword'], axis=0, how='any')
    # 对应tables.v10
    
    # select valuable columns and rename them
    raw_columns = ['order_id', 'user_address', 'keyword', 'user_latitude', 'user_longitude', \
                     'home_cwifisdk_lat', 'home_cwifisdk_lng', 'home_cwifisdk_timestamp', \
                     'home_system_lat', 'home_system_lng', 'home_system_timestamp', \
                     'home_poi_lat', 'home_poi_lng', 'home_poi_timestamp', \
                     'ssid', 'bssid', 'rssi', 'wifi_scan_list2']

    column_rename_rules = {'user_address': 'user_text', 'keyword': 'key_text', \
                    'user_latitude':'key_lat', 'user_longitude': 'key_lon', \
                    'home_cwifisdk_lat': 'user_cwifi_lat', 'home_cwifisdk_lng': 'user_cwifi_lon', 'home_cwifisdk_timestamp': 'user_cwifi_timestamp', \
                    'home_system_lat': 'user_system_lat', 'home_system_lng': 'user_system_lon', 'home_system_timestamp': 'user_system_timestamp', \
                    'home_poi_lat': 'user_poi_lat', 'home_poi_lng': 'user_poi_lon', 'home_poi_timestamp': 'user_poi_timestamp', \
                    'wifi_scan_list2': 'user_wifi_list'}
    if cfg.DATASET.VERSION == 8:
        pass

    elif cfg.DATASET.VERSION == 10:
        raw_columns = raw_columns[:11] + ['home_amap_lat', 'home_amap_lng', 'home_amap_timestamp'] + raw_columns[11:]
        # raw_columns = ['order_id', 'user_address', 'keyword', 'user_latitude', 'user_longitude', \
        #              'home_cwifisdk_lat', 'home_cwifisdk_lng', 'home_cwifisdk_timestamp', \
        #              'home_system_lat', 'home_system_lng', 'home_system_timestamp', \
        #              'home_amap_lat', 'home_amap_lng', 'home_amap_timestamp', \
        #              'home_poi_lat', 'home_poi_lng', 'home_poi_timestamp', \
        #              'ssid', 'bssid', 'rssi', 'wifi_scan_list2']
        column_rename_rules['home_amap_lat'] = 'user_amap_lat'
        column_rename_rules['home_amap_lng'] = 'user_amap_lon'
        column_rename_rules['home_amap_timestamp'] = 'user_amap_timestamp'
        # column_rename_rules = {'user_address': 'user_text', 'keyword': 'key_text', \
        #             'user_latitude':'key_lat', 'user_longitude': 'key_lon', \
        #             'home_cwifisdk_lat': 'user_cwifi_lat', 'home_cwifisdk_lng': 'user_cwifi_lon', \
        #             'home_system_lat': 'user_system_lat', 'home_system_lng': 'user_system_lon', \
        #             'home_amap_lat': 'user_amap_lat', 'home_amap_lng': 'user_amap_lon', \
        #             'home_poi_lat': 'user_poi_lat', 'home_poi_lng': 'user_poi_lon', \
        #             'wifi_scan_list2': 'user_wifi_list', \
        #             'courier_lat': 'cour_lat', 'courier_lng': 'cour_lon', 'gps_trace':'cour_trace', \
        #             }
    if 'poi_id' in raw_data.columns and 'poi_name' in raw_data.columns and 'is_off_site' in raw_data.columns:
        print(f"Half annotated!")
        raw_columns = raw_columns + ['poi_id', 'poi_name', 'key_type', 'is_off_site']
    else:
        print(f"No annotation!")
    data = raw_data[raw_columns]
    data = data.rename(columns=column_rename_rules)

    # remove rows with no efficient geo_coordinates
    if cfg.DATASET.VERSION == 8:
        data = data.dropna(subset=['user_cwifi_lat', 'user_system_lat', 'user_poi_lat'], how='all')
    elif cfg.DATASET.VERSION == 10:
        data = data.dropna(subset=['user_cwifi_lat', 'user_system_lat', 'user_amap_lat', 'user_poi_lat'], how='all')

    # deal with geo_coordinates and wifi
    user_geo_coord = list()
    key_geo_coord = list()
    user_wifi_ap = list()
    user_wifi_list = list()
    for i, r in data.iterrows():
        if pd.isna(r['user_cwifi_lat']) == False:
            this_user_geo = list(r[['user_cwifi_lon', 'user_cwifi_lat']])
            # this_user_geo = str(round(r['user_cwifi_lon'], cfg.DATA.GEO_PRECISION)).ljust(4 + cfg.DATA.GEO_PRECISION, '0') + '|' + str(round(r['user_cwifi_lat'], cfg.DATA.GEO_PRECISION)).ljust(3 + cfg.DATA.GEO_PRECISION, '0')
        elif pd.isna(r['user_system_lat']) == False:
            this_user_geo = list(r[['user_system_lon', 'user_system_lat']])
            # this_user_geo = str(round(r['user_system_lon'], cfg.DATA.GEO_PRECISION)).ljust(4 + cfg.DATA.GEO_PRECISION, '0') + '|' + str(round(r['user_system_lat'], cfg.DATA.GEO_PRECISION)).ljust(3 + cfg.DATA.GEO_PRECISION, '0')
        elif 'user_amap_lat' in data.columns and pd.isna(r['user_amap_lat']) == False:
            this_user_geo = list(r[['user_amap_lon', 'user_amap_lat']])
            # this_user_geo = str(round(r['user_amap_lon'], cfg.DATA.GEO_PRECISION)).ljust(4 + cfg.DATA.GEO_PRECISION, '0') + '|' + str(round(r['user_amap_lat'], cfg.DATA.GEO_PRECISION)).ljust(3 + cfg.DATA.GEO_PRECISION, '0')
        elif pd.isna(r['user_poi_lat']) == False:
            this_user_geo = list(r[['user_poi_lon', 'user_poi_lat']])
            # this_user_geo = str(round(r['user_poi_lon'], cfg.DATA.GEO_PRECISION)).ljust(4 + cfg.DATA.GEO_PRECISION, '0') + '|' + str(round(r['user_poi_lat'], cfg.DATA.GEO_PRECISION)).ljust(3 + cfg.DATA.GEO_PRECISION, '0')
        this_key_geo = list(r[['key_lon', 'key_lat']])
        # this_key_geo = str(round(r['key_lon'], cfg.DATA.GEO_PRECISION)).ljust(4 + cfg.DATA.GEO_PRECISION, '0') + '|' + str(round(r['key_lat'], cfg.DATA.GEO_PRECISION)).ljust(3 + cfg.DATA.GEO_PRECISION, '0')
        user_geo_coord.append(this_user_geo)
        key_geo_coord.append(this_key_geo)

        this_wifi_ap = str(r['bssid']) + '|' + str(r['ssid']) + '|' + str(r['rssi'])
        user_wifi_ap.append(this_wifi_ap)

        this_wifi_list = list()
        try:
            wifi_list = literal_eval(r['user_wifi_list'])
            for wifi in wifi_list:
                wifi = literal_eval(wifi)
                wifi = str(wifi['bssid']) + '|' + str(wifi['ssid']) + '|' + str(wifi['rssi'])
                this_wifi_list.append(wifi)
        except ValueError as e:
            this_wifi_list = []
        user_wifi_list.append(this_wifi_list)
    data.insert(1, 'user_geo', user_geo_coord)
    data.insert(2, 'key_geo', key_geo_coord)
    data.insert(3, 'wifi_fp', user_wifi_ap)
    data.insert(4, 'wifi_scanlist', user_wifi_list)

    final_columns = ['order_id', 'user_text', 'key_text', 'user_geo', 'key_geo', 'wifi_fp', 'wifi_scanlist']

    dataset = data[final_columns]
    print(dataset.iloc[0])
    print(dataset.iloc[0]['user_geo'][0])

    return dataset

def levenshtein(text1: str, text2: str):
    if len(text1) > len(text2):
        text1, text2 = text2, text1

    distances = range(len(text1) + 1)
    for i2, c2 in enumerate(text2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(text1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def get_text_distance_matrix(text_list: List[str]):
    n_data = len(text_list)
    text_distance_matrix = torch.zeros(n_data, n_data)
    for i, text_i in enumerate(text_list):
        for j in range(i+1, n_data):
            text_j = text_list[j]
            text_distance_matrix[i, j] = text_distance_matrix[j, i] = levenshtein(text_i, text_j)
    return text_distance_matrix

def get_geo_distance_matrix(geo_list):
    n_data = len(geo_list)
    geo_distance_matrix = torch.zeros(n_data, n_data)
    for i, geo_i in enumerate(geo_list):
        lon_i, lat_i = geo_i
        for j in range(i+1, n_data):
            lon_j, lat_j = geo_list[j]
            geo_distance_matrix[i, j] = geo_distance_matrix[j, i] = geodesic([lat_i, lon_i], [lat_j, lon_j]).km
    return geo_distance_matrix

def get_wifi_distance_matrix(wifi_list):    # wifi_list: List[Set]
    n_data = len(wifi_list)
    wifi_distance_matrix = torch.zeros(n_data, n_data)
    for i, wifiset_i in enumerate(wifi_list):
        if len(wifiset_i) < 1:
            continue
        for j in range(i+1, n_data):
            wifiset_j = wifi_list[j]
            if len(wifiset_j) < 1:
                continue
            intersection_size = len(wifiset_i & wifiset_j) / len(wifiset_i | wifiset_j)
            wifi_distance_matrix[i, j] = wifi_distance_matrix[j, i] = intersection_size
    wifi_distance_matrix = torch.ones_like(wifi_distance_matrix) - wifi_distance_matrix
    return wifi_distance_matrix

def pairwise_distance(x1: torch.Tensor, x2: torch.Tensor, metric: str):
    assert len(x1.shape) >=2 and len(x1.shape) <= 3
    assert x1.shape[1:] == x2.shape[1:]
    if len(x1.shape) <= 2:
        return torch.tensor(cdist(x1, x2, metric=metric))
    else:
        n_channel = x1.shape[1]
        n_channel_distmat = list()
        for c in range(n_channel):
            dist = torch.as_tensor(cdist(x1[:, c], x2[:, c], metric=metric))
            n_channel_distmat.append(dist)
        return torch.stack(n_channel_distmat, dim=-1)

def top_k(x: torch.Tensor, k: int, find_maximum: bool):
    unique_values, unique_indices = torch.unique(x, return_inverse=True)
    sorted_values, sorted_indices = torch.sort(unique_values, descending=find_maximum)
    topk_unique_indices = sorted_indices[:k]
    # original_indices = torch.nonzero(unique_indices == topk_unique_indices, as_tuple=False).squeeze(1)
    original_indices = torch.nonzero(torch.isin(unique_indices, topk_unique_indices), as_tuple=False).squeeze(1)
    return original_indices, sorted_values[:k]

def multimodal_distance(textmat=None, geomat=None, 
                        wifimat=None, featmat=None):
    if textmat.any() == None and geomat.any() == None and wifimat.any() == None:
        distance_matrix = featmat
    elif featmat.any() == None:
        distance_matrix = textmat + geomat + wifimat
    else:
        # distance_matrix = textmat + geomat + wifimat + featmat
        distance_matrix = textmat + torch.mul(geomat, wifimat) + featmat
        # distance_matrix = torch.mul(textmat + torch.mul(geomat, wifimat), featmat)
    return distance_matrix

def normalize(x):
    x_min = torch.min(x)
    if torch.isinf(x).any():
        x_masked = torch.where(torch.isinf(x), -1, x)
        x_max = torch.max(x_masked)
        x = torch.where(x_masked==-1, x_max, x)
    else:
        x_max = torch.max(x)
    if x_max == x_min:
        normalized_x = torch.ones_like(x)
    else:
        normalized_x = torch.div((x - x_min), (x_max - x_min))
    # print(normalized_x)
    # x_mean = torch.mean(normalized_x)
    # x_std = torch.std(normalized_x)
    # normalized_x = torch.div(normalized_x - x_mean, x_std)

    return normalized_x


if __name__== "__main__" :
    args = arg_parser().parse_args()
    cfg = load_config(args)

    database_dir = os.path.join(cfg.GLOBAL.ROOT_DIR, 'database')
    curr_data_dir = os.path.join(cfg.GLOBAL.CURR_DIR, 'data')
    data_path = os.path.join(database_dir, '37841_1d_3336.csv')
    save_path = os.path.basename(data_path)

    data = data_process(cfg, data_path)
    print(save_path)
    # shutil.move(args.data_path, os.path.join(database_dir, save_path))
    save_path = os.path.join(curr_data_dir, save_path.split('.')[0] + '.dat')
    data.to_pickle(save_path)
