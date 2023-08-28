import os
import pickle
import random
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from transformers improt BertForMaskedLM
from torch.utils.data import Dataset
import torch

root_dir = '/mnt/c/Users/ALSC/Documments/POI/'
data_dir = os.path.join(root_dir, 'data')
curr_dir = os.path.join(root_dir, 'method/models/gpsbert/')
corpus_path = os.path.join(curr_dir, 'traj_corpus.txt')

class Trajectory(Dataset):
    def __init__(self, data_dir=corpus_path):
        dataset = pd.read_csv(data_dir)
        dataset = dataset.dropna(subset=['cour_trace'], how='any')
        cour_traj = list(dataset['cour_trace'])
        user_coor = list()
        key_coor = list()
        for i, r in dataset.iterrows():
            if pd.isna(r['user_cwifi_lat']) == False:
                coor_word = str(round(r['user_cwifi_lon'], 6)).ljust(10, '0') + '|' + str(round(r['user_cwifi_lat'], 6)).ljust(9, '0')
            elif pd.isna(r['user_system_lat']) == False:
                coor_word = str(round(r['user_system_lon'], 6)).ljust(10, '0') + '|' + str(round(r['user_system_lat'], 6)).ljust(9, '0')
            # elif pd.isna(r['user_amap_lat']) == False:
            #     user_geo_coor.append(list(r[['user_amap_lon', 'user_amap_lat']]))
            elif pd.isna(r['user_poi_lat']) == False:
                coor_word = str(round(r['user_poi_lon'], 6)).ljust(10, '0') + '|' + str(round(r['user_poi_lat'], 6)).ljust(9, '0')
            user_coor.append(coor_word)
            key_coor.append(str(round(r['key_lon'], 6)).ljust(10, '0') + '|' + str(round(r['key_lat'], 6)).ljust(9, '0'))
        n_data = dataset.shape[0]
        print(n_data)
        assert n_data == len(cour_traj)
        assert n_data == len(user_coor)
        assert n_data == len(key_coor)
        
        self.max_length = 0
        self.tokenizer = BertTokenizer(
            vocab_file='./geo_vocab.txt', do_basic_tokenize=False, is_split_into_words=False)

        self.traj_corpus = list()
        
        for idx in range(n_data):
            this_traj_sent = list()
            this_traj = str(cour_traj[idx]).split(';')
            # this_coor_word = user_coor[idx]
            # user_lon = float(this_coor_word.split('|')[0])
            # user_lat = float(this_coor_word.split('|')[1])
            # this_dists = list()
            for cour_coor in this_traj:
                cour_lon = float(cour_coor.split(',')[0])
                cour_lat = float(cour_coor.split(',')[1])
                coor_word = str(round(cour_lon, 6).ljust(10, '0')) + '|' + str(round(cour_lat, 6).ljust(9, '0'))
                # this_dists.append(np.sqrt(np.power(cour_lon - user_lon, 2) + np.power(cour_lat - user_lat, 2)))
                this_traj_sent.append(coor_word)
            # if this_coor_word not in this_traj_sent:
            #     closest_index = np.argmin(this_dists)
            #     if this_dists[closest_index - 1] > this_dists[closest_index + 1]:
            #         this_traj_sent.insert(closest_index, this_coor_word)
            #     else:
            #         this_traj_sent.insert(closest_index - 1, this_coor_word)
            #     print(this_coor_word, this_traj_sent[closest_index])
            this_traj_sent = " ".join(this_traj_sent)
            self.traj_corpus.append(this_traj_sent)

        for traj_sent in self.traj_corpus:
            encode = self.tokenizer.encode(traj_sent)
            print(encode)
            self.max_length = max(self.max_length, len(encode))

        # for input, label in dataset:
        #     encode = torch.tensor(tokenizer.encode(input),dtype=torch.long)
        #     toadd = torch.ones(self.max_length, dtype=torch.long)
        #     toadd[0:len(encode)] = encode
        #     self.input.append(toadd)
        #     self.label.append(torch.tensor(label,dtype=torch.long))

        # self.input=torch.stack(self.input)
        # self.dataset = dataset


    def __getitem__(self, index):
        this_cour_traj = self.cour_traj[index]
        this_user_coor = self.user_coor[index]
        encode = torch.tensor(tokenizer.encode(this_cour_traj))

        return self.input[index], self.label[index]

    def __len__(self):
        return len(self.dataset)


class DataLoader:
    def __init__(self, in_dir=dataset_path,
                 out_dir=os.path.join(curr_dir, 'gpsbert-base/'),
                 bert_dir='bert-base-uncased',
                 train_source='title',
                 batch_size: int = 64,
                 max_len: int = 64,
                 shuffle: bool = True,
                 mask_token='[MASK]',
                 mask_rate=0.15):
        super(DataLoader, self).__init__()
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.bert_dir = bert_dir
        self.train_source = train_source
        if len(os.listdir(out_dir)) == 0:   # 如果没有保存好的checkpoints，那么就使用BERT的tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_dir)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.out_dir)
        self.get_data()
        # 定义BERT数据加载的迭代器
        self.bert_iter = BERTMLMDataIter(datas=self.datas, tokenizer=self.tokenizer,
                                         max_len=max_len, batch_size=batch_size)
        self.model = BertForMaskedLM.from_pretrained(self.bert_dir)
        # 注意在使用之前resize bemedding大小
        self.model.resize_token_embeddings(len(self.tokenizer))

    def get_data(self):
        ''' 加载源数据获取raw的文本,文本是以excel(csv)形式存放的，并且只加载'title'字段的文本进行训练
        :return:
        '''
        self.datas = []
        frame = list(pd.read_csv(self.in_dir)[self.train_source].values)
        self.datas.extend(frame)

    def get_new_tokens(self):
        '''
        为BERT词表添加新的tokens
        :return:
        '''
        self.new_tokens = []
        for data in self.datas:
            # tokens = self.tokenizer.tokenize(data)
            # print(tokens)
            for word in data:
                if word not in self.tokenizer.vocab:  
                	# 由于是中文的模型，因此这里剔除一些非中文的特殊字符
                    if u'\u4e00' <= word <=u'\u9fff' and word not in self.new_tokens:
                        self.new_tokens.append(word)
        
        self.tokenizer.add_tokens(self.new_tokens)
        self.tokenizer.save_pretrained(self.out_dir)  #保存增加的词表

class BERTMLMDataIter():
    '''
    BertForMaskedLM的数据加载工具，其输入的格式为：The capital of France is [MASK]转化之后的ids，
    输出则为[Mask]的预测
    '''
    def __init__(self, datas:list, tokenizer: BertTokenizer, batch_size: int = 32,
                 max_len: int = 128, shuffle:bool=True, mask_token='[MASK]', mask_rate=0.15):
        super(BERTMLMDataIter).__init__()
        self.datas = datas
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.shuffle = shuffle
        self.Mask_id = self.tokenizer.convert_tokens_to_ids(mask_token)
        self.mask_rate = mask_rate
        # 首次初始化
        self.reset()
        self.ipts = 0

    def reset(self):
        print("dataiter reset, 读取数据")
        if self.shuffle:
            random.shuffle(self.datas)
        self.data_iter = iter(self.datas)

    def random_mask(self, tokens, rate):
        '''
        :param token: 需要mask的初始字符串
        :param rate:
        :return:
        '''
        mask_tokens, label = [], []
        for word in tokens:
            mmm = random.random()
            if mmm <= rate:
                mask_tokens.append(self.Mask_id)
                label.append(word)
            else:
                mask_tokens.append(word)
                label.append(-100)    # -100表示计算损失函数的时候不计算该值
        return mask_tokens, label

    def get_data(self):
        ''' 获取mask的data数据以及标签的程序
        :return:
        '''
        data_ids = []
        labels = []
        att_masks = []
        for data in self.datas:
            data_id = self.tokenizer.encode(data)
            masked_data, label = self.random_mask(data_id, self.mask_rate)
            att_mask = [1]*len(masked_data)+[0]*(self.max_len-len(masked_data))
            masked_data = masked_data + [0]*(self.max_len-len(masked_data))
            label = label + [-100]*(self.max_len-len(label))
            data_ids.append(masked_data)
            labels.append(label)
            att_masks.append(att_mask)

    def get_batch_data(self):
        ''''''
        batch_data = []
        for i in self.data_iter:
            batch_data.append(i)
            if len(batch_data) == self.batch_size:
                break
        if len(batch_data) < 1:
            return None
        data_ids = []
        labels = []
        att_masks = []
        for data in batch_data:
            data_id = self.tokenizer.encode(data)
            masked_data, label = self.random_mask(data_id, self.mask_rate)
            if len(masked_data) < self.max_len:
                att_mask = [1] * len(masked_data) + [0] * (self.max_len - len(masked_data))
            else:
                att_mask = [1]*self.max_len
            masked_data = masked_data[:self.max_len]
            label = label[:self.max_len]
            att_mask = att_mask[:self.max_len]
            masked_data = masked_data + [0] * (self.max_len - len(masked_data))
            label = label + [-100] * (self.max_len - len(label))
            data_ids.append(masked_data)
            labels.append(label)
            att_masks.append(att_mask)
        batch_ipts = {}
        batch_ipts['ids'] = torch.LongTensor(data_ids)
        batch_ipts['mask'] = torch.LongTensor(att_masks)
        batch_ipts['label'] = torch.LongTensor(labels)
        return batch_ipts

    def __iter__(self):
        return self

    def __next__(self):
        if self.ipts is None:
            self.reset()
        self.ipts = self.get_batch_data()
        if self.ipts is None:
            raise StopIteration
        else:
            return self.ipts