#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : bert_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2021/7/5 11:25
@version: 1.0
@desc  : BERT model
"""
import json
import os
from typing import List, Dict, Iterator, Optional, Union

import tokenizers
import torch
from pypinyin import pinyin, Style
from tokenizers import BertWordPieceTokenizer


class ChinBertTokenizer(BertWordPieceTokenizer):

    def __init__(self, pretrained_path, max_length: int = 512):
        self.max_length = max_length
        # load pinyin map dict
        vocab = os.path.join(pretrained_path, 'vocab.txt')
        config = os.path.join(pretrained_path, 'config')
        
        with open(os.path.join(config, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map tensor
        with open(os.path.join(config, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # load pinyin map tensor
        with open(os.path.join(config, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

        super().__init__(vocab)

    def encode(self, sentence):
        # convert sentence to ids
        tokenizer_output = super().encode(sentence)
        bert_tokens = tokenizer_output.ids
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output)
        # assert，token nums should be same as pinyin token nums
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        # convert list to tensor
        input_ids = torch.LongTensor(bert_tokens)
        pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        return input_ids, pinyin_ids

    def convert_sentence_to_pinyin_ids(self, 
                                       sentence: str, 
                                       tokenizer_output: tokenizers.Encoding) -> List[List[int]]:
        # get pinyin of a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        # find chinese character location, and generate pinyin ids
        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)

        return pinyin_ids

