import os
import sys
import numpy as np
import tensorflow as tf
import json
import pickle
import collections
import Vocabulary
import Config
import utils
import torch


class BatchInput(collections.namedtuple("BatchInput",
                                        ("key_input", "val_input", "input_lens",
                                         "target_input", "target_output", "output_lens",
                                         "group", "group_lens", "group_cnt",
                                         "target_type", "target_type_lens",
                                         "text", "slens",
                                         "category"))):
    pass

class EPWDataset:
    def __init__(self, file_name):
        self.config = Config.config
        self.texts = open(file_name, 'r', encoding='UTF-8').read().splitlines()
        
        if not os.path.exists(self.config.vocab_file):
            pickle.dump(Vocabulary.Vocabulary(), open(self.config.vocab_file, "wb"))
        self.vocab = pickle.load(open(self.config.vocab_file, "rb"))
        self.cate2FK = {
            "裙": ["类型", "版型", "材质", "颜色", "风格", "图案", "裙型", "裙下摆", "裙腰型", "裙长", "裙衣长", "裙袖长", "裙领型", "裙袖型", "裙衣门襟",
                  "裙款式"],
            "裤": ["类型", "版型", "材质", "颜色", "风格", "图案", "裤长", "裤型", "裤款式", "裤腰型", "裤口"],
            "上衣": ["类型", "版型", "材质", "颜色", "风格", "图案", "衣样式", "衣领型", "衣长", "衣袖长", "衣袖型", "衣门襟", "衣款式"]}
        for key, val in self.cate2FK.items():
            self.cate2FK[key] = dict(zip(val, range(len(val))))

    def sort(self, cate, lst):
        assert cate in self.cate2FK
        tgt = self.cate2FK[cate]
        return sorted(lst, key=lambda x: tgt.get(x[0], len(tgt) + 1))
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        line = self.texts[idx]
        line = json.loads(line)
        res = {"feats" + suffix: [] for suffix in ['_key', '_val']}
        cate = dict(line['feature'])['类型']
        val_tpe = 1
        feats = self.sort(cate, line['feature'])
        for item in feats:
            res["feats_key"].append(self.vocab.lookup(item[0], 0))
            res["feats_val"].append(self.vocab.lookup(item[1], val_tpe))

        text = [self.vocab.lookup(word, 2) for word in line['desc'].split(" ")]     # 目标文案，text对应的id号
        slens = len(text)
        res["feats_key_len"] = len(res["feats_key"])

        category = self.vocab.category2id[cate]

        key_input = [self.vocab.lookup("<SENT>", 0)] + res['feats_key']
        val_input = [self.vocab.lookup("<ADJ>", val_tpe)] + res['feats_val']
        input_lens = len(key_input)

        target_input = []
        target_output = []
        output_lens = []

        group = []
        group_lens = []

        target_type = []
        target_type_lens = []

        key_val = list(zip(key_input, val_input))
        for _, segment in line['segment'].items():
            sent = [self.vocab.lookup(w, 2) for w in segment['seg'].split(" ")]
            target_output.append(sent + [self.vocab.end_token])
            target_input.append([self.vocab.start_token] + sent)
            output_lens.append(len(target_output[-1]))

            order = [item[:2] for item in segment['order']]
            if len(order) == 0:
                order = [['<SENT>', '<ADJ>']]
            gid = [key_val.index((self.vocab.lookup(k, 0), self.vocab.lookup(v, val_tpe))) for k, v in order]
            group.append(sorted(gid))
            group_lens.append(len(group[-1]))

            target_type.append([self.vocab.type2id[t] for t in segment['key_type']])
            target_type_lens.append(len(target_type[-1]))

        group_cnt = len(group)

        for item in [target_input, target_output, group, target_type]:
            max_len = -1
            for lst in item:
                max_len = max(max_len, len(lst))
            for idx, lst in enumerate(item):
                if len(lst) < max_len:
                    item[idx] = lst + [0] * (max_len - len(lst))
        
        return np.array(key_input, dtype=torch.int32), np.array(val_input, dtype=torch.int32),   # 整条文案的key和value对
        np.array(input_lens, dtype=torch.int32),
        np.array(target_input, dtype=torch.int32), np.array(target_output, dtype=torch.int32),    # 每条文案所有句子的列表，句子列表的项是句子所有词的列表
        np.array(output_lens, dtype=torch.int32),
        np.array(group, dtype=torch.int32), np.array(group_lens, dtype=torch.int32),      # 每个句子包含的key和value对
        np.array(group_cnt, dtype=torch.int32),
        np.array(target_type, dtype=torch.int32), np.array(target_type_lens, dtype=torch.int32),
        np.array(text, dtype=torch.int32), np.array(slens, dtype=torch.int32),     # 目标文案所包含词对应的id号
        np.array(category, dtype=torch.int32),      # 描述目标的类型


class alignCollate(object):
    def __init__(self):
        pass
    
    def __call__(self, batch):
        key_input, val_input, input_lens, target_input, target_output, output_lens, groups, glens, group_cnt, target_type, target_type_lens, text, slens, cate_input = zip(*batch)
        
        stack_input_lens = np.concatenate(input_lens)
        max_key_len = cat_input_lens.max().item()
        padded_key_input = []
        padded_val_input = []
        for key, val in zip(key_input, val_input):
            np_key = np.zeros(max_key_len)
            np_val = np.zeros(max_key_len)
            k_size = key.size()
            v_size = val.size()
            np_key[:k_size] = key
            np_val[:v_size] = val
            padded_key_input.append(np_key)
            padded_val_input.append(np_val)
        stack_key_input = np.stack(padded_key_input, axis=0)
        stack_val_input = np.stack(padded_val_input, axis=0)
        
        max_g_cnt = 0
        max_w_cnt = 0
        for output_len in output_lens:
            g_cnt = output_len.shape
            if g_cnt > max_g_cnt:
                max_g_cnt = g_cnt
            w_cnt = output_len.max().item()
            if w_cnt > max_w_cnt:
                max_w_cnt = w_cnt
        
        padded_output_lens = []     
        for output_len in output_lens:
            np_output_len = np.zeros(max_g_cnt)
            g_cnt = output_len.shape
            np_output_len[:g_cnt] = output_len
            padded_output_lens.append(np_output_len)
        stack_output_lens = np.stack(padded_output_lens, axis=0)
        
        padded_target_input = []
        padded_target_output = []
        for t_input, t_output in zip(target_input, target_output):
            np_target_input = np.zeros((max_g_cnt, max_w_cnt))
            np_target_output = np.zeros((max_g_cnt, max_w_cnt))
            i_g_len, i_w_len = t_input.shape
            np_target_input[:i_g_len, :i_w_len] = t_input
            o_g_len, o_w_len = t_output.shape
            np_target_output[:o_g_len, :o_w_len] = t_output
            padded_target_input.append(np_target_input)
            padded_target_output.append(np_target_output)
        stack_target_input = np.stack(padded_target_input, axis=0)
        stack_target_output = np.stack(padded_target_output, axis=0)
        
        
        


