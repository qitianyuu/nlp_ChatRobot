"""
# File       :  my_dataset.py
# Time       :  2022/1/21 1:12 上午
# Author     : Qi
# Description:
"""
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
from itertools import chain


class MyDataset(Dataset):
    def __init__(self, prefix, config):
        # 检验是否是训练或者测试前缀
        assert prefix == config['train_prefix'] or prefix == config['valid_prefix']

        print(f"Loading {prefix}_id.pickle.")
        with open(f"{config['data_dir']}/{prefix}_ids.pickle", 'rb') as f:
            dials = pickle.load(f)

        self.input_ids = []        # (N, L)
        self.token_type_ids = []   # (N, L)
        self.labels = []           # (N, L)

        # 对每一段对话循环
        for dial in tqdm(dials):
            hists = []
            # 对每一句话循环，加上对话者ID
            for u, utter in enumerate(dial):
                if u % 2 == 0:
                    hists.append([config['sp1_id']] + utter)
                else:
                    hists.append([config['sp2_id']] + utter)

            # 对于每句话循环
            for h in range(len(hists)):
                # 如果是 <speaker2>
                if hists[h][0] == config['sp2_id']:
                    for s in range(0, h):
                        # 对话
                        contexts = hists[s:h + 1]
                        # 选出对话轮数合适的数据 大于2句，小于5句
                        if len(contexts) > config['max_turns']:
                            num_exceeded = len(contexts) - config['max_turns']
                            contexts = contexts[num_exceeded:]
                        if len(contexts) < 2:
                            break

                        input_ids = [config['bos_id']] + list(chain.from_iterable(contexts)) + [config['eos_id']]

                        if len(input_ids) <= config['max_len']:
                            start_sp_id, next_sp_id = contexts[0][0], contexts[1][0]
                            # 为 input_ids 的每个位置标注上是谁说的话, 同时加上开始标志和下个speaker的开始标志
                            token_type_ids = [[start_sp_id] * len(ctx) if c % 2 == 0 else [next_sp_id] * len(ctx) for c, ctx in enumerate(contexts)]
                            assert token_type_ids[-1][0] == config['sp2_id']
                            token_type_ids = [start_sp_id] + list(chain.from_iterable(token_type_ids)) + [config['sp2_id']]
                            assert len(input_ids) == len(token_type_ids)

                            # 除了最后一句话，其他全部标注为 -100
                            labels = [[-100] * len(ctx) if c < len(contexts) - 1 else [-100] + ctx[1:] for c, ctx in enumerate(contexts)]
                            assert labels[-1][1:] == contexts[-1][1:]
                            labels = [-100] + list(chain.from_iterable(labels)) + [config['eos_id']]
                            assert len(input_ids) == len(labels)

                            self.input_ids.append(input_ids)
                            self.token_type_ids.append(token_type_ids)
                            self.labels.append(labels)
                            break

    def __getitem__(self, idx):
        return self.input_ids[idx], self.token_type_ids[idx], self.labels[idx]

    def __len__(self):
        return len(self.input_ids)


class PadCollate():
    def __init__(self, eos_id):
        self.eos_id = eos_id

    def pad_collate(self, batch):
        input_ids, token_type_ids, labels = [], [], []
        for idx, seqs in enumerate(batch):
            input_ids.append(torch.LongTensor(seqs[0]))
            token_type_ids.append(torch.LongTensor(seqs[0]))
            labels.append(torch.LongTensor(seqs[2]))

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.eos_id)
        token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=self.eos_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return input_ids, token_type_ids, labels