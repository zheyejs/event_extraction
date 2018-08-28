# @Author : bamtercelboo
# @Datetime : 2018/8/17 16:06
# @File : BiLSTM.py
# @Last Modify Time : 2018/8/17 16:06
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  BiLSTM.py
    FUNCTION : None
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable as Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
import torch.nn.init as init
import numpy as np
import time

from wheel.signatures import assertTrue

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class BiLSTM(nn.Module):

    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.config = config

        V = config.embed_num
        D = config.embed_dim
        C = config.class_num
        paddingId = config.paddingId

        # self.embed = nn.Embedding(V, D, padding_idx=paddingId)
        self.embed = nn.Embedding(V, D)
        # self.embed = nn.Embedding(V, D)

        if config.pretrained_embed:
            self.embed.weight.data.copy_(config.pretrained_weight)
        # self.embed.weight.requires_grad = self.config.embed_finetune

        self.dropout_embed = nn.Dropout(config.dropout_emb)
        self.dropout = nn.Dropout(config.dropout)

        self.bilstm = nn.LSTM(input_size=D, hidden_size=config.lstm_hiddens, dropout=0.0, num_layers=config.lstm_layers,
                              bidirectional=True, batch_first=True, bias=True)
        # self.init_lstm()

        self.linear = nn.Linear(in_features=config.lstm_hiddens * 2, out_features=C, bias=True)
        # init.xavier_uniform(self.linear.weight)
        # self.linear.bias.data.uniform_(-np.sqrt(6 / (config.lstm_hiddens * 2 + 1)), np.sqrt(6 / (config.lstm_hiddens * 2 + 1)))

    def forward(self, batch_features):
        word = batch_features.word_features
        sentence_length = batch_features.sentence_length
        x = self.embed(word)  # (N,W,D)
        x = self.dropout_embed(x)
        # print(x.size())
        # x, _ = self.bilstm(x)
        packed_embed = pack_padded_sequence(x, sentence_length, batch_first=True)
        x, _ = self.bilstm(packed_embed)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = x[batch_features.desorted_indices]
        x = self.dropout(x)
        # x = F.tanh(x)
        logit = self.linear(x)
        return logit
