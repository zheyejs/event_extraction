# @Author : bamtercelboo
# @Datetime : 2018/1/31 9:24
# @File : model_PNC.py
# @Last Modify Time : 2018/1/31 9:24
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  model_PNC.py
    FUNCTION : Part-of-Speech Tagging(POS), Named Entity Recognition(NER) and Chunking
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
        self.cat_size = 5

        V = config.embed_num
        D = config.embed_dim
        C = config.class_num
        paddingId = config.paddingId

        self.embed = nn.Embedding(V, D, padding_idx=paddingId)
        # self.embed = nn.Embedding(V, D)

        if config.pretrained_embed:
            self.embed.weight.data.copy_(config.pretrained_weight)
        self.embed.weight.requires_grad = self.config.embed_finetune

        self.dropout_embed = nn.Dropout(config.dropout_emb)
        self.dropout = nn.Dropout(config.dropout)

        self.bilstm = nn.LSTM(input_size=D * config.windows_size, hidden_size=config.lstm_hiddens, dropout=config.dropout, num_layers=config.lstm_layers,
                              bidirectional=True, bias=True)
        # self.init_lstm()

        self.linear = nn.Linear(in_features=config.lstm_hiddens * 2, out_features=C, bias=True)
        # init.xavier_uniform(self.linear.weight)
        # self.linear.bias.data.uniform_(-np.sqrt(6 / (config.lstm_hiddens + 1)), np.sqrt(6 / (config.lstm_hiddens + 1)))

    def init_lstm(self):
        if self.bilstm.bidirectional is True:   weight = 2
        else:   weight = 1
        for i in range(weight):
            for j in range(2):
                init.xavier_uniform(self.bilstm.all_weights[i][j])

    def context_embed(self, embed, batch_features):
        context_indices = batch_features.context_indices
        B, T, WS = context_indices.size()
        B_embed, T_embed, dim = embed.size()
        if assertTrue((B == B_embed) and (T == T_embed)) is False:
            print("invalid argument")
            exit()
        context_indices = context_indices.view(B, T * WS)
        if self.config.use_cuda is True:
            context_np = context_indices.data.cpu().numpy()
        else:
            context_np = np.copy(context_indices.data.numpy())
        for i in range(B):
            for j in range(T * WS):
                context_np[i][j] = T * i + context_np[i][j]
        if self.config.use_cuda is True:
            context_indices = Variable(torch.from_numpy(context_np)).cuda()
        else:
            context_indices = Variable(torch.from_numpy(context_np))
        context_indices = context_indices.view(context_indices.size(0) * context_indices.size(1))

        embed = embed.view(B * T, dim)
        if self.config.use_cuda is True:
            pad_embed = Variable(torch.zeros(1, dim)).cuda()
        else:
            pad_embed = Variable(torch.zeros(1, dim))
        embed = torch.cat((embed, pad_embed), 0)
        context_embed = torch.index_select(embed, 0, context_indices)
        context_embed = context_embed.view(B, T, -1)

        return context_embed

    def forward(self, batch_features):
        word = batch_features.word_features
        sentence_length = batch_features.sentence_length
        x = self.embed(word)  # (N,W,D)
        context_embed = self.context_embed(x, batch_features)
        context_embed = self.dropout_embed(context_embed)
        packed_embed = pack_padded_sequence(context_embed, sentence_length, batch_first=True)
        x, _ = self.bilstm(packed_embed)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = x[batch_features.desorted_indices]
        x = F.tanh(x)
        logit = self.linear(x)
        return logit

