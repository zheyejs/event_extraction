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
import random
import torch.nn.init as init
import numpy as np

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class PNC(nn.Module):

    def __init__(self, config):
        super(PNC, self).__init__()
        self.args = config
        self.cat_size = 5

        V = config.embed_num
        D = config.embed_dim
        C = config.class_num
        paddingId = config.paddingId

        self.embed = nn.Embedding(V, D, padding_idx=paddingId)

        if config.pretrained_embed:
            self.embed.weight.data.copy_(config.pretrained_weight)
        # self.embed.weight.requires_grad = False

        self.dropout_embed = nn.Dropout(config.dropout_emb)
        self.dropout = nn.Dropout(config.dropout)

        # self.batchNorm = nn.BatchNorm1d(D * 5)

        self.bilstm = nn.LSTM(input_size=D, hidden_size=config.lstm_hiddens, num_layers=config.lstm_layers,
                              bidirectional=True, bias=True)
        self.init_lstm()
        # init.xavier_uniform(self.bilstm.all_weights[0][0])
        # self.bilstm.bias.uniform_(-np.sqrt(6 / (config.lstm_hiddens + 1)), np.sqrt(6 / (config.lstm_hiddens + 1)))

        # self.linear = nn.Linear(in_features=D * self.cat_size, out_features=C, bias=True)
        self.linear = nn.Linear(in_features=config.lstm_hiddens * 2, out_features=C, bias=True)
        init.xavier_uniform(self.linear.weight)
        self.linear.bias.data.uniform_(-np.sqrt(6 / (config.lstm_hiddens + 1)), np.sqrt(6 / (config.lstm_hiddens + 1)))

    def init_lstm(self):
        if self.bilstm.bidirectional is True:   weight = 2
        else:   weight = 1
        for i in range(weight):
            for j in range(2):
                # print(i, "  ", j)
                init.xavier_uniform(self.bilstm.all_weights[i][j])
            # for j in range(3, 4):
            #     print(i, "  ", j)
            #     self.bilstm.all_weights[i][j].uniform_(-np.sqrt(6 / (self.args.lstm_hiddens + 1)), np.sqrt(6 / (self.args.lstm_hiddens + 1)))

    def cat_embedding(self, x):
        # print("source", x)
        batch = x.size(0)
        word_size = x.size(1)
        cated_embed = torch.zeros(batch, word_size, self.args.embed_dim * self.cat_size)
        for id_batch in range(batch):
            batch_word_list = np.array(x[id_batch].data).tolist()
            batch_word_list.insert(0, [0] * self.args.embed_dim)
            batch_word_list.insert(1, [0] * self.args.embed_dim)
            batch_word_list.insert(word_size, [0] * self.args.embed_dim)
            batch_word_list.insert(word_size + 1, [0] * self.args.embed_dim)
            batch_word_embed = torch.from_numpy(np.array(batch_word_list)).type(torch.FloatTensor)
            cat_list = []
            for id_word in range(word_size):
                cat_list.append(torch.cat(batch_word_embed[id_word:(id_word + self.cat_size)]).unsqueeze(0))
            sentence_cated_embed = torch.cat(cat_list)
            cated_embed[id_batch] = sentence_cated_embed
        if self.args.use_cuda is True:
            cated_embed = Variable(cated_embed).cuda()
        else:
            cated_embed = Variable(cated_embed)
        # print("cated", cated_embed)
        return cated_embed

    def forward(self, batch_features):
        word = batch_features.word_features
        # print(word)
        # print(self.args.create_alphabet.word_alphabet.from_id(word.data[0][0]))

        x = self.embed(word)  # (N,W,D)
        # cated_embed = self.cat_embedding(x)
        cated_embed = self.dropout_embed(x)
        x, _ = self.bilstm(cated_embed)
        # cated_embed = self.batchNorm(cated_embed.permute(0, 2, 1))
        # cated_embed = F.tanh(cated_embed)
        logit = self.linear(x)
        # print(logit.size())
        return logit

