# @Author : bamtercelboo
# @Datetime : 2018/9/14 8:43
# @File : Sequence_Label.py
# @Last Modify Time : 2018/9/14 8:43
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Sequence_Label.py
    FUNCTION : None
"""

import torch
import torch.nn as nn
import random
import numpy as np
import time
from models.BiLSTM import BiLSTM
from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Sequence_Label(nn.Module):
    """
        Sequence_Label
    """

    def __init__(self, config):
        super(Sequence_Label, self).__init__()
        self.config = config
        # embed
        self.embed_num = config.embed_num
        self.embed_dim = config.embed_dim
        self.label_num = config.class_num
        self.paddingId = config.paddingId
        # dropout
        self.dropout_emb = config.dropout_emb
        self.dropout = config.dropout
        # lstm
        self.lstm_hiddens = config.lstm_hiddens
        self.lstm_layers = config.lstm_layers
        # pre train
        self.pretrained_embed = config.pretrained_embed
        self.pretrained_weight = config.pretrained_weight

        self.encoder_model = BiLSTM(embed_num=self.embed_num, embed_dim=self.embed_dim, label_num=self.label_num,
                                    paddingId=self.paddingId, dropout_emb=self.dropout_emb, dropout=self.dropout,
                                    lstm_hiddens=self.lstm_hiddens, lstm_layers=self.lstm_layers,
                                    pretrained_embed=self.pretrained_embed, pretrained_weight=self.pretrained_weight)

    def forward(self, word, sentence_length, desorted_indices, train=False):
        """
        :param word:
        :param sentence_length:
        :param desorted_indices:
        :param train:
        :return:
        """
        x = self.encoder_model(word, sentence_length, desorted_indices)
        return x


