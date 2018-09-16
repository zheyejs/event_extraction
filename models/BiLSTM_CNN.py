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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from models.initialize import *
from DataUtils.Common import *
from models.modelHelp import prepare_pack_padded_sequence
torch.manual_seed(seed_num)
random.seed(seed_num)


class BiLSTM_CNN(nn.Module):
    """
        BiLSTM_CNN
    """

    def __init__(self, **kwargs):
        super(BiLSTM_CNN, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        V = self.embed_num
        D = self.embed_dim
        C = self.label_num
        paddingId = self.paddingId
        char_paddingId = self.char_paddingId

        # word embedding layer
        self.embed = nn.Embedding(V, D, padding_idx=paddingId)
        if self.pretrained_embed:
            self.embed.weight.data.copy_(self.pretrained_weight)

        # char embedding layer
        self.char_embedding = nn.Embedding(self.char_embed_num, self.char_dim, padding_idx=char_paddingId)
        # init_embedding(self.char_embedding.weight)

        # dropout
        self.dropout_embed = nn.Dropout(self.dropout_emb)
        self.dropout = nn.Dropout(self.dropout)

        # cnn
        self.char_encoders = nn.ModuleList()
        for i, filter_size in enumerate(self.conv_filter_sizes):
            f = nn.Conv3d(in_channels=1, out_channels=self.conv_filter_nums[i], kernel_size=(1, filter_size, self.char_dim))
            self.char_encoders.append(f)
        lstm_input_dim = D + sum(self.conv_filter_nums)
        self.bilstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=self.lstm_hiddens, num_layers=self.lstm_layers,
                              bidirectional=True, batch_first=True, bias=True)

        self.linear = nn.Linear(in_features=self.lstm_hiddens * 2, out_features=C, bias=True)
        # init.xavier_uniform(self.linear.weight)
        # self.linear.bias.data.uniform_(-np.sqrt(6 / (config.lstm_hiddens * 2 + 1)), np.sqrt(6 / (config.lstm_hiddens * 2 + 1)))

    def _char_forward(self, inputs):
        """
        Args:
            inputs: 3D tensor, [bs, max_len, max_len_char]

        Returns:
            char_conv_outputs: 3D tensor, [bs, max_len, output_dim]
        """
        # print(inputs.size())
        max_len, max_len_char = inputs.size(1), inputs.size(2)
        inputs = inputs.view(-1, max_len * max_len_char)  # [bs, -1]
        input_embed = self.char_embedding(inputs)  # [bs, ml*ml_c, feature_dim]
        # [bs, 1, max_len, max_len_char, feature_dim]
        # print(input_embed.size())
        input_embed = input_embed.view(-1, 1, max_len, max_len_char, self.char_dim)
        # print(input_embed.size())
        # conv
        char_conv_outputs = []
        for char_encoder in self.char_encoders:
            conv_output = char_encoder(input_embed)
            # print(conv_output.size())
            # print(torch.max(conv_output, -2)[0].size())
            pool_output = torch.squeeze(torch.max(conv_output, -2)[0], -1)
            # print(pool_output.size())
            char_conv_outputs.append(pool_output)
        char_conv_outputs = torch.cat(char_conv_outputs, dim=1)
        # exit()

        # size=[bs, max_len, output_dim]
        char_conv_outputs = char_conv_outputs.transpose(-2, -1).contiguous()

        return char_conv_outputs

    def forward(self, word, char, sentence_length):
        """
        :param char:
        :param word:
        :param sentence_length:
        :return:
        """
        char_conv = self._char_forward(char)
        word = self.embed(word)  # (N,W,D)
        # x = torch.cat([word, char_conv], -1)
        # print(word.size())
        # print(char_conv.size())
        x = torch.cat((word, char_conv), -1)
        # print(char_conv.size())
        # print(word.size())
        # print(x.size())
        # exit()

        # word, sentence_length, desorted_indices = prepare_pack_padded_sequence(word, sentence_length, use_cuda=self.use_cuda)

        x = self.dropout_embed(x)
        # packed_embed = pack_padded_sequence(x, sentence_length, batch_first=True)
        x, _ = self.bilstm(x)
        # x, _ = pad_packed_sequence(x, batch_first=True)
        # x = x[desorted_indices]
        x = self.dropout(x)
        # x = F.tanh(x)
        logit = self.linear(x)
        return logit
