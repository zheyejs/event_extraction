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
import re
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable as Variable
import random
import torch.nn.init as init
import numpy as np
from Dataloader.Common import *
from Dataloader.Embed_From_Pretrained import Pretrain_Embed
import hyperparams
torch.manual_seed(hyperparams.seed_num)
random.seed(hyperparams.seed_num)


class PNC(nn.Module):

    def __init__(self, args):
        super(PNC, self).__init__()
        self.args = args
        self.cat_size = 2

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        paddingId = args.paddingId

        # self.embed = nn.Embedding(V, D, padding_idx=paddingId)
        if self.args.ininital_from_Pretrained is True:
            self.embed, self.pretrained_embed_dim = Pretrain_Embed(file=args.word_Embedding_Path,
                                                                   vocab_size=self.args.create_alphabet.pretrained_alphabet.vocab_size,
                                                                   words2id=self.args.create_alphabet.pretrained_alphabet.words2id,
                                                                   padding=paddingkey, unk=unkkey)
        self.embed.weight.requires_grad = False

        self.bilstm = nn.LSTM(input_size=500, hidden_size=100, bidirectional=False, bias=True)

        self.dropout_embed = nn.Dropout(args.dropout_embed)
        self.dropout = nn.Dropout(args.dropout)

        self.batchNorm = nn.BatchNorm1d(D * 5)

        # self.linear = nn.Linear(in_features=D, out_features=C, bias=True)
        self.linear = nn.Linear(in_features=D * 5, out_features=C, bias=True)
        init.xavier_uniform(self.linear.weight)
        self.linear.bias.data.uniform_(-np.sqrt(6 / (D + 1)), np.sqrt(6 / (D + 1)))

    def cat_embedding(self, x):
        # print("source", x)
        batch = x.size(0)
        word_size = x.size(1)
        cated_embed = torch.zeros(batch, word_size, self.args.embed_dim * 5)
        for id_batch in range(batch):
            batch_word_list = np.array(x[id_batch].data).tolist()
            batch_word_list.insert(0, [0] * self.args.embed_dim)
            batch_word_list.insert(1, [0] * self.args.embed_dim)
            batch_word_list.insert(word_size, [0] * self.args.embed_dim)
            batch_word_list.insert(word_size + 1, [0] * self.args.embed_dim)
            batch_word_embed = torch.from_numpy(np.array(batch_word_list)).type(torch.FloatTensor)
            cat_list = []
            for id_word in range(word_size):
                cat_list.append(torch.cat(batch_word_embed[id_word:(id_word + 5)]).unsqueeze(0))
            sentence_cated_embed = torch.cat(cat_list)
            cated_embed[id_batch] = sentence_cated_embed
        if self.args.use_cuda is True:
            cated_embed = Variable(cated_embed).cuda()
        else:
            cated_embed = Variable(cated_embed)
        # print("cated", cated_embed)
        return cated_embed

    def clean_conll(self, string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", "", string)
        string = re.sub(r"\'s", "", string)
        string = re.sub(r"\'ve", "", string)
        string = re.sub(r"n\'t", "", string)
        string = re.sub(r"\'re", "", string)
        string = re.sub(r"\'d", "", string)
        string = re.sub(r"\'ll", "", string)
        string = re.sub(r",", "", string)
        string = re.sub(r"!", "", string)
        string = re.sub(r"\(", "", string)
        string = re.sub(r"\)", "", string)
        string = re.sub(r"\?", "", string)
        string = re.sub(r"\s{2,}", "", string)
        return string.strip()

    def word_n_gram(self, word=None, feat_embedding_dict=None):
        feat_count = 0

        word = "<" + word + ">"
        feat_embedding_list = []
        # print("word", word)
        # if len(word) < 3:

        for feat_num in range(3, 7):
            for i in range(0, len(word) - feat_num + 1):
                feat = word[i:(i + feat_num)]
                # print("feat", feat)
                if feat in feat_embedding_dict:
                    feat_count += 1
                    # print(feat)
                    featID = feat_embedding_dict[feat]
                    # print(featID)
                    list_float = self.embed.weight.data[featID]
                    # list_float = [float(i) for i in feat_embedding_dict[feat.strip()]]
                    # print(np.array(list_float))
                    feat_embedding_list.append(np.array(list_float))
                    # feat_embedding = np.array(feat_embedding) + np.array(list_float)
        feat_embedding = np.sum(feat_embedding_list, axis=0)
        return feat_embedding, feat_count

    def handle_word_context(self, sentence=None, word=None, windows_size=5):
        # print(sentence)
        data_dict = {}
        index = (len(sentence) // 2)
        left = sentence[:index]
        right = sentence[(index + 1):]
        context_dict = {}
        for i in range(len(left)):
            if left[i] == judge_flag or left[i] == "":
                continue
            context_dict["F-" + str(len(left) - i) + "@" + left[i]] = 0
        for i in range(len(right)):
            if right[i] == judge_flag or right[i] == "":
                continue
            context_dict["F" + str(i + 1) + "@" + right[i]] = 0
        data_dict[word] = set(context_dict)
        return data_dict

    def context(self, context_dict=None, stoi=None, itos=None):
        # print(context_dict)
        context_num = 0
        context_embed_list = []
        context_embed = 0
        for context in context_dict:
            if context in stoi:
                context_num += 1
                contextID = stoi[context]
                # print("context", context)
                # print("contextID", contextID)
                list_float = self.embed.weight.data[contextID]
                # print("list_float", list_float)
                context_embed_list.append(np.array(list_float))
        context_embed = np.sum(context_embed_list, axis=0)
        return context_embed, context_num

    def handle_embedding_input(self, x):
        windows_size = 5
        itos = self.args.create_alphabet.word_alphabet.id2words
        stoi = self.args.create_alphabet.pretrained_alphabet.words2id
        feat_context_embed = torch.zeros(x.size(0), x.size(1), self.pretrained_embed_dim)
        # feat_context_embed = torch.randn(x.size(0), x.size(1), self.pretrained_embed_dim)
        for id_batch in range(x.size(0)):
            # sentence = [self.clean_str(itos[word]) for word in x.data[id_batch]]
            sentence = [itos[word] for word in x.data[id_batch]]
            sentence_set = set(sentence)
            # print(sentence)
            if paddingkey in sentence_set:
                sentence = sentence[:sentence.index(paddingkey)]

            sentence = [self.clean_conll(w) for w in sentence]

            # context_dict = self.handle_word_context(sentence=sentence, windows_size=5)
            # print("sentence", sentence)
            for id_word in range(x.size(1)):
                word = itos[x.data[id_batch][id_word]]
                # print("sentence", sentence)
                # print("word", word)
                if word != paddingkey:
                    word = self.clean_conll(word)
                    start = id_word
                    sentence_paded = []
                    for i in range((start - windows_size), (start + windows_size + 1)):
                        if i >= len(sentence):
                            break
                        if i < 0:
                            sentence_paded.append(judge_flag)
                            continue
                        else:
                            sentence_paded.extend([sentence[i]])
                    sentence_paded_len = (2 * windows_size + 1 - len(sentence_paded))
                    if sentence_paded_len > 0:
                        sentence_paded.extend([judge_flag] * sentence_paded_len)
                    # print(sentence_paded)
                    context_dict = self.handle_word_context(sentence=sentence_paded, word=word,
                                                            windows_size=windows_size)
                    # print(context_dict)
                    feat_sum_embedding, feat_ngram_num = self.word_n_gram(word=word, feat_embedding_dict=stoi)
                    n_gram_flag = True
                    if not isinstance(feat_sum_embedding, np.ndarray):
                        n_gram_flag = True
                        # continue
                        # if the word no n-gram in feature, replace with zero
                        # feat_sum_embedding = np.array(list([0] * self.pretrained_embed_dim))
                        feat_sum_embedding = np.array(np.random.uniform(-0.25, 0.25, self.pretrained_embed_dim).round(6).tolist())
                        # feat_sum_embedding = np.array(self.embed.weight.data[]))
                        feat_ngram_num = 1
                    # print(context_dict)
                    # context_embed, context_num = self.context(context_dict=context_dict[word], stoi=stoi)
                    context_embed, context_num = 0, 0
                    # context_embed /= 10
                    feat_embed = np.divide(np.add(feat_sum_embedding, context_embed),
                                           np.add(feat_ngram_num, context_num))
                    # print(feat_embed)
                    # feat_embed = np.square(feat_embed)
                    feat_context_embed[id_batch][id_word] = torch.from_numpy(feat_embed)
                    # print(feat_context_embed)
        if self.args.use_cuda is True:
            feat_context_embed = Variable(feat_context_embed).cuda()
        else:
            feat_context_embed = Variable(feat_context_embed)
        return feat_context_embed

    def forward(self, batch_features):
        word = batch_features.word_features
        x = self.handle_embedding_input(word)
        cated_embed = self.cat_embedding(x)

        # lstm_out, _ = self.bilstm(cated_embed)
        # print(lstm_out.size())
        # print(cated_embed.data)
        # file = open("./Embedding.txt", mode="a")
        # for i in range(cated_embed.size(0)):
        #     for j in range(cated_embed.size(1)):
        #         file.write(str(np.array(cated_embed.data[i][j])))
        logit = self.linear(cated_embed)
        return logit
        # file = open("./Embedding.txt", encoding="UTF-8", mode="a")
        # for i in range(x.size(0)):
        #     for j in range(x.size(1)):
        #         # print(cated_embed[i][j])
        #         file.write(str(np.array(x[i][j].data).tolist()))
        # cated_embed = self.cat_embedding(x)
        # print(cated_embed.size())
        # cated_embed = self.batchNorm(cated_embed.permute(0, 2, 1))
        # print(cated_embed.size())
        # cated_embed = F.tanh(cated_embed)
        # print(cated_embed.permute(0, 2, 1))

        # logit = self.linear(cated_embed.permute(0, 2, 1))
        # logit = self.linear(cated_embed)
        # print(logit.size())


