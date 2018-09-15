# @Author : bamtercelboo
# @Datetime : 2018/9/15 19:09
# @File : modelHelp.py
# @Last Modify Time : 2018/9/15 19:09
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  modelHelp.py
    FUNCTION : None
"""

import torch
import random
from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


def prepare_pack_padded_sequence(inputs_words, seq_lengths, use_cuda=False, descending=True):
    """
    :param use_cuda:
    :param inputs_words:
    :param seq_lengths:
    :param descending:
    :return:
    """
    # if use_cuda is True:
    #     sorted_seq_lengths, indices = torch.sort(torch.cuda.LongTensor(seq_lengths), descending=descending)
    # else:
    sorted_seq_lengths, indices = torch.sort(torch.LongTensor(seq_lengths), descending=descending)
    if use_cuda is True:
        sorted_seq_lengths, indices = sorted_seq_lengths.cuda(), indices.cuda()
    _, desorted_indices = torch.sort(indices, descending=False)
    sorted_inputs_words = inputs_words[indices]
    return sorted_inputs_words, sorted_seq_lengths.cpu().numpy(), desorted_indices


