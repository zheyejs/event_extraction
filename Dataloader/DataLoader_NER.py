# @Author : bamtercelboo
# @Datetime : 2018/1/30 15:58
# @File : DataConll2003_Loader.py
# @Last Modify Time : 2018/1/30 15:58
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :
    FUNCTION :
"""
import re
import random
import torch
from Dataloader.Instance import Instance

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class DataLoader(object):
    def __init__(self, config):
        print("Loading Data......")
        self.data_list = []
        self.max_count = config.max_count

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def dataLoader(self, path=None, shuffle=True):
        assert isinstance(path, list), "Path Must Be In List"
        print("Data Path {}".format(path))
        for id_data in range(len(path)):
            print("Loading Data Form {}".format(path[id_data]))
            insts = self.Load_Each_Data(path=path[id_data], shuffle=shuffle)
            if shuffle is True:
                print("shuffle data......")
                random.shuffle(insts)
            # sorted(inst)
            # sorted_insts = self.sort(insts)
            # sorted_insts = self.sort(insts)
            self.data_list.append(insts)
        # return train/dev/test data
        if len(self.data_list) == 3:
            return self.data_list[0], self.data_list[1], self.data_list[2]
        elif len(self.data_list) == 2:
            return self.data_list[0], self.data_list[1]

    def Load_Each_Data(self, path=None, shuffle=False):
        assert path is not None, "The Data Path Is Not Allow Empty."
        insts = []
        with open(path, encoding="UTF-8") as f:
            inst = Instance()
            for line in f.readlines():
                line = line.strip()
                if line == "" and len(inst.words) != 0:
                    inst.words_size = len(inst.words)
                    insts.append(inst)
                    inst = Instance()
                else:
                    line = line.strip().split(" ")
                    # print(line)
                    word = line[0]
                    # print(word)
                    # word = self.normalize_word(word)
                    # print(word)
                    inst.words.append(word.lower())
                    inst.labels.append(line[-1])
                if len(insts) == self.max_count:
                    break
            if len(inst.words) != 0:
                inst.words_size = len(inst.words)
                insts.append(inst)
            # print("\n")
        return insts

    def normalize_word(self, word):
        new_word = ""
        for char in word:
            if char.isdigit():
                new_word += '0'
            else:
                new_word += char
        return new_word

    def sort(self, insts):
        sorted_insts  = []
        sorted_dict = {}
        for id_inst, inst in enumerate(insts):
            sorted_dict[id_inst] = inst.words_size
        dict = sorted(sorted_dict.items(), key=lambda d: d[1], reverse=True)
        for key, value in dict:
            sorted_insts.append(insts[key])
        print("Sort Finished.")
        return sorted_insts


if __name__ == "__main__":
    path = ["../Data/test/test.txt", "../Data/test/test.txt", "../Data/test/test.txt"]
    conll2000data = DataLoader()
    conll2000data.dataLoader(path=path, shuffle=True)