# @Author : bamtercelboo
# @Datetime : 2018/1/30 15:58
# @File : DataConll2003_Loader.py
# @Last Modify Time : 2018/1/30 15:58
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  DataConll2003_Loader.py
    FUNCTION :
        Loading conll2003 dataset
        Download From Here : https://www.clips.uantwerpen.be/conll2003/ner/
"""
import sys
import re
import random
import torch
from DataUtils.Instance import Instance

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class DataLoader():
    def __init__(self):
        print("Loading Data......")
        self.data_list = []

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

    def dataLoader(self, path=None, shuffle=False):
        assert isinstance(path, list), "Path Must Be In List"
        print("Data Path {}".format(path))
        for id_data in range(len(path)):
            print("Loading Data Form {}".format(path[id_data]))
            insts = self.Load_Each_Data(path=path[id_data], shuffle=shuffle)
            if shuffle is True:
                print("shuffle data......")
                random.shuffle(insts)
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
                    inst.words.append(word.lower())
                    inst.labels.append(line[1])
                # if len(insts) == 32:
                #     break
            if len(inst.words) != 0:
                inst.words_size = len(inst.words)
                insts.append(inst)
            # print("\n")
        return insts


if __name__ == "__main__":
    path = ["../Data/test/test.txt", "../Data/test/test.txt", "../Data/test/test.txt"]
    conll2000data = DataLoader()
    conll2000data.dataLoader(path=path, shuffle=True)