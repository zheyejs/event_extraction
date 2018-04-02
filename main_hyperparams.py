# @Author : bamtercelboo
# @Datetime : 2018/1/30 19:50
# @File : main_hyperparams.py.py
# @Last Modify Time : 2018/1/30 19:50
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  main_hyperparams.py.py
    FUNCTION : main
"""

import os
import sys
import argparse
import datetime
import torch
import Dataloader.config as config
from Dataloader.Alphabet import *
from Dataloader.Batch_Iterator import *
from Dataloader import DataConll2000_Loader_Chunking
from Dataloader import DataConll2000_Loader_POS
from Dataloader import DataConll2003_Loader_NER
from Dataloader import DataConll2003_Loader_Chunking
from Dataloader.Load_Pretrained_Embed import *
from Dataloader.Common import unkkey, paddingkey
from models.model_PNC import *
import random
import shutil
import hyperparams as hy
# solve default encoding problem
from imp import reload
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# random seed
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)

# init hyperparams instance



# load data / create alphabet / create iterator
def load_Conll2000_Chunking(args):
    print("Loading Conll2000 Chunking Data......")
    # read file
    data_loader = DataConll2000_Loader_Chunking.DataLoader()
    train_data, test_data = data_loader.dataLoader(path=[args.train_path, args.test_path], shuffle=args.shuffle)

    # create the alphabet
    create_alphabet = CreateAlphabet(min_freq=args.min_freq)
    create_alphabet.build_vocab(train_data=train_data, test_data=test_data)

    # create iterator
    create_iter = Iterators()
    train_iter, test_iter = create_iter.createIterator(batch_size=[args.batch_size, len(test_data)],
                                                       data=[train_data, test_data], operator=create_alphabet,
                                                       args=args)
    return train_iter, test_iter, create_alphabet


# load data / create alphabet / create iterator
def load_Conll2000_POS(args):
    print("Loading Conll2000 POS Data......")
    # read file
    data_loader = DataConll2000_Loader_POS.DataLoader()
    train_data, test_data = data_loader.dataLoader(path=[args.train_path, args.test_path], shuffle=args.shuffle)

    # create the alphabet
    create_alphabet = CreateAlphabet(min_freq=args.min_freq)
    create_alphabet.build_vocab(train_data=train_data, test_data=test_data)

    # create iterator
    create_iter = Iterators()
    train_iter, test_iter = create_iter.createIterator(batch_size=[args.batch_size, len(test_data)],
                                                       data=[train_data, test_data], operator=create_alphabet,
                                                       args=args)
    return train_iter, test_iter, create_alphabet


# load data / create alphabet / create iterator
def load_Conll2003_NER(args):
    print("Loading Conll2003 NER Data......")
    # read file
    data_loader = DataConll2003_Loader_NER.DataLoader()
    train_data, dev_data, test_data = data_loader.dataLoader(path=[args.train_path, args.dev_path, args.test_path],
                                                             shuffle=args.shuffle)

    # create the alphabet
    create_alphabet = CreateAlphabet(min_freq=args.min_freq)
    create_alphabet.build_vocab(train_data=train_data, dev_data=dev_data, test_data=test_data)

    # create iterator
    create_iter = Iterators()
    train_iter, dev_iter, test_iter = create_iter.createIterator(batch_size=[args.batch_size, len(dev_data), len(test_data)],
                                                       data=[train_data, dev_data, test_data], operator=create_alphabet,
                                                       args=args)
    return train_iter, dev_iter, test_iter, create_alphabet


# load data / create alphabet / create iterator
def load_Conll2003_Chunking(args):
    print("Loading Conll2003 Chunking Data......")
    # read file
    data_loader = DataConll2003_Loader_Chunking.DataLoader()
    train_data, dev_data, test_data = data_loader.dataLoader(path=[args.train_path, args.dev_path, args.test_path],
                                                             shuffle=args.shuffle)

    # create the alphabet
    create_alphabet = CreateAlphabet(min_freq=args.min_freq)
    create_alphabet.build_vocab(train_data=train_data, dev_data=dev_data, test_data=test_data)

    # create iterator
    create_iter = Iterators()
    train_iter, dev_iter, test_iter = create_iter.createIterator(batch_size=[args.batch_size, len(dev_data), len(test_data)],
                                                       data=[train_data, dev_data, test_data], operator=create_alphabet,
                                                       args=args)
    return train_iter, dev_iter, test_iter, create_alphabet


def show_params():
    print("\nParameters:")
    if os.path.exists("./Parameters.txt"):
        os.remove("./Parameters.txt")
    file = open("Parameters.txt", "a", encoding="UTF-8")
    for attr, value in sorted(args.__dict__.items()):
        if attr.upper() != "PRETRAINED_WEIGHT":
            print("\t{}={}".format(attr.upper(), value))
        file.write("\t{}={}\n".format(attr.upper(), value))
    file.close()
    shutil.copy("./Parameters.txt", args.save_dir)
    shutil.copy("./hyperparams.py", args.save_dir)


def main():
    # save file
    config.mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config.save_dir = os.path.join(config.save_direction, config.mulu)
    if not os.path.isdir(config.save_dir):
        os.makedirs(config.save_dir)

    # get iter
    create_alphabet = None
    if args.Conll2000 is True and args.Chunking is True:
        train_iter, test_iter, create_alphabet = load_Conll2000_Chunking(args)
    if args.Conll2000 is True and args.POS is True:
        train_iter, test_iter, create_alphabet = load_Conll2000_POS(args)
    if args.Conll2003 is True and args.NER is True:
        train_iter, dev_iter, test_iter, create_alphabet = load_Conll2003_NER(args)
    if args.Conll2003 is True and args.Chunking is True:
        train_iter, dev_iter, test_iter, create_alphabet = load_Conll2003_Chunking(args)

    args.embed_num = create_alphabet.word_alphabet.vocab_size
    args.class_num = create_alphabet.label_alphabet.vocab_size
    args.paddingId = create_alphabet.word_paddingId
    args.create_alphabet = create_alphabet
    print("embed_num : {}, class_num : {}".format(args.embed_num, args.class_num))
    print("PaddingID {}".format(args.paddingId))

    if args.word_Embedding:
        print("Using Pre_Trained Embedding.")
        pretrain_embed = load_pretrained_emb_zeros(path=args.word_Embedding_Path,
                                                   text_field_words_dict=create_alphabet.word_alphabet.id2words,
                                                   pad=paddingkey)
        # calculate_oov(path=args.word_Embedding_Path, text_field_words_dict=text_field.vocab.itos,
        #               pad=text_field.pad_token)
        args.pretrained_weight = pretrain_embed

    # print params
    show_params()

    model = None
    if args.model_PNC is True:
        print("loading PNC(POS,NER,Chunking) model.....")
        model = PNC(args)
        shutil.copy("./models/model_PNC.py", args.save_dir)
        print(model)
        if args.use_cuda is True:
            print("Using Cuda Speed Up......")
            model = model.cuda()
        print("Training Start......")
        if os.path.exists("./Test_Result.txt"):
            os.remove("./Test_Result.txt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Chinese NER & POS")
    parser.add_argument('--config_file', default="./Config/config.cfg")
    args = parser.parse_args()

    config = config.Configurable(config_file=args.config_file)
    if config.use_cuda is False:
        print("Using GPU To Train......")

    main()

