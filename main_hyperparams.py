# @Author : bamtercelboo
# @Datetime : 2018/1/30 19:50
# @File : main_hyperparams.py.py
# @Last Modify Time : 2018/1/30 19:50
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  main_hyperparams.py.py
    FUNCTION : main
"""

import argparse
import datetime
import Config.config as configurable
from DataUtils.Alphabet import *
from DataUtils.Batch_Iterator import *
from Dataloader import DataLoader_NER
from DataUtils.Load_Pretrained_Embed import *
from DataUtils.Common import seed_num, paddingkey
from models.model_PNC import *
import train
import random
import shutil

# solve default encoding problem
from imp import reload
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# random seed
torch.manual_seed(seed_num)
random.seed(seed_num)


# load data / create alphabet / create iterator
def load_Data(config):
    print("Loading Data......")
    # read file
    data_loader = DataLoader_NER.DataLoader()
    train_data, dev_data, test_data = data_loader.dataLoader(path=[config.train_file, config.dev_file, config.test_file],
                                                             shuffle=config.shuffle)

    # create the alphabet
    create_alphabet = CreateAlphabet(min_freq=config.min_freq)
    # create_alphabet.build_vocab(train_data=train_data, dev_data=dev_data, test_data=test_data)
    create_alphabet.build_vocab(train_data=train_data)

    # create iterator
    create_iter = Iterators()
    train_iter, dev_iter, test_iter = create_iter.createIterator(
        # batch_size=[config.batch_size, len(dev_data), len(test_data)],
        batch_size=[config.batch_size, config.batch_size, config.batch_size],
        data=[train_data, dev_data, test_data], operator=create_alphabet,
        args=config)
    return train_iter, dev_iter, test_iter, create_alphabet


def main():
    # save file
    config.mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config.save_dir = os.path.join(config.save_direction, config.mulu)
    if not os.path.isdir(config.save_dir):
        os.makedirs(config.save_dir)

    # get iter
    create_alphabet = None
    train_iter, dev_iter, test_iter, create_alphabet = load_Data(config)

    config.embed_num = create_alphabet.word_alphabet.vocab_size
    config.class_num = create_alphabet.label_alphabet.vocab_size
    config.paddingId = create_alphabet.word_paddingId
    config.label_paddingId = create_alphabet.label_paddingId
    config.create_alphabet = create_alphabet
    print("embed_num : {}, class_num : {}".format(config.embed_num, config.class_num))
    print("PaddingID {}".format(config.paddingId))

    if config.pretrained_embed:
        print("Using Pre_Trained Embedding.")
        pretrain_embed = load_pretrained_emb_zeros(path=config.pretrained_embed_file,
                                                   text_field_words_dict=create_alphabet.word_alphabet.id2words,
                                                   pad=paddingkey)
        config.pretrained_weight = pretrain_embed

    model = None
    if config.model_BiLstm is True:
        print("loading model.....")
        model = PNC(config)
        # shutil.copytree("./models", config.save_dir)
        print(model)
        if config.use_cuda is True:
            model = model.cuda()
        print("Training Start......")
        if os.path.exists("./Test_Result.txt"):
            os.remove("./Test_Result.txt")

    train.train(train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter, model=model, args=config)
    # train.train(train_iter=train_iter, dev_iter=train_iter, test_iter=train_iter, model=model, args=config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Chinese NER & POS")
    parser.add_argument('--config_file', default="./Config/config.cfg")
    args = parser.parse_args()

    config = configurable.Configurable(config_file=args.config_file)
    if config.use_cuda is False:
        print("Using GPU To Train......")

    main()

