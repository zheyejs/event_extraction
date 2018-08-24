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
from optparse import OptionParser
import Config.config as configurable
from DataUtils.Alphabet import *
from DataUtils.Batch_Iterator import *
from DataUtils.Pickle import pcl
from Dataloader import DataLoader_NER
from DataUtils.Load_Pretrained_Embed import *
from DataUtils.Common import seed_num, paddingkey
from models.BiLSTM_Context import *
from models.BiLSTM import BiLSTM
from test import load_test_model, load_test_data
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
def preprocessing(config):
    print("Processing Data......")
    # read file
    data_loader = DataLoader_NER.DataLoader(config)
    train_data, dev_data, test_data = data_loader.dataLoader(path=[config.train_file, config.dev_file, config.test_file],
                                                             shuffle=config.shuffle)
    print("train sentence {}, dev sentence {}, test sentence {}.".format(len(train_data), len(dev_data), len(test_data)))
    data_dict = {"train_data": train_data, "dev_data": dev_data, "test_data": test_data}
    pcl.save(obj=data_dict, path=os.path.join(config.pkl_directory, config.pkl_data))

    # create the alphabet
    alphabet = CreateAlphabet(min_freq=config.min_freq)
    if config.embed_finetune is False:
        alphabet.build_vocab(train_data=train_data, dev_data=dev_data, test_data=test_data)
    if config.embed_finetune is True:
        alphabet.build_vocab(train_data=train_data)
    alphabet_dict = {"alphabet": alphabet}
    pcl.save(obj=alphabet_dict, path=os.path.join(config.pkl_directory, config.pkl_alphabet))

    # create iterator
    create_iter = Iterators()
    train_iter, dev_iter, test_iter = create_iter.createIterator(
        # batch_size=[config.batch_size, len(dev_data), len(test_data)],
        batch_size=[config.batch_size, config.dev_batch_size, config.test_batch_size],
        data=[train_data, dev_data, test_data], operator=alphabet,
        config=config)
    iter_dict = {"train_iter": train_iter, "dev_iter": dev_iter, "test_iter": test_iter}
    pcl.save(obj=iter_dict, path=os.path.join(config.pkl_directory, config.pkl_iter))
    return train_iter, dev_iter, test_iter, alphabet


def save_dict2file(dict, path):
    print("Saving dictionary")
    if os.path.exists(path):
        print("path {} is exist, deleted.".format(path))
    file = open(path, encoding="UTF-8", mode="w")
    for word, index in dict.items():
        # print(word, index)
        file.write(str(word) + "\t" + str(index) + "\n")
    file.close()
    print("Save dictionary finished.")


def save_dictionary(config):
    if config.save_dict is True:
        if os.path.exists(config.dict_directory):
            shutil.rmtree(config.dict_directory)
        if not os.path.isdir(config.dict_directory):
            os.makedirs(config.dict_directory)

        config.word_dict_path = "/".join([config.dict_directory, config.word_dict])
        config.label_dict_path = "/".join([config.dict_directory, config.label_dict])
        print("word_dict_path : {}".format(config.word_dict_path))
        print("label_dict_path : {}".format(config.label_dict_path))
        save_dict2file(config.create_alphabet.word_alphabet.words2id, config.word_dict_path)
        save_dict2file(config.create_alphabet.label_alphabet.words2id, config.label_dict_path)
        # copy to mulu
        print("copy dictionary to {}".format(config.save_dir))
        shutil.copytree(config.dict_directory, "/".join([config.save_dir, config.dict_directory]))


def pre_embed(config, alphabet):
    print("***************************************")
    pretrain_embed = None
    if config.pretrained_embed and config.zeros:
        print("Using Pre_Trained Embedding.")
        pretrain_embed = load_pretrained_emb_zeros(path=config.pretrained_embed_file,
                                                   text_field_words_dict=alphabet.word_alphabet.id2words,
                                                   pad=paddingkey)
    elif config.pretrained_embed and config.avg:
        print("Using Pre_Trained Embedding.")
        pretrain_embed = load_pretrained_emb_avg(path=config.pretrained_embed_file,
                                                 text_field_words_dict=alphabet.word_alphabet.id2words,
                                                 pad=paddingkey)
    elif config.pretrained_embed and config.uniform:
        print("Using Pre_Trained Embedding.")
        pretrain_embed = load_pretrained_emb_uniform(path=config.pretrained_embed_file,
                                                     text_field_words_dict=alphabet.word_alphabet.id2words,
                                                     pad=paddingkey)
    elif config.pretrained_embed and config.nnembed:
        print("Using Pre_Trained Embedding.")
        pretrain_embed = load_pretrained_emb_Embedding(path=config.pretrained_embed_file,
                                                       text_field_words_dict=alphabet.word_alphabet.id2words,
                                                       pad=paddingkey)
    embed_dict = {"pretrain_embed": pretrain_embed}
    pcl.save(obj=embed_dict, path=os.path.join(config.pkl_directory, config.pkl_embed))
    return pretrain_embed


def get_learning_algorithm(config):
    algorithm = None
    if config.adam is True:
        algorithm = "Adam"
    elif config.sgd is True:
        algorithm = "SGD"
    print("learning algorithm is {}.".format(algorithm))
    return algorithm


def get_params(config, alphabet):
    # get algorithm
    config.learning_algorithm = get_learning_algorithm(config)

    # save best model path
    config.save_best_model_path = config.save_best_model_dir
    if config.test is False:
        if os.path.exists(config.save_best_model_path):
            shutil.rmtree(config.save_best_model_path)

    # get params
    config.embed_num = alphabet.word_alphabet.vocab_size
    config.class_num = alphabet.label_alphabet.vocab_size
    config.paddingId = alphabet.word_paddingId
    config.label_paddingId = alphabet.label_paddingId
    config.create_alphabet = alphabet
    print("embed_num : {}, class_num : {}".format(config.embed_num, config.class_num))
    print("PaddingID {}".format(config.paddingId))


def load_model(config):
    print("***************************************")
    model = None
    if config.model_bilstm is True:
        print("loading BiLSTM model......")
        model = BiLSTM(config)
    if config.model_bilstm_context is True:
        print("loading BiLSTM_Context model.....")
        model = BiLSTM_Context(config)
    if config.use_cuda is True:
        model = model.cuda()
    if config.test is True:
        model = load_test_model(model, config)
    print(model)
    return model


def load_data(config):
    print("load data for process or pkl data.")
    train_iter, dev_iter, test_iter = None, None, None
    alphabet = None
    if (config.train is True) and (config.process is True):
        print("process data")
        if os.path.exists(config.pkl_directory): shutil.rmtree(config.pkl_directory)
        if not os.path.isdir(config.pkl_directory): os.makedirs(config.pkl_directory)
        train_iter, dev_iter, test_iter, alphabet = preprocessing(config)
        # load Pre_Trained Embedding
        config.pretrained_weight = pre_embed(config=config, alphabet=alphabet)
    elif ((config.train is True) and (config.process is False)) or (config.test is True):
        print("load data from pkl file")
        # load alphabet from pkl
        alphabet_dict = pcl.load(path=os.path.join(config.pkl_directory, config.pkl_alphabet))
        print(alphabet_dict.keys())
        alphabet = alphabet_dict["alphabet"]
        # load iter from pkl
        iter_dict = pcl.load(path=os.path.join(config.pkl_directory, config.pkl_iter))
        print(iter_dict.keys())
        train_iter, dev_iter, test_iter = iter_dict.values()
        # train_iter, dev_iter, test_iter = iter_dict["train_iter"], iter_dict["dev_iter"], iter_dict["test_iter"]
        # load embed from pkl
        embed_dict = pcl.load(os.path.join(config.pkl_directory, config.pkl_embed))
        print(embed_dict.keys())
        embed = embed_dict["pretrain_embed"]
        config.pretrained_weight = embed

    return train_iter, dev_iter, test_iter, alphabet


def start_train(train_iter, dev_iter, test_iter, model, config):
    print("\nTraining Start......")
    train.train(train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter, model=model, config=config)


def start_test(train_iter, dev_iter, test_iter, model, alphabet, config):
    print("\nTesting Start......")
    data, path_source, path_result = load_test_data(train_iter, dev_iter, test_iter, config)
    print(data, path_source, path_result)
    # train.train(train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter, model=model, config=config)


def main():
    # save file
    config.mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # config.add_args(key="mulu", value=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    config.save_dir = os.path.join(config.save_direction, config.mulu)
    if not os.path.isdir(config.save_dir): os.makedirs(config.save_dir)

    # get data, iter, alphabet
    train_iter, dev_iter, test_iter, alphabet = load_data(config=config)

    # get params
    get_params(config=config, alphabet=alphabet)

    # save dictionary
    save_dictionary(config=config)

    model = load_model(config)

    # print("Training Start......")
    if config.train is True:
        start_train(train_iter, dev_iter, test_iter, model, config)
        exit()
    elif config.test is True:
        start_test(train_iter, dev_iter, test_iter, model, alphabet, config)
        exit()


def parse_argument():
    parser = argparse.ArgumentParser(description="NER & POS")
    parser.add_argument("-c", "--config", dest="config_file", type=str, default="./Config/config.cfg",
                        help="config path")
    parser.add_argument("--train", dest="train", action="store_true", default=True, help="train model")
    parser.add_argument("-p", "--process", dest="process", action="store_true", default=False, help="data process")
    parser.add_argument("-t", "--test", dest="test", action="store_true", default=True, help="test model")
    parser.add_argument("--t_model", dest="t_model", type=str, default=None, help="model for test")
    parser.add_argument("--t_data", dest="t_data", type=str, default=None,
                        help="data[train dev test None] for test model")
    parser.add_argument("--predict", dest="predict", action="store_true", default=False, help="predict model")
    args = parser.parse_args()
    # print(vars(args))
    config = configurable.Configurable(config_file=args.config_file)
    config.train = args.train
    config.process = args.process
    config.test = args.test
    config.t_model = args.t_model
    config.t_data = args.t_data
    config.predict = args.predict
    # config
    if config.test is True:
        config.train = False
    if config.t_data not in [None, "train", "dev", "test"]:
        print("\nUsage")
        parser.print_help()
        print("t_data : {}, not in [None, 'train', 'dev', 'test']".format(config.t_data))
        exit()
    print("***************************************")
    print("Data Process : {}".format(config.process))
    print("Train model : {}".format(config.train))
    print("Test model : {}".format(config.test))
    print("t_model : {}".format(config.t_model))
    print("t_data : {}".format(config.t_data))
    print("predict : {}".format(config.predict))
    print("***************************************")

    return config


if __name__ == "__main__":

    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    config = parse_argument()
    if config.use_cuda is True:
        print("Using GPU To Train......")
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        print("torch.cuda.initial_seed", torch.cuda.initial_seed())

    main()

