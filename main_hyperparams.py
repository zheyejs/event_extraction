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
from models.BiLSTM_Context import *
from models.BiLSTM import BiLSTM
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
    print("train sentence {}, dev sentence {}, test sentence {}.".format(len(train_data), len(dev_data), len(test_data)))

    # create the alphabet
    create_alphabet = CreateAlphabet(min_freq=config.min_freq)
    if config.embed_finetune is False:
        create_alphabet.build_vocab(train_data=train_data, dev_data=dev_data, test_data=test_data)
    if config.embed_finetune is True:
        create_alphabet.build_vocab(train_data=train_data)

    # create iterator
    create_iter = Iterators()
    train_iter, dev_iter, test_iter = create_iter.createIterator(
        # batch_size=[config.batch_size, len(dev_data), len(test_data)],
        batch_size=[config.batch_size, config.dev_batch_size, config.test_batch_size],
        data=[train_data, dev_data, test_data], operator=create_alphabet,
        config=config)
    return train_iter, dev_iter, test_iter, create_alphabet


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


def pre_embed(config, create_alphabet):
    pretrain_embed = None
    if config.pretrained_embed and config.zeros:
        print("Using Pre_Trained Embedding.")
        pretrain_embed = load_pretrained_emb_zeros(path=config.pretrained_embed_file,
                                                   text_field_words_dict=create_alphabet.word_alphabet.id2words,
                                                   pad=paddingkey)
    elif config.pretrained_embed and config.avg:
        print("Using Pre_Trained Embedding.")
        pretrain_embed = load_pretrained_emb_avg(path=config.pretrained_embed_file,
                                                 text_field_words_dict=create_alphabet.word_alphabet.id2words,
                                                 pad=paddingkey)
    elif config.pretrained_embed and config.uniform:
        print("Using Pre_Trained Embedding.")
        pretrain_embed = load_pretrained_emb_uniform(path=config.pretrained_embed_file,
                                                     text_field_words_dict=create_alphabet.word_alphabet.id2words,
                                                     pad=paddingkey)
    elif config.pretrained_embed and config.nnembed:
        print("Using Pre_Trained Embedding.")
        pretrain_embed = load_pretrained_emb_Embedding(path=config.pretrained_embed_file,
                                                       text_field_words_dict=create_alphabet.word_alphabet.id2words,
                                                       pad=paddingkey)
    return pretrain_embed


def get_learning_algorithm(config):
    algorithm = None
    if config.adam is True:
        algorithm = "Adam"
    elif config.sgd is True:
        algorithm = "SGD"
    print("learning algorithm is {}.".format(algorithm))
    return algorithm


def get_params(config, create_alphabet):
    # get algorithm
    # config.learning_algorithm = get_learning_algorithm(config)

    # get params
    config.embed_num = create_alphabet.word_alphabet.vocab_size
    config.class_num = create_alphabet.label_alphabet.vocab_size
    config.paddingId = create_alphabet.word_paddingId
    config.label_paddingId = create_alphabet.label_paddingId
    config.create_alphabet = create_alphabet
    print("embed_num : {}, class_num : {}".format(config.embed_num, config.class_num))
    print("PaddingID {}".format(config.paddingId))


def load_model(config):
    model = None
    if config.model_bilstm is True:
        print("loading BiLSTM model......")
        model = BiLSTM(config)
    if config.model_bilstm_context is True:
        print("loading BiLSTM_Context model.....")
        model = BiLSTM_Context(config)
    print(model)
    if config.use_cuda is True:
        model = model.cuda()
    print(model)
    return model


def start_train(train_iter, dev_iter, test_iter, model, config):
    print("\nTraining Start......")
    train.train(train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter, model=model, config=config)


def main():
    # save file
    config.mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config.save_dir = os.path.join(config.save_direction, config.mulu)
    if not os.path.isdir(config.save_dir):
        os.makedirs(config.save_dir)

    # get iter
    create_alphabet = None
    train_iter, dev_iter, test_iter, create_alphabet = load_Data(config)

    # get params
    get_params(config=config, create_alphabet=create_alphabet)

    # load Pre_Trained Embedding
    config.pretrained_weight = pre_embed(config=config, create_alphabet=create_alphabet)

    # save dictionary
    save_dictionary(config=config)

    model = load_model(config)

    # print("Training Start......")
    start_train(train_iter, dev_iter, test_iter, model, config)


if __name__ == "__main__":

    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    parser = argparse.ArgumentParser(description="Chinese NER & POS")
    parser.add_argument('--config_file', default="./Config/config.cfg")
    args = parser.parse_args()

    config = configurable.Configurable(config_file=args.config_file)
    if config.use_cuda is True:
        print("Using GPU To Train......")
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        print("torch.cuda.initial_seed", torch.cuda.initial_seed())

    main()

