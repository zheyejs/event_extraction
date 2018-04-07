# @Author : bamtercelboo
# @Datetime : 2018/1/31 10:01
# @File : train.py
# @Last Modify Time : 2018/1/31 10:01
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  train.py
    FUNCTION : None
"""

import sys
import torch
import torch.nn.functional as F
import torch.nn.utils as utils
import random
import time
from DataUtils.eval import Eval, EvalPRF
from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


def train(train_iter, dev_iter, test_iter, model, config):
    if config.use_cuda:
        model.cuda()

    optimizer = None
    if config.adam is True:
        print("Adam Training......")
        if config.embed_finetune is True:
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate,
                                         weight_decay=config.weight_decay)

    if config.sgd is True:
        print("SGD Training......")
        if config.embed_finetune is True:
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate,
                                        weight_decay=config.weight_decay)

    best_fscore = Best_Result()

    steps = 0
    model_count = 0
    model.train()
    max_dev_acc = -1
    train_eval = Eval()
    dev_eval = Eval()
    test_eval = Eval()
    for epoch in range(1, config.epochs+1):
        print("\n## The {} Epoch, All {} Epochs ! ##".format(epoch, config.epochs))
        print("now lr is {}".format(optimizer.param_groups[0].get("lr")))
        start_time = time.time()
        random.shuffle(train_iter)
        model.train()
        for batch_count, batch_features in enumerate(train_iter):
            model.zero_grad()
            optimizer.zero_grad()
            # if config.use_cuda is True:
            #     batch_features.label_features = batch_features.label_features.cuda()
            logit = model(batch_features)
            getAcc(train_eval, batch_features, logit, config)
            # loss_logit = logit.view(logit.size(0) * logit.size(1), logit.size(2))
            loss = F.cross_entropy(logit.view(logit.size(0) * logit.size(1), -1), batch_features.label_features,
                                   ignore_index=config.label_paddingId)
            loss.backward()
            # if config.clip_max_norm is not None:
            #     utils.clip_grad_norm(model.parameters(), max_norm=config.clip_max_norm)
            optimizer.step()


            steps += 1
            if steps % config.log_interval == 0:
                sys.stdout.write("\rbatch_count = [{}] , loss is {:.6f}, [TAG-ACC is {:.6f}%]".format(batch_count + 1,
                                 loss.data[0], train_eval.acc()))
        end_time = time.time()
        print("\nTrain Time {:.3f}".format(end_time - start_time), end="")
        if steps is not 0:
            dev_eval.clear_PRF()
            eval_start_time = time.time()
            eval(dev_iter, model, dev_eval, best_fscore, epoch, config, test=False)
            eval_end_time = time.time()
            print("Dev Time {:.3f}".format(eval_end_time - eval_start_time))
            # model.train()
        if steps is not 0:
            test_eval.clear_PRF()
            eval_start_time = time.time()
            eval(test_iter, model, test_eval, best_fscore, epoch, config, test=True)
            eval_end_time = time.time()
            print("Test Time {:.3f}".format(eval_end_time - eval_start_time))
            # model.train()


def eval(data_iter, model, eval_instance, best_fscore, epoch, config, test=False):
    model.eval()
    # eval time
    eval_acc = Eval()
    eval_PRF = EvalPRF()
    gold_labels = []
    predict_labels = []
    for batch_features in data_iter:
        logit = model(batch_features)
        getAcc(eval_acc, batch_features, logit, config)
        for id_batch in range(batch_features.batch_length):
            inst = batch_features.inst[id_batch]
            predict_label = []
            for id_word in range(inst.words_size):
                maxId = getMaxindex(logit[id_batch][id_word], logit.size(2), config)
                predict_label.append(config.create_alphabet.label_alphabet.from_id(maxId))
            gold_labels.append(inst.labels)
            predict_labels.append(predict_label)
            eval_PRF.evalPRF(predict_labels=predict_label, gold_labels=inst.labels, eval=eval_instance)

    p, r, f = eval_instance.getFscore()
    test_flag = "Test"
    if test is False:
        print()
        test_flag = "Dev"
        if f >= best_fscore.best_dev_fscore:
            best_fscore.best_dev_fscore = f
            best_fscore.best_epoch = epoch
            best_fscore.best_test = True
    if test is True and best_fscore.best_test is True:
        best_fscore.p = p
        best_fscore.r = r
        best_fscore.f = f
    print("{} eval: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%,  [TAG-ACC = {:.6f}%]".format(test_flag, p, r, f, eval_acc.acc()))
    if test is True:
        print("The Current Best Dev F-score: {:.6f}, Locate on {} Epoch.".format(best_fscore.best_dev_fscore,
                                                                                 best_fscore.best_epoch))
        print("The Current Best Test Result: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%".format(
            best_fscore.p, best_fscore.r, best_fscore.f))
    if test is True:
        best_fscore.best_test = False


def getMaxindex(model_out, label_size, args):
    # model_out.data[0] = -9999
    max = model_out.data[0]
    maxIndex = 0
    for idx in range(1, label_size):
        if model_out.data[idx] > max:
            max = model_out.data[idx]
            maxIndex = idx
    return maxIndex


def getAcc(eval_acc, batch_features, logit, args):
    eval_acc.clear_PRF()
    for id_batch in range(batch_features.batch_length):
        inst = batch_features.inst[id_batch]
        predict_label = []
        gold_lable = inst.labels
        for id_word in range(inst.words_size):
            maxId = getMaxindex(logit[id_batch][id_word], logit.size(2), args)
            predict_label.append(args.create_alphabet.label_alphabet.from_id(maxId))
        assert len(predict_label) == len(gold_lable)
        cor = 0
        for p_lable, g_lable in zip(predict_label, gold_lable):
            if p_lable == g_lable:
                cor += 1
        eval_acc.correct_num += cor
        eval_acc.gold_num += len(gold_lable)


class Best_Result:
    def __init__(self):
        self.best_dev_fscore = -1
        self.best_fscore = -1
        self.best_epoch = 1
        self.best_test = False
        self.p = -1
        self.r = -1
        self.f = -1


