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
import os
import torch
import torch.nn.functional as F
import torch.nn.utils as utils
import random
import numpy as np
import time
import shutil
from DataUtils.eval_bio import entity_evalPRF_exact, entity_evalPRF_propor, entity_evalPRF_binary
from DataUtils.eval import Eval, EvalPRF
from DataUtils.Common import *
from DataUtils.utils import *
from DataUtils.Optim import Optimizer
torch.manual_seed(seed_num)
random.seed(seed_num)


def train(train_iter, dev_iter, test_iter, model, config):
    """
    :param train_iter:  train batch iterator
    :param dev_iter:  dev batch iterator
    :param test_iter: test batch iterator
    :param model: nn model
    :param config: config
    :return: None
    """

    # optimizer = None
    optimizer = Optimizer(name=config.learning_algorithm, model=model, lr=config.learning_rate,
                          weight_decay=config.weight_decay, grad_clip=config.clip_max_norm)

    best_fscore = Best_Result()

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
        steps = 0
        for batch_count, batch_features in enumerate(train_iter):
            model.zero_grad()
            optimizer.zero_grad()
            logit = model(batch_features)
            loss = F.cross_entropy(logit.view(logit.size(0) * logit.size(1), -1), batch_features.label_features,
                                   ignore_index=config.label_paddingId, size_average=False)
            loss.backward()
            if config.clip_max_norm_use is True:
                gclip = None if config.clip_max_norm == "None" else float(config.clip_max_norm)
                assert isinstance(gclip, float)
                utils.clip_grad_norm(model.parameters(), max_norm=gclip)
            optimizer.step()
            steps += 1
            if steps % config.log_interval == 0:
                getAcc(train_eval, batch_features, logit, config)
                sys.stdout.write("\nbatch_count = [{}] , loss is {:.6f}, [TAG-ACC is {:.6f}%]".format(batch_count + 1,
                                 loss.data[0], train_eval.acc()))

        end_time = time.time()
        print("\nTrain Time {:.3f}".format(end_time - start_time), end="")
        if steps is not 0:
            dev_eval.clear_PRF()
            eval_start_time = time.time()
            # eval(dev_iter, model, dev_eval, best_fscore, epoch, config, test=False)
            eval_batch(dev_iter, model, dev_eval, best_fscore, epoch, config, test=False)
            eval_end_time = time.time()
            print("Dev Time {:.3f}".format(eval_end_time - eval_start_time))
            # model.train()
        if steps is not 0:
            test_eval.clear_PRF()
            eval_start_time = time.time()
            # eval(test_iter, model, test_eval, best_fscore, epoch, config, test=True)
            eval_batch(test_iter, model, test_eval, best_fscore, epoch, config, test=True)
            eval_end_time = time.time()
            print("Test Time {:.3f}".format(eval_end_time - eval_start_time))
            # model.train()
        if config.save_model and config.save_all_model:
            save_model_all(model, config.save_dir, config.model_name, epoch)
        elif config.save_model and config.save_best_model:
            save_best_model(model, config.save_best_model_path, config.model_name, best_fscore)
        else:
            print()


def eval_batch(data_iter, model, eval_instance, best_fscore, epoch, config, test=False):
    """
    :param data_iter:  eval batch data iterator
    :param model: eval model
    :param eval_instance:
    :param best_fscore:
    :param epoch:
    :param config: config
    :param test:  whether to test
    :return: None
    """
    model.eval()
    # eval time
    eval_acc = Eval()
    eval_PRF = EvalPRF()
    gold_labels = []
    predict_labels = []
    for batch_features in data_iter:
        logit = model(batch_features)
        for id_batch in range(batch_features.batch_length):
            inst = batch_features.inst[id_batch]
            maxId_batch = getMaxindex_batch(logit[id_batch])
            predict_label = []
            for id_word in range(inst.words_size):
                predict_label.append(config.create_alphabet.label_alphabet.from_id(maxId_batch[id_word]))
            gold_labels.append(inst.labels)
            predict_labels.append(predict_label)
    for p_label, g_label in zip(predict_labels, gold_labels):
        eval_PRF.evalPRF(predict_labels=p_label, gold_labels=g_label, eval=eval_instance)
    if eval_acc.gold_num == 0:
        eval_acc.gold_num = 1
    p, r, f = eval_instance.getFscore()
    # p, r, f = entity_evalPRF_exact(gold_labels=gold_labels, predict_labels=predict_labels)
    # p, r, f = entity_evalPRF_propor(gold_labels=gold_labels, predict_labels=predict_labels)
    # p, r, f = entity_evalPRF_binary(gold_labels=gold_labels, predict_labels=predict_labels)
    test_flag = "Test"
    if test is False:
        print()
        test_flag = "Dev"
        best_fscore.current_dev_fscore = f
        if f >= best_fscore.best_dev_fscore:
            best_fscore.best_dev_fscore = f
            best_fscore.best_epoch = epoch
            best_fscore.best_test = True
    if test is True and best_fscore.best_test is True:
        best_fscore.p = p
        best_fscore.r = r
        best_fscore.f = f
    # print("{} eval: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%,  [TAG-ACC = {:.6f}%]".format(test_flag, p, r, f, eval_acc.acc()))
    print("{} eval: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%,  [TAG-ACC = {:.6f}%]".format(test_flag, p, r, f, 0.0000))
    if test is True:
        print("The Current Best Dev F-score: {:.6f}, Locate on {} Epoch.".format(best_fscore.best_dev_fscore,
                                                                                 best_fscore.best_epoch))
        print("The Current Best Test Result: precision = {:.6f}%  recall = {:.6f}% , f-score = {:.6f}%".format(
            best_fscore.p, best_fscore.r, best_fscore.f))
    if test is True:
        best_fscore.best_test = False


def getAcc(eval_acc, batch_features, logit, args):
    eval_acc.clear_PRF()
    for id_batch in range(batch_features.batch_length):
        inst = batch_features.inst[id_batch]
        predict_label = []
        gold_lable = inst.labels
        maxId_batch = getMaxindex_batch(logit[id_batch])
        for id_word in range(inst.words_size):
            # maxId = getMaxindex(logit[id_batch][id_word], logit.size(2), args)
            predict_label.append(args.create_alphabet.label_alphabet.from_id(maxId_batch[id_word]))
        assert len(predict_label) == len(gold_lable)
        cor = 0
        for p_lable, g_lable in zip(predict_label, gold_lable):
            if p_lable == g_lable:
                cor += 1
        eval_acc.correct_num += cor
        eval_acc.gold_num += len(gold_lable)








