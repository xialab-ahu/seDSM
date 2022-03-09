#! /usr/bin/python3
# -*- coding: utf8 -*-
# @Time:  16:12
# : amira
# @FileName: utils.py

import numpy as np
import random
import pickle
import os
from sklearn.model_selection import StratifiedKFold

from metric_utils import Scorer


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def split_data(X, y, n_splits, seed=0):
    kfolder = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, test_index in kfolder.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        yield X_train, X_test, y_train, y_test


def random_sample(X, y, n_times, frac=1.0, seed=0):
    """
    Randomly sample from the given data.
    Args:
        X: futures.
        y: labels
        n_times: sample times.
        frac: the ratio of negative samples to positive sample

    Returns:
        A generator can yield balanced data.

    """
    np.random.seed(seed)
    pos_ind = y == 1
    neg_ind = y == 0
    pos = X[pos_ind]
    neg = X[neg_ind]
    index = np.arange(len(neg))
    for i in range(n_times):
        ind = np.random.choice(index, size=int(frac*len(pos)))
        neg_ = neg[ind]

        data = np.concatenate([pos, neg_], axis=0)
        label = np.array([1] * len(pos) + [0] * len(neg_))
        yield data, label


class Sco(object):
    """
    This class collect methods to get predictions and metrics of models.
    """

    @classmethod
    def get_prediction(cls, model, data):
        y_pred = model.predict_proba(data[0])[:, 1]
        return data[1], y_pred

    @classmethod
    def get_predictions(cls, models, data, mean=False):
        res = [cls.get_prediction(model, data) for model in models]
        if mean:
            preds = np.array([x[1] for x in res]).mean(axis=0)
            return data[1], preds
        return [cls.get_prediction(model, data) for model in models]

    @classmethod
    def get_score(cls, y_true, y_pred, return_dict=True):
        return Scorer.get_scores(y_pred, y_true, return_dict=return_dict)

    @classmethod
    def get_scores(cls, outputs):
        scores = []
        for y_true, y_pred in outputs:
            scores.append(cls.get_score(y_true, y_pred))
        return np.array(scores)

    @classmethod
    def eval_model(cls, model, data):
        y_pred, y_label = cls.get_prediction(model, data)
        return cls.get_score(y_pred, y_label)

    @classmethod
    def eval_models(cls, models, data, mean=False):
        outputs = cls.get_predictions(models, data, mean=mean)
        if mean:
            score = cls.get_score(*outputs)
            return score
        scores = cls.get_scores(outputs)
        return scores


def to_pickle(file, obj):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def from_pickle(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def make_dirs(*args):
    for dir in args:
        if not os.path.exists(dir):
            os.makedirs(dir)


sco = Sco()

feature_names = ['PrDSM', 'TraP', 'SilVA', 'PhD-SNPg', 'FATHMM-MKL', 'CADD', 'DANN', 'FATHMM-XF', 'priPhCons',
                     'mamPhCons', 'verPhCons', 'priPhyloP', 'mamPhyloP', 'verPhyloP', 'GerpS', 'TFBs', 'TE', 'dPSIZ',
                     'DSP', 'RSCU', 'dRSCU', 'CpG?', 'CpG_exon', 'SR-', 'SR+', 'FAS6-', 'FAS6+', 'MES', 'dMES', 'MES+',
                     'MES-', 'MEC-MC?', 'MEC-CS?', 'MES-KM?', 'PESE-', 'PESE+', 'PESS-', 'PESS+', 'f_premrna', 'f_mrna']