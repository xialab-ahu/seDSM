#! /usr/bin/python3
# -*- coding: utf8 -*-
# @Time:  ${TIME}
# : amira
# @FileName: data_preprocessing.py

# The main function of this demo is to impute missing values from raw datasets of training and independent test sets.


import argparse
import functools
import pickle
import numpy as np
from sklearn.impute import KNNImputer
import os

from utils import make_dirs, set_seed, random_sample
from datasets import get_data


class EKNNImputer(object):
    def __init__(self, n=10):
        self.n = n
        self.pipelines = [KNNImputer() for i in range(n)]

    def fit(self, X, y):
        for i, (x, y) in enumerate(random_sample(*(X, y), n_times=10, frac=1., seed=0)):
            self.pipelines[i].fit(x)

    def transform(self, X):
        _X = np.zeros_like(X)
        for p in self.pipelines:
            _X += p.transform(X)
        _X /= 10
        return _X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--strategy", type=str, default="EKNNI")
    argparser.add_argument("--results_dir", type=str, default="preprocessed_data")
    args = argparser.parse_args()
    return args


def impute(train, *tests, strategy="EKNNI"):
    assert strategy in IMPUTER_STRATEGIES
    imputer = IMPUTER_STRATEGIES[strategy]()  # KNNImputer() or EKNNImputer()

    imputer.fit(*train)

    _results = [(imputer.transform(train[0]), train[1])]
    for test in tests:
        _results.append((imputer.transform(test[0]), test[1]))

    if strategy == "EKNNI":
        imputer = imputer.pipelines  # only save the pipelines in the `EKNNImputer`.
    return _results, imputer


def save_results(objects: dict, root_dir):
    for key, value in objects.items():
        with open(os.path.join(root_dir, key+".pkl"), "wb") as f:
            pickle.dump(value, f)


def get_data_without_preprocessing():
    train = get_data("train", col=range(40), shuffle=True)
    test1, test2 = map(functools.partial(get_data, col=range(40), shuffle=False), ["test1", "test2"])
    return train, test1, test2


IMPUTER_STRATEGIES = {"KNNI": KNNImputer, "EKNNI": EKNNImputer}


if __name__ == "__main__":
    set_seed(0)
    args = get_args()
    root_dir = os.path.join(args.results_dir, args.strategy)
    make_dirs(root_dir)

    # Obtain the raw datasets: imbalanced training set, balanced independent test set 1 and 2
    train, test1, test2 = get_data_without_preprocessing()

    # Impute missing values according strategy based on training sets.
    results, imputer = impute(train, test1, test2, strategy=args.strategy)
    train_, test1_, test2_ = results

    # Save results to root_dir.
    objects = {"train": train_, "test1": test1_, "test2": test2_, "pipelines": imputer}
    save_results(objects, root_dir)







