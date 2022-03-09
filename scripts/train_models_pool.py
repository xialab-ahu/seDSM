#! /usr/bin/python3
# -*- coding: utf8 -*-
# @Time:  14:09
# : amira
# @FileName: train_models_pool

# Train base classifiers.


import argparse
from collections import defaultdict
import functools
import numpy as np
from pathlib import Path
import pickle
import os

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from utils import set_seed, split_data, random_sample, sco, to_pickle, from_pickle, make_dirs
from modelbuilder import get_builder, ModelBuilder, PipeBuilder


def load_data(data_dir):
    files = ["train.pkl", "test1.pkl", "test2.pkl"]
    res = []
    for file in files:
        data_file = os.path.join(data_dir, file)
        with open(data_file, "rb") as f:
            res.append(pickle.load(f))
    return res


def get_args():
    pass


def train_a_model(train_data, model_builder, pipeline_builder):
    """Randomly build a model and train it with given datasets"""
    model = model_builder.product()  # algorithm randomly selected from SVM, DT, LR.
    pipeline = pipeline_builder.product()  # pipeline to select subspace of the features account for 20% to 40%.

    train_data = pipeline.fit_transform(train_data[0]), train_data[1]
    random_index = np.arange(len(train_data[0]))
    np.random.shuffle(random_index)

    train_data = train_data[0][random_index], train_data[1][random_index]  # shuffle training data
    model.fit(*train_data)
    estimator = make_pipeline(pipeline, model)
    return estimator


def get_experiment_utils(args):
    imputer_strategy = args.strategy
    train, test1, test2 = load_data(args.data_dir)

    model_builder, pipeline_builder = get_builder(fill_methods=imputer_strategy)
    return (train, test1, test2), (model_builder, pipeline_builder)


class Cache(object):
    def __init__(self):
        # cache models
        self.models_pools = defaultdict(list)
        self.mp_key = "cv-%d"

        # cache outputs of models pools
        self.records = {}

        # cache eval dataset
        self.eval_datasets = {}
        self.data_key = "cv-%d eval-%d"

    def load(self, output_dir):
        mp_file = os.path.join(output_dir, "model_pools.pickle")
        r_file = os.path.join(output_dir, "records.pickle")
        ed_file = os.path.join(output_dir, "eval_data.pickle")
        self.models_pools = from_pickle(mp_file)
        self.records = from_pickle(r_file)
        self.eval_datasets = from_pickle(ed_file)

    def freeze(self, output_dir):
        mp_file = os.path.join(output_dir, "model_pools.pickle")
        to_pickle(mp_file, self.models_pools)

        r_file = os.path.join(output_dir, "records.pickle")
        to_pickle(r_file, self.records)

        ed_file = os.path.join(output_dir, "eval_data.pickle")
        to_pickle(ed_file, self.eval_datasets)

    def save_data(self, cv, i, data):
        self.eval_datasets[self.data_key % (cv, i)] = data

    def get_data(self, cv, i):
        return self.eval_datasets[self.data_key % (cv, i)]

    def save_model(self, cv, model):
        self.models_pools[self.mp_key % cv].append(model)

    def save_output(self, cv, output):
        self.records[self.mp_key % cv] = output

    def get_model(self, cv):
        return self.models_pools[self.mp_key % cv]

    def get_output(self, cv):
        return self.records[self.mp_key % cv]


def main():
    args = get_args()
    set_seed(args.seed)

    (train, test1, test2), (model_builder, pipeline_builder) = get_experiment_utils(args)
    cache = Cache()

    for cv, (train_x, eval_x, train_y, eval_y) in enumerate(split_data(*train, n_splits=args.cv)):
        # Train models

        # 1. Split the validation sets into two parts.One is used for calculate the diversity measure and the other for
        # model evaluation.
        eval_data = next(random_sample(eval_x, eval_y, n_times=1, frac=1, seed=args.seed))  # obtain balanced datasets.

        x_eval0, x_eval1, y_eval0, y_eval1 = train_test_split(*eval_data, random_state=args.seed, shuffle=True,
                                                              stratify=eval_data[1])
        # cache the two parts.
        cache.save_data(cv, 0, (x_eval0, y_eval0))
        cache.save_data(cv, 1, (x_eval1, y_eval1))

        # 2. Start sampling loop and train base classifiers.
        for ind, (x, y) in enumerate(
            random_sample(train_x, train_y, n_times=args.num_models, frac=1., seed=args.seed),
            start=1
        ):

            estimator = train_a_model((x, y), model_builder, pipeline_builder)
            cache.save_model(cv, estimator)

        output = sco.get_predictions(cache.get_model(cv), (x_eval0, y_eval0))
        cache.save_output(cv, output)
    cache.freeze(args.output_dir)


if __name__ == "__main__":
    main()









