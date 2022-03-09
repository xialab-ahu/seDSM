#! /usr/bin/python3
# -*- coding: utf8 -*-
# @Time:  15:00
# : amira
# @FileName: model_selection.py

# Calculate diversity measures of base classifiers and select some models to integrate.


import argparse
import functools
import pickle
import pandas as pd
import random
from pathlib import Path
import numpy as np
import operator
import os
from sklearn.model_selection import train_test_split


from train_models_pool import Cache, load_data
from utils import set_seed, split_data, random_sample, sco
from collections import defaultdict


def get_args():
    pass


def select(metrics: list, do_reverse: bool=False):
    """"""


def Record(object):
    def __init__(self):
        self.scores = defaultdict(list)
        self.s_key = "s-%d"
        self.model_selected = defaultdict(list)
        self.ms_key = "cv-%d diversity-%s"
        self.total_results = defaultdict(
            functools.partial(pd.DataFrame,
                              columns="sen, spe, pre, f1, mcc, acc, auc, aupr, tn, fp, fn, tp".split(", "))
        )
        self.tr_key = "%s-%s"

    def add_score(self, cv, score):
        self.scores[self.s_key % cv].append(score)

    def get_score(self, cv):
        return self.scores[self.s_key % cv]

    def add_selected_models(self, cv, idf, models_list):
        key = self.ms_key % (cv, idf)
        self.model_selected[key] = models_list

    def get_selected_models(self, cv, idf):
        key = self.ms_key % (cv, idf)
        return self.model_selected[key]

    def add_record(self, cv, idf, data_idf, results):
        key_ = self.tr_key % (data_idf, idf)
        self.total_results[key_].loc["cv-%d" % cv] = results

    def save_results_and_models(self, output_dir):
        # 1. save results
        for k in self.total_results:
            file = Path(output_dir) / (k + ".csv")
            self.total_results[k].loc["mean"] = self.total_results[k].mean(axis=0)
            self.total_results[k].to_csv(file)

        # 2. save selected models
        file = Path(output_dir) / "selected_models.pickle"
        file_obj = file.open(mode="wb")
        pickle.dump(self.model_selected, file_obj)
        file_obj.close()


def select(metrics: list, do_reverse: bool=False, n: int=100):
    sorted_scores = np.argsort(metrics)
    if do_reverse:
        sorted_scores = sorted_scores[::-1]
    return sorted_scores[:n]


def compute_diversity(by, prediction_matrix, targets):
    """compute diversity"""
    from deslib.util import diversity
    diversity_func_sets = {"Q_statistic": diversity.Q_statistic,
                           "agreement_measure": diversity.agreement_measure,
                           "correlation_coefficient": diversity.correlation_coefficient,
                           "disagreement_measure": diversity.disagreement_measure,
                           "double_fault": diversity.double_fault,
                           "ratio_errors": diversity.ratio_errors}

    diversity_reverse = {"Q_statistic": False,
                         "agreement_measure": False,
                         "correlation_coefficient": False,
                         "disagreement_measure": True,
                         "double_fault": False,
                         "ratio_errors": True}

    assert by in diversity_func_sets
    diversity_func = diversity_func_sets[by]
    do_reverse = diversity_reverse[by]
    return diversity.compute_pairwise_diversity(targets, prediction_matrix, diversity_func), do_reverse


def main():
    args = get_args()
    set_seed(args.seed)

    train, test1, test2 = load_data()
    cache = Cache()
    cache.load(os.path.dirname(args.output_dir))

    record = Record()  # object to cache for saving results and finally selected models

    # start cv
    for cv, (train_x, eval_x, train_y, eval_y) in enumerate(split_data(*train, n_splits=args.cv)):
        # 1. split eval data for selection and evaluation, (X_eval0, y_eval0) is not used in this section.
        eval_data = next(random_sample(eval_x, eval_y, n_times=1, frac=1, seed=args.seed))
        x_eval0, x_eval1, y_eval0, y_eval1 = train_test_split(*eval_data, random_state=args.seed, shuffle=True,
                                                              stratify=eval_data[1])

        # assert eval0 and eval1 remaining the same between `train_models_pool` step and `model_selection` step.
        saved_eval_0 = cache.get_data(cv, 0)
        np.testing.assert_equal(saved_eval_0[0], x_eval0)
        np.testing.assert_equal(saved_eval_0[1], y_eval0)
        saved_eval_1 = cache.get_data(cv, 1)
        np.testing.assert_equal(saved_eval_1[0], x_eval1)
        np.testing.assert_equal(saved_eval_1[1], y_eval1)

        # load models pool and restored output of each model on eval0.
        models_pool = cache.get_model(cv=cv)
        output = cache.get_output(cv=cv)

        # 2. calculate diversity measure.
        indexes = []  # To cache index of all selected model.

        # 2.1. no selection (NS)
        ind = list(range(1000))
        indexes.append(ind)

        # 2.2. diversity measure
        diversity_keys = ["Q_statistic", "correlation_coefficient", "disagreement_mreasure",
                          "double_fault", "ratio_errors"]

        # 2.2.1 prediction matrix and targets
        prediction_matrix = np.array([prediction[1] for prediction in output]).T

        # 2.2.2 discretize the prediction matrix
        prediction_matrix[prediction_matrix >= 0.5] = 1
        prediction_matrix[prediction_matrix < 0.5] = 0
        target = output[0][0]

        # 2.2.3 compute scores and sort models by the measures.
        for key in diversity_keys:
            diversity_score, reverse = compute_diversity(key, prediction_matrix, target)
            ind = select(diversity_score, do_reverse=reverse)
            indexes.append(ind)

        # 3. start selection
        identifiers = ["no_selection"]
        identifiers.extend(diversity_keys)

        datasets = {
            "test1": test1,
            "test2": test2,
            "eval": (x_eval1, y_eval1)
        }

        for i, idf in enumerate(identifiers):
            selector = operator.itemgetter(*list(indexes[i]))
            record.add_selected_models(cv, idf, selector(models_pool))

            for data_idf in datasets:
                results = sco.eval_models(record.get_selected_models(cv, idf),
                                          datasets[data_idf],
                                          mean=True)
                record.add_record(cv, idf, data_idf, results)

    # save outputs and selected models
    record.save_results_and_models(args.output_dir)


if __name__ == "__main__":
    main()








