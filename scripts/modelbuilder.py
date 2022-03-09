#! /usr/bin/python3
# -*- coding: utf8 -*-
# @Time:  16:04
# : amira
# @FileName: modelbuilder.py

# Some utils for training base classifiers.


import random
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from typing import Callable, Dict

from transforms import RandomFeatureSelector


class ModelBuilder(object):
    def __init__(self, random_state=0, select_func: Callable[[Dict, random.Random], Dict] = None):
        self.models = {}
        self.models_kwargs = {}
        self.rs = random.Random(random_state)
        self.select_func = select_func

    def product(self):
        model_type = self.rs.choice(list(self.models))
        if not self.select_func:
            parameters = self.rs.choice(self.models_kwargs[model_type])
        else:
            parameters = self.select_func(self.models_kwargs[model_type], self.rs)
        print(model_type, parameters)
        return self.models[model_type](**parameters)

    def add(self, models, model_name: str, model_parameters: [dict]):
        self.models[model_name] = models
        self.models_kwargs[model_name] = model_parameters


class PipeBuilder(object):
    def __init__(self, fill_methods="INNER", p=None):
        self.rs = random.Random(fill_methods)
        self.fill_methods = fill_methods
        self.p = p

    def product(self):
        scale = list(range(int(40*0.2), int(40*0.8)))
        n = self.rs.choice(scale)

        # todo
        if self.fill_methods == "INNER":
            return make_pipeline(RandomFeatureSelector(n, self.p), KNNImputer(), MinMaxScaler())
        else:
            return make_pipeline(RandomFeatureSelector(n, self.p), MinMaxScaler())


# model builder
def get_builder(version, fill_methods, p=None):
    assert fill_methods in ["INNER", "OUTER", "E-OUTER"]
    # v1
    models = {"dt": DecisionTreeClassifier, "svm": SVC, "lr": LogisticRegression}
    parameters = {"dt": [{"criterion": "gini", "splitter": "random", "max_depth": 6},
                         {"criterion": "entropy", "min_samples_split": 10},
                         {"min_samples_leaf": 3},
                         {"class_weight": {0: 0.3, 1: 0.7}, "max_depth": 8}],
                  "svm": [{"C": 1.5, "kernel": "linear", "probability": True},
                          {"kernel": "poly", "probability": True},
                          {"probability": True, "kernel": "rbf"}],
                  "lr": [{"penalty": "l2"},
                         {"penalty": "none", "max_iter": 5000}],
                  }
    model_builder = ModelBuilder(random_state=0)
    for key in models:
        model_builder.add(models[key], key, parameters[key])

    # make a pipeline_builder
    pipeline_builder = PipeBuilder(fill_methods=fill_methods)
    version_builders = {"v1": (model_builder, pipeline_builder)}
    return version_builders[version]
