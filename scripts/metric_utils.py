#! /usr/bin/python3
# -*- coding: utf8 -*-
# @Time:  14:57
# : amira
# @FileName: metric_utils.py

# Help to calculate the performance of a classifier.


from sklearn import metrics
import numpy as np
import pandas as pd


class Scorer(object):
    def __init__(self, columns=[]):
        self.table = pd.DataFrame(columns=columns)

    @staticmethod
    def get_scores(y_score, y_true, threshold=0.5, return_dict=True):
        y_pred = [int(i >= threshold) for i in y_score]
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix.flatten()
        sen = tp / (fn + tp)
        spe = tn / (fp + tn)
        pre = metrics.precision_score(y_true, y_pred)
        auc = metrics.roc_auc_score(y_true, y_score)
        pr, rc, _ = metrics.precision_recall_curve(y_true, y_score)
        aupr = metrics.auc(rc, pr)
        f1 = metrics.f1_score(y_true, y_pred)
        mcc = metrics.matthews_corrcoef(y_true, y_pred)
        acc = metrics.accuracy_score(y_true, y_pred)
        if return_dict:
            return dict(sen=sen, spe=spe, pre=pre, f1=f1, mcc=mcc, acc=acc, auc=auc, aupr=aupr, tn=tn, fp=fp, fn=fn,
                        tp=tp)
        return np.array([sen, spe, pre, f1, mcc, acc, auc, aupr, tn, fp, fn, tp])

    def add_score(self, score_dict, index):
        self.table.loc[index] = score_dict

    def get_scores_for_augment(self, y_score, y_true, return_dict=True, repeat=1):
        y_score = np.array(y_score).reshape(-1, repeat).mean(axis=1)
        y_label = np.array(y_true).reshape(-1, repeat).mean(axis=1)
        return self.get_scores(y_score, y_label, return_dict=return_dict)

