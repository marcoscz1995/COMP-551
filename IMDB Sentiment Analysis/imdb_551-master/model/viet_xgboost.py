#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 23:15:58 2019
@author: Viet
Quick XGBoosting stuff. Should take a while to run but that's okay.
"""

from model.pre2 import retrieve_and_pre, dump_to_csv
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

def main():
    X, y, final, names = retrieve_and_pre(fromsave=True, tfidfpca=True, n_components=50000)
    y = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)
    xgb = XGBClassifier(n_estimators=3000, max_depth=5, silent=False)
    training_start = time.perf_counter()
    xgb.fit(X_train, y_train)
    training_end = time.perf_counter()
    prediction_start = time.perf_counter()
    preds = xgb.predict(X_test)
    prediction_end = time.perf_counter()
    acc_xgb = (preds == y_test).sum().astype(float) / len(preds) * 100
    xgb_train_time = training_end - training_start
    xgb_prediction_time = prediction_end - prediction_start
    print("XGBoost's prediction accuracy is: %3.2f" % (acc_xgb))
    print("Time consumed for training: %4.3f seconds" % (xgb_train_time))
    print("Time consumed for prediction: %6.5f seconds" % (xgb_prediction_time))

    dump_to_csv(xgb, final, names, 'xgboost-3k')


    f1_scr = f1_score(y_test, preds)
    print("F1 score: %f%%" % (f1_scr * 100))
    #dump_to_csv(clf, test, names, 'autosvm')

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    print("true neg: %i\tfalse pos: %i\nfalse neg: %i\ttrue pos: %i\n" % (tn, fp, fn, tp))

if __name__ == '__main__':
    main()
    