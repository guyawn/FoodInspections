import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import re
import sys
sys.path.insert(0, './Helpers/')
import Helpers.SequentialSearch as ss
import dill

def get_logistic_set(name):

    X_train = pd.read_csv("Data/Modelling/" + name + "PredictorsTrain.csv")
    X_test = pd.read_csv("Data/Modelling/" + name + "PredictorsTest.csv")
    y_train = pd.read_csv("Data/Modelling/" + name + "OutcomesTrain.csv", header=None)
    y_test = pd.read_csv("Data/Modelling/" + name + "OutcomesTest.csv", header=None)

    clf = LogisticRegression(solver='lbfgs')
    seq = ss.SequentialSearch(clf, X_train, y_train, 10)
    seq.run(3, 1, 30)
    logisticSet = seq.bestSet

    X_train_log = X_train[logisticSet]
    X_test_log = X_test[logisticSet]

    clf.fit(X_train_log, y_train)

    scores = clf.predict_proba(X_test_log)
    scores = [score[1] for score in scores]
    fp, tp, _ = metrics.roc_curve(y_test, scores)
    metrics.auc(fp, tp)

    pd.DataFrame({'LogisticSet': logisticSet}).to_csv("Data/Outcomes/" + name + "Logistic.csv")

    return clf, logisticSet


def check_model(name, clf, vars):

    x_train = pd.read_csv("Data/Modelling/" + name + "PredictorsTrain.csv")
    x_test = pd.read_csv("Data/Modelling/" + name + "PredictorsTest.csv")
    y_train = pd.read_csv("Data/Modelling/" + name + "OutcomesTrain.csv", header=None)
    y_test = pd.read_csv("Data/Modelling/" + name + "OutcomesTest.csv", header=None)

    c, r = y_train.shape
    y_train = y_train.values.reshape(c, )
    c, r = y_test.shape
    y_test = y_test.values.reshape(c, )

    x_train_good = x_train[vars]
    x_test_good = x_test[vars]

    scores = clf.predict_proba(x_train_good)
    scores = [score[1] for score in scores]
    fp, tp, _ = metrics.roc_curve(y_train, scores)
    train_auc = metrics.auc(fp, tp)

    scores = clf.predict_proba(x_test_good)
    scores = [score[1] for score in scores]
    fp, tp, _ = metrics.roc_curve(y_test, scores)
    test_auc = metrics.auc(fp, tp)

    print("---------------------------")
    print("Model: " + name)
    print("Variables: " + str(len(vars)))
    print("Train AUC: " + str(train_auc))
    print("Test AUC: " + str(test_auc))

    return train_auc, test_auc

if __name__ == "__main__":

    # Simple Model
    logSimple, varsSimple = get_logistic_set("Simple")

    # Full Model
    logFull, varsFull = get_logistic_set("Full")

    # No Reviews Model
    logNoReviews, varsNoReviews = get_logistic_set("NoReviews")

    # No Categories Model
    logNoCategories, varsNoCategories = get_logistic_set("NoCategories")

    dill.dump_session("Log-Models.pkl")

