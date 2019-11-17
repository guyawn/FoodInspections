import pandas as pd

from skrvm import RVC
from sklearn import metrics
import sys
sys.path.insert(0, './Helpers/')
import Helpers.SequentialSearch as ss
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import dill

def get_rvm_set(name, step=2, indexMin=1, indexMax=9999):

    x_train = pd.read_csv("Data/Modelling/" + name + "PredictorsTrain.csv")
    x_test = pd.read_csv("Data/Modelling/" + name + "PredictorsTest.csv")
    y_train = pd.read_csv("Data/Modelling/" + name + "OutcomesTrain.csv", header=None)
    y_test = pd.read_csv("Data/Modelling/" + name + "OutcomesTest.csv", header=None)

    clf = RVC()
    c, r = y_train.shape
    y_train = y_train.values.reshape(c, )
    c, r = y_test.shape
    y_test = y_test.values.reshape(c, )

    clf.fit(x_train, y_train)

    indices_ordered = np.flip(np.argsort(np.var(clf.relevance_, axis=0)))
    indices_zero = np.where(np.var(clf.relevance_, axis=0) == 0)
    indices_ordered = indices_ordered[~np.in1d(indices_ordered, indices_zero)]
    indices_ordered = indices_ordered[0:(int(len(indices_ordered) / 2))]

    indices_to_include = np.arange(1, len(indices_ordered)+1, step)
    indices_to_include = np.clip(indices_to_include, indexMin, indexMax)
    indices_to_include = np.unique(indices_to_include)

    aucs = []

    for i in indices_to_include:

        good_indices = indices_ordered[:i]
        x_train_limited = x_train.iloc[:, good_indices]

        x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(x_train_limited,
                                                            y_train,
                                                            test_size=0.4,
                                                            shuffle=True)

        y_train_temp = [list(a)[0] for a in list(y_train_temp.values)]
        y_train_temp = np.array(y_train_temp)

        clf.fit(x_train_temp, y_train_temp)

        scores = clf.predict_proba(x_test_temp)
        scores = [score[1] for score in scores]
        fp, tp, _ = metrics.roc_curve(y_test_temp, scores)
        train_auc = metrics.auc(fp, tp)

        aucs.append(train_auc)

        print(str(i) + " of " + str(len(indices_ordered)+1) + ": " + str(train_auc))


    best_indices_include = indices_to_include[np.array(aucs).argmax()]
    best_indices = indices_ordered[0:best_indices_include]

    x_train_best = x_train.iloc[:, best_indices]
    x_test_best = x_test.iloc[:, best_indices]

    y_train = [list(a)[0] for a in list(y_train.values)]
    y_train = np.array(y_train)

    clf.fit(x_train_best, y_train)

    scores = clf.predict_proba(x_train_best)
    scores = [score[1] for score in scores]
    fp, tp, _ = metrics.roc_curve(y_train, scores)
    train_auc = metrics.auc(fp, tp)

    scores = clf.predict_proba(x_test_best)
    scores = [score[1] for score in scores]
    fp, tp, _ = metrics.roc_curve(y_test, scores)
    test_auc = metrics.auc(fp, tp)

    print("---------------------------")
    print("Model: " + name)
    print("Variables: " + str(np.sum(np.sum(abs(clf.relevance_), axis=0) > 0)))
    print("Train AUC: " + str(train_auc))
    print("Test AUC: " + str(test_auc))

    return clf, best_indices

if __name__ == "__main__":

    # Simple Model
    rvmSimple, varsSimple = get_rvm_set("Simple", 2)

    # No Reviews Model
    rvmNoReviews, varsNoReviews = get_rvm_set("NoReviews", 2)

    # No Categories Model
    rvmNoCategories, varsNoCategories = get_rvm_set("NoCategories", 5)

    # Full Model
    rvmFull, varsFull = get_rvm_set("Full", 30)

    #Save the models
    dill.dump_session("RVM-Models.pkl")


