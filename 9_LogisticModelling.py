import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import re
import sys
sys.path.insert(0, './Helpers/')
import Helpers.SequentialSearch as ss


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



if __name__ == "__main__":

    # Simple Model
    #get_logistic_set("Simple")

    # Full Model
    #get_logistic_set("Full")

    # No Reviews Model
    #get_logistic_set("NoReviews")

    # No Categories Model
    get_logistic_set("NoCategories")