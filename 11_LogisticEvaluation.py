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

    variable_set = pd.read_csv("Data/Outcomes/" + name + "Logistic.csv")
    variable_set = list(variable_set['LogisticSet'].values)

    X_train = X_train[variable_set]
    X_test = X_test[variable_set]

    clf = LogisticRegression(solver='lbfgs')
    clf.fit(X_train, y_train)

    seq = ss.SequentialSearch(clf, X_train, y_train, 10)
    trainAUC = seq.getFoldedAUC(variable_set)

    scores = clf.predict_proba(X_test)
    scores = [score[1] for score in scores]
    fp, tp, _ = metrics.roc_curve(y_test, scores)
    testAUC = metrics.auc(fp, tp)

    print("---------------------------")
    print("Model: " + name)
    print("Variables: " + str(len(variable_set)))
    print("Train AUC: " + str(trainAUC))
    print("Test AUC: " + str(testAUC))



if __name__ == "__main__":

    # Simple Model
    get_logistic_set("Simple")

    # Full Model
    get_logistic_set("Full")

    # No Reviews Model
    get_logistic_set("NoReviews")

    # No Categories Model
    get_logistic_set("NoCategories")