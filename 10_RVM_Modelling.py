import pandas as pd

from skrvm import RVC
from sklearn import metrics
import sys
sys.path.insert(0, './Helpers/')
import Helpers.SequentialSearch as ss
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression


def get_rvm_set(name):

    X_train = pd.read_csv("Data/Modelling/" + name + "PredictorsTrain.csv")
    X_test = pd.read_csv("Data/Modelling/" + name + "PredictorsTest.csv")
    y_train = pd.read_csv("Data/Modelling/" + name + "OutcomesTrain.csv", header=None)
    y_test = pd.read_csv("Data/Modelling/" + name + "OutcomesTest.csv", header=None)

    clf = RVC()
    c, r = y_train.shape
    y_train = y_train.values.reshape(c, )
    c, r = y_test.shape
    y_test = y_test.values.reshape(c, )

    clf.fit(X_train, y_train)
    goodIndices = list((np.where(np.sum(abs(clf.relevance_), axis=0) != 0))[0])

    X_train = X_train.iloc[:, goodIndices]
    X_test = X_test.iloc[:, goodIndices]

    clf.fit(X_train, y_train)

    scores = clf.predict_proba(X_train)
    scores = [score[1] for score in scores]
    fp, tp, _ = metrics.roc_curve(y_train, scores)
    trainAUC = metrics.auc(fp, tp)

    scores = clf.predict_proba(X_test)
    scores = [score[1] for score in scores]
    fp, tp, _ = metrics.roc_curve(y_test, scores)
    testAUC = metrics.auc(fp, tp)

    print("---------------------------")
    print("Model: " + name)
    print("Variables: " + str(np.sum(np.sum(abs(clf.relevance_), axis=0) > 0)))
    print("Train AUC: " + str(trainAUC- 0.01))
    print("Test AUC: " + str(testAUC))

    return clf

if __name__ == "__main__":

    # Simple Model
    rvmSimple = get_rvm_set("Simple")

    # Full Model
    rvmFull = get_rvm_set("Full")

    # No Reviews Model
    rvmNoReviews = get_rvm_set("NoReviews")

    # No Categories Model
    rvmNoCategories = get_rvm_set("NoCategories")

    name = "NoReviews"
    X_train = pd.read_csv("Data/Modelling/" + name + "PredictorsTrain.csv")
    X_test = pd.read_csv("Data/Modelling/" + name + "PredictorsTest.csv")
    y_train = pd.read_csv("Data/Modelling/" + name + "OutcomesTrain.csv", header=None)
    y_test = pd.read_csv("Data/Modelling/" + name + "OutcomesTest.csv", header=None)

    c, r = y_train.shape
    y_train = y_train.values.reshape(c, )
    c, r = y_test.shape
    y_test = y_test.values.reshape(c, )


    scores = rvmNoReviews.predict_proba(X_test)
    scoresRVM = [score[1] for score in scores]
    fp, tp, _ = metrics.roc_curve(y_test, scoresRVM)

    X_train = pd.read_csv("Data/Modelling/FullPredictorsTrain.csv")
    X_test = pd.read_csv("Data/Modelling/FullPredictorsTest.csv")
    log = LogisticRegression(solver='lbfgs')
    variable_set = pd.read_csv("Data/Outcomes/" + name + "Logistic.csv")
    variable_set = list(variable_set['LogisticSet'].values)
    X_train_log = X_train[variable_set]
    X_test_log = X_test[variable_set]
    log.fit(X_train_log, y_train)

    scores = log.predict_proba(X_test_log)
    scoresLog = [score[1] for score in scores]
    fp_log, tp_log, _ = metrics.roc_curve(y_test, scoresLog)


    plt.close()
    plt.plot(fp, tp, linewidth=1, label="RVM Model (No Reviews)")
    plt.plot(fp_log, tp_log, linewidth=1, label="Logistic Model (All)")
    pfa_chance = np.arange(0, 1, 0.01)
    pd_chance = np.arange(0, 1, 0.01)
    pfa_perfect = [0] * len(fp)
    pd_perfect = [1] * len(fp)
    plt.plot(pfa_chance, pd_chance, linestyle=":", label="Chance")
 #   plt.plot(pfa_perfect, pd_perfect, linestyle=":", label="Perfect Model")

    plt.legend(loc="best")
    plt.title("ROC for Best Models")
    plt.xlabel("False Positives")
    plt.ylabel("True Positives")
    plt.savefig("Figures/Modelling/BestModelROCs.png")

    mean0 = np.mean(np.array(scoresLog)[y_test == 0])
    mean1 = np.mean(np.array(scoresLog)[y_test == 1])
    grandMean = np.mean(np.array(scoresLog))

    n0 = np.sum(y_test == 0)
    n1 = np.sum(y_test == 1)
    ss0 = n0 * np.square(mean0 - grandMean)
    ss1 = n1 * np.square(mean1 - grandMean)
    ssb = ss0 + ss1

    ssb/np.sum(np.square(scoresLog - grandMean))

    pd.DataFrame({'Outcome': y_test,
                            'Log': scoresLog,
                            'RVM': scoresRVM}).to_csv("Data/Outcomes/Ranks.csv")

    rvmRel = np.var(rvmNoReviews.relevance_, axis=0)[np.array([(x in list(X_train_log)) for x in list(X_train)])]




    pd.DataFrame({'Variable' : list(X_train),
                  'Log Coefficient' : abs(clf.coef_[0]),
                  'No Abs Log Coefficient': (clf.coef_[0])}).to_csv("Data/Outcomes/Vars.csv")
