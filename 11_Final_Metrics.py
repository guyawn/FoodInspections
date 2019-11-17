import pandas as pd
from sklearn import metrics
import sys
sys.path.insert(0, './Helpers/')
import matplotlib.pyplot as plt
import numpy as np
import dill

import statsmodels.formula.api as sm

def check_log_model(name, clf, vars):

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

def check_rvm_model(name, clf, vars):

    x_train = pd.read_csv("Data/Modelling/" + name + "PredictorsTrain.csv")
    x_test = pd.read_csv("Data/Modelling/" + name + "PredictorsTest.csv")
    y_train = pd.read_csv("Data/Modelling/" + name + "OutcomesTrain.csv", header=None)
    y_test = pd.read_csv("Data/Modelling/" + name + "OutcomesTest.csv", header=None)

    c, r = y_train.shape
    y_train = y_train.values.reshape(c, )
    c, r = y_test.shape
    y_test = y_test.values.reshape(c, )

    x_train_good = x_train.iloc[:, vars]
    x_test_good = x_test.iloc[:, vars]

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
    print("Variables: " + str(np.sum(np.sum(abs(clf.relevance_), axis=0) > 0)))
    print("Relevance Vectors: " + str(len(clf.relevance_)))
    print("Train AUC: " + str(train_auc))
    print("Test AUC: " + str(test_auc))

    return train_auc, test_auc

def eight_group_plot(firstMeans, secondMeans, title, labels, output_name):

    N = 4

    fig, ax = plt.subplots()

    ind = np.arange(N)
    width = 0.35

    p1 = ax.bar(ind, firstMeans, width, bottom=0)
    p2 = ax.bar(ind + width, secondMeans, width, bottom=0)

    ax.set_title(title)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(labels)
    plt.ylim(bottom=0, top=1.0)
    plt.xlabel("Variable Set")
    plt.ylabel("AUC")

    ax.legend((p1[0], p2[0]), ('RVM', 'Logistic'))
    ax.autoscale_view()

    plt.savefig(output_name)


if __name__ == "__main__":

    #Initialize variables to avoid a compiler error
    rvmSimple = True
    logSimple = True
    varsSimple = True

    rvmFull = True
    logFull = True
    varsFull = True

    rvmNoCategories = True
    logNoCategories = True
    varsNoCategories = True

    rvmNoReviews = True
    logNoReviews = True
    varsNoReviews = True

    # Save the models
    dill.load_session("RVM-Models.pkl")

    # Evaluate the models
    simple_rvm_train_auc, simple_rvm_test_auc = check_rvm_model("Simple", rvmSimple, varsSimple)
    noreviews_rvm_train_auc, noreviews_rvm_test_auc = check_rvm_model("NoReviews", rvmNoReviews, varsNoReviews)
    nocategories_rvm_train_auc, nocategories_rvm_test_auc = check_rvm_model("NoCategories", rvmNoCategories, varsNoCategories)
    full_rvm_train_auc, full_rvm_test_auc = check_rvm_model("Full", rvmFull, varsFull)

    # Save the models
    dill.load_session("Log-Models.pkl")

    # Evaluate the model
    simple_log_train_auc, simple_log_test_auc = check_log_model("Simple", logSimple, varsSimple)
    noreviews_log_train_auc, noreviews_log_test_auc = check_log_model("NoReviews", logNoReviews, varsNoReviews)
    nocategories_log_train_auc, nocategories_log_test_auc = check_log_model("NoCategories", logNoCategories, varsNoCategories)
    full_log_train_auc, full_log_test_auc = check_log_model("Full", logFull, varsFull)


    eight_group_plot((simple_log_train_auc, noreviews_log_train_auc, nocategories_log_train_auc, full_log_train_auc),
                     (simple_rvm_train_auc, noreviews_rvm_train_auc, nocategories_rvm_train_auc, full_rvm_train_auc),
                     "Train Outcomes", ("Simple", "No Reviews", "No Categories", "All"), "Figures/Modelling/Train-AUCs.png")

    eight_group_plot((simple_log_test_auc, noreviews_log_test_auc, nocategories_log_test_auc, full_log_test_auc),
                     (simple_rvm_test_auc, noreviews_rvm_test_auc, nocategories_rvm_test_auc, full_rvm_test_auc),
                     "Validation Outcomes", ("Simple", "No Reviews", "No Categories", "All"), "Figures/Modelling/Test-AUCs.png")

    dill.load_session("Log-Models.pkl")
    x_train = pd.read_csv("Data/Modelling/FullPredictorsTrain.csv")
    y_train = pd.read_csv("Data/Modelling/FullOutcomesTrain.csv", header=None)
    x_train = x_train[varsFull]
    model = sm.Logit(y_train, x_train)
    result = model.fit()


    dill.load_session("RVM-Models.pkl")
    x_train = pd.read_csv("Data/Modelling/NoReviewsPredictorsTrain.csv")
    y_train = pd.read_csv("Data/Modelling/NoReviewsOutcomesTrain.csv", header=None)
    rvmRelevances = pd.DataFrame(rvmNoReviews.relevance_)

    rvmRelevancesNames = np.array(list(x_train))[varsNoReviews]

    renaming = dict(zip(list(rvmRelevances), rvmRelevancesNames))
    rvmRelevances.rename(columns=renaming, inplace=True)
    rvmRelevances.to_csv("RVM_Relevances.csv")