import statsmodels.api as sm
import numpy as np

class CV:

    def __init__(self, classifier, predictors, outcomes, folds, **kwargs):

        self.folds = list(range(0, folds)) * int(np.ceil(len(predictors) / 10))
        self.folds = self.folds[0:len(predictors)]
        self.folds = np.array(self.folds)
        self.classifiers = []

        for i in range(0, folds):

            predictorsTrain = predictors.iloc[folds != i]
            predictorsTest = predictors.iloc[folds == i]

            outcomeTrain = outcomes.iloc[folds != i]
            outcomeTest = outcomes.iloc[folds == i]

            if classifier == sm.GLM:

                family = kwargs['family']

                currentClassifier = classifier(predictorsTrain, outcomeTrain, family=family)
                currentClassifier.fit()




    # def roc(self, train_indices, outfile):
    #
    #     data_train = self.data.loc[self.folds.apply(lambda x: x in train_indices)]
    #     data_test = self.data.loc[self.folds.apply(lambda x: x not in train_indices)]
    #
    #     if len(data_test) == 0:
    #         data_test = data_train
    #
    #     if self.classifier == knnClassifier:
    #         knn = self.classifier(data_train, self.metric)
    #         knn.plotROC(self.k, outfile, data_test)
    #
    # def decision_surface(self, train_indices, outfile, direction='regular'):
    #
    #     data_train = self.data.loc[self.folds.apply(lambda x: x in train_indices)]
    #
    #     if self.classifier == knnClassifier:
    #
    #         if direction == 'rotated':
    #             knn = self.classifier(data_train, self.metric)
    #             knn.plotDecisionSurfaceRotated(self.k, self.threshold, outfile)
    #         else:
    #             knn = self.classifier(data_train, self.metric)
    #             knn.plotDecisionSurface(self.k, self.threshold, outfile)
    #
    # def bias_variance_plot(self, train_indices, outfile):
    #
    #     data_train = self.data.loc[self.folds.apply(lambda x: x in train_indices)]
    #     data_test = self.data.loc[self.folds.apply(lambda x: x not in train_indices)]
    #
    #     if len(data_test) == 0:
    #         data_test = data_train
    #
    #     if self.classifier == knnClassifier:
    #
    #         knn = self.classifier(data_train, self.metric)
    #         knn.plotBiasVariance(outfile + "-train.png")
    #         knn.plotBiasVariance(outfile + "-test.png"
    #                                        "", data_test)

