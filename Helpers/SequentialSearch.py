from sklearn import metrics
import numpy as np
import copy

class SequentialSearch:

    def __init__ (self, classifier, predictors, outcome, nfolds=10):

        self.classifier = classifier
        self.predictors=predictors
        self.outcome = outcome

        self.folds = list(range(0, nfolds)) * len(predictors)
        self.folds = np.array(self.folds[0:len(predictors)])
        self.names = list(predictors)
        self.setsAttempted = []


        self.currentSet = []
        self.currentScore = 0

        self.bestSet = []
        self.bestScore = 0

    def run(self, forward, backward, iterations):

        for i in range(0, iterations):

            for j in range(0, forward+1):

                self.iterationForward()

                print("-----------------------------------------")
                print("Iteration " + str(i) + ", Forward " + str(j))
                print("Best set: " + str(self.bestSet))
                print("Best score: " + str(self.bestScore))
                print("Current set: " + str(self.currentSet))
                print("Current score: " + str(self.currentScore))


            for k in range(0, backward+1):

                self.iterationBackward()

                print("-----------------------------------------")
                print("Iteration " + str(i) + ", Backward " + str(k))
                print("Best set: " + str(self.bestSet))
                print("Best score: " + str(self.bestScore))
                print("Current set: " + str(self.currentSet))
                print("Current score: " + str(self.currentScore))



    def getFoldedAUC(self, set):

        currentPredictors = self.predictors[set]

        AUCs = []
        for fold in np.unique(self.folds):

            trainPredictors = currentPredictors[self.folds != fold]
            testPredictors = currentPredictors[self.folds == fold]

            trainOutcomes = self.outcome[self.folds != fold]
            testOutcomes = self.outcome[self.folds == fold]

            self.classifier.fit(trainPredictors, trainOutcomes)
            scores = self.classifier.predict_proba(testPredictors)
            scores = [score[1] for score in scores]

            fp, tp, _ = metrics.roc_curve(testOutcomes, scores)
            AUCs.append(metrics.auc(fp, tp))

        return np.mean(AUCs)


    def iterationForward(self):

        newSets = []
        for name in self.names:
            if name not in self.currentSet:
                newSet = copy.deepcopy(self.currentSet)
                newSet.append(name)
                if newSet not in self.setsAttempted:
                    self.setsAttempted.append(newSet)
                    newSets.append(newSet)

        self.currentSet = []
        self.currentScore = 0

        for newSet in newSets:
            score = self.getFoldedAUC(newSet)

            if score > self.currentScore:
                self.currentSet = newSet
                self.currentScore = score

            if score > self.bestScore:
                self.bestSet = newSet
                self.bestScore = score

    def iterationBackward(self):

        newSets = []
        for name in self.currentSet:
            newSet = copy.deepcopy(self.currentSet)
            newSet.remove(name)

            if newSet not in self.setsAttempted:
                self.setsAttempted.append(newSet)
                newSets.append(newSet)

        self.currentSet = []
        self.currentScore = 0

        for newSet in newSets:
            score = self.getFoldedAUC(newSet)

            if score > self.currentScore:
                self.currentSet = newSet
                self.currentScore = score

            if score > self.bestScore:
                self.bestSet = newSet
                self.bestScore = score