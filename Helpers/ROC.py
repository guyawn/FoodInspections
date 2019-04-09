import matplotlib.pyplot as plt
import numpy as np
import math


# Object to hold the pairs
class ROCPairs:

    def __init__(self, Pd, Pfa):
        self.Pd = Pd
        self.Pfa = Pfa


# Generates thresholds at every value in the sample.
def decisionThresholds(scores):

    # Remove duplicates
    thresholds = list(set(scores))

    # Add infinity bookends and sort
    thresholds.append(math.inf)
    thresholds.append(-math.inf)
    thresholds.sort()

    #Return
    return thresholds


# Function that takes in the true values and the decisions stats
def getROCPairs(scores, truth):

    # Performs the specified type of sampling
    thresholds = decisionThresholds(scores)

    # Splits the data into h0 and h1
    scores_0 = np.array(scores)[truth == 0]
    scores_1 = np.array(scores)[truth == 1]

    # Finds Pd and Pfa for each threshold.
    Pd = list()
    Pfa = list()
    for threshold in thresholds:
        Pd.append(sum([(stat > threshold) for stat in scores_1]) / len(scores_1))
        Pfa.append(sum([(stat > threshold) for stat in scores_0]) / len(scores_0))

    # Returns the Pd and Pfa lists as a dictionary
    return ROCPairs(Pd, Pfa)


# Identify the max operating point for a given prior
def getMaxOperatingPoint(pairs, h0toh1ratio=1):

    # Get the data from the data pairs
    Pd = pairs.Pd
    Pfa = pairs.Pfa

    # Define the priors
    Ph0 = h0toh1ratio /(h0toh1ratio + 1)
    Ph1 = 1 - Ph0

    # Identify the probability of a correct decision at each pair
    Pcds = []
    for i in range(0, len(Pd)):
        Pcds.append((Pd[i]*Ph1) + ((1-Pfa[i])*Ph0))

    # Identify the point that maximizes correct decision for the given prior
    maxIndex = 0
    for i, Pcd in enumerate(Pcds):
        if Pcd == max(Pcds):
            maxIndex = i

    # Return the best point
    return {'D': Pd[maxIndex], 'FA': Pfa[maxIndex], 'CD': round(max(Pcds),2)}


# Get the area under the curve for a set of ROC pairs
def getAUC(pairs):

    # Get and sort the pair data
    Pd = pairs.Pd
    Pfa = pairs.Pfa
    Pd.sort()
    Pfa.sort()

    # Use trapezoids to estimate AUC
    totalArea = 0
    for i in range(1, len(Pfa)):
        height = Pd[i] + Pd[i-1]
        width = Pfa[i] - Pfa[i-1]
        totalArea = totalArea + ((height * width)/2)

    # Return the data
    return round(totalArea, 2)


# Generates a single ROC curve using the specified threshold method
def plotROC(scores, truth, prior_ratios=[]):
    pairs = getROCPairs(scores, truth)
    plt.plot(pairs.Pfa , pairs.Pd, linewidth=1)

    # Add 0.5 if needed
    if len(prior_ratios) == 0:

        prior_ratios.append(0.5)

    # Go through each of the ratio
    for prior_ratio in prior_ratios:

        # Get the pax operating point and add it as a new point to the plot
        maxPoint = getMaxOperatingPoint(pairs, prior_ratio)
        label = "0-1 Ratio: " + str(prior_ratio) + " Pcd=" + str(maxPoint['CD'])
        plt.scatter(maxPoint['FA'], maxPoint['D'], color="black", label=label)

    # Add the chance diagonal
    pfa_chance = np.arange(0, 1, 0.01)
    pd_chance = np.arange(0, 1, 0.01)
    plt.plot(pfa_chance, pd_chance, linestyle=":", label="Chance")

    auc = getAUC(pairs)
    plt.title("ROC Plot, AUC: " + str(auc))

    # Plot the curve
    plt.legend(loc='best')