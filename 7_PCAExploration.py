import pandas as pd
import numpy as np

from sklearn.manifold import TSNE
from sklearn import neighbors
import matplotlib.pyplot as plt

import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA



def generate_plot(outfile):
    plt.savefig(outfile)
    plt.close()


if __name__ == "__main__":

    # Read the data in
    reviews = pd.read_csv("Data/Yelp/Reviews.csv")

    ##
    # Recreate the bag of words representation
    ##

    # Pull the most recent three reviews for each business
    reviewText = reviews['text']

    # Turn the review text into word tokens
    reviewText = [word_tokenize(review) for review in reviewText]

    # Turn text into lowercase
    reviewText = [[word.lower() for word in text] for text in reviewText]

    # Remove the stopwords
    stopwords_eng = stopwords.words('english')
    reviewText = [[word for word in text if word not in stopwords_eng] for text in reviewText]

    # Turn text into word stems and recombine
    stemmer = PorterStemmer()
    reviewText = [[stemmer.stem(word) for word in text] for text in reviewText]
    reviewText = [" ".join(text) for text in reviewText]

    # Turn reviews into bag-of-words representations, keeping only the 1000 most common words
    vectorizer = CountVectorizer(max_features=1000)
    reviewBOW = vectorizer.fit_transform(reviewText).toarray()

    # Get the bag-of-words representation into a dataframe
    reviewBOW = pd.DataFrame(reviewBOW)
    reviewBOW.columns = vectorizer.get_feature_names()

    ##
    # Get the outcome data
    ##

    outcomes = pd.read_csv("Data/Full/FoodInspectionsPCA300.csv")
    outcomes = outcomes[['business_id', 'had_critical_violation']]

    ##
    # PCA
    ##

    # Perform PCA with increasing number of components
    explainedVar = 0
    comps = 1
    precisions = []
    compsCount = []
    while explainedVar < 90:

        # Add five components to the count.
        comps = comps + 5
        compsCount.append(comps)

        # Perform PCA
        pca = PCA(n_components=comps)
        pca.fit(reviewBOW)

        # Count the total variance
        explainedVar = max(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3) * 100))

        # Get it into a data frame and sum across the reviews
        reviewPCA = pd.DataFrame(pca.fit_transform(reviewBOW))
        reviewPCA['business_id'] = reviews['business_id']
        reviewPCA = pd.pivot_table(reviewPCA, index='business_id', aggfunc=np.mean)

        # Perform TSNE
        reviewTSNE = TSNE(n_components=2).fit_transform(reviewPCA)
        reviewTSNE = pd.DataFrame(reviewTSNE)
        reviewTSNE['business_id'] = reviewPCA.index
        reviewTSNE = pd.merge(reviewTSNE, outcomes, how='inner')

        # Perform KNN on the TSNE
        reviewTSNEData = reviewTSNE[[0,1]]
        reviewTSNEOutcome = reviewTSNE['had_critical_violation']
        reviewKNN = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='brute')
        reviewKNN.fit(reviewTSNEData, reviewTSNEOutcome)

        # Identify the precision in the model
        noViolationPredictions = reviewKNN.predict(reviewTSNEData)[reviewTSNEOutcome == 0]
        precision = sum(noViolationPredictions == 0) / len(noViolationPredictions)
        precisions.append(precision)

    # Plot the resulting interplay between the number of components and the identified precision
    plt.scatter(compsCount, precisions)
    plt.xlabel("Number of Primary Components")
    plt.ylabel("Detection Rate for Non-Violation Restaurants")
    plt.title("Identifying Choice of Components")
    generate_plot("Figures/ReviewPCA/ComponentQuality.png")


    # Plot the t-SNE with the limited number of components
    reviewTSNE = TSNE(n_components=2).fit_transform(reviewPCA)
    reviewTSNE = pd.DataFrame(reviewTSNE)
    reviewTSNE['business_id'] = reviews['business_id']
    reviewTSNE = pd.merge(reviewTSNE, outcomes, how='inner')
    plt.scatter(reviewTSNE[0][reviewTSNE['had_critical_violation'] == 1],
                reviewTSNE[1][reviewTSNE['had_critical_violation'] == 1],
                label='Had Critical Violation')
    plt.scatter(reviewTSNE[0][reviewTSNE['had_critical_violation'] == 0],
                reviewTSNE[1][reviewTSNE['had_critical_violation'] == 0],
                label='None')
    plt.legend(loc="best")
    plt.title("T-SNE for 100 PCA Terms")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    generate_plot("Figures/ReviewPCA/TSNE-100PCA.png")

    # Use 100 as the chosen number of components and perform the TSNE
    pca = PCA(n_components=100)
    reviewPCA = pca.fit_transform(reviewBOW)
    reviewPCA = pd.DataFrame(reviewPCA)

    # Go through the components and identify the words that have highest loadings
    ComponentNo = []
    TopWords = []
    i = 0
    components = pca.components_
    for component in components:
        topLoadings = np.argsort(component)[::-1][0:10]
        words = np.array(list(reviewBOW))
        topWords = words[topLoadings]
        i = i + 1
        ComponentNo.append(i)
        TopWords.append(', '.join(topWords.tolist()))
    componentList = pd.DataFrame({'Component': ComponentNo, 'Words': TopWords})
    componentList.to_csv("Data/PCA/TopComponents.csv", index=False)

    #Collapse PCAs by review
    reviewPCA['business_id'] = reviews['business_id']
    reviewPCA = pd.pivot_table(reviewPCA, index='business_id', aggfunc=np.mean)
    reviewPCA.columns = [("review_pca_" + str(col)) for col in reviewPCA.columns]
    reviewPCA.reset_index(inplace=True)

    #Read the original data and remove the review problems
    foodInspections = pd.read_csv("Data/Full/FoodInspections.csv")
    reviewCols = [col for col in list(foodInspections) if re.search('review_', col)]
    foodInspections.drop(columns=reviewCols, inplace=True)

    #Write the reduced PCA output
    foodInspectionsPCA100 = pd.merge(foodInspections, reviewPCA)

    foodInspectionsPCA100.to_csv("Data/Full/FoodInspectionsPCA100.csv", index=False)