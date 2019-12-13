import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import torch
import os
import pickle

if __name__ == "__main__":

    inspections = pd.read_csv("Data/FoodInspections.csv")
    reviews = pd.read_csv("Data/Yelp/Reviews.csv")

    reviews = reviews.loc[reviews['business_id'].isin(inspections['business_id'])]
    reviews = reviews.reset_index(drop=True)

    outcome = [(inspections.loc[review_id == inspections['business_id']]['had_critical_violation']).tolist()[0]
               for review_id in reviews['business_id']]

    reviewTexts = reviews['text']
    reviewIds = reviews['business_id']

    # Turn the review text into word tokens
    reviewWords = [word_tokenize(text) for text in reviewTexts]

    # Turn text into lowercase
    reviewWords = [[word.lower() for word in sentence] for sentence in reviewWords]

    # Remove the stopwords
    stopwords_eng = stopwords.words('english')
    reviewWords = [[word for word in sentence if word not in stopwords_eng] for sentence in reviewWords]

    # Turn text into word stems and recombine
    stemmer = PorterStemmer()
    reviewStems = [[stemmer.stem(word) for word in sentence] for sentence in reviewWords]
    reviewTextStemmed = [" ".join(text) for text in reviewStems]

    # Turn reviews into bag-of-words representations, keeping only the 5550 most common words
    vectorizer = CountVectorizer(max_features=5550)
    reviewBOW = vectorizer.fit_transform(reviewTextStemmed).toarray()

    # Find the maximum text length
    maxTextLength = max([len(sentence) for sentence in reviewStems])

    # Capture reviews in a matrix
    review_BOW_matrix = -np.ones((len(reviewTexts),
                                  maxTextLength,
                                  5550), dtype=np.int8)

    # Encode
    for i in range(len(reviewTexts)):
        for j in range(len(reviewStems[i])):
            review_BOW_matrix[i][j] = vectorizer.transform([reviewStems[i][j]]).todense().astype(np.int8)

    # Convert to tensor
    X = torch.tensor(review_BOW_matrix)
    y = torch.tensor(outcome)

    if not os.path.exists("Data/Tensors/BOW"):
        os.mkdir("Data/Tensors/BOW")

    with open("Data/Tensors/BOW/X.pkl", "wb") as f:
        pickle.dump(X, f)

    with open("Data/Tensors/BOW/y.pkl", "wb") as f:
        pickle.dump(y, f)
