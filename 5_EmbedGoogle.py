import pandas as pd
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import gensim
import torch
import os
import pickle
import numpy as np

def encode_w2v(model, word):

    if word in model.vocab:
        return model.get_vector(word)
    else:
        return np.zeros(300, dtype=np.float32)


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

    # load the google w2v model
    model = gensim.models.KeyedVectors.load_word2vec_format('Data/EmbeddingModels/GoogleNews-vectors-negative300.bin', binary=True)

    # Run model on training words
    reviewW2V = [[encode_w2v(model, word) for word in sentence] for sentence in reviewWords]

    # Matrix to store the encoded inputs
    reviewW2V_matrix = -np.ones((len(reviewW2V),
                                max([len(x) for x in reviewWords]),
                                reviewW2V[0][0].shape[0]))

    for i in range(len(reviewW2V)):
        for j in range(len(reviewW2V[i])):
            reviewW2V_matrix[i, j, :] = reviewW2V[i][j]

    # Convert to tensor
    X = torch.tensor(reviewW2V_matrix)
    y = torch.tensor(outcome)

    if not os.path.exists("Data/Tensors/W2V_Google"):
        os.mkdir("Data/Tensors/W2V_Google")

    with open("Data/Tensors/W2V_Google/X.pkl", "wb") as f:
        pickle.dump(X, f)

    with open("Data/Tensors/W2V_Google/y.pkl", "wb") as f:
        pickle.dump(y, f)
