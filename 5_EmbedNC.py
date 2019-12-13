import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
import torch
import os
import pickle
import numpy as np
import glob
import gensim
import re


def encode_w2v(model, word):
    try:
        return model[word]
    except:
        return np.zeros(model.layer1_size, dtype=np.float32)


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

    # Get all of the W2V models trained on NC data.
    model_files = glob.glob("Data/EmbeddingModels/*model")

    for model_file in model_files:

        # Trace
        print("Embedding with " + model_file)

        # load the google w2v model
        model = gensim.models.Word2Vec.load(model_file)

        # Run model on training words
        reviewW2V = [[encode_w2v(model, re.sub('[.,\/#!$%\^&\*;:{}=\-_`~()]', '', word.lower())) for word in sentence]
                     for sentence in reviewWords]

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

        output_name = re.sub(".model", "", re.sub("Data/EmbeddingModels\\\\", "", model_file))

        if not os.path.exists("Data/Tensors/" + output_name):
            os.mkdir("Data/Tensors/" + output_name)

        with open("Data/Tensors/" + output_name + "/X.pkl", "wb") as f:
            pickle.dump(X, f)

        with open("Data/Tensors/" + output_name + "/y.pkl", "wb") as f:
            pickle.dump(y, f)
