from bert_embedding import BertEmbedding
import pandas as pd
import torch
import os
import re
import pickle
import numpy as np


if __name__ == "__main__":

    inspections = pd.read_csv("Data/FoodInspections.csv")
    reviews = pd.read_csv("Data/Yelp/Reviews.csv")

    reviews = reviews.loc[reviews['business_id'].isin(inspections['business_id'])]
    reviews = reviews.reset_index(drop=True)

    outcome = [(inspections.loc[review_id == inspections['business_id']]['had_critical_violation']).tolist()[0]
               for review_id in reviews['business_id']]

    reviewTexts = reviews['text'].tolist()
    reviewTexts = [re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()]', '', text) for text in reviewTexts]
    reviewTexts = [re.sub(r'\n', '; ', text) for text in reviewTexts]

    # load the google w2v model
    model = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')

    sentencesEmbedded = model(reviewTexts)

    maxEmbeddingLength = max([len(sentenceEmbedded[1]) for sentenceEmbedded in sentencesEmbedded])

    # Matrix to store the encoded inputs
    reviewBERT_matrix = -np.ones((len(sentencesEmbedded),
                                  maxEmbeddingLength,
                                  sentencesEmbedded[0][1][0].shape[0]),
                                 dtype=np.float32)

    for i in range(len(sentencesEmbedded)):

        if i % 1000 == 0:
            print(i, " of ", len(sentencesEmbedded))

        for j in range(len(sentencesEmbedded[i][0])):
            reviewBERT_matrix[i, j, :] = sentencesEmbedded[i][1][j]

    # Convert to tensor
    X = torch.tensor(reviewBERT_matrix)
    y = torch.tensor(outcome)

    if not os.path.exists("Data/Tensors/BERT"):
        os.mkdir("Data/Tensors/BERT")

    with open("Data/Tensors/BERT/X.pkl", "wb") as f:
        pickle.dump(X, f)

    with open("Data/Tensors/BERT/y.pkl", "wb") as f:
        pickle.dump(y, f)
