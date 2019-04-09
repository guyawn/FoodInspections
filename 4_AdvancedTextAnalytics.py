import pandas as pd
import numpy as np
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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
    # PCA
    ##

    explainedVar = 0
    comps = 1
    while explainedVar < 80:
        print(explainedVar)
        comps = comps + 1
        pca = PCA(n_components=comps)
        pca.fit(reviewBOW)
        explainedVar = max(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100))

    pca = PCA(n_components=comps)
    reviewPCA = pca.fit_transform(reviewBOW)
    reviewPCA = pd.DataFrame(reviewPCA)

    reviewPCA['business_id'] = reviews['business_id']
    reviewPCA= pd.pivot_table(reviewPCA, index='business_id', aggfunc=np.mean)
    reviewPCA.columns = [("review_pca_" + str(col)) for col in reviewPCA.columns]
    reviewPCA.reset_index(inplace=True)

    # Showing this is no longer a sparse representation
    sum(reviewPCA.values == 0)

    ##
    # Doc2Vec
    ##

    # Recombine and turn into NLTK documents
    reviewTexts = [" ".join(text) for text in reviewText]
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(reviewTexts)]

    # Build the doc2vec embeddings
    model = Doc2Vec(documents, vector_size=comps, window=2, min_count=1, workers=4)
    docEmbeddings = [model.infer_vector(reviewText.split(" ")) for reviewText in reviewTexts]

    reviewDoc2Vec = pd.DataFrame(docEmbeddings)
    reviewDoc2Vec['business_id'] = reviews['business_id']
    reviewDoc2Vec=pd.pivot_table(reviewDoc2Vec, index='business_id', aggfunc=np.mean)
    reviewDoc2Vec.columns = [("review_d2v_" + str(col)) for col in reviewDoc2Vec]
    reviewDoc2Vec.reset_index(inplace=True)

    ##
    # Finalize datasets
    ##

    inspections = pd.read_csv('Data/FoodInspections.csv')
    reviewColumns = [var for var in list(inspections) if re.match("review_.*", var)]
    inspections.drop(columns = reviewColumns, inplace=True)

    # Combine the data sets together
    inspectionsPCA = pd.merge(inspections, reviewPCA, on="business_id")
    inspectionsD2V = pd.merge(inspections, reviewDoc2Vec, on="business_id")

    # Add the name name matching variable
    # Write data
    inspectionsPCA.to_csv("Data/FoodInspectionsPCA300.csv", index=False)
    inspectionsD2V.to_csv("Data/FoodInspectionsD2V.csv", index=False)










