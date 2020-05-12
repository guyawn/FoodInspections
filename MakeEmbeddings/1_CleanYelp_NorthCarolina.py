import pandas as pd
from nltk import sent_tokenize
from nltk.stem.porter import PorterStemmer
import re

if __name__ == "__main__":

    # Read the data in
    businesses = pd.read_csv("Data/Yelp/BusinessesNC.csv")
    categories = pd.read_csv("Data/Yelp/CategoriesNC.csv")
    reviews = pd.read_csv("Data/Yelp/ReviewsNC.csv")

    ##

    # Process Businesses
    ##

    # Remove alias, a fairly redundant variable
    businesses.drop(['alias'], axis=1, inplace=True)

    # Turn price into a binary variable
    businesses = pd.get_dummies(businesses, columns=["price"])

    # Drop duplicates (arise from incorrect pulls).
    # Sort the busineses by edit distance between names, in order to more likely remove
    # the unmatched cases
    businesses = businesses.drop_duplicates()
    categories = categories.drop_duplicates()
    reviews = reviews.drop_duplicates()

    businesses = businesses.reset_index(drop=True)
    categories = categories.reset_index(drop=True)
    reviews = reviews.reset_index(drop=True)

    ##
    # Process categories
    ##

    # Turn the set of identified categories into a binary vector for each business
    categories = pd.crosstab(categories['business_id'], categories['alias'])
    categories.columns = [("category_" + col) for col in categories.columns]
    categories.reset_index(inplace=True)

    # Remove categories that imply this isn't a restaurant
    # Will cause the merge to later filter out non-restaurants.
    with open('Data/Utils/stopCategories.txt') as f:
        stopCategories = f.read().splitlines()
        stopCategories = [category for category in stopCategories if category in categories.columns]
        categories.drop(stopCategories, axis=1, inplace=True)
        categories = categories.loc[categories.sum(axis=1) != 0]

    # Combine the data sets together
    businesses = pd.merge(businesses, categories, on="business_id")
    reviews = pd.merge(reviews, businesses, on="business_id")

    ##
    # Process Reviews
    ##

    # Pull the most recent three reviews for each business
    reviewText = reviews['text']

    # Turn the review text into sentences
    reviewSentences = [sent_tokenize(review) for review in reviewText]
    reviewSentences = [item for sublist in reviewSentences for item in sublist]

    # Turn text into lowercase
    reviewSentences = [sentence.lower() for sentence in reviewSentences]
    reviewSentences = [re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()]', '', sentence) for sentence in reviewSentences]
    reviewSentences = [re.sub(r'[^\x00-\x7F]+', ' ', sentence) for sentence in reviewSentences]

    # Write data to doc lines format
    with open('Data/EmbeddingModels/reviews_nc.cor', 'w') as f:
        for sentence in reviewSentences:
            f.write(sentence)
            f.write("\n")

    # Saved stemmed version of the text
    stemmer = PorterStemmer()
    reviewSentences = [' '.join([stemmer.stem(word) for word in sentence.split(' ')]) for sentence in reviewSentences]

    # Write data to doc lines format
    with open('Data/EmbeddingModels/reviews_nc_stemmed.cor', 'w') as f:
        for sentence in reviewSentences:
            f.write(sentence)
            f.write("\n")




