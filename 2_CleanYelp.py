import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.metrics import edit_distance

if __name__ == "__main__":

    # Read the data in
    businesses = pd.read_csv("Data/Yelp/Businesses.csv")
    categories = pd.read_csv("Data/Yelp/Categories.csv")
    reviews = pd.read_csv("Data/Yelp/Reviews.csv")

    ##

    # Process Businesses
    ##

    # Remove alias, a fairly redundant variable
    businesses.drop(['alias'], axis=1, inplace=True)

    # Turn price into a binary variable
    businesses = pd.get_dummies(businesses, columns=["price"])

    # Drop duplicates (arise from incorrect pulls). Temporarily ignores the wake_name for now.
    # Sort the businsesse by edit distance between names, in order to more likely remove
    # the unmatched cases
    yelp_name_lower = [name.lower() for name in businesses['yelp_name']]
    wake_county_name_lower = [name.lower() for name in businesses['wake_county_name']]
    businessesOrder = []
    for i in range(0, len(yelp_name_lower)):
        distance = edit_distance(yelp_name_lower[i], wake_county_name_lower[i])
        businessesOrder.append(distance)
    businesses['order'] = businessesOrder
    businesses.sort_values(['order'], inplace=True)
    businesses.drop('order', axis=1, inplace=True)

    wake_county_name = businesses['wake_county_name']
    businesses.drop(columns=['wake_county_name'], inplace=True)
    restaurant_open_date = businesses['restaurant_open_date']
    businesses.drop(columns=['restaurant_open_date'], inplace=True)

    businesses = businesses.drop_duplicates()
    categories = categories.drop_duplicates()
    reviews = reviews.drop_duplicates()

    businesses.insert(2, 'wake_county_name', wake_county_name)
    businesses.insert(5, 'restaurant_open_date', restaurant_open_date)


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

    ##
    # Process Reviews
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
    reviewBOW['business_id'] = reviews['business_id']
    reviewBOW = pd.pivot_table(reviewBOW, index='business_id', aggfunc=np.mean)
    reviewBOW.columns = [("review_" + col) for col in reviewBOW.columns]
    reviewBOW.reset_index(inplace=True)

    ##
    # Finalize dataset
    ##

    # Combine the data sets together
    full = pd.merge(businesses, categories, on="business_id")
    full = pd.merge(full, reviewBOW, on="business_id")

    # Add the name name matching variable
    bad_match = ''
    full.insert(0, 'bad_name_match', bad_match)

    # Write data
    full.to_csv("Data/IncludesBadMatches/NeedsBadMatchRemoval.csv", index=False)









