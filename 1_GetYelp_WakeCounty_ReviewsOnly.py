import json
import requests
import pandas as pd


# Yelp Review Search Endpoint (see https://www.yelp.com/developers/documentation/v3/business_reviews)
def api_host_reviews(id):
    return 'https://api.yelp.com/v3/businesses/' + str(id) + '/reviews'


if __name__ == "__main__":

    # Read API keys and create authorization header
    with open('config.json') as f:
        config = json.load(f)
        api_key = config['api']['api_key']
    headers = {
        'Authorization': 'Bearer %s' % api_key
    }

    reviews = pd.DataFrame()


    businesses = pd.read_csv("data/Yelp/Businesses.csv")
    businesses.drop_duplicates(inplace=True)
    businesses.index = range(0, businesses.shape[0])

    counter = 0
    for i in range(businesses.shape[0]):

        print(i, " of ", businesses.shape[0])

        business_id = businesses.loc[i]['business_id']

        responseReviews = requests.request('GET', api_host_reviews(business_id), headers=headers).json()

        if 'reviews' in responseReviews.keys():

            for review in responseReviews['reviews']:
                reviews = reviews.append({
                    'business_id': business_id,
                    'review_id': review['id'],
                    'text': review['text'],
                    'rating': review['rating'],
                    'time_created': review['time_created']
                }, ignore_index=True)

                counter += 1

                if (counter % 100) == 0:
                    reviews.to_csv("data/Yelp/Reviews.csv",
                                   columns=['business_id', 'review_id', 'text', 'rating', 'time_created'],
                                   index=False)

    reviews.to_csv("data/Yelp/Reviews.csv",
                   columns=['business_id', 'review_id', 'text', 'rating', 'time_created'],
                   index=False)

