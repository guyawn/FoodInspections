import json
import requests
import pandas as pd
import numpy as np

# Other Cities to Query
cities = ["Charlotte, NC", "Greensboro, NC", "Durham, NC", "Winston-Salem, NC",
          "Fayetteville, NC"]

cities_coordinates = [
    [(35.363455, -80.977736), (35.043558, -80.686807)],
    [(36.182138, -79.988422), (35.996020, -79.693273)],
    [(36.129520, -79.000237), (35.870256, -78.805146)],
    [(36.204910, -80.384233), (36.001845, -80.113049)],
    [(35.151583, -79.093694), (34.989393, -78.815789)]
]

# Yelp Business Search Endpoint (see https://www.yelp.com/developers/documentation/v3/business_search)
api_host_search = 'https://api.yelp.com/v3/businesses/search'


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

    # DataFrames to store the Yelp output
    businesses = pd.DataFrame()
    categories = pd.DataFrame()
    reviews = pd.DataFrame()

    counter = 0

    city_coordinates = cities_coordinates[0]
    for city_coordinates in cities_coordinates:

        print(city_coordinates)

        city_latitude_minimum = city_coordinates[0][0]
        city_latitude_maximum = city_coordinates[1][0]

        city_longitude_minimum = city_coordinates[0][1]
        city_longitude_maximum = city_coordinates[1][1]

        current_longitude = np.arange(city_longitude_minimum, city_longitude_maximum, 0.01)[12]
        current_latitude = np.arange(city_latitude_minimum, city_latitude_maximum, -0.01)[10]
        for current_longitude in np.arange(city_longitude_minimum, city_longitude_maximum, 0.01):
            for current_latitude in np.arange(city_latitude_minimum, city_latitude_maximum, -0.01):

                current_latitude = round(current_latitude, 3)
                current_longitude = round(current_longitude, 3)

                url_params = {
                    'latitude': current_latitude,
                    'longitude': current_longitude,
                    'radius': 500,
                    'sort_by': 'best_match',
                    'limit': 50
                }

                # Performs the query.
                response = requests.request('GET', api_host_search, headers=headers, params=url_params).json()
                if 'businesses' in response:

                    # If at least one restaurant was found
                    if len(response['businesses']) > 0:

                        business = response['businesses'][0]
                        for business in response['businesses']:

                            price = "?"
                            if 'price' in business.keys():
                                price = business['price']

                            # Save business data
                            businesses = businesses.append({
                                'business_id': business['id'],
                                'alias': business['alias'],
                                'yelp_name': business['name'],
                                'is_closed': business['is_closed'],
                                'review_count': business['review_count'],
                                'rating': business['rating'],
                                'latitude': business['coordinates']['latitude'],
                                'longitude': business['coordinates']['longitude'],
                                'price': price,
                                'location_city': business['location']['city'],
                                'location_zip_code': business['location']['zip_code'],
                                'url': business['url']},
                                ignore_index=True)

                            # For each business category, save it in another dataset
                            response_categories = business['categories']
                            for category in response_categories:
                                categories = categories.append({
                                    'business_id': business['id'],
                                    'alias': category['alias'],
                                    'title': category['title']
                                }, ignore_index=True)

                            counter += 1

                            # Write outputs to csv every 100 cases.
                            if (counter % 100) == 0:
                                businesses.to_csv("data/Yelp/BusinessesNC.csv",
                                                  columns=['business_id', 'alias', 'yelp_name', 'wake_county_name',
                                                           'had_critical_violation', 'years_open',
                                                           'is_closed', 'review_count', 'rating',
                                                           'latitude', 'longitude', 'price', 'location_city',
                                                           'location_zip_code', 'url'],
                                                  index=False)
                                categories.to_csv("data/Yelp/CategoriesNC.csv",
                                                  columns=['business_id', 'alias', 'title'],
                                                  index=False)

    # Write the final set of outputs
    businesses.to_csv("data/Yelp/BusinessesNC.csv",
                      columns=['business_id', 'alias', 'yelp_name',
                               'is_closed', 'review_count', 'rating',
                               'latitude', 'longitude', 'price',
                               'location_city', 'location_zip_code', 'url'],
                      index=False)
    categories.to_csv("data/Yelp/CategoriesNC.csv",
                      columns=['business_id', 'alias', 'title'],
                      index=False)

    businesses = pd.read_csv("data/Yelp/BusinessesNC.csv")
    businesses.drop_duplicates(inplace=True)
    businesses.index = range(0, businesses.shape[0])

    counter = 0
    for i in range(businesses.shape[0]):
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
                    reviews.to_csv("data/Yelp/ReviewsNC.csv",
                                   columns=['business_id', 'review_id', 'text', 'rating', 'time_created'],
                                   index=False)

    reviews.to_csv("data/Yelp/ReviewsNC.csv",
                   columns=['business_id', 'review_id', 'text', 'rating', 'time_created'],
                   index=False)

