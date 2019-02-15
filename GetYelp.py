import json
import requests
import csv
import pandas as pd
import re

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

    # Open the dataset from Wake County (
    with open('data/Wake_County/Restaurants_in_Wake_County__Combined2.csv', 'r') as csvRead:

        # DataFrames to store the Yelp output
        businesses = pd.DataFrame()
        categories = pd.DataFrame()
        reviews = pd.DataFrame()
        i = 0

        # Read rows from the Wake county data
        # See(http://data-wake.opendata.arcgis.com/datasets/restaurants-in-wake-county-yelp)
        reader = csv.reader(csvRead, delimiter=',')
        csvRead.seek(0)
        next(csvRead)
        for row in reader:

            # Build the query based on the restaurant name and address
            restaurantName = re.sub("#\d+", "", row[1])
            restaurantLongitude = row[5]
            restaurantLatitude = row[6]

            url_params = {
                'term': restaurantName,
                'latitude': restaurantLatitude,
                'longitude': restaurantLongitude,
                'radius': 500,
                'sort_by': 'best_match',
                'limit': 5
            }

            # Performs the query.
            response = requests.request('GET', api_host_search, headers=headers, params=url_params).json()
            if 'businesses' in response:

                # If at least one restaurant was found
                if len(response['businesses']) > 0:

                    # Track the processing
                    i = i + 1
                    print(str(i) + " Input Name: " + restaurantName + ", Yelp Name: ", response['businesses'][0]['name'])

                    # Some restaurants don't have price listed. Use ? instead
                    price = response['businesses'][0]['price'] if "price" in (response['businesses'][0]).keys() else "?"

                    # Add business data
                    businesses = businesses.append({
                        'business_id': response['businesses'][0]['id'],
                        'alias': response['businesses'][0]['alias'],
                        'yelp_name': response['businesses'][0]['name'],
                        'wake_county_name' : restaurantName,
                        'had_critical_violation': row[7],
                        'restaurant_open_date': row[4],
                        'is_closed': response['businesses'][0]['is_closed'],
                        'review_count': response['businesses'][0]['review_count'],
                        'rating': response['businesses'][0]['rating'],
                        'latitude': response['businesses'][0]['coordinates']['latitude'],
                        'longitude': response['businesses'][0]['coordinates']['longitude'],
                        'price': price,
                        'location_city': response['businesses'][0]['location']['city'],
                        'location_zip_code': response['businesses'][0]['location']['zip_code'],
                        'url': response['businesses'][0]['url']},
                        ignore_index=True)

                    # For each business category, save it in another dataset
                    responseCategories = response['businesses'][0]['categories']
                    for category in responseCategories:
                        categories = categories.append({
                            'business_id': response['businesses'][0]['id'],
                            'alias': category['alias'],
                            'title': category['title']
                        },
                            ignore_index=True)

                    # Each restaurant has up to 3 reviews accessible. Get those and add them to a third DataFrame
                    responseReviews = requests.request('GET', api_host_reviews(response['businesses'][0]['id']),
                                                       headers=headers).json()
                    for review in responseReviews['reviews']:
                        reviews = reviews.append({
                            'business_id': response['businesses'][0]['id'],
                            'review_id': review['id'],
                            'text': review['text'],
                            'rating': review['rating'],
                            'time_created': review['time_created']
                        },
                            ignore_index=True)

                    # Write outputs to csv every 100 cases.
                    if (i % 100) == 0:
                        businesses.to_csv("data/Yelp/Businesses2.csv",
                                          columns=['business_id', 'alias', 'yelp_name', 'wake_county_name',
                                                   'had_critical_violation', 'restaurant_open_date',
                                                   'is_closed', 'review_count', 'rating',
                                                   'latitude', 'longitude', 'price', 'location_city',
                                                   'location_zip_code', 'url'],
                                          index=False)
                        categories.to_csv("data/Yelp/Categories2.csv",
                                          columns=['business_id', 'alias', 'title'],
                                          index=False)
                        reviews.to_csv("data/Yelp/Reviews2.csv",
                                       columns=['business_id', 'review_id', 'text', 'rating', 'time_created'],
                                       index=False)

    # Write the final set of outputs
    businesses.to_csv("data/Yelp/Businesses2.csv",
                      columns=['business_id', 'alias', 'yelp_name', 'wake_county_name',
                               'had_critical_violation', 'restaurant_open_date',
                                'is_closed', 'review_count', 'rating',
                               'latitude', 'longitude', 'price', 'location_city',
                               'location_zip_code', 'url'],
                      index=False)
    categories.to_csv("data/Yelp/Categories2.csv",
                      columns=['business_id', 'alias', 'title'],
                      index=False)
    reviews.to_csv("data/Yelp/Reviews2.csv",
                   columns=['business_id', 'review_id', 'text', 'rating', 'time_created'],
                   index=False)
