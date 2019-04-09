import pandas as pd
import numpy as np
import re
from nltk.metrics import edit_distance

if __name__ == "__main__":

    # Read the Wake County data
    wake_county_locations = pd.read_csv("Data/Wake_County/Restaurants_in_Wake_County__Locations.csv")

    # Number of restaurants in Wake County
    sum(wake_county_locations['FACILITYTYPE'] == 'Restaurant')

    # Read the final dataset
    inspections = pd.read_csv('Data/Full/FoodInspections.csv')

    # Number of restaurants
    len(inspections)

    #Number of unique names
    len(set(inspections['yelp_name'].values.tolist()))

    #Highest edit distance between names
    yelp_name_lower = [name.lower() for name in inspections['yelp_name']]
    wake_county_name_lower = [name.lower() for name in inspections['wake_county_name']]
    editDistances = []
    for i in range(0, len(yelp_name_lower)):
        distance = edit_distance(yelp_name_lower[i], wake_county_name_lower[i])
        editDistances.append(distance)
    max(editDistances)

    # Number of categories
    len([col for col in inspections.columns if bool(re.search("category", col))])

    # Number of restaurants with a critical violation
    sum(inspections['had_critical_violation'])
    sum(inspections['had_critical_violation']) / len(inspections)

    # Number of restaurants which closed
    sum(inspections['is_closed'])
    sum(inspections['is_closed']) / len(inspections)

    # Summary means and standard deviations
    np.mean(inspections['years_open'])
    np.std(inspections['years_open'])

    np.mean(inspections['review_count'])
    np.std(inspections['review_count'])

    np.mean(inspections['rating'])
    np.std(inspections['rating'])

    np.min(inspections['latitude'])
    np.max(inspections['latitude'])
    np.min(inspections['longitude'])
    np.max(inspections['longitude'])

    # Number of cities and zip codes
    len(set(inspections['location_city'].values.tolist()))
    len(set(inspections['location_zip_code'].values.tolist()))

    # Counts at each price point
    sum(inspections['price_$'])
    sum(inspections['price_$$'])
    sum(inspections['price_$$$'])
    sum(inspections['price_$$$$'])
    sum(inspections['price_?'])

    #Find review sparsity
    zeroReviewWords = sum(sum(inspections.filter(regex="review_.*").values == 0))
    reviewWordsAll = len(inspections) * inspections.filter(regex="review_.*").shape[1]
    zeroReviewWords / reviewWordsAll