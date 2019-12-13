import pandas as pd
from nltk.metrics import edit_distance

if __name__ == "__main__":

    # Read the data in
    businesses = pd.read_csv("Data/Yelp/Businesses.csv")
    categories = pd.read_csv("Data/Yelp/Categories.csv")

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
    years_open = businesses['years_open']
    businesses.drop(columns=['years_open'], inplace=True)

    businesses = businesses.drop_duplicates()
    categories = categories.drop_duplicates()

    businesses.insert(2, 'wake_county_name', wake_county_name)
    businesses.insert(5, 'years_open', years_open)


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


    ##
    # Finalize dataset
    ##

    # Combine the data sets together
    full = pd.merge(businesses, categories, on="business_id")

    # Add the name name matching variable
    bad_match = ''
    full.insert(0, 'bad_name_match', bad_match)

    # Write data
    full.to_csv("Data/IncludesBadMatches/NeedsBadMatchRemoval.csv", index=False)









