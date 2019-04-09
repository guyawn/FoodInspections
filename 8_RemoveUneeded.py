import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV

import Helpers.ROC as roc


if __name__ == "__main__":

    # Read the food inspections data
    FoodInspections = pd.read_csv("Data/FoodInspectionsPCA100.csv")

    # Drop Variables identified as unneeded during the EDA
    FoodInspections.drop(columns=['business_id',
                                  'yelp_name',
                                  'wake_county_name',
                                  'is_closed',
                                  'location_city',
                                  'location_zip_code',
                                  'url',
                                  'price_$$$',
                                  'price_$$$$',
                                  'price_?'], inplace=True)

    # Extract the categories out
    category_columns = [i for i, name in enumerate(list(FoodInspections)) if re.search('category_', name)]
    categories = FoodInspections.iloc[:, category_columns]
    FoodInspections.drop(columns=list(categories), inplace=True)

    # Find the top categories
    top_category_names = categories.sum().sort_values(ascending=False)[1:25].index.values
    top_categories = categories[top_category_names]

    # Add the categories back
    FoodInspections = pd.concat([FoodInspections, top_categories], axis=1)