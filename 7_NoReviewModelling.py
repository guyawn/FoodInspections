import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    # Read the food inspections data
    foodInspections = pd.read_csv("Data/FoodInspections.csv")

    # Drop unecessary varaibles
    foodInspections.drop(columns=['business_id',
                                  'yelp_name',
                                  'wake_county_name',
                                  'is_closed',
                                  'location_city',
                                  'location_zip_code',
                                  'url',
                                  'price_$$$',
                                  'price_$$$$',
                                  'price_?'], inplace=True)

    # Extract the reviews out
    review_columns = [i for i, name in enumerate(list(foodInspections)) if re.search('review_', name)]
    review_columns = review_columns[1:]
    reviews = foodInspections.iloc[:, review_columns]
    foodInspections.drop(columns=list(reviews), inplace=True)

    # Extract the categories out
    category_columns = [i for i, name in enumerate(list(foodInspections)) if re.search('category_', name)]
    categories = foodInspections.iloc[:, category_columns]
    foodInspections.drop(columns=list(categories), inplace=True)

    # Find the top categories
    top_category_names = categories.sum().sort_values(ascending=False)[1:25].index.values
    top_categories = categories[top_category_names]

    # Add the categories back
    foodInspections = pd.concat([foodInspections, top_categories], axis=1)

    # Remove the outcome
    had_critical_violation = foodInspections['had_critical_violation']
    foodInspections.drop(columns=['had_critical_violation'], inplace=True)

    # Split out the validation set
    X_train, X_test, y_train, y_test = train_test_split(foodInspections, had_critical_violation, test_size=0.2, shuffle=True)

    #