import pandas as pd
import os.path

if __name__ == "__main__":

    #Check if the hand-coded name removal file exists.
    filePath = "Data/IncludesBadMatches/BadMatchRemoval.csv"

    if os.path.isfile(filePath):

        # Read the data in
        foodInspections = pd.read_csv(filePath)

        # Filter out the bad matches
        foodInspections = foodInspections.loc[foodInspections['bad_name_match'] != 'x']

        # Remove restaurants that were founded in 2018 or later
        foodInspections = foodInspections.loc[foodInspections['years_open'] > 0]

        # Remove some restaurants that got added in from San Francisco for some reason.
        foodInspections = foodInspections.loc[foodInspections['longitude'] > -85]

        #Drop the name matching column
        foodInspections.drop(['bad_name_match'], axis=1, inplace=True)

        # Write data
        foodInspections.to_csv("Data/FoodInspections.csv", index=False)

    else:

        # Message that
        print(""" Need to hand-code the bad name matches! 
        Go to Data/IncludesBadMatches/NeedsBadMatchRemoval.csv,'
        and fill the bad_name_match column, adding __ for the 
        rows you want to keep.
        Save your result as BadNameMatches.csv""")







