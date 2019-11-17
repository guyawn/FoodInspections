import json
import requests
import csv
import pandas as pd
import re
import urllib.request

# Geocode information for the restaurants. Includes the date a restaurant was founded.
restaurant_locations_url = "https://opendata.arcgis.com/datasets/124c2187da8c41c59bde04fa67eb2872_0.csv?outSR=%7B%22wkid%22%3A102719%2C%22latestWkid%22%3A2264%7D"

# The list of all violations by restaurants
restaurant_violations_url = "https://opendata.arcgis.com/datasets/9b04d0c39abd4e049cbd4656a0a04ba3_2.csv?outSR=%7B%22wkid%22%3A102719%2C%22latestWkid%22%3A2264%7D"


if __name__ == "__main__":

    # Download data files from the Wake county ARCGIS site
    urllib.request.urlretrieve(restaurant_locations_url, 'Data/Wake_County/Restaurants_in_Wake_County__Locations.csv')
    urllib.request.urlretrieve(restaurant_violations_url, 'Data/Wake_County/Restaurants_in_Wake_County__Violations.csv')

    # Read files in
    locations = pd.read_csv("Data/Wake_County/Restaurants_in_Wake_County__Locations.csv")
    violations = pd.read_csv("Data/Wake_County/Restaurants_in_Wake_County__Violations.csv")

    # Remove unneeded variables from locations and filter to only restaurants
    # (the original file includes things like school cafeterias and food festival booths)
    locations = locations.loc[locations['FACILITYTYPE'] == 'Restaurant']
    locations.drop(columns=['OBJECTID', 'ADDRESS2', 'STATE', 'PHONENUMBER', 'POSTALCODE',
                            'PHONENUMBER','GEOCODESTATUS', 'PERMITID', 'FACILITYTYPE'], inplace=True)

    # Limit only to violations in 2018
    violations['INSPECTYEAR'] = [date[0:4] for date in violations['INSPECTDATE']]
    violations = violations.loc[violations['INSPECTYEAR'] == '2018']

    # Limit only to critical violations
    violations = violations.loc[violations['CRITICAL'] == 'Yes']

    # Transform data into binary for had critical violation or not
    violations = pd.DataFrame(violations['HSISID'].value_counts())
    violations.reset_index(inplace=True)
    violations.columns = ['HSISID', 'CRITICAL_VIOLATIONS']

    # Merge the datasets together and replace NaN with 0 (for no critical violations
    wake_county = pd.merge(locations, violations, on="HSISID", how="left")
    wake_county.fillna(0, axis=1, inplace=True)

    # Output the combined dataset
    wake_county.to_csv("Data/Wake_County/Restaurants_in_Wake_County__Combined.csv", index=False)