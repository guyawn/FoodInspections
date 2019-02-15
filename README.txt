ECE 681: 
Food Inspections Prediction
---------------------------

This github project serves as a reproducible proposal for an Pattern Recognition course completed 
for the Duke Masters in Electrical Engineering. 

The starting pont for this data was catalog of all restaurants in Wake County, North Carolina, 
found at https://catalog.data.gov/dataset/restaurants-in-wake-county-yelp.

Using the Yelp Business search API, these resetaurants were queried to acquire a large account
of the information available to Yelp users. This includes location data, review counts, 
pricing, category, and the text of the three most recent reviews. 

Data were cleaned to remove non-restaurants, as well as any duplicates identified during the 
API pulls. Categories were one-hot encoded (note, some restaurants have listed multiple categories).
Reviews were transformed into a bag-of-words representation, and the first 1000 features were 
selected. Note that both the cateogires and reviews result in fairly sparse representations.

Scripts for pulling (GetYelp.py) and cleaning data (CleanYelp.py) are presented for reproducibility.


Available Data
--------------

Wake_County/Restaurants_in_Wake_County__Location.csv - Data provided by Data-Wake.opendata.arcgis.com.
Has restaurant list and details

Wake_County/Restaurants_in_Wake_County__Violations.csv - Data provided by Data-Wake.opendata.arcgis.com.
Has the set of violations committed by each restauratn

Yelp/Businesses.csv - The raw business data pulled from Yelp. One row per restaurant. 
See https://www.yelp.com/developers/documentation/v3/business_search

Yelp/Categories.csv - The raw category data pulled from Yelp. 1 - 3 rows per restaurant.
See https://www.yelp.com/developers/documentation/v3/all_category_list

Yelp/Reviews.csv - The raw review data pulled from Yelp. 0-3 rows per restaurant 

Utils/stopCategories.txt - Non restaurant categories that were removed (see CleanYelp.py)

FoodInspections.csv - "Analytics-ready" dataset for this project. One row per restaurant.
	Only includes businesses with at least one category and one review.


Extending the Effort 
--------------------

For those interested in replicating the data pull, you'll need to create an application 
and get a api_key from the Yelp developers site. You can then put it into the format specified
in config-format.json and rename the file config.json. That will allow you to pull from your 
own account. 

The natural language processing pipeline here, used to build RestaurantClosures.csv was pretty
basic, Intro-to-NLP-level processing. One major room for additional development is the application
of additional techniques to extract information from the reviews.







