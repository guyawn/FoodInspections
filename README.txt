ECE 590-13: 
Food Inspections Prediction using SoTA NLP
------------------------------------------

This branch represents an extention of the original project, with work performed and presented for the
final project of the Text Data and Analytics Course of the Duke ECE Masters.

The starting pont for this data was catalog of all restaurants in Wake County, North Carolina, 
found at https://catalog.data.gov/dataset/restaurants-in-wake-county-yelp.

Using the Yelp Business search API, these resetaurants were queried to acquire a large account
of the information available to Yelp users. This includes location data, review counts, 
pricing, category, and the text of the three most recent reviews. 

Data were cleaned to remove non-restaurants, as well as any duplicates identified during the 
API pulls. Categories were one-hot encoded (note, some restaurants have listed multiple categories).
Reviews were transformed into a bag-of-words representation, and the first 1000 features were 
selected. Note that both the cateogires and reviews result in fairly sparse representations.

The scripts in the main directory will create the Wake county dataset. To run 5_EmbedNC, you'll
need to run the scripts in the MakeEmbeddings directory as well. 

Scripts should be run in the specified order; note that running them will take a couple days (to
pull all of the data from yelp), since you'll hit the rate limit during the data pull. 

Files 6 and 7 represent the modelling. These require running on a GPU. I used Google Colab to do so;
colab only accepts Jupyter notebooks, hence why these are not standard python files. 

Contact Me
----------

If you have any additional questions about the data or the processes used to build them, please e-mail
me at gayan.seneviratna@duke.edu.
