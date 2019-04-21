import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn import preprocessing

def generate_plot(outfile, width=8, height=6):
    plt.savefig(outfile)
    plt.close()


if __name__ == "__main__":

    # Read the food inspections data
    FoodInspections = pd.read_csv("Data/Full/FoodInspectionsPCA300.csv")

    # Drop Variables identified as unneeded during the EDA
    FoodInspections.drop(columns=['yelp_name',
                                  'wake_county_name',
                                  'is_closed',
                                  'latitude',
                                  'longitude',
                                  'location_zip_code',
                                  'url',
                                  'price_$$',
                                  'price_$$$',
                                  'price_$$$$',
                                  'price_?'], inplace=True)
    FoodInspectionsOriginal = FoodInspections

    # Transform some variables

    # Log-normal Transform for years open
    FoodInspections['years_open_log'] = st.zscore(np.log(FoodInspections['years_open']))
    years_no = FoodInspections['years_open_log'][FoodInspections['had_critical_violation'] == 0]
    years_yes = FoodInspections['years_open_log'][FoodInspections['had_critical_violation'] == 1]
    years_yes.plot.kde(label="Had Critical Violation")
    years_no.plot.kde(label="None")
    plt.xlabel("Years Open, Z-Score of Log-Normal Transform")
    plt.legend(loc='best')
    plt.title("Distribution of Years Open")
    #generate_plot('Figures/Transform/YearsOpenLogZ.png')

    # Log-normal for review count
    FoodInspections['review_count_log'] = st.zscore((np.log(FoodInspections['review_count'])))
    years_no = FoodInspections['review_count_log'][FoodInspections['had_critical_violation'] == 0]
    years_yes = FoodInspections['review_count_log'][FoodInspections['had_critical_violation'] == 1]
    years_yes.plot.kde(label="Had Critical Violation")
    years_no.plot.kde(label="None")
    plt.xlabel("Review Count, Z-Score of Log-Normal Transform")
    plt.legend(loc='best')
    plt.title("Distribution of Review Count")
    #generate_plot('Figures/Transform/ReviewCountLogZ.png')

    #Rating z-score
    FoodInspections['rating'] = st.zscore(FoodInspections['rating'])
    countsNo = FoodInspections[FoodInspections['had_critical_violation'] == 0].groupby('rating'). \
        count().apply(lambda x: x / x.sum())
    countsYes = FoodInspections[FoodInspections['had_critical_violation'] == 1].groupby('rating'). \
        count().apply(lambda x: x / x.sum())
    countsNo.reset_index(inplace=True)
    countsYes.reset_index(inplace=True)
    plt.bar(countsYes['rating'],
            countsYes['business_id'],
            width=0.5,
            label='Had Critical Violation',
            edgecolor='k',
            alpha=0.5)
    plt.bar(countsNo['rating'],
            countsNo['business_id'],
            width=0.5,
            label='None',
            edgecolor='k',
            alpha=0.5)
    plt.legend(loc="best")
    plt.xlabel('Rating, Z-Scored')
    plt.ylabel("Frequency")
    plt.title("Distribution of Ratings")
    #generate_plot('Figures/Transform/RatingsZ.png')

    # Plotting categories by commonalisty
    CategoryFields = [field for field in list(FoodInspections) if re.match("category_.*", field)]
    CategoryData = FoodInspections[CategoryFields]
    CategoryData.columns = [re.sub("category_", "", name) for name in list(CategoryData)]
    CategorySum = CategoryData.sum()
    CategorySum = CategorySum[CategorySum > 10]
    pd.DataFrame(CategorySum).sort_values(0, ascending=True).plot.barh()
    plt.xlim(0, 240)
    plt.xticks(fontsize=4, rotation=90)
    plt.title("Categories with more than 10 cases")
    plt.ylabel("Category")
    plt.xlabel("Count")
    plt.gca().get_legend().remove()
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.xticks(fontsize=8)
    #generate_plot('Figures/Transform/CategoryLimited.png')

    # City One-hot and limit
    LocationCity = FoodInspections['location_city'].values
    LocationCity = pd.get_dummies(LocationCity)
    CitySum = LocationCity.sum()
    CitySum = CitySum[CitySum > 10]
    pd.DataFrame(CitySum).sort_values(0, ascending=True).plot.barh()
    plt.xlim(0, 700)
    plt.xticks(fontsize=4, rotation=90)
    plt.title("Cities with more than 10 cases")
    plt.ylabel("City")
    plt.xlabel("Count")
    plt.gca().get_legend().remove()
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.xticks(fontsize=8)
    #generate_plot('Figures/Transform/CityLimited.png')

    # Make the modifications themselves
    FoodInspections = FoodInspectionsOriginal
    FoodInspections['years_open'] = st.zscore((np.log(FoodInspections['years_open'])))
    FoodInspections['review_count'] = st.zscore((np.log(FoodInspections['review_count'])))
    FoodInspections['rating'] = st.zscore(FoodInspections['rating'])

    TopCategoryFields = ["category_" + cat for cat in CategorySum.index]
    TopCategoryData = FoodInspections[TopCategoryFields]
    FoodInspections.drop(columns=CategoryFields, inplace=True)
    FoodInspections = pd.concat([FoodInspections, TopCategoryData], axis=1)

    #LocationCity[list(CitySum.index)]
    LocationCity.columns = ["" \
                            "city_" + re.sub(" ", "_", name) for name in list(LocationCity)]
    FoodInspections.drop(columns=["location_city", "business_id"], inplace=True)
    FoodInspections = pd.concat([FoodInspections, LocationCity], axis=1)


    ReviewFields = [name for name in list(FoodInspections) if re.match('review_pca', name)]
    for field in ReviewFields:
        FoodInspections[field] = st.zscore(FoodInspections[field])


    FoodInspections['review_pca_12'].plot.kde()
    plt.xlabel("Primary Component Value")
    plt.legend(loc='best')
    plt.title("Standard Normal for PCA 12")
    generate_plot('Figures/Transform/PCA12.png')

    FoodInspections['review_pca_117'].plot.kde()
    plt.xlabel("Primary Component Value")
    plt.legend(loc='best')
    plt.title("Standard Normal for PCA 117")
    generate_plot('Figures/Transform/PCA117.png')

    FoodInspections['review_pca_335'].plot.kde()
    plt.xlabel("Primary Component Value")
    plt.legend(loc='best')
    plt.title("Standard Normal for PCA 335")
    generate_plot('Figures/Transform/PCA335.png')


    FoodInspections.to_csv("Data/Transformed/FoodInspectionsTransformed.csv", index=False)