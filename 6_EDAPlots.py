import pandas as pd
import numpy as np
import os
from sklearn.manifold import TSNE
os.environ['PROJ_LIB'] = 'C://Users/gsene/Anaconda3/Library/share'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def generate_plot(outfile):
    plt.savefig(outfile)
    plt.close()


if __name__ == "__main__":

    # Read the food inspections
    foodInspections = pd.read_csv("Data/FoodInspections.csv")
    foodInspections['Price'] = ''
    foodInspections.loc[foodInspections['price_$']==1, 'Price'] = '$'
    foodInspections.loc[foodInspections['price_$$']==1, 'Price'] = '$$'
    foodInspections.loc[foodInspections['price_$$$']==1, 'Price'] = '$$$'
    foodInspections.loc[foodInspections['price_$$$$']==1, 'Price'] = '$$$$'
    foodInspections.loc[foodInspections['price_?']==1, 'Price'] = '?'

    # Plot the outcome
    fig, ax = plt.subplots()
    percents_outcome = [sum(foodInspections['had_critical_violation']) / len(foodInspections),
                        1 - sum(foodInspections['had_critical_violation']) / len(foodInspections)]
    plt.pie(x=percents_outcome, explode=(0, 0.1),
            labels=['Had Critical Violation', 'None'], autopct='%1.1f%%',
            startangle=90)
    ax.axis('equal')
    generate_plot("Figures/EDA/Outcome_Pie.png")

    # Means and SDs
    means = foodInspections.groupby('had_critical_violation').mean()
    sd = foodInspections.groupby('had_critical_violation').mean()
    means['had_critical_violation'] = means.index
    means['had_critical_violation'] = means['had_critical_violation'].map({0: 'No', 1: 'Yes'})

    # Closure
    plt.bar(means['had_critical_violation'], means['is_closed'], align='center')
    plt.xlabel("Had Critical Violation")
    plt.ylabel("Percentage Closed")
    plt.title("Proportion of Closure")
    generate_plot("Figures/EDA/Closed.png")

    # Years Open
    years_no = foodInspections['years_open'][foodInspections['had_critical_violation'] == 0]
    years_yes = foodInspections['years_open'][foodInspections['had_critical_violation'] == 1]
    years_yes.plot.kde(label="Had Critical Violation")
    years_no.plot.kde(label="None")
    plt.xlim(left=0)
    plt.xlabel("Years Open")
    plt.legend(loc='best')
    plt.title("Distribution of Years Open")
    generate_plot("Figures/EDA/YearsOpen.png")

    # Review Count
    reviews_yes = foodInspections['review_count'][foodInspections['had_critical_violation'] == 1]
    reviews_no = foodInspections['review_count'][foodInspections['had_critical_violation'] == 0]
    reviews_yes.plot.kde(label="Had Critical Violation")
    reviews_no.plot.kde(label="None")
    plt.xlim(left=0, right=800)
    plt.xlabel("Review Count")
    plt.legend(loc='best')
    plt.title("Distribution of Reviews")
    generate_plot("Figures/EDA/ReviewCount.png")

    # Rating
    countsNo = foodInspections[foodInspections['had_critical_violation'] == 0].groupby('rating').\
        count().apply(lambda x: x / x.sum())
    countsYes = foodInspections[foodInspections['had_critical_violation'] == 1].groupby('rating').\
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
    plt.xticks(np.arange(0.5, 5.5, 0.5))
    plt.xlabel('Rating')
    plt.ylabel("Frequency")
    plt.title("Distribution of Ratings")
    generate_plot("Figures/EDA/Ratings.png")

    # Location City - Markers
    fig, ax = plt.subplots()
    uniqueCities = np.unique(foodInspections['location_city']).tolist()
    cityList = [("$" + str(i) + "$") for i in range(0, len(uniqueCities))]
    cityMap = dict(zip(uniqueCities, cityList))
    for city in uniqueCities:
        inspectionsNo = foodInspections[(foodInspections['had_critical_violation'] == 0) & (foodInspections['location_city'] == city)]
        inspectionsYes = foodInspections[(foodInspections['had_critical_violation'] == 1) & (foodInspections['location_city'] == city)]
        plt.scatter(
            inspectionsYes['longitude'],
            inspectionsYes['latitude'],
            marker=cityMap[city],
            c="orange"
        )
        plt.scatter(
            inspectionsNo['longitude'],
            inspectionsNo['latitude'],
            marker=cityMap[city],
            c="blue"
        )
    plt.xlim(left=np.min(foodInspections['longitude']),
               right=np.max(foodInspections['longitude']))
    plt.ylim(bottom=np.min(foodInspections['latitude']),
               top=np.max(foodInspections['latitude']))
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.xlim(right=max(foodInspections['longitude']) + 0.4,
             left=min(foodInspections['longitude'] - 0.05))
    plt.ylim(top=max(foodInspections['latitude']) + 0.05,
             bottom=min(foodInspections['latitude'] - 0.1))
    plt.title("Outcome by Location | City")
    legend_elements = []
    for i, shape in enumerate(cityList):
        new_patch = plt.Line2D([0], [0], marker=shape, color='k', label=uniqueCities[i],
                                markerfacecolor='w', markersize=8)
        legend_elements.append(new_patch)
    legend_elements.append(new_patch)
    ax.legend(handles=legend_elements, loc=1)
    generate_plot("Figures/EDA/Location-City-Marker.png")

    #Location City - Colors
    fig, ax = plt.subplots()
    uniqueCities = np.unique(foodInspections['location_city']).tolist()
    cityList = [("$" + str(i) + "$") for i in range(0, len(uniqueCities))]
    cityMap = dict(zip(uniqueCities, cityList))
    for city in uniqueCities:
        inspectionsNo = foodInspections[(foodInspections['had_critical_violation'] == 0) & (foodInspections['location_city'] == city)]
        inspectionsYes = foodInspections[(foodInspections['had_critical_violation'] == 1) & (foodInspections['location_city'] == city)]
        plt.scatter(
            inspectionsYes['longitude'],
            inspectionsYes['latitude'],
            marker=cityMap[city],
            c="orange"
        )
        plt.scatter(
            inspectionsNo['longitude'],
            inspectionsNo['latitude'],
            marker=cityMap[city],
            c="blue"
        )
    plt.xlim(left=np.min(foodInspections['longitude']),
               right=np.max(foodInspections['longitude']))
    plt.ylim(bottom=np.min(foodInspections['latitude']),
               top=np.max(foodInspections['latitude']))
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.xlim(right=max(foodInspections['longitude']) + 0.4,
             left=min(foodInspections['longitude'] - 0.05))
    plt.ylim(top=max(foodInspections['latitude']) + 0.05,
             bottom=min(foodInspections['latitude'] - 0.1))
    plt.title("Outcome by Location | City")
    legend_elements2 = []
    new_patch1 = mpatches.Patch(label='Had a Critical Violation', color='tab:blue')
    new_patch2 = mpatches.Patch(label='None', color='tab:orange')
    legend_elements2.append(new_patch1)
    legend_elements2.append(new_patch2)
    ax.legend(handles=legend_elements2, loc=3)
    generate_plot("Figures/EDA/Location-City-Color.png")


    # Location City Big
    plt.figure(figsize=[6.4*4, 4.8*4])
    fig, ax = plt.subplots()
    uniqueCities = np.unique(foodInspections['location_city']).tolist()
    cityList = [("$" + str(i) + "$") for i in range(0, len(uniqueCities))]
    cityMap = dict(zip(uniqueCities, cityList))
    for city in uniqueCities:
        inspectionsNo = foodInspections[(foodInspections['had_critical_violation'] == 0) & (foodInspections['location_city'] == city)]
        inspectionsYes = foodInspections[(foodInspections['had_critical_violation'] == 1) & (foodInspections['location_city'] == city)]
        plt.scatter(
            inspectionsYes['longitude'],
            inspectionsYes['latitude'],
            marker=cityMap[city],
            c="orange",
            s=20
        )
        plt.scatter(
            inspectionsNo['longitude'],
            inspectionsNo['latitude'],
            marker=cityMap[city],
            c="blue",
            s=20
        )
    plt.xlim(left=np.min(foodInspections['longitude']),
               right=np.max(foodInspections['longitude']))
    plt.ylim(bottom=np.min(foodInspections['latitude']),
               top=np.max(foodInspections['latitude']))
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Outcome by Location | City")
    legend_elements = []
    generate_plot("Figures/EDA/Location-City-Big.png")

    # Price
    priceMeans = foodInspections.groupby(['Price', 'had_critical_violation']).count()['business_id'].\
        unstack(fill_value=0).stack().reset_index()
    priceMeans.loc[priceMeans['had_critical_violation'] == 0, 0] = priceMeans.loc[priceMeans['had_critical_violation'] == 0, 0] / sum(priceMeans.loc[priceMeans['had_critical_violation'] == 0, 0])
    priceMeans.loc[priceMeans['had_critical_violation'] == 1, 0] = priceMeans.loc[priceMeans['had_critical_violation'] == 1, 0] / sum(priceMeans.loc[priceMeans['had_critical_violation'] == 1, 0])

    priceMeans0 = priceMeans[priceMeans['had_critical_violation'] == 0][0]
    priceMeans1 = priceMeans[priceMeans['had_critical_violation'] == 1][0]

    N = 5
    ind = np.arange(N)
    width = 0.35
    fig, ax = plt.subplots()
    plt.bar(ind, priceMeans1.tolist(), width, label="Had Critical Violation")
    plt.bar(ind + width, priceMeans0.tolist(), width, label="None")
    ax.set_title('Percent with Violations')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('\$', '\$\$', '\$\$\$', '\$\$\$\$', '?'))
    plt.legend(loc='best')
    generate_plot("Figures/EDA/Prices.png")


    #Category TSNE
    foodCategories = foodInspections.filter(regex="category_.*")
    foodTSNE = TSNE(n_components=2).fit_transform(foodCategories)
    plt.scatter(foodTSNE[:, 0][foodInspections['had_critical_violation'] == 1],
                foodTSNE[:, 1][foodInspections['had_critical_violation'] == 1],
                label='Had Critical Violation')
    plt.scatter(foodTSNE[:, 0][foodInspections['had_critical_violation'] == 0],
                foodTSNE[:, 1][foodInspections['had_critical_violation'] == 0],
                label='None')
    plt.legend(loc="best")
    plt.title("T-SNE for " + str(foodCategories.shape[1]-1) + " Categories")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    generate_plot("Figures/EDA/CategoryTSNE.png")

    # Review TSNE
    foodReviews = foodInspections.filter(regex="review_.*")
    reviewTSNE = TSNE(n_components=2).fit_transform(foodReviews)
    plt.scatter(reviewTSNE[:, 0][foodInspections['had_critical_violation'] == 1],
                reviewTSNE[:, 1][foodInspections['had_critical_violation'] == 1],
                label='Had Critical Violation')
    plt.scatter(reviewTSNE[:, 0][foodInspections['had_critical_violation'] == 0],
                reviewTSNE[:, 1][foodInspections['had_critical_violation'] == 0],
                label='None')
    plt.legend(loc="best")
    plt.title("T-SNE for " + str(foodReviews.shape[1]-1) + " Reviews Terms")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    generate_plot("Figures/EDA/ReviewTSNE.png")


    # Read the PCA food inspections
    foodInspectionsPCA = pd.read_csv("Data/FoodInspectionsPCA.csv")

    # Review PCA TSNE
    foodReviews = foodInspectionsPCA.filter(regex="review_pca_.*")
    reviewTSNE = TSNE(n_components=2).fit_transform(foodReviews)
    plt.scatter(reviewTSNE[:, 0][foodInspectionsPCA['had_critical_violation'] == 1],
                reviewTSNE[:, 1][foodInspectionsPCA['had_critical_violation'] == 1],
                label='Had Critical Violation')
    plt.scatter(reviewTSNE[:, 0][foodInspectionsPCA['had_critical_violation'] == 0],
                reviewTSNE[:, 1][foodInspectionsPCA['had_critical_violation'] == 0],
                label='None')
    plt.legend(loc="best")
    plt.title("T-SNE for " + str(foodReviews.shape[1]-1) + " PCA Terms")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    generate_plot("Figures/EDA/Review-PCA-TSNE.png")

    # Read the D2V food inspections
    foodInspectionsD2V = pd.read_csv("Data/FoodInspectionsD2V.csv")

    # Review PCA TSNE
    foodReviews = foodInspectionsD2V.filter(regex="review_d2v_.*")

    generate_plot("Figures/EDA/Review-D2V-TSNE.png")