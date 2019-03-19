import pandas as pd
import numpy as np
import os
import re
from sklearn.manifold import TSNE
os.environ['PROJ_LIB'] = 'C://Users/gsene/Anaconda3/Library/share'
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def generate_plot(outfile):
    plt.savefig(outfile)
    plt.close()


if __name__ == "__main__":

    # Read the food inspections
    foodInspections = pd.read_csv("Data/FoodInspections.csv")

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
    generate_plot("Figures/EDA/Closed.png")

    # Years Open
    years_no = foodInspections['years_open'][foodInspections['had_critical_violation'] == 0]
    years_yes = foodInspections['years_open'][foodInspections['had_critical_violation'] == 1]
    years_no.plot.kde(label="No")
    years_yes.plot.kde(label="Yo")
    plt.xlim(left=0)
    plt.xlabel("Years Open")
    plt.legend(loc='best')
    generate_plot("Figures/EDA/YearsOpen.png")

    #Review Count
    reviews_no = foodInspections['review_count'][foodInspections['had_critical_violation'] == 0]
    reviews_yes = foodInspections['review_count'][foodInspections['had_critical_violation'] == 1]
    reviews_no.plot.kde(label="No")
    reviews_yes.plot.kde(label="Yo")
    plt.xlim(left=0, right=800)
    plt.xlabel("Review Count")
    plt.legend(loc='best')
    generate_plot("Figures/EDA/ReviewCount.png")

    #Rating
    countsNo = foodInspections[foodInspections['had_critical_violation'] == 0].groupby('rating').\
        count().apply(lambda x: x / x.sum())
    countsYes = foodInspections[foodInspections['had_critical_violation'] == 1].groupby('rating').\
        count().apply(lambda x: x / x.sum())
    countsNo.reset_index(inplace=True)
    countsYes.reset_index(inplace=True)
    plt.bar(countsNo['rating'],
            countsNo['business_id'],
            width=0.5,
            label='Yes',
            edgecolor='k',
            alpha=0.5)
    plt.bar(countsYes['rating'],
            countsYes['business_id'],
            width=0.5,
            label='No',
            edgecolor='k',
            alpha=0.5)
    plt.legend(loc="best")
    plt.xticks(np.arange(0.5, 5.5, 0.5))
    plt.xlabel('Rating')
    plt.ylabel("Frequency")
    plt.show()



    # Location
    # noViolation = foodInspections[foodInspections['had_critical_violation'] == 0]
    # yesViolation = foodInspections[foodInspections['had_critical_violation'] == 1]
    # fig = plt.figure(figsize=(8, 8))
    # m = Basemap(projection='lcc', resolution='h',
    #                 lat_0=np.mean(foodInspections['latitude']),
    #                 lon_0=np.mean(foodInspections['longitude']),
    #                 width=1E6, height=1.2E6)
    # m.scatter(#yesViolation['longitude'].tolist(),
    #     np.mean(foodInspections['latitude']),
    #     np.mean(foodInspections['longitude']),
    #     #yesViolation['latitude'].tolist(),
    #           s=999999,
    #           alpha=0.5,
    #           cmap='Reds',#,
    #           #latlon=True,
    #           zorder=10
    #           )
    # m.drawcoastlines(color='gray')
    # m.drawstates(color='gray')
    # m.fillcontinents(color='lightgray')
    # plt.show()



    #Category TSNE
    foodCategories = foodInspections.filter(regex="category_.*")
    foodTSNE = TSNE(n_components=2).fit_transform(foodCategories)
    plt.scatter(foodTSNE[:, 0][foodInspections['had_critical_violation'] == 1],
                foodTSNE[:, 1][foodInspections['had_critical_violation'] == 1],
                c='y',
                label='Yes')
    plt.scatter(foodTSNE[:, 0][foodInspections['had_critical_violation'] == 0],
                foodTSNE[:, 1][foodInspections['had_critical_violation'] == 0],
                c='b',
                label='No')
    plt.legend(loc="best")


    # Reviews

    foodInspections = pd.read_csv("Data/FoodInspections2.csv")
    foodReviews = foodInspections.filter(regex="review_embedding_.*")
    reviewTSNE = TSNE(n_components=2).fit_transform(foodReviews)
    plt.scatter(reviewTSNE[:, 0][foodInspections['had_critical_violation'] == 1],
                reviewTSNE[:, 1][foodInspections['had_critical_violation'] == 1],
                c='y',
                label='Yes')
    plt.scatter(reviewTSNE[:, 0][foodInspections['had_critical_violation'] == 0],
                reviewTSNE[:, 1][foodInspections['had_critical_violation'] == 0],
                c='b',
                label='No')
    plt.legend(loc="best")
    plt.show()


    foodInspections = pd.read_csv("Data/FoodInspections2.csv")
    foodReviews = foodInspections.filter(regex="review_.*")
    reviewTSNE = TSNE(n_components=2).fit_transform(foodReviews)
    plt.scatter(reviewTSNE[:, 0][foodInspections['had_critical_violation'] == 1],
                reviewTSNE[:, 1][foodInspections['had_critical_violation'] == 1],
                c='y',
                label='Yes')
    plt.scatter(reviewTSNE[:, 0][foodInspections['had_critical_violation'] == 0],
                reviewTSNE[:, 1][foodInspections['had_critical_violation'] == 0],
                c='b',
                label='No')
    plt.legend(loc="best")
    plt.show()