import pandas as pd
import re

from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    # Read the data
    FoodInspections = pd.read_csv("Data/Transformed/FoodInspectionsTransformed.csv")

    # Split outcome data out
    Outcome = FoodInspections['had_critical_violation']

    # Take out the outcome
    FoodInspections.drop(columns=['had_critical_violation'], inplace=True)

    # Do a test/train split
    X_train, X_test, y_train, y_test = train_test_split(FoodInspections, Outcome, test_size=0.2, shuffle=True)
    X_train.to_csv("Data/Modelling/FullPredictorsTrain.csv", index=False)
    X_test.to_csv("Data/Modelling/FullPredictorsTest.csv", index=False)
    y_train.to_csv("Data/Modelling/FullOutcomesTrain.csv", index=False)
    y_test.to_csv("Data/Modelling/FullOutcomesTest.csv", index=False)

    # Take out the reviews
    ReviewName = [name for name in list(FoodInspections) if re.match("review_.*", name)]
    NoReviews = FoodInspections.drop(columns=ReviewName)

    # Do a test/train split
    X_train, X_test, y_train, y_test = train_test_split(NoReviews, Outcome, test_size=0.2, shuffle=True)
    X_train.to_csv("Data/Modelling/NoReviewsPredictorsTrain.csv", index=False)
    X_test.to_csv("Data/Modelling/NoReviewsPredictorsTest.csv", index=False)
    y_train.to_csv("Data/Modelling/NoReviewsOutcomesTrain.csv", index=False)
    y_test.to_csv("Data/Modelling/NoReviewsOutcomesTest.csv", index=False)

    # Take out the categories
    CategoryName = [name for name in list(FoodInspections) if re.match("category_.*", name)]
    NoCategories = FoodInspections.drop(columns=CategoryName)

    # Do a test/train split
    X_train, X_test, y_train, y_test = train_test_split(NoCategories, Outcome, test_size=0.2, shuffle=True)
    X_train.to_csv("Data/Modelling/NoCategoriesPredictorsTrain.csv", index=False)
    X_test.to_csv("Data/Modelling/NoCategoriesPredictorsTest.csv", index=False)
    y_train.to_csv("Data/Modelling/NoCategoriesOutcomesTrain.csv", index=False)
    y_test.to_csv("Data/Modelling/NoCategoriesOutcomesTest.csv", index=False)

    # Take out the categories
    NoCategoriesNoReviews = FoodInspections.drop(columns= (CategoryName + ReviewName))

    # Do a test/train split
    X_train, X_test, y_train, y_test = train_test_split(NoCategoriesNoReviews, Outcome, test_size=0.2, shuffle=True)
    X_train.to_csv("Data/Modelling/SimplePredictorsTrain.csv", index=False)
    X_test.to_csv("Data/Modelling/SimplePredictorsTest.csv", index=False)
    y_train.to_csv("Data/Modelling/SimpleOutcomesTrain.csv", index=False)
    y_test.to_csv("Data/Modelling/SimpleOutcomesTest.csv", index=False)
