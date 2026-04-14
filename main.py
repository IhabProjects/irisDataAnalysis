# Load libraries
from math import gamma
from uuid import NAMESPACE_DNS
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Lading Data Set

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset = read_csv(url, names=names)


def getIdeaAboutDataSet():
    # Dimension
    print(dataset.shape)
    # Have a peak
    print(dataset.head(20))
    # description
    print(dataset.describe())  # summary of each attribute.
    # Class Distribution
    print(dataset.groupby("class").size())


def splitDataSet():
    # We will split the dataset into two parts 80% to train and 20% as a validation DataSet
    array = dataset.values
    X = array[:, 0:4]  # : All rows and Attributes 0, 1, 2, 3 (4 : Collumn is excluded)
    y = array[:, 4]  # All Rows, Only Collumn 4 Which is Class Collumn
    # We randomly shuffle and split into testing and validation
    X_train, X_Validation, Y_train, Y_Validation = train_test_split(
        X, y, test_size=0.20, random_state=1
    )
    return X_train, X_Validation, Y_train, Y_Validation


def buildModels():
    """
    Let’s test 6 different algorithms:

        Logistic Regression (LR)
        sLinear Discriminant Analysis (LDA)
        K-Nearest Neighbors (KNN).
        Classification and Regression Trees (CART).
        Gaussian Naive Bayes (NB).
        Support Vector Machines (SVM).

    This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms.

    """

    models = []
    models.append(("LR", LogisticRegression(max_iter=200)))
    models.append(("LDA", LinearDiscriminantAnalysis()))
    models.append(("KNN", KNeighborsClassifier()))
    models.append(("CART", DecisionTreeClassifier()))
    models.append(("NB", GaussianNB()))
    models.append(("SVM", SVC(gamma="auto")))
    return models


def testModels():
    """
    We will use stratified 10-fold cross validation to estimate model accuracy.
    This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits.
    """
    models = buildModels()
    results = []
    names = []

    X_train, X_Validation, Y_train, Y_Validation = splitDataSet()

    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(
            model, X_train, Y_train, cv=kfold, scoring="accuracy"
        )
        results.append(cv_results)
        names.append(name)
        print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))
    return results, names


def compareModels():
    results, names = testModels()
    plt.boxplot(results, tick_labels=names)
    plt.title("Algorithm Comparison")
    plt.show()


def predictOnValidationDataSet():
    X_train, X_Validation, Y_train, Y_Validation = splitDataSet()
    model = SVC(gamma="auto")
    model.fit(X_train, Y_train)
    model.predict(X_Validation)
    predictions = model.predict(X_Validation)
    # Evaluate predictions
    print(accuracy_score(Y_Validation, predictions))
    print(confusion_matrix(Y_Validation, predictions))
    print(classification_report(Y_Validation, predictions))


def getUniVariatePlot():
    # Box and Whisker plotting
    dataset.plot(kind="box", subplots=True, layout=(2, 2), sharex=False, sharey=False)
    # Histogram
    dataset.hist()


def getMultiVaribalePlot():
    # scatterplots Matrix
    scatter_matrix(dataset)


def ideationPlots():
    getUniVariatePlot()
    getIdeaAboutDataSet()
    plt.show()


def main():
    # getIdeaAboutDataSet()
    # ideationPlots()
    # testModels()
    # compareModels()
    predictOnValidationDataSet()


if __name__ == "__main__":
    main()
