import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.lda import LDA
from sklearn.qda import QDA


"""
sklearn-version : 0.17.1
pandas-version:   0.18.0
"""

# Classifiers
decisionTreeClf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=20,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort='True', random_state=None, splitter='best')
supportVectorClf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', kernel='rbf',
    max_iter=-1,
    tol=0.001, verbose=False)
randomForestClf = RandomForestClassifier(n_jobs=-1, class_weight=None, criterion='gini', max_depth=20,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0)
extraTreeClf = ExtraTreesClassifier(n_estimators=80, criterion='gini', max_depth=20, min_samples_split=2,
                           min_samples_leaf=1,max_features=None, max_leaf_nodes=None,
                           bootstrap=False, oob_score=False, n_jobs=1,random_state=None,
                           warm_start=False, class_weight=None)
logisticRegressionClf = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=1000, multi_class='ovr', n_jobs=-1,
          penalty='l2', random_state=None, solver='lbfgs', tol=0.0001,
          verbose=0, warm_start=False)
kNearestNeighborClf = KNeighborsClassifier(algorithm='auto', leaf_size=2, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=9, p=2,
           weights='distance')
naiveBayesClf = GaussianNB()
adaBoostClf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=100, random_state=2)

ldaClf = LDA(n_components=2, priors=None, shrinkage=None,
              solver='svd', store_covariance='True', tol=0.0001)
qdaClf = QDA()

classifiers = [decisionTreeClf, randomForestClf, extraTreeClf, kNearestNeighborClf, naiveBayesClf,
               ldaClf, qdaClf, adaBoostClf,  logisticRegressionClf]


# Load the dataset and preparing the features and the target labels
# In addition to that I was using random sampling to see how the performance
# improves as the data is increaased.
def prepare_data(filename):
    # Load the preprocessed dataset
    data = pd.read_csv(filename)
    #data = pd.read_csv('/home/pawan/Desktop/wavelet/test.csv')
    # Take a sample of data and see the performance with increasing data
    #data = data.sample(frac=0.80, replace=False)
    # Target/user_ids
    target = data.iloc[:, 0]
    # Features containing (feat2, feat3, feat5, feat6, feat7, feat9, feat10, feat11, feat12,
    # feat15, feat17, feat18, feat20, feat21
    features = data.iloc[:, 1:]
    return features, target


# returns the train test split for training
def return_train_test_split(features, target):
    # Preparing Training and Testing Data
    train_features, test_features, train_target, test_target = \
        train_test_split(features, target, test_size=0.25, random_state=100)
    return train_features, test_features, train_target, test_target


# Scale the features using the MinMaxScaler
def scaled_data(train_features, test_features):
    # Scaling data for better convergence and avoid certain feature dominating others
    scaler = MinMaxScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.fit_transform(test_features)
    return train_features, test_features


# Using pca to find the principal components and get pca transformed training features.
def return_pca_transformed_data(train_features, test_features):
    # Principal Component Analysis: we are going to find the optimum principal
    # components and transform the training and testing data accordingly for other classifier
    # that I will using while pipelining them.
    pca = RandomizedPCA(n_components=11, whiten=True).fit(train_features)
    # Array containing means of individual features for the entire dataset.
    means = pca.mean_
    total = sum(pca.explained_variance_)
    print "PCA trained"
    print "The first component expains: ", (float(pca.explained_variance_[0]) / total) * 100, "% of the variance"
    # Training and Testing data transformed by PCA
    features_train_pca = pca.transform(train_features)
    features_test_pca = pca.transform(test_features)
    return features_train_pca, features_test_pca


# Returns the trained classifier
def train_classifer(clf, train_features, train_target):
    clf.fit(train_features, train_target)
    return clf


# Returns the predicted values by the classifier
def predict_values(clf, test_features):
    predicted_values = clf.predict(test_features)
    return predicted_values


# Return the accuracy of the classifer
def return_accuracy(predicted_values, test_target):
    acc = accuracy_score(test_target, predicted_values)
    return acc


# Return Classification Report
def return_classification_report(predicted_values, test_target):
    class_report = classification_report(test_target, predicted_values)
    return class_report


# Return confusion Matrix
def return_confusion_matrix(predicted_values, test_target):
    confusion_mat = confusion_matrix(test_target, predicted_values)
    return confusion_mat


# Stratified Cross Validation
# Since the I was not over fitting and the training time was too long I just checked it with
# the DecisionTreeClassifier
def stratified_cross_validation(clf, features, target):
    skf = StratifiedKFold(target, n_folds=5, shuffle=False)
    for train_index, test_index in skf:
        train_features, test_features = features.iloc[train_index], features.iloc[test_index]
        train_target, test_target = target[train_index], target[test_index]
        clf.fit(train_features, train_target)
        pred = clf.predict(test_features)
        print "The classifier is : "
        print clf
        acc = accuracy_score(test_target, pred)
        print acc
        print '#' *30
        print classification_report(test_target, pred)
        print confusion_matrix(test_target, pred)


# Main function that reads all the functions:
def main():
    # uncomment this line for the filename
    # test.csv : file is the preprocessed file
    #filename = sys.argv[1]
    filename = '/home/pawan/Desktop/wavelet/test.csv'
    features, target = prepare_data(filename)
    train_features, test_features, train_target, test_target = \
        return_train_test_split(features, target)
    scaled_train_features, scaled_test_features = \
        scaled_data(train_features, test_features)
    features_train_pca, features_test_pca = \
        return_pca_transformed_data(train_features,test_features)
    # Run the following code for all the classifiers tuned by the GridSearchCV
    for clf in classifiers:
        clf = train_classifer(clf, train_features, train_target)
        predicted_values = predict_values(clf, test_features)
        accuracy = return_accuracy(predicted_values , test_target)
        classificaton_report = return_classification_report(predicted_values, test_target)
        confusionn_matrix = return_confusion_matrix(predicted_values, test_target)
        print "The name of the classifier is : ", clf
        print "The accuracy of the classifier is : ", accuracy
        print "#" * 30
        print "The classification report for individual classes is : "
        print classificaton_report
        print "#" * 30
        print "The confusion matrix is as follow : "
        print confusionn_matrix
    """
    # This is the pipelining of pca along with support vector classifier and it basically generates N * (N-1)
    # classifiers and will take a huge amount of training time approx 90 minutes. and will result in an accuracy of
    # of approximately 0.38

    pca = RandomizedPCA()
    pipe = Pipeline(steps=[('pca', pca), ('logistic', supportVectorClf)])
    clf = GridSearchCV(pipe,dict(pca__n_components=11))
    clf = clf.fit(scaled_train_features, train_target)
    pred = clf.predict(scaled_test_features)
    accuracy = return_accuracy(pred, test_target)
    print "Accuracy of the pipeling classifier is: ", accuracy
    """



# Boiler plate code to call main
if __name__ == '__main__':
    main()


# The parameters that I used for different classifiers for GridSearchCV
# This step was one offline, since incorporating it into code would have
# increased the training time exponentially.
"""
param_grid_decisionTree = {
         'max_features': ['auto', 'sqrt', 'log2',None],
          'max_depth': [5, 10, 20, 40, 100,None],
          'presort': ['True']
          }


param_grid_SVC = {
    'kernel': ['rbf', 'poly', 'linear','sigmoid'],
    'gamma': ['auto', 0.1],
    'degree': [3],
    'random_state': [2]
}

param_grid_logRegression = {
    'multi_class': ['ovr'],
    'n_jobs': [-1],
    'solver': ['lbfgs'],
    'penalty': ['l2'],
    'max_iter': [100, 1000, 10000]
}

param_grid_k_neighbor = {
    'n_neighbors': [5, 9, 15, 20],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto'],
    'leaf_size': [2, 10, 30]
}

param_grid_adaboost = {
    'n_estimators': [50, 100],
    'learning_rate': [0.1, 1.0],
    'algorithm': ['SAMME', 'SAMME.R'],
    'random_state': [2, 10]
}

param_grid_LDA = {
    'solver': ['svd', 'lsqr', 'eigen'],
    'store_covariance': ['True'],
    'n_components': [2, 3, 5, 7, 8]
}


"""