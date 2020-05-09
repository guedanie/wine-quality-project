import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ---------------------- #
#        Modeling        #
# ---------------------- #

# Decision Tree

def run_clf(X_train, y_train, max_depth):
    '''
    Function used to create and fit decision tree models. It requires a max_depth parameter. Returns model and predictions.
    '''
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=123)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    return clf, y_pred


# KNN

def run_knn(X_train, y_train, n_neighbors):
    '''
    Function used to create and fit KNN model. Requires to specify n_neighbors. Returns model and predictions.
    '''
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_train)
    return knn, y_pred

# Random_forest

def run_rf(X_train, y_train, leaf, max_depth):
    ''' 
    Function used to create and fit random forest models. Requires to specif leaf and max_depth. Returns model and predictions.
    '''
    rf = RandomForestClassifier(random_state= 123, min_samples_leaf = leaf, max_depth = max_depth).fit(X_train, y_train)
    y_pred = rf.predict(X_train)
    return rf, y_pred

# Logistic Regression

def run_lg(X_train, y_train):
    '''
    Function used to create and fit logistic regression models. Returns model and predictions.
    '''
    logit = LogisticRegression().fit(X_train, y_train)
    y_pred = logit.predict(X_train)
    return logit, y_pred

def create_report(y_train, y_pred):
    '''
    Helper function used to create a classification evaluation report, and return it as df
    '''
    report = classification_report(y_train, y_pred, output_dict = True)
    report = pd.DataFrame.from_dict(report)
    return report


def accuracy_report(model, y_pred, y_train):
    '''
    Main function used to create printable versions of the classification accuracy score, confusion matrix and classification report.
    '''
    report = classification_report(y_train, y_pred, output_dict = True)
    report = pd.DataFrame.from_dict(report)
    accuracy_score = f'Accuracy on dataset: {report.accuracy[0]:.2f}'

    labels = sorted(y_train.unique())
    matrix = pd.DataFrame(confusion_matrix(y_train, y_pred), index = labels, columns = labels)

    return accuracy_score, matrix, report

# ------------------ #
#    Clustering      #
# ------------------ #


# Elbow code
def create_elbow_graph(df, col):
    '''
    Function to draw and elbow graph. Takes a dataframe and the name of a column to create a cluster
    '''
    X = df[col]
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(9, 6))
        pd.Series({k: KMeans(k).fit(X).inertia_ for k in range(2, 12)}).plot(marker='x')
        plt.xticks(range(2, 12))
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Change in inertia as k increases')

# Cluster
def create_cluster(df, n_cluster, col, cluster_name):
    '''
    Function to cluster features. Takes a dataframe, the number of clusters and the columns to cluster by
    '''
    X = df[col]
    kmeans = KMeans(n_clusters=n_cluster).fit(X)
    centroid = kmeans.cluster_centers_
    df[cluster_name] = kmeans.predict(X)
    df[cluster_name] = 'cluster_' + (df[cluster_name] + 1).astype('str')
    return df, centroid
    