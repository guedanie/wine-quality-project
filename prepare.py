

import pandas as pd
import split_scale
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split


# ------------------ #
#       Split        #
# ------------------ #


def split_data(wine):
    # Data split
    train, test = train_test_split(wine, random_state = 123, train_size=.8)
    train, validate = train_test_split(train, random_state = 123, train_size=.75)
    return train, test, validate

# ----------------- #
#       Scale       #
# ----------------- #

def scale_data(X_train, X_validate, X_test):
    # Data Scale
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    scaler, train_scaled, validate_scaled, test_scaled = return_values(scaler, X_train, X_validate, X_test)
    return scaler, train_scaled, validate_scaled, test_scaled



# ------------------ #
#   Preprocessing    #
# ------------------ #

def return_values(scaler, train, validate, test):
        train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
        validate_scaled = pd.DataFrame(scaler.transform(validate), columns=validate.columns.values).set_index([validate.index.values])
        test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
        return scaler, train_scaled, validate_scaled, test_scaled

def get_data_for_modeling():
    # Data acquire and prep
    wines = pd.read_csv("wine-quality-red.csv")
    wines["mso2"] = wines["free sulfur dioxide"] / (1 + 10** (wines.pH - 1.81))
    wines = wines.drop(columns= "fixed acidity")
    
    # Data split
    train, test = train_test_split(wines, random_state = 123, train_size=.8)
    train, validate = train_test_split(train, random_state = 123, train_size=.75)
    X_train = train.drop(columns=["quality"])
    y_train = train.quality
    X_validate = validate.drop(columns=["quality"])
    y_validate = validate.quality
    X_test = test.drop(columns="quality")
    y_test = test.quality
    
    # Data Scale
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    scaler, train_scaled, validate_scaled, test_scaled = return_values(scaler, X_train, X_validate, X_test)
    
    return train_scaled, y_train, validate_scaled, y_validate, test_scaled, y_test

# ------------------ #
#      Outliers      #
# ------------------ #


def get_upper_outliers_iqr(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.
    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))

def outliers_z_score(ys):
    '''
    Function used to detect outliers using z_score
    '''
    threshold = 3

    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.where(np.abs(z_scores) > threshold)

def outliers_percentile(s):
    '''
    Function used to detect outliers using percentiles
    '''
    return s > s.quantile(.99)

def detect_outliers(s, k, method="iqr"):
    ''' 
    Main function to detect outliers. Takes a series, a value for k, and a method for detecting outliers. Standard method for detecting outliers is IQR
    '''
    if method == "iqr":
        upper_bound = get_upper_outliers_iqr(s, k)
        return upper_bound
    elif method == "z_score":
        z_score = outliers_z_score(s)
        return z_score
    elif method == "percentile":
        percentile = outliers_percentile(s)
        return percentile
    
def detect_columns_outliers(df, k, method="iqr"):
    '''
    Function used to detect outliers across the entire dataframe
    '''
    outlier = pd.DataFrame()
    for col in df.select_dtypes("number"):
        is_outlier = detect_outliers(df[col], k, method=method)
        outlier[col] = is_outlier
    return outlier


    