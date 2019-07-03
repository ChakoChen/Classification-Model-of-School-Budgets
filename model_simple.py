# model_simple.py - fit a multi-class logistic regression model to the NUMERIC_COLUMNS of your feature data

import numpy as np
import pandas as pd
from mod_multilabel import multilabel_train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


######################################################
# 1. Build a simple model with only numeric features #
######################################################

# import dataframe
df = pd.read_csv('../data/DataCamp/TrainingData.csv', index_col=0)

# define numeric columns (features) and create the new DataFrame: numeric_data_only
NUMERIC_COLUMNS = ['FTE','Total']
numeric_data_only = df[NUMERIC_COLUMNS].fillna(-1000)

# define labels (target) and convert to dummy variables: label_dummies
LABELS = ['Function','Use','Sharing','Reporting','Student_Type','Position_Type','Object_Type','Pre_K','Operating_Status'] 
label_dummies = pd.get_dummies(df[LABELS])
"""
# define a lambda function to convert column x to category
categorize_label = lambda x: x.astype('category')
df[LABELS] = df[LABELS].apply(categorize_label, axis=0)
num_unique_labels = df[LABELS].apply(pd.Series.nunique)
print(num_unique_labels)
"""

# Create training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(numeric_data_only, label_dummies, size=0.2, seed=123, min_count=5)
"""
# Print the info
print("X_train info:")
print(X_train.info())
print("\nX_test info:")  
print(X_test.info())
print("\ny_train info:")  
print(y_train.info())
print("\ny_test info:")  
print(y_test.info()) 
"""

# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression())

# fit the classifier to the training data
clf.fit(X_train, y_train)

# test and print the accuracy with the .score() method
print("Accuracy: {}".format(clf.score(X_test, y_test)))  # 0.0!


##########################################
# 2. Make predictions on holdout dataset #
##########################################

# Load the holdout data: holdout
holdout = pd.read_csv('../data/DataCamp/HoldoutData.csv', index_col=0)

# Generate predictions: predictions
predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS].fillna(-1000))

# Format predictions in DataFrame: prediction_df
prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS]).columns,
                             index=holdout.index,
                             data=predictions)

# Save prediction_df to csv
prediction_df.to_csv('../data/DataCamp/predictions.csv')

