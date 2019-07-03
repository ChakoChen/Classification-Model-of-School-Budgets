# model.py - fit a classification model to text and numeric features

import numpy as np
import pandas as pd

from mod_multilabel import multilabel_train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion


# Import dataframe
df = pd.read_csv('../data/DataCamp/TrainingData.csv', index_col=0)


#####################################################
# (1) Define features (text and numeric) and target #
#####################################################

# Define numeric columns (numeric features)
NUMERIC_COLUMNS = ['FTE','Total']

# Define labels (target) and get the dummy encoding of the labels
LABELS = ['Function','Use','Sharing','Reporting','Student_Type','Position_Type','Object_Type','Pre_K','Operating_Status'] 
dummy_labels = pd.get_dummies(df[LABELS])

# Get the columns that are features in the original df
NON_LABELS = [c for c in df.columns if c not in LABELS]


def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """
    Converts all text in each row of data_frame to single vector 
    """

    # Drop non-text columns that are in the df
    # The set() constructor constructs a Python set from the given iterable and returns it.
    to_drop = set(to_drop) & set(data_frame.columns.tolist())  # intersect of two sets
    text_data = data_frame.drop(to_drop, axis=1)

    # Replace nans with blanks
    text_data.fillna("", inplace=True)

    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)


###############################################
# (2) Split the data into train and test sets #
###############################################

# Split into training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(df[NON_LABELS],
                                                               dummy_labels,
                                                               0.2, 
                                                               seed=123)

#############################################
# (3) Create Pipeline with nested pipelines #
#############################################

# Preprocess the text data: get_text_data
get_text_data = FunctionTransformer(combine_text_columns, validate=False)

# Preprocess the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

# Create nested pipelines, which process our text and numeric data separately
numeric_pipeline = Pipeline([
                              ('selector', get_numeric_data),
                              ('imputer', Imputer())
                           ])
text_pipeline = Pipeline([
                           ('selector', get_text_data),
                           ('vectorizer', CountVectorizer())
                        ])

# Join the nested pipelines                         
union = FeatureUnion([
                       ('numeric', numeric_pipeline),
                       ('text', text_pipeline)
                    ])

# Create the pipeline
"""Random forest classifier, which uses the statistics of an ensemble of decision trees to generate predictions."""
pl = Pipeline([
                ('union', union),
#               ('clf', OneVsRestClassifier(LogisticRegression()))
#               ('clf', RandomForestClassifier())  # default n_estimators=10
                ('clf', RandomForestClassifier(n_estimators=15))
             ])

"""
   The structure of the pipeline is exactly the same as earlier in this chapter:
   (1) the preprocessing step uses FeatureUnion to join the results of nested pipelines that each rely on FunctionTransformer to select multiple datatypes
   (2) the model step stores the model object
"""
"""
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
"""

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)
