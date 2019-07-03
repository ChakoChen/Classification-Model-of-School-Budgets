# model_expert.py - the classification model from model.py with improvements

import numpy as np
import pandas as pd

from SparseInteractions import SparseInteractions
from mod_multilabel import multilabel_train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.pipeline import FeatureUnion

from sklearn.feature_selection import chi2, SelectKBest


def model():
    """
    The classification model with improvements
    """

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

    # Select 300 best features
    chi_k = 300

    # Preprocess the text data: get_text_data
    get_text_data = FunctionTransformer(combine_text_columns, validate=False)

    # Preprocess the numeric data: get_numeric_data
    get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

    # Create the token pattern: TOKENS_ALPHANUMERIC
    TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

    # Create nested pipelines, which process our text and numeric data separately
    numeric_pipeline = Pipeline([
                                  ('selector', get_numeric_data),
                                  ('imputer', Imputer())
                               ])
    text_pipeline = Pipeline([
                               ('selector', get_text_data),
    #                          ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC, ngram_range=(1,2))),
                               ('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC, norm=None, binary=False, ngram_range=(1, 2)))
    #                          ('dim_red', SelectKBest(chi2, chi_k))
                            ])
    """
    'vectorizer' is updated to tokenize on punctuation and compute multiple n-gram features
    'dim_red' uses the SelectKBest() function, which applies chi-squared test to select the K 'best' features.
   
    'dim_red' and 'scale' have to be used to account for the fact that you're using a reduced-size sample of the full dataset in this course. To make sure the models perform as the expert competition winner intended, we have to apply a dimensionality reduction technique, which is the 'dim_red' step does, and we have to scale the features to lie between -1 and 1, which is the 'scale' step does
  
    Note: 'dim_red' does not work here; replace CountVectorizer with HashingVectorizer to speed up. 
    """

    # Join the nested pipelines                         
    union = FeatureUnion([
                           ('numeric_features', numeric_pipeline),
                           ('text_features', text_pipeline)
                        ])

    # Create the pipeline
    pl = Pipeline([
                    ('union', union),
    #               ('int', SparseInteractions(degree=2)),
                    ('scale', MaxAbsScaler()),
                    ('clf', OneVsRestClassifier(LogisticRegression()))
                 ])
    """
    'int' adds interaction terms to features (n features -> n-squared features)
    'scale' squashes the relevant features into the interval -1 to 1.
  
    Note, 'int' is extremely slow
    """

    # Fit to the training data
    pl.fit(X_train, y_train)

    # Compute and print accuracy
    accuracy = pl.score(X_test, y_test)
    print("\nAccuracy on budget dataset: ", accuracy)


def main():
    #text_processing_trick()
    model()

if __name__ == "__main__":
    main()
