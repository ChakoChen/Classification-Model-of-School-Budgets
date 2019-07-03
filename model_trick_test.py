# model_expert.py - tricks from expert to improve the model 

import numpy as np
import pandas as pd

from mod_multilabel import multilabel_train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.pipeline import FeatureUnion

from sklearn.feature_selection import chi2, SelectKBest

from sklearn.feature_extraction.text import HashingVectorizer



def combine_text_columns(data_frame, to_drop=None):
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


def text_processing_trick():
    """
    Create n-grams of words
    """

    # Import dataframe
    df = pd.read_csv('../data/DataCamp/TrainingData.csv', index_col=0)
 
    # Define numeric columns (numeric features)
    NUMERIC_COLUMNS = ['FTE','Total'] 

    # Define labels (target) and get the dummy encoding of the labels
    LABELS = ['Function','Use','Sharing','Reporting','Student_Type','Position_Type','Object_Type','Pre_K','Operating_Status'] 
    dummy_labels = pd.get_dummies(df[LABELS])

    # Get the columns that are features in the original df
    NON_LABELS = [c for c in df.columns if c not in LABELS]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = multilabel_train_test_split(df[NON_LABELS],
                                                                   dummy_labels,
                                                                   0.2, 
                                                                   seed=123)

    # Create the text vector
    text_vector = combine_text_columns(X_train, to_drop=NUMERIC_COLUMNS + LABELS)

    # Create the token pattern: TOKENS_ALPHANUMERIC
    TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

    # Instantiate the CountVectorizer: text_features
    text_features = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

    # Fit text_features to the text vector
    text_features.fit(text_vector)

    # Print the first 10 tokens
    print(text_features.get_feature_names()[:10])


def hashing_trick():
    """
    Check out the scikit-learn implementation of HashingVectorizer

    Why hashing useful?
    Some problems are memory-bound and not easily parallelizable, and hashing enforces a fixed length computation instead of using a mutable datatype (like a dictionary).
    Enforcing a fixed length can speed up calculations drastically, especially on large datasets!
    """
    # In fact, python dictionaries ARE hash tables!

    # Import dataframe
    df = pd.read_csv('../data/DataCamp/TrainingData.csv', index_col=0)

    # Define numeric columns (numeric features)
    NUMERIC_COLUMNS = ['FTE','Total']

    # Define labels (target) and get the dummy encoding of the labels
    LABELS = ['Function','Use','Sharing','Reporting','Student_Type','Position_Type','Object_Type','Pre_K','Operating_Status']
    dummy_labels = pd.get_dummies(df[LABELS])

    # Get the columns that are features in the original df
    NON_LABELS = [c for c in df.columns if c not in LABELS]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = multilabel_train_test_split(df[NON_LABELS],
                                                                   dummy_labels,
                                                                   0.2,
                                                                   seed=123)

    # Get text data: text_data
    text_data = combine_text_columns(X_train, to_drop=NUMERIC_COLUMNS + LABELS)

    # Create the token pattern: TOKENS_ALPHANUMERIC
    TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)' 

    # Instantiate the HashingVectorizer: hashing_vec
    hashing_vec = HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

    # Fit and transform the Hashing Vectorizer
    hashed_text = hashing_vec.fit_transform(text_data)

    # Create DataFrame and print the head
    hashed_df = pd.DataFrame(hashed_text.data)
    print(hashed_df.head())


def main():
    #text_processing_trick()
    hashing_trick()

if __name__ == "__main__":
    main()
