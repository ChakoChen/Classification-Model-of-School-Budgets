# bag_of_words.py - study the effects of tokenizing in different ways by comparing the bag-of-words representations resulting from different token patterns.

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def tokenization_test(): 
    """
    Creating a bag-of-words from one feature only
    WARNING: This code has a bug--the token pattern 
             ignores any word that doesn't have a 
             trailing space, instead only these with 
             a leading space!
    """ 

    # Create the alphanumeric token pattern
    TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

    # Import dataframe
    df = pd.read_csv('../data/DataCamp/TestData.csv', index_col=0)

    # Fill missing values in df.Position_Extra
    df.Position_Extra.fillna('', inplace=True)

    # Instantiate the CountVectorizer: vec_alphanumeric
    vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

    # Fit to the data
    vec_alphanumeric.fit(df.Position_Extra)
    #vec_alphanumeric.fit(['This is a test','apple'])

    # Print the number of tokens and first 15 tokens
    msg = "There are {} tokens in Position_Extra if we split on non-alpha numeric"
    print(msg.format(len(vec_alphanumeric.get_feature_names())))
    print(vec_alphanumeric.get_feature_names()[:15])
    """
    WARNING: alphanumeric only finds words trailing with space(s), 
             igorning last word with no trailing space!!!
    """


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

def tokenization_test2():
    """
    Convert all training text data to a single vector, 
    pass it to a vectorizer object and make a bag-of-words using .fit_transform().
    Two tokens: non-whitespace characters vs alphanumeric characters
    """

    # Import dataframe
    df = pd.read_csv('../data/DataCamp/TrainingData.csv', index_col=0)    
 
    # Define features and targets
    NUMERIC_COLUMNS = ['FTE', 'Total']
    LABELS = ['Function','Use','Sharing','Reporting','Student_Type','Position_Type','Object_Type','Pre_K','Operating_Status']

    # Create the basic token pattern
    TOKENS_BASIC = '\\S+(?=\\s+)'

    # Create the alphanumeric token pattern
    TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

    # Instantiate basic CountVectorizer: vec_basic
    vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC)

    # Instantiate alphanumeric CountVectorizer: vec_alphanumeric
    vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

    # Create the text vector
    text_vector = combine_text_columns(df, to_drop=NUMERIC_COLUMNS + LABELS)

    # Fit and transform vec_basic
    vec_basic.fit_transform(text_vector)

    # Print number of tokens of vec_basic
    print("There are {} tokens in the dataset".format(len(vec_basic.get_feature_names())))

    # Fit and transform vec_alphanumeric
    vec_alphanumeric.fit_transform(text_vector)

    # Print number of tokens of vec_alphanumeric
    print("There are {} alpha-numeric tokens in the dataset".format(len(vec_alphanumeric.get_feature_names())))


def main():
    tokenization_test()
    #tokenization_test2()


if __name__ == "__main__":
    main()



