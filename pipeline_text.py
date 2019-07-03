# pipeline_text.py - use a Pipeline to train a model with only a text feature

# Import Pipeline
from sklearn.pipeline import Pipeline

# Import other necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd

# Import data frame
sample_df = pd.read_csv('../data/DataCamp/exercise/sample.csv', index_col=0)
sample_df['text'].fillna("", inplace=True)  # pipeline can't fit text with NaNs


##################################
# (1) Split and select text data #
##################################

# Split out only the text data
X_train, X_test, y_train, y_test = train_test_split(sample_df['text'],
                                                    pd.get_dummies(sample_df['label']), 
                                                    random_state=456)

# Instantiate Pipeline object: pl
pl = Pipeline([
        ('vec', CountVectorizer()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - just text data: ", accuracy)

