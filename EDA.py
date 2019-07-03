# EDA.py - Exploratory analysis of the school buget data

import pandas as pd
import matplotlib.pyplot as plt

# Load the data set
df = pd.read_csv('../data/DataCamp/TrainingData.csv', index_col=0)

##########################################
# Task I: check the data set numerically #
##########################################

# Print the information of the data set, e.g, data types
print(df.info())

# Print the first 5 rows with column names
print(df.head())

# Print the last 5 rows with column names
print(df.tail())

# Print the summary statistics
print(df.describe())


########################################
# Task II: check the data set visually #
########################################
"""
FTE: Stands for "full-time equivalent". If the budget item is associated to an employee, this number tells us the percentage of full-time that the employee works. A value of 1 means the associated employee works for the schooll full-time. A value close to 0 means the item is associated to a part-time or contracted employee.

Total: Stands for the total cost of the expenditure. This number tells us how much the budget item cost.
"""

# Create the histogram
plt.hist(df['FTE'].dropna())

# Add title and labels
plt.title('Distribution of %full-time \n employee works')
plt.xlabel('% of full-time')
plt.ylabel('num employees')
plt.show()
