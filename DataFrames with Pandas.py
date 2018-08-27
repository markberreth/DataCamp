import pandas as pd

# Assign the row position of election.loc['Bedford']: x
x = 4

# Assign the column position of election['winner']: y
y = 4

# Print the boolean equivalence
print(election.iloc[x, y] == election.loc['Bedford', 'winner'])

# Read in filename and set the index: election
election = pd.read_csv(filename, index_col='county')

# Create a separate dataframe with the columns ['winner', 'total', 'voters']: results
results = election[['winner', 'total', 'voters']]

# Print the output of results.head()
print(results.head())

# Slice the row labels 'Perry' to 'Potter': p_counties
p_counties = election.loc['Perry': 'Potter']

# Print the p_counties DataFrame
print(p_counties)

# Slice the row labels 'Potter' to 'Perry' in reverse order: p_counties_rev
p_counties_rev = election.loc['Potter':'Perry':-1]

# Print the p_counties_rev DataFrame
print(p_counties_rev)

# Slice the columns from the starting column to 'Obama': left_columns
left_columns = election.loc[:,'state':'Obama']

# Print the output of left_columns.head()
print(left_columns.head())

# Slice the columns from 'Obama' to 'winner': middle_columns
middle_columns = election.loc[:,'Obama':'winner']

# Print the output of middle_columns.head()
print(middle_columns.head())

# Slice the columns from 'Romney' to the end: 'right_columns'
right_columns = election.loc[:,'Romney':'voters']

# Print the output of right_columns.head()
print(right_columns.head())

# Create the list of row labels: rows
rows = ['Philadelphia', 'Centre', 'Fulton']

# Create the list of column labels: cols
cols = ['winner','Obama','Romney']

# Create the new DataFrame: three_counties
three_counties = election.loc[rows, cols]

# Print the three_counties DataFrame
print(three_counties)

# Create the boolean array: high_turnout
high_turnout = election['turnout'] > 70

# Filter the election DataFrame with the high_turnout array: high_turnout_df
high_turnout_df = election.loc[high_turnout]

# Print the high_turnout_results DataFrame
print(high_turnout_df)

# Import numpy
import numpy as np

# Create the boolean array: too_close
too_close = election['margin'] < 1

# Assign np.nan to the 'winner' column where the results were too close to call
election.winner.loc[too_close] = np.nan

# Print the output of election.info()
print(election.info())

# Select the 'age' and 'cabin' columns: df
df = titanic[['age', 'cabin']]

# Print the shape of df
print(df.shape)

# Drop rows in df with how='any' and print the shape
print(df.dropna(how='any').shape)

# Drop rows in df with how='all' and print the shape
print(df.dropna(how='all').shape)

# Drop columns in titanic with less than 1000 non-missing values
print(titanic.dropna(thresh=1000, axis='columns').info())
