import pandas as pd
import numpy as np

# Reading in the CSV
us_cases_df = pd.read_csv("US_Covid_Cases.csv")

# Counting the Number of Days after Start and putting this in a list
indices = []
for x in range(len(us_cases_df)):
    indices.append(x)

# Changing the index list to a Pandas series
indices_series = pd.Series(indices)

# Adding Series to existing Dataframe
us_cases_df['Days Since Start'] = indices_series

# Extracting the Days and Cases from the Old Dataframe into a new one
days_cases_df = us_cases_df[['Days Since Start', 'cases']]

# Removing the header row
new_header = days_cases_df.iloc[0] #grab the first row for the header
days_cases_df = days_cases_df[1:] #take the data less the header row
days_cases_df.columns = new_header #set the header row as the df header

# Training data will be first 80% of data and Test will be last 20% of data
n = 80
m = 20
us_cases_train = days_cases_df.head(int(len(us_cases_df)*(n/100)+1))
us_cases_test = days_cases_df.tail(int(len(us_cases_df)*(m/100)))
us_cases_test = us_cases_test.iloc[1:]
# Removing the header row
new_header = us_cases_test.iloc[0] #grab the first row for the header
us_cases_test = us_cases_test[1:] #take the data less the header row
us_cases_test.columns = new_header #set the header row as the df header

#Exporting to txt files
us_cases_train.to_csv('us_train.txt', sep=' ', index=False)
us_cases_test.to_csv('us_test.txt', sep=' ', index=False)





"""
# Training data will be first 80% of data and Test will be last 20% of data
n = 80
m = 20
us_cases_train = us_cases_df.head(int(len(us_cases_df)*(n/100)+1))
us_cases_test = us_cases_df.tail(int(len(us_cases_df)*(m/100)))
train_indices = []
test_indices = []
j = 0
for x in range(len(us_cases_train)):
    train_indices.append(x)
    j = j + 1
for y in range(len(us_cases_test)):
    test_indices.append(j)
    j = j + 1
train_series = pd.Series(train_indices)
test_series = pd.Series(test_indices)
us_cases_train_series = us_cases_train[0]
us_cases_test_series = us_cases_test[0]

clean_train = pd.concat([train_series, us_cases_train_series], axis = 1)
new_header = clean_train.iloc[0] #grab the first row for the header
clean_train = clean_train[1:] #take the data less the header row
clean_train.columns = new_header
clean_test = pd.concat([test_series, us_cases_test_series], axis = 1)
new_header = clean_test.iloc[0] #grab the first row for the header
clean_test = clean_test[1:] #take the data less the header row
clean_test.columns = new_header
print(clean_train)
print(clean_test)

clean_train.to_csv('us_train.txt', sep=' ', index=False)
clean_test.to_csv('us_test.txt', sep=' ', index=False)
"""