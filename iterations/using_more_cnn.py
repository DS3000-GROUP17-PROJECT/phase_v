import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('american_bankruptcy.csv')
# Assuming 'df' is your original DataFrame
# Step 1: Group by 'company_name' and count the number of distinct years
grouped = df.groupby('company_name')['year'].nunique()

# Step 2: Create a new column to represent the number of years reported
df['years_reported'] = df['company_name'].map(grouped)

# Step 3: Initialize a list to store the DataFrames
max_years_reported = grouped.max()
dfs_by_years = [None] * (max_years_reported + 1)

# Step 4: Group by the number of years reported and create DataFrames for each group
for years in range(1, max_years_reported + 1):
    companies_in_group = grouped[grouped == years].index
    df_group = df[df['company_name'].isin(companies_in_group)]
    df_group.reset_index(inplace=True)
    df_group.drop(columns='years_reported', inplace=True)
    dfs_by_years[years] = df_group

#print (dfs_by_years[1])
# Now `dfs_by_years` contains DataFrames for each group (1 year, 2 years, ..., max_years_reported)
