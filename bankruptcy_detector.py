import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import matplotlib.pyplot as plt

#This will make american bankrupcty csv into a dataframe we can use
df = pd.read_csv('american_bankruptcy.csv')

#removing all the "C_" from the start of the company_name and making them into integers
for index, row in df.iterrows():
    if row['company_name'].startswith("C_"):
        df.loc[index, 'company_name'] = int(row['company_name'][2:])

#we need to split this dataframe into several dataframes for each company
#then we can make a new dataframe containing each company dataframe

columns = df.columns.tolist()
columns = [col for col in columns if col != 'company_name']

new_df_dict = {'company_id':[],'subdataframe' :[]}
num_companies = df['company_name'].nunique()

for x in range(1, num_companies + 1):
    new_df_dict['company_id'].append(x)
    new_df_dict['subdataframe'].append({col: [] for col in columns})

num_rows = df.shape[0]


for x in range(num_rows):
    company_id = df.loc[x,'company_name']

    for y in columns:
        new_df_dict['subdataframe'][company_id - 1][y].append(df.loc[x,y])

subdf_list = []
for subdf_data in new_df_dict['subdataframe']:
    subdf = pd.DataFrame(subdf_data)
    subdf_list.append(subdf)

new_df_dict['subdataframe'] = subdf_list

final_df = pd.DataFrame(new_df_dict)