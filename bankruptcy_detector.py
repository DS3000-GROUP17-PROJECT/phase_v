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

#removing all the 'C_" from the start of the company_name and making them into integers
for index, row in df.iterrows():
    df.loc[index, 'company_name'] = int(row['company_name'][2:])
