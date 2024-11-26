import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


#This will make american bankrupcty csv into a dataframe we can use
#we have already checked that this data is clean and does not contain nulls or errors
df = pd.read_csv('american_bankruptcy.csv')

#reframing the dataframe into a collection of smaller dataframes 
#reframing the data will help us analyze the data from 3 dimensions
#companies, features and time

def data_reframing(df):

    #removing all the "C_" from the start of the company_name and making them into integers
    for index, row in df.iterrows():
        if row['company_name'].startswith("C_"):
            df.loc[index, 'company_name'] = int(row['company_name'][2:])

    #we need the set of columns for the subdataframes without the company_name column
    columns = df.columns.tolist()
    columns = [col for col in columns if col != 'company_name']

    #we need a dictionary to convert into our final df
    new_df_dict = {'company_id':[],'status_label':[],'subdataframe' :[]}
    
    #we need the number of companies for logical purposes
    num_companies = df['company_name'].nunique()

    #create a structure for the dictionary for the final df
    for x in range(1, num_companies + 1):
        new_df_dict['company_id'].append(x)
        new_df_dict['subdataframe'].append({col: [] for col in columns})

    #we need number of rows for logical purposes
    num_rows = df.shape[0]

    #for all rows we need to extract data and put into dictionaries for the new subdataframes
    for x in range(num_rows):
        company_id = df.loc[x,'company_name']
        
        for y in columns:
            new_df_dict['subdataframe'][company_id - 1][y].append(df.loc[x,y])

    #turn these dictionaries into subdataframes and make a list of subdataframes
    subdf_list = []
    for subdf_data in new_df_dict['subdataframe']:
        subdf = pd.DataFrame(subdf_data)
        subdf_list.append(subdf)

    for subdf in subdf_list:
        # Get the 'status_label' from the last row
        status = subdf.iloc[-1]['status_label']
        new_df_dict['status_label'].append(status)  # Append to dictionary
        
    # Remove 'status_label' from all sub-DataFrames in one go
    subdf_list = [subdf.drop(columns=['status_label'], inplace = True) for subdf in subdf_list]

    # Update the dictionary with the cleaned sub-DataFrames
    new_df_dict['subdataframe'] = subdf_list


    #make the final dataframe
    final_df = pd.DataFrame(new_df_dict)

    return final_df

df = data_reframing(df)

#we need to transform each subdataframe from the time domain to a new feature domain.

def time_feature_transform(subdf):
    #The following objects are needed for logical purposes
    n = subdf.shape[0]  # number of rows
    columns = subdf.shape[1]  # number of columns
    x = subdf['year']
    c = x[-1]

    x_transformed = x - c

    output_dict = {}

    for y in range (columns):
        if subdf.columns[y] != 'year':
            #The 'A' set of coefficients are the final values of a company's report
            #A0 is the final value
            #A1 is the difference between the last and the second last value
            #A2 is the difference between the last and third last value divided by 2, the second order difference
            A0 = subdf.iloc[-1,y]
            A1 = subdf.iloc[-2,y] - A0
            A2 = (subdf.iloc[-3,y] - A0)/2

            #The 'B' set of coefficients are the weighted average of a company's report
            #The weight is 1/(the year's before the last report + 1)
            #ex. the year before has a weight of 1/2, 2 years before has a weight of 1/3
            #B0 is the weighted average of individual data points
            #B1 is the weighted average of the differences between subsequent points
            #B2 is the weighted average of the second-order differences
            B0 = sum((1 / k) * subdf.iloc[-k, y] for k in range(1, n + 1)) / sum((1 / k) for k in range(1, n + 1))
            B1 = sum((1 / k) * (subdf.iloc[-(k+1), y] - subdf.iloc[-k, y]) for k in range(1, n)) / sum((1 / k) for k in range(1, n))
            B2 = sum((1 / k) * (subdf.iloc[-(k+2), y] - subdf.iloc[-k, y]) for k in range(1, n - 1)) / sum((1 / k) for k in range(1, n - 1))
            
            #The 'C' set of coefficients are the coefficients of a curve fitting to the datasets
            #This helps us analyze the data points from a continuous perspective
            #we use scikitlearn to fit a polynomial to the each column
            poly = PolynomialFeatures(degree=3)
            x_poly = poly.fit_transform(x_transformed)

            model = LinearRegression()
            model.fit(x_poly, subdf.iloc[:, y])

            coefficients = model.coef_
        
            #The following coefficients are meant to fit the graph:
            #Y(x-c) = C0 + C1x + C2x^2 + C3x^3
            #where c is the last year reported by the company
            C0 = coefficients[0]
            C1 = coefficients[1]
            C2 = coefficients[2]
            C3 = coefficients[3]

            output_dict[subdf.columns[y]] = [A0,A1,A2,B0,B1,B2,C0,C1,C2,C3]
        
    final_subdf = pd.DataFrame(output_dict)
    return final_subdf

#Now that the dataframe is no longer time-dependent, the dataframe no longer has 3 dimensions
#The dataframe now has 2 dimensions: companies and features

def dimension_2_1_transform(subdf):
    final_dict = {}

#10 new columns for every old column, representing a 10 features for every previous feature.
    for col in subdf.columns:
        final_dict[(col + 'A0')] = subdf[col][0]
        final_dict[(col + 'A1')] = subdf[col][1]
        final_dict[(col + 'A2')] = subdf[col][2]
        final_dict[(col + 'B0')] = subdf[col][3]
        final_dict[(col + 'B1')] = subdf[col][4]
        final_dict[(col + 'B2')] = subdf[col][5]
        final_dict[(col + 'C0')] = subdf[col][6]
        final_dict[(col + 'C1')] = subdf[col][7]
        final_dict[(col + 'C2')] = subdf[col][8]
        final_dict[(col + 'C3')] = subdf[col][9]

    return final_dict

#Now a final transform is necessary 
def full_transform(df):
    rows = df.shape[0]
    
    subdf = df['subdataframe'][0]
    subdf = time_feature_transform(subdf)
    final_subdict = dimension_2_1_transform(subdf)
    rows_to_drop = []

    for row in range(rows):
        subdf = df['subdataframe'][row]
        if subdf.shape[0] < 3:
            rows_to_drop.append(row)
        else:
            subdf = time_feature_transform(subdf)
            subdf_dict = dimension_2_1_transform(subdf)

            for key in final_subdict:
                final_subdict[key] = [final_subdict[key],subdf_dict[key]]

    final_df = df

    if rows_to_drop != []:
        final_df.drop(index=rows_to_drop, inplace=True)

    final_df.drop(columns = ['subdataframe'], implace = True)
    subdf_reframed = pd.DataFrame(final_subdict)
    final_df = pd.concat([final_df, subdf_reframed])

    return final_df
            

df = full_transform(df)

#make df a method that we can use to access the processed dataframe for the bankruptcy detector

"""def processed_df():
    df = pd.read_csv('american_bankruptcy.csv')
    df = data_reframing(df)
    df = full_transform(df)
    return df"""

print(df['subdataframe'][0].head())