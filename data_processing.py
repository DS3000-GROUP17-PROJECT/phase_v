import pandas as pd

#This will make american bankrupcty csv into a dataframe we can use
df = pd.read_csv('american_bankruptcy.csv')

#we need to clean the data and check for errors, nulls and so on


#reframing the dataframe into a collection of smaller dataframes 
#for every company will help us do chronological analysis on each company
def data_reframing(df):

    #removing all the "C_" from the start of the company_name and making them into integers
    for index, row in df.iterrows():
        if row['company_name'].startswith("C_"):
            df.loc[index, 'company_name'] = int(row['company_name'][2:])

    #we need the set of columns for the subdataframes without the company_name column
    columns = df.columns.tolist()
    columns = [col for col in columns if col != 'company_name']

    #we need a dictionary to convert into our final df
    new_df_dict = {'company_id':[],'subdataframe' :[]}
    
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

    #use the list of subdataframes as the values for a column in the full dataframe
    new_df_dict['subdataframe'] = subdf_list

    #make the final dataframe
    final_df = pd.DataFrame(new_df_dict)
    return final_df

processed_df = data_reframing(df)

print(processed_df['subdataframe'][0].head())