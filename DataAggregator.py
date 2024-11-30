import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler

class DataAggregator:
    def __init__(self):
        pass
    def Rename_Columns(self):
          # Load the dataset
        df = pd.read_csv('american_bankruptcy.csv')

        # Rename the columns to make it easier to understand
        df.columns = [
            "company_name", 
            "status_label", 
            "year",
            "Current Assets", 
            "Cost of Goods Sold", 
            "Depreciation", 
            "EBITDA", 
            "Inventory", 
            "Net Income", 
            "Total Receivables",
            "Market Value", 
            "Net Sales", 
            "Total Assets", 
            "Total Long Term Debt",
            "Earnings Before Interest & Taxes", 
            "Gross Profit", 
            "Total Current Liabilities", 
            "Retained Earnings", 
            "Total Revenue", 
            "Total Liabilities", 
            "Total Operating Expenses"
        ]

        # Preprocess the dataset
        df['company_name'] = df['company_name'].str.replace("C_", "").astype(int)
        df['status_label'] = df['status_label'].map({'alive': 0, 'failed': 1})
        df = df.sort_values(by=['company_name', 'year']).reset_index(drop=True)


        return df
    
    def featureCreation(self):
        # Call the Rename_Columns function to get the data
        df = self.Rename_Columns()

        #Create 3 Growth Metrics
        df['Revenue Growth'] = df['Total Revenue'].pct_change().round(2)
        df['Income Growth'] = df['Net Income'].pct_change().round(2)
        df['Debt Growth'] = df['Total Long Term Debt'].pct_change().round(2)

        #Create 3 Financial Ratios
        df['Debt to Equity Ratio'] = (df['Total Long Term Debt'] / df['Total Assets']).round(2)
        df['Gross Margin'] = (df['Gross Profit'] / df['Net Sales']).round(2)
        df['Return on Assets'] = (df['Net Income'] / df['Total Assets']).round(2)

        #Create a csv with only new features and status label in index 1
        df2 = df[['company_name','status_label', 'Revenue Growth', 'Income Growth', 'Debt Growth', 'Debt to Equity Ratio', 'Gross Margin', 'Return on Assets']]

        #Replace infinite values with NaN
        df2 = df2.replace([np.inf, -np.inf], np.nan)

        #Drop rows with NaN values in any of the specified columns
        df2 = df2.dropna(subset=['Revenue Growth', 'Income Growth', 'Debt Growth', 
                                'Debt to Equity Ratio', 'Gross Margin', 'Return on Assets'], how="any")

        #Drop status label column
        df2 = df2.drop(columns=['status_label'])

        return df2
    
    def mergeData(self):
        #Load the renamed data
        renamed_data = self.Rename_Columns()

        #Load the feature data
        feature_data = self.featureCreation()

        #Merge the two DataFrames
        merged_data = renamed_data.merge(feature_data, on='company_name')

        #Save the merged data to a new CSV
        merged_data.to_csv('american_bankruptcy_merged.csv', index=False)

        return merged_data
    
    def Data_Aggregator(self):
        #Call the mergeData function to get the data
        df = self.mergeData()

        # Aggregate data by company
        named_data = df.groupby('company_name').agg({
            'status_label': list,           
            'year': list,
            'Current Assets': list,
            'Cost of Goods Sold': list,
            'Depreciation': list,
            'EBITDA': list,
            'Inventory': list,
            'Net Income': list,
            'Total Receivables': list,
            'Market Value': list,
            'Net Sales': list,
            'Total Assets': list,
            'Total Long Term Debt': list,
            'Earnings Before Interest & Taxes': list,
            'Gross Profit': list,
            'Total Current Liabilities': list,
            'Retained Earnings': list,
            'Total Revenue': list,
            'Total Liabilities': list,
            'Total Operating Expenses': list,
            'Revenue Growth': list,
            'Income Growth': list,
            'Debt Growth': list,
            'Debt to Equity Ratio': list,
            'Gross Margin': list,
            'Return on Assets': list
        }).reset_index()

        # Note: Each company's status_label is consistent (either all 0s or all 1s across years).
        # Summing and dividing by count will always result in a binary value (0 or 1)
        named_data['count'] = named_data['year'].map(len)

        #Sum of the values of the columns
        named_data['status_label'] = named_data['status_label'].map(sum)
        named_data['Current Assets'] = named_data['Current Assets'].map(sum)
        named_data['Cost of Goods Sold'] = named_data['Cost of Goods Sold'].map(sum)
        named_data['Depreciation'] = named_data['Depreciation'].map(sum)
        named_data['EBITDA'] = named_data['EBITDA'].map(sum)
        named_data['Inventory'] = named_data['Inventory'].map(sum)
        named_data['Net Income'] = named_data['Net Income'].map(sum)
        named_data['Total Receivables'] = named_data['Total Receivables'].map(sum)
        named_data['Market Value'] = named_data['Market Value'].map(sum)
        named_data['Net Sales'] = named_data['Net Sales'].map(sum)
        named_data['Total Assets'] = named_data['Total Assets'].map(sum)
        named_data['Total Long Term Debt'] = named_data['Total Long Term Debt'].map(sum)
        named_data['Earnings Before Interest & Taxes'] = named_data['Earnings Before Interest & Taxes'].map(sum)
        named_data['Gross Profit'] = named_data['Gross Profit'].map(sum)
        named_data['Total Current Liabilities'] = named_data['Total Current Liabilities'].map(sum)
        named_data['Retained Earnings'] = named_data['Retained Earnings'].map(sum)
        named_data['Total Revenue'] = named_data['Total Revenue'].map(sum)
        named_data['Total Liabilities'] = named_data['Total Liabilities'].map(sum)
        named_data['Total Operating Expenses'] = named_data['Total Operating Expenses'].map(sum)
        named_data['Revenue Growth'] = named_data['Revenue Growth'].map(sum)
        named_data['Income Growth'] = named_data['Income Growth'].map(sum)
        named_data['Debt Growth'] = named_data['Debt Growth'].map(sum)
        named_data['Debt to Equity Ratio'] = named_data['Debt to Equity Ratio'].map(sum)
        named_data['Gross Margin'] = named_data['Gross Margin'].map(sum)
        named_data['Return on Assets'] = named_data['Return on Assets'].map(sum)

        #create a csv with the new data
        named_data.to_csv('sumOfValuesColumn.csv', index=False)

        #Take the mean of values (divide summed column by count)
        named_data['status_label'] = named_data['status_label'] / named_data['count']
        named_data['Current Assets'] = named_data['Current Assets'] / named_data['count']
        named_data['Cost of Goods Sold'] = named_data['Cost of Goods Sold'] / named_data['count']
        named_data['Depreciation'] = named_data['Depreciation'] / named_data['count']
        named_data['EBITDA'] = named_data['EBITDA'] / named_data['count']
        named_data['Inventory'] = named_data['Inventory'] / named_data['count']
        named_data['Net Income'] = named_data['Net Income'] / named_data['count']
        named_data['Total Receivables'] = named_data['Total Receivables'] / named_data['count']
        named_data['Market Value'] = named_data['Market Value'] / named_data['count']
        named_data['Net Sales'] = named_data['Net Sales'] / named_data['count']
        named_data['Total Assets'] = named_data['Total Assets'] / named_data['count']
        named_data['Total Long Term Debt'] = named_data['Total Long Term Debt'] / named_data['count']
        named_data['Earnings Before Interest & Taxes'] = named_data['Earnings Before Interest & Taxes'] / named_data['count']
        named_data['Gross Profit'] = named_data['Gross Profit'] / named_data['count']
        named_data['Total Current Liabilities'] = named_data['Total Current Liabilities'] / named_data['count']
        named_data['Retained Earnings'] = named_data['Retained Earnings'] / named_data['count']
        named_data['Total Revenue'] = named_data['Total Revenue'] / named_data['count']
        named_data['Total Liabilities'] = named_data['Total Liabilities'] / named_data['count']
        named_data['Total Operating Expenses'] = named_data['Total Operating Expenses'] / named_data['count']
        named_data['Revenue Growth'] = named_data['Revenue Growth'] / named_data['count']
        named_data['Income Growth'] = named_data['Income Growth'] / named_data['count']
        named_data['Debt Growth'] = named_data['Debt Growth'] / named_data['count']
        named_data['Debt to Equity Ratio'] = named_data['Debt to Equity Ratio'] / named_data['count']
        named_data['Gross Margin'] = named_data['Gross Margin'] / named_data['count']
        named_data['Return on Assets'] = named_data['Return on Assets'] / named_data['count']

        #create a csv with the new data
        named_data.to_csv('meanOfValuesColumn.csv', index=False)

        #Drop the year and count columns
        named_data = named_data.drop(columns=['year', 'count'])

        # Split the data into training and test sets based on 85% training and 15% test using train_test_split
        train_data, test_data = train_test_split(named_data, test_size=0.15, random_state=42)

        # Create a csv with the training data
        train_data.to_csv('train_data.csv', index=False)

        # Create a csv with the test data
        test_data.to_csv('test_data.csv', index=False)

        # Prepare the data for desired model (HistGradientBoostingClassifier is best for now)
        X_train = train_data.drop(columns=['status_label'])
        y_train = train_data['status_label']

        X_test = test_data.drop(columns=['status_label'])
        y_test = test_data['status_label']

        return X_train, y_train, X_test, y_test
    
    def main(self):
        self.Data_Aggregator()
        print("Data Aggregator complete!")

if __name__ == "__main__":
    DataAggregator().main()
    