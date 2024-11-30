import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler

class TreeBankruptcyDetector:
    
    def __init__(self):
    #replace string to change model
    # LogisticRegression(penalty='l2', random_state=42,)
    # estimator=RandomForestClassifier(n_estimators=100, random_state=42)
    # estimator=HistGradientBoostingClassifier(random_state=42)
    # estimator=GradientBoostingClassifier(random_state=42)
    

    #Create a pipeline with StandardScaler and model of choice 
        self.pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(penalty='l2', random_state=42,)
        )

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

        #make a xslx file with the new data
        df.to_excel('american_bankruptcy_renamed.xlsx', index=False)

        return df


    def Data_Aggregator(self, df):

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


        #create a csv with the new data
        named_data.to_csv('named_data.csv', index=False)

        # Split the data into training and test sets based on the year
        train_data = df[df['year'].between(1999, 2014)]
        test_data = df[df['year'].between(2015, 2018)]

        # Prepare the data for logistic regression
        X_train = train_data.drop(columns=['status_label'])
        y_train = train_data['status_label']

        X_test = test_data.drop(columns=['status_label'])
        y_test = test_data['status_label']

        return X_train, y_train, X_test, y_test
    
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

        # Replace infinite values with NaN
        df2 = df2.replace([np.inf, -np.inf], np.nan)

        # Drop rows with NaN values in any of the specified columns
        df2 = df2.dropna(subset=['Revenue Growth', 'Income Growth', 'Debt Growth', 
                                'Debt to Equity Ratio', 'Gross Margin', 'Return on Assets'], how="any")

        # Save the cleaned DataFrame to a new CSV
        df2.to_csv('american_bankruptcy_features.csv', index=False)

        return df2
    
    
    def trainingModel(self, X_train, y_train):
        # Apply RandomUnderSampler
        print("Resampling the data...") #Print Statement to update user on the status of the model
        undersampler = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
        
        # Train the pipeline on the resampled data
        print("Training the model...") #Print Statement to update user on the status of the model
        self.pipeline.fit(X_train_resampled, y_train_resampled)

    
    def predictModel(self, X_test):
        print("Making predictions...") #Print Statement to update user on the status of the model

        y_pred_test = self.pipeline.predict(X_test) #Predict the model

        print("Predictions complete!") #Print Statement to update user on the status of the model

        return y_pred_test

    def plotPRC(self, X, y):
        # Get the predicted probabilities
        y_scores = self.pipeline.predict_proba(X)[:, 1]

        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y, y_scores)
        prc_auc = auc(recall, precision)

        # Plot the precision-recall curve
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, marker='.', label=f'PRC (AUC = {prc_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.show()


    def main():
        detector = TreeBankruptcyDetector()
        """X_train, y_train, X_test, y_test = detector.Data_Aggregator()

        # Train the model
        detector.trainingModel(X_train, y_train)

        # Test the model
        y_pred_test = detector.predictModel(X_test)
        print("\nTest Set Metrics:")
        print("Accuracy:", accuracy_score(y_test, y_pred_test))
        print("Classification Report:\n", classification_report(y_test, y_pred_test))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))

        # Plot the Precision-Recall Curve
        detector.plotPRC(X_test, y_test)
"""

        #test the feature creation function
        detector.featureCreation()

if __name__ == '__main__':
    TreeBankruptcyDetector.main()