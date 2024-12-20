import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, auc
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

class BankruptcyDetector:
    
    def __init__(self):
        # Define the pipeline with StandardScaler and LogisticRegression using L2 regularization
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Standardize features
            ('classifier', LogisticRegression(penalty='l2', solver='liblinear', random_state=42))  # Logistic regression with L2 regularization
        ])

    def Data_Splitter(self):
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

        # Aggregate data by company
        aggregated_data = df.groupby('company_name').agg({
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
            'status_label': 'first'
        }).reset_index()

        # Feature engineering: Calculate differences between consecutive years
        for col in aggregated_data.columns[2:-1]:  # Skip 'company_name', 'year', and 'status_label'
            aggregated_data[col + '_diff'] = aggregated_data[col].apply(lambda x: np.diff(x).tolist())

        # Flatten the lists into individual columns
        feature_cols = []
        for col in aggregated_data.columns[2:-1]:  # Skip 'company_name', 'year', and 'status_label'
            max_len = max(aggregated_data[col].apply(len))
            for i in range(max_len):
                aggregated_data[col + f'_{i}'] = aggregated_data[col].apply(lambda x: x[i] if i < len(x) else np.nan)
                feature_cols.append(col + f'_{i}')

        # Prepare features and labels
        X = aggregated_data[feature_cols]
        y = aggregated_data['status_label']

        # Split data into training, validation, and test sets
        X_train, X_valid, X_test = np.split(X, [int(.6*len(X)), int(.8*len(X))])
        y_train, y_valid, y_test = np.split(y, [int(.6*len(y)), int(.8*len(y))])

        return X_train, y_train, X_valid, y_valid, X_test, y_test, df

    def trainingModel(self, X_train, y_train):
        # Train the pipeline on the training data
        self.pipeline.fit(X_train, y_train)

    def validateModel(self, X_valid):
        # Make prediction on the validation set
        y_pred_valid = self.pipeline.predict(X_valid)
        return y_pred_valid
    
    def predictModel(self, X_test):
        # Make prediction on the test set
        y_pred_test = self.pipeline.predict(X_test)
        return y_pred_test
    
    def RidgeRegressionCoef(self, X, y):
        # Define the range of regularization coefficients
        lambda_values = np.logspace(-8, 6, 15)  # 15 values from exp(-8) to exp(6)

        # Initialize an array to store mean squared errors
        mse_list = []

        # Perform 10-fold CV for each λ value
        for lam in lambda_values:
            self.pipeline.set_params(classifier__C=1/lam)  # Set the inverse of lambda as C
            mse_scores = cross_val_score(self.pipeline, X, y, cv=10, scoring='neg_mean_squared_error')
            mean_mse = -mse_scores.mean()  # Negate because cross_val_score returns negative MSE
            mse_list.append(mean_mse)

        # Plot mean squared error vs log(λ)
        plt.figure(figsize=(10, 6))
        plt.plot(np.log(lambda_values), mse_list, marker='o')
        plt.title('Mean Squared Error vs log(λ)')
        plt.xlabel('log(λ)')
        plt.ylabel('Mean Squared Error')
        plt.grid(True)
        plt.show()

        # Determine the best regularization parameter
        best_lambda = lambda_values[np.argmin(mse_list)]
        best_mse = min(mse_list)

        print(f"Best regularization parameter (λ): {best_lambda:.6f}")
        print(f"Corresponding Mean Squared Error: {best_mse:.4f}")

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

    def main(self):
        # Storing all the split data into a holder array
        ArrayOfSplitData = self.Data_Splitter()

        # Declaring the training data
        X_train = ArrayOfSplitData[0]
        y_train = ArrayOfSplitData[1]

        # Declaring the validation data
        X_valid = ArrayOfSplitData[2]
        y_valid = ArrayOfSplitData[3]

        # Declaring the test data
        X_test = ArrayOfSplitData[4]
        y_test = ArrayOfSplitData[5]

        # Declaring the full dataframe
        df = ArrayOfSplitData[6]

        # Calling the training model function (Training)
        self.trainingModel(X_train, y_train)

        # Calling the validate model function (Validation)
        y_pred_valid = self.validateModel(X_valid)

        # Calling the predict model function (Test)
        y_pred_test = self.predictModel(X_test)

        # Evaluate the model on the validation set
        print("Validation Set Metrics:")
        print("Accuracy:", accuracy_score(y_valid, y_pred_valid))
        print("Classification Report:\n", classification_report(y_valid, y_pred_valid))
        print("Confusion Matrix:\n", confusion_matrix(y_valid, y_pred_valid))

        # Evaluate the model on the test set
        print("\nTest Set Metrics:")
        print("Accuracy:", accuracy_score(y_test, y_pred_test))
        print("Classification Report:\n", classification_report(y_test, y_pred_test))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))

        # Analyze the ridge coefficients
        self.RidgeRegressionCoef(X_train, y_train)

        # Plot the Precision-Recall Curve
        self.plotPRC(X_test, y_test)

if __name__ == "__main__":
    BankruptcyDetector().main()