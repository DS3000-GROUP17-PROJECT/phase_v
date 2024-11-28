import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class TreeBankruptcyDetector:
    
    def __init__(self):
        # Define the pipeline with StandardScaler and RandomForestClassifier with AdaBoost
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Standardize features
            ('classifier', AdaBoostClassifier(
                estimator=RandomForestClassifier(n_estimators=100, random_state=42),
                n_estimators=50,
                random_state=42
            ))  # Random Forest with AdaBoost
        ])

    def Data_Splitter(self):
        # Load the dataset
        df = pd.read_csv('american_bankruptcy.csv')

        # Preprocess the dataset
        df['company_name'] = df['company_name'].str.replace("C_", "").astype(int)
        df['status_label'] = df['status_label'].map({'alive': 0, 'failed': 1})
        df = df.sort_values(by=['company_name', 'year']).reset_index(drop=True)

        # Split data into training, validation, and test sets by year (as the dataset recommends)
        train_df = df[df['year'] <= 2011]
        valid_df = df[(df['year'] > 2011) & (df['year'] <= 2014)]
        test_df = df[df['year'] > 2014]

        # Prepare features and labels
        X_train = train_df.drop(columns=['status_label', 'company_name', 'year'])
        y_train = train_df['status_label']

        X_valid = valid_df.drop(columns=['status_label', 'company_name', 'year'])
        y_valid = valid_df['status_label']

        X_test = test_df.drop(columns=['status_label', 'company_name', 'year'])
        y_test = test_df['status_label']

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

    def main():
        detector = TreeBankruptcyDetector()
        X_train, y_train, X_valid, y_valid, X_test, y_test, df = detector.Data_Splitter()

        # Train the model
        detector.trainingModel(X_train, y_train)

        # Validate the model
        y_pred_valid = detector.validateModel(X_valid)
        print("Validation Set Metrics:")
        print("Accuracy:", accuracy_score(y_valid, y_pred_valid))
        print("Classification Report:\n", classification_report(y_valid, y_pred_valid))
        print("Confusion Matrix:\n", confusion_matrix(y_valid, y_pred_valid))

        # Test the model
        y_pred_test = detector.predictModel(X_test)
        print("\nTest Set Metrics:")
        print("Accuracy:", accuracy_score(y_test, y_pred_test))
        print("Classification Report:\n", classification_report(y_test, y_pred_test))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))

# Run the main function       
if __name__ == '__main__':
    TreeBankruptcyDetector.main()