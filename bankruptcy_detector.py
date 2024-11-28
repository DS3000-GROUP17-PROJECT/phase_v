import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, auc
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import data_processing  # Ensure this module is available and correct

class BankruptcyDetector:
    def __init__(self):
        # Define the pipeline with StandardScaler and LogisticRegression
        self.pipeline = Pipeline([
        ('scaler', StandardScaler()),  
        ('classifier', LogisticRegression(penalty='l2', solver='liblinear', random_state=42, class_weight='balanced'))  # Custom class weights
        ])


    def Data_Splitter(self):
        # Load the dataset
        df = data_processing.final_df()

        # Map target variable
        if 'status_label' not in df.columns:
            raise KeyError("Column 'status_label' not found in the dataset.")
        df['status_label'] = df['status_label'].map({'alive': 0, 'failed': 1})

        # Split data into training and test sets
        train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)

        # Prepare features and labels
        X_train = train_df.drop(columns=['status_label', 'company_id'], errors='ignore')
        y_train = train_df['status_label']
        X_test = test_df.drop(columns=['status_label', 'company_id'], errors='ignore')
        y_test = test_df['status_label']

        return X_train, y_train, X_test, y_test, df

    def trainingModel(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def predictModel(self, X_test):
        return self.pipeline.predict(X_test)

    def plotPRC(self, X, y):
        try:
            y_scores = self.pipeline.predict_proba(X)[:, 1]
        except AttributeError:
            raise RuntimeError("Pipeline is not fitted. Call 'trainingModel' before this method.")

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y, y_scores)
        prc_auc = auc(recall, precision)

        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, marker='.', label=f'PRC (AUC = {prc_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    def main(self):
        # Split data
        X_train, y_train, X_test, y_test, _ = self.Data_Splitter()

        # Train model
        self.trainingModel(X_train, y_train)

        # Predict on test set
        y_pred_test = self.predictModel(X_test)

        # Metrics
        print("\nTest Set Metrics:")
        print("Accuracy:", accuracy_score(y_test, y_pred_test))
        print("Classification Report:\n", classification_report(y_test, y_pred_test))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))

        # Plot Precision-Recall Curve
        self.plotPRC(X_test, y_test)


if __name__ == "__main__":
    try:
        BankruptcyDetector().main()
    except Exception as e:
        print(f"An error occurred: {e}")
