import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class BankruptcyDetector:
    
    # Program Fields 
    def __init__(self):

        # Define the pipeline with StandardScaler and LogisticRegression using L2 regularization
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Standardize features
            ('classifier', LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=42))  # Logistic regression with L2 regularization
        ])

    # We can account for a way to use data_processing instead of this data splitter. I didnt use it because I wanted to see how accurtate thwe model can be without tracking companies over years
    # Our model (unless its lying to me) says its 97.3% accurate on the validation set and 97.5% accurate on the test set
    # Ill keep working on it in a bit, eating rn

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
    
    def heatmap(self, df):
        # Create a heatmap of the correlation matrix
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')


    def trainingModel(self,X_train, y_train):
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

if __name__ == "__main__":
    BankruptcyDetector().main()