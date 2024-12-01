
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, auc, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

import officialDataProcessing

class TreeBankruptcyDetector:
    
    def __init__(self):

    #Create a pipeline with StandardScaler and model of choice 
        self.pipeline = make_pipeline(
            StandardScaler(),
            AdaBoostClassifier(
                estimator=HistGradientBoostingClassifier(),
                n_estimators=1000,
                random_state=42,
            )
        )
    
    def trainingModel(self, X_train, y_train):

        print("Training the model...") #Print Statement to update user on the status of the model
        self.pipeline.fit(X=X_train, y=y_train)

    
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


    def main(self):
        detector = TreeBankruptcyDetector()
        df = officialDataProcessing.processed_df()

        # Map target variable
        if 'status_label' not in df.columns:
            raise KeyError("Column 'status_label' not found in the dataset.")
        df['status_label'] = df['status_label'].map({'alive': 0, 'failed': 1})

        # Split data into training and test sets
        train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)

        # Prepare features and labels
        X_train = train_df.drop(columns=['status_label', 'company_id'], errors='ignore')
        y_train = train_df['status_label']
        X_test = test_df.drop(columns=['status_label', 'company_id'], errors='ignore')
        y_test = test_df['status_label']

        # Train the model
        detector.trainingModel(X_train, y_train)

        # Test the model
        y_pred_test = detector.predictModel(X_test)
        print("\nTest Set Metrics:")
        print("Accuracy:", accuracy_score(y_test, y_pred_test))
        print("Classification Report:\n", classification_report(y_test, y_pred_test))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))

        # Assuming result_df is already created as per your code

        # Assume 'status_label' is the target variable, and the rest are features
        X = df.drop(columns=['company_id', 'status_label'])
        y = df['status_label']

        # Split data into training and test sets (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on training and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Mean Squared Error
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        # Print results
        print(f"Training MSE: {train_mse}")
        print(f"Test MSE: {test_mse}")

        #Plot the Precision-Recall Curve
        detector.plotPRC(X_test, y_test)

if __name__ == '__main__':
    TreeBankruptcyDetector.main(self=None)