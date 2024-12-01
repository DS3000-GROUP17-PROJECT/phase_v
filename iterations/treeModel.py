import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

import DataAggregator

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
            AdaBoostClassifier(
                estimator=HistGradientBoostingClassifier(),
                n_estimators=200,
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
        data = DataAggregator.DataAggregator()

        X_train, y_train, X_test, y_test, df = data.Data_Aggregator()

        # Train the model
        detector.trainingModel(X_train, y_train)

        # Test the model
        y_pred_test = detector.predictModel(X_test)
        print("\nTest Set Metrics:")
        print("Accuracy:", accuracy_score(y_test, y_pred_test))
        print("Classification Report:\n", classification_report(y_test, y_pred_test))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))

        #Plot the Precision-Recall Curve
        detector.plotPRC(X_test, y_test)

        #make a csv of merged data
        detector.mergeData()



if __name__ == '__main__':
    TreeBankruptcyDetector.main(self=None)