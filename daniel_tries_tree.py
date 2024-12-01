import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import DataAggregator

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#bring in the csv
df = pd.read_csv('american_bankruptcy.csv')
df.drop(columns=['X4','X13','X15','X16'], inplace=True)


#make string company name into an integer company id
df['company_id'] = df['company_name'].str.extract(r'(\d+)').astype(int)

#sort by `company_id` and `year`
df = df.sort_values(by=['company_id', 'year']).reset_index(drop=True)

#filter companies appearing less than 3 times
df_filtered = df.groupby('company_id').filter(lambda x: len(x) >= 3)

#initialize the new dataframe
result = []

#process each company
for company, group in df_filtered.groupby('company_id'):
    group = group.sort_values('year')  # Sort by year within each company
    M = len(group)
    latest_status = group['status_label'].iloc[-1]
    features = {}

    #iterate over each feature X1 to X18
    for Xn in [f'X{i}' for i in range(1, 19) if f'X{i}' in group.columns]:
        x_values = group[Xn].values
        years = group['year'].values
        t = years - years[-1]  # Adjust time with respect to the latest year

        #A0, A1, A2
        #These are the instantaneous coefficients
        #A0 is the last reported instance
        #A1 is the last reported difference
        #A2 is the last reported second order difference
        A0 = x_values[-1]
        A1 = x_values[-1] - x_values[-2] if M > 1 else np.nan
        A2 = (x_values[-1] - x_values[-3]) / 2 if M > 2 else np.nan

        #B0, B1, B2
        #There are the discrete weighted average coefficients
        #B1 is the weight average sum of points
        #B2 is the weighted average sum of differences
        #B3 is the weighted average sum of second order differences
        
        weights = 1 / np.arange(1, M + 1)
        B0 = np.sum(weights * x_values) / np.sum(weights)
        B1 = (
            np.sum(weights[:-1] * (x_values[:-1] - x_values[1:])) / np.sum(weights[:-1])
            if M > 1
            else np.nan
        )
        B2 = (
            np.sum(weights[:-2] * (x_values[:-2] - x_values[2:])) / np.sum(weights[:-2])
            if M > 2
            else np.nan
        )
        
        #C0, C1, C2, C3
        #These are the fitted polynomial coefficients
        #A continuous function may give an accurate prediction
        
        if M > 3:
            poly = PolynomialFeatures(degree=3)
            X_poly = poly.fit_transform(t.reshape(-1, 1))
            model = LinearRegression().fit(X_poly, x_values)
            C0, C1, C2, C3 = model.intercept_, *model.coef_[1:]
        else:
            C0, C1, C2, C3 = [np.nan] * 4
        
        #add to features dictionary
        features.update(
            {
                f'{Xn}A0': A0,
                f'{Xn}A1': A1,
                f'{Xn}A2': A2,
                f'{Xn}B0': B0,
                f'{Xn}B1': B1,
                f'{Xn}B2': B2,
                f'{Xn}C0': C0,
                f'{Xn}C1': C1,
                f'{Xn}C2': C2,
                f'{Xn}C3': C3
            }
        )

    #append to result
    result.append({'company_id': company, 'status_label': latest_status, **features})

#create the final dataframe
result_df = pd.DataFrame(result)
result_df = result_df.dropna()
result_df = result_df.reset_index(drop=True)


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
        df = result_df

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
        X = result_df.drop(columns=['company_id', 'status_label'])
        y = result_df['status_label']

        # Split data into training and test sets (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on training and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 1. Mean Squared Error (MSE)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        # 2. Mean Absolute Error (MAE)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # 3. R-squared (R²)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # 4. Bias and Variance Estimation
        # Bias is the difference between predicted and true values on the training set
        bias_train = np.mean(y_train - y_train_pred)

        # Variance is the variability of predictions (or errors) on the test set
        variance_test = np.var(y_test_pred)

        # Print results
        print(f"Training MSE: {train_mse}")
        print(f"Test MSE: {test_mse}")
        print(f"Training MAE: {train_mae}")
        print(f"Test MAE: {test_mae}")
        print(f"Training R²: {train_r2}")
        print(f"Test R²: {test_r2}")
        print(f"Bias (Training): {bias_train}")
        print(f"Variance (Test): {variance_test}")

        #Plot the Precision-Recall Curve
        detector.plotPRC(X_test, y_test)




if __name__ == '__main__':
    TreeBankruptcyDetector.main(self=None)