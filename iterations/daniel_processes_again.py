import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load the CSV file
df = pd.read_csv('american_bankruptcy.csv')

# Select the relevant columns
data = df[['company_name', 'year', 'status_label']]

# Use .loc[] to modify the dataframe and avoid the warning
data.loc[:, 'X1'] = df['X1'] / df['X14']
data.loc[:, 'X2'] = (df['X1'] - df['X5']) / df['X14']

# Calculate Profitability Ratios
data.loc[:, 'X3'] = df['X6'] / df['X16']
data.loc[:, 'X4'] = df['X13'] / df['X16']
data.loc[:, 'X5'] = df['X6'] / df['X10']

# Calculate Leverage Ratios
data.loc[:, 'X6'] = df['X11'] / (df['X10'] - df['X17'])
data.loc[:, 'X7'] = df['X17'] / df['X10']

# Calculate Efficiency Ratios
data.loc[:, 'X8'] = df['X9'] / df['X7']
data.loc[:, 'X9'] = df['X9'] / df['X5']

# Calculate Solvency & Distress Indicators
data.loc[:, 'X10'] = df['X15'] / df['X10']

df = 0
df = data
"""
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['X1', 'X5', 'X6', 'X9', 'X10', 'X14', 'X15', 'X17'])

# Use .loc[] to modify the dataframe and avoid the warning
data.loc[:, 'X1'] = df['X1'] / df['X14']
data.loc[:, 'X2'] = (df['X1'] - df['X5']) / df['X14']

# Calculate Profitability Ratios
data.loc[:, 'X3'] = df['X6'] / df['X16']
data.loc[:, 'X4'] = df['X13'] / df['X16']
data.loc[:, 'X5'] = df['X6'] / df['X10']

# Calculate Leverage Ratios
data.loc[:, 'X6'] = df['X11'] / (df['X10'] - df['X17'])
data.loc[:, 'X7'] = df['X17'] / df['X10']

# Calculate Efficiency Ratios
data.loc[:, 'X8'] = df['X9'] / df['X7']
data.loc[:, 'X9'] = df['X9'] / df['X5']

# Calculate Solvency & Distress Indicators
# Replace 'interest_expense' with the correct column name if available
# If interest expense is not in the dataset, skip this line
# data['Interest_Coverage_Ratio'] = data['X12'] / data['interest_expense']

data.loc[:, 'X10'] = df['X15'] / df['X10']

df = data
"""
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Make string company name into an integer company id
df['company_id'] = df['company_name'].str.extract(r'(\d+)').astype(int)

# Sort by `company_id` and `year`
df = df.sort_values(by=['company_id', 'year']).reset_index(drop=True)

# Filter companies appearing less than 3 times
df_filtered = df.groupby('company_id').filter(lambda x: len(x) >= 3)

# Initialize the new dataframe
result = []

# Process each company
for company, group in df_filtered.groupby('company_id'):
    group = group.sort_values('year')  # Sort by year within each company
    M = len(group)
    latest_status = group['status_label'].iloc[-1]
    features = {}

    # Iterate over each feature X1 to X18
    for Xn in [f'X{i}' for i in range(1, 11) if f'X{i}' in group.columns]:
        x_values = group[Xn].values
        years = group['year'].values
        t = years - years[-1]  # Adjust time with respect to the latest year

        # A0, A1, A2
        A0 = x_values[-1]
        A1 = x_values[-1] - x_values[-2] if M > 1 else np.nan
        A2 = (x_values[-1] - x_values[-3]) / 2 if M > 2 else np.nan

        # B0, B1, B2
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
        
        # C0, C1, C2, C3
        if M > 3:
            poly = PolynomialFeatures(degree=3)
            X_poly = poly.fit_transform(t.reshape(-1, 1))
            model = LinearRegression().fit(X_poly, x_values)
            C0, C1, C2, C3 = model.intercept_, *model.coef_[1:]
        else:
            C0, C1, C2, C3 = [np.nan] * 4
        
        # Add to features dictionary
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

    # Append to result
    result.append({'company_id': company, 'status_label': latest_status, **features})

# Create the final dataframe
result_df = pd.DataFrame(result)
result_df = result_df.dropna()
result_df = result_df.reset_index(drop=True)

# Output the result
def final_df():
    return result_df

# Final dataframe has 6553 companies
print(result_df)

#print (data)