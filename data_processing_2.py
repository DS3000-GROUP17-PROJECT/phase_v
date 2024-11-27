import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Example DataFrame (replace with your actual data loading)
df = pd.read_csv('american_bankruptcy.csv')

# Step 1: Convert `company_name` to integer
df['company_id'] = df['company_name'].str.extract(r'(\d+)').astype(int)

# Step 2: Sort by `company_id` and `year`
df = df.sort_values(by=['company_id', 'year']).reset_index(drop=True)

# Step 3: Filter companies appearing less than 3 times
df_filtered = df.groupby('company_id').filter(lambda x: len(x) >= 3)

# Step 4: Initialize the new dataframe
result = []

# Step 5: Process each company
for company, group in df_filtered.groupby('company_id'):
    group = group.sort_values('year')  # Sort by year within each company
    M = len(group)
    latest_status = group['status_label'].iloc[-1]
    features = {}

    # Iterate over each feature X1 to X18
    for Xn in [f'X{i}' for i in range(1, 19) if f'X{i}' in group.columns]:
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
                f'{Xn}C3': C3,
            }
        )

    # Append to result
    result.append({'company_id': company, 'status_label': latest_status, **features})

# Step 6: Create the final dataframe
result_df = pd.DataFrame(result)

# Step 7: Output the result
print(result_df)
