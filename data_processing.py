import pandas as pd
import numpy as np
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

#output the result
def final_df():
    return result_df

#Final dataframe has 6553 companies
print(result_df)

