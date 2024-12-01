# Using Exploratory Data Analysis to understand the data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
class EDA:
    def __init__(self):
        pass

    # Load the dataset
    def load_data(self):
        df = pd.read_csv('american_bankruptcy.csv')
        return df
    
    # Create a heatmap of the correlation matrix
    def heatmap(self, df):

        # Select only numeric columns (Essentially dropping Company# & )
        numeric_df = df.select_dtypes(include=[np.number])

        # Rename the columns
        numeric_df.columns = ["Year",
                      "Current Assets", 
                      "Cost of Goods Sold", 
                      "Depreciation", 
                      "EBITDA", 
                      "Inventory", 
                      "Net Income", 
                      "Total Recievables",
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
                      "Total Operating Expenses"]
        
        # Create a heatmap
        cols = numeric_df.columns
        cm = np.corrcoef(numeric_df[cols].values.T)
        sns.set(font_scale=1.5)
        hm = sns.heatmap(cm,
                         cbar=True,
                         annot=True,
                         square=True,
                         fmt='.2f',
                         annot_kws={'size': 10},
                         yticklabels=cols,
                         xticklabels=cols)
        plt.tight_layout()
        plt.show()

    # Main function
    def main(self):
        df = self.load_data()
        self.heatmap(df)

if __name__ == "__main__":
    EDA().main()