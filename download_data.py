import pandas as pd
import sys

print("Downloading Online Retail dataset...")
try:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    df = pd.read_excel(url, engine="openpyxl")
    df.to_csv("data/retail_data.csv", index=False)
    print("Successfully downloaded and saved to data/retail_data.csv")
except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)
