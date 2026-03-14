import pandas as pd
import numpy as np

# Load the CSV with appropriate encoding
df = pd.read_csv('/workspaces/Sales-trend-and-forecast-reporting-data/sales_data_sample.csv', encoding='latin1')

# Clean ORDERDATE: remove ' 0:00' and convert to datetime
df['ORDERDATE'] = df['ORDERDATE'].str.replace(' 0:00', '', regex=False)
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], format='%m/%d/%Y')

# Handle missing values
# Fill empty POSTALCODE with 'Unknown'
df['POSTALCODE'] = df['POSTALCODE'].fillna('Unknown')

# Fill empty ADDRESSLINE2 with empty string
df['ADDRESSLINE2'] = df['ADDRESSLINE2'].fillna('')

# Fix encoding issues - replace specific problematic strings
df = df.replace({'Berguvsv�gen': 'Berguvsvägen', 'Mart�n': 'Martín'}, regex=True)

# Replace � in string columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.replace('�', 'ä', regex=False)
    df[col] = df[col].str.replace('�', 'í', regex=False)

# Ensure data types
df['QUANTITYORDERED'] = df['QUANTITYORDERED'].astype(int)
df['PRICEEACH'] = df['PRICEEACH'].astype(float)
df['SALES'] = df['SALES'].astype(float)
df['MSRP'] = df['MSRP'].astype(int)
df['QTR_ID'] = df['QTR_ID'].astype(int)
df['MONTH_ID'] = df['MONTH_ID'].astype(int)
df['YEAR_ID'] = df['YEAR_ID'].astype(int)

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# If duplicates, remove them
if duplicates > 0:
    df = df.drop_duplicates()

# Save cleaned data
df.to_csv('/workspaces/Sales-trend-and-forecast-reporting-data/sales_data_cleaned.csv', index=False)

print("Data cleaning completed. Cleaned data saved to sales_data_cleaned.csv")
print(f"Original rows: {len(df) + duplicates}")
print(f"Cleaned rows: {len(df)}")
print("Columns:", list(df.columns))