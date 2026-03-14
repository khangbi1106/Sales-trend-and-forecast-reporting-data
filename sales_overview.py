import pandas as pd

# Load the cleaned sales data
df = pd.read_csv('/workspaces/Sales-trend-and-forecast-reporting-data/sales_data_cleaned.csv')

# Calculate basic KPIs
total_revenue = df['SALES'].sum()
total_quantity_sold = df['QUANTITYORDERED'].sum()
number_of_orders = df['ORDERNUMBER'].nunique()  # Unique order numbers
average_sales_per_order = total_revenue / number_of_orders

# Print the Sales Overview
print("=== SALES OVERVIEW ===")
print(f"Total Revenue: ${total_revenue:,.2f}")
print(f"Total Quantity Sold: {total_quantity_sold:,}")
print(f"Number of Orders: {number_of_orders:,}")
print(f"Average Sales per Order: ${average_sales_per_order:,.2f}")

# Additional insights
print("\n=== ADDITIONAL INSIGHTS ===")
print(f"Total Records: {len(df):,}")
print(f"Average Quantity per Order: {total_quantity_sold / number_of_orders:.2f}")
print(f"Date Range: {df['ORDERDATE'].min()} to {df['ORDERDATE'].max()}")

# Top product lines by revenue
top_product_lines = df.groupby('PRODUCTLINE')['SALES'].sum().sort_values(ascending=False)
print(f"\nTop Product Line by Revenue: {top_product_lines.index[0]} (${top_product_lines.iloc[0]:,.2f})")

# Top countries by revenue
top_countries = df.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False)
print(f"Top Country by Revenue: {top_countries.index[0]} (${top_countries.iloc[0]:,.2f})")