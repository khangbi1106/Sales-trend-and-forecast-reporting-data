import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned sales data
df = pd.read_csv('/workspaces/Sales-trend-and-forecast-reporting-data/sales_data_cleaned.csv')

# Convert ORDERDATE to datetime if not already
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])

# Create subplots for visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Sales Overview Dashboard', fontsize=16, fontweight='bold')

# 1. Sales by Product Line
product_sales = df.groupby('PRODUCTLINE')['SALES'].sum().sort_values(ascending=False)
axes[0, 0].bar(product_sales.index, product_sales.values, color='skyblue')
axes[0, 0].set_title('Sales by Product Line')
axes[0, 0].set_ylabel('Sales ($)')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Sales by Country
country_sales = df.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False).head(10)
axes[0, 1].bar(country_sales.index, country_sales.values, color='lightgreen')
axes[0, 1].set_title('Top 10 Countries by Sales')
axes[0, 1].set_ylabel('Sales ($)')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Monthly Sales Trend
df['YearMonth'] = df['ORDERDATE'].dt.to_period('M')
monthly_sales = df.groupby('YearMonth')['SALES'].sum()
axes[1, 0].plot(monthly_sales.index.astype(str), monthly_sales.values, marker='o', color='orange')
axes[1, 0].set_title('Monthly Sales Trend')
axes[1, 0].set_ylabel('Sales ($)')
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. Order Status Distribution
status_counts = df['STATUS'].value_counts()
axes[1, 1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
axes[1, 1].set_title('Order Status Distribution')

plt.tight_layout()
plt.savefig('/workspaces/Sales-trend-and-forecast-reporting-data/sales_overview_charts.png', dpi=300, bbox_inches='tight')
print("Sales overview charts saved to sales_overview_charts.png")

# Print key metrics again for reference
total_revenue = df['SALES'].sum()
total_quantity = df['QUANTITYORDERED'].sum()
num_orders = df['ORDERNUMBER'].nunique()
avg_order_value = total_revenue / num_orders

print("\n=== KEY METRICS ===")
print(f"Total Revenue: ${total_revenue:,.2f}")
print(f"Total Quantity Sold: {total_quantity:,}")
print(f"Number of Orders: {num_orders:,}")
print(f"Average Order Value: ${avg_order_value:,.2f}")