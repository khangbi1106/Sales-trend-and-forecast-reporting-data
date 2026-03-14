import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned sales data
df = pd.read_csv('/workspaces/Sales-trend-and-forecast-reporting-data/sales_data_cleaned.csv')

# Convert ORDERDATE to datetime
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])

# Extract time components
df['Year'] = df['ORDERDATE'].dt.year
df['Month'] = df['ORDERDATE'].dt.month
df['Quarter'] = df['ORDERDATE'].dt.quarter
df['YearMonth'] = df['ORDERDATE'].dt.to_period('M')

print("=== SALES TREND ANALYSIS ===\n")

# 1. Sales by Year
print("1. SALES BY YEAR")
yearly_sales = df.groupby('Year')['SALES'].sum().sort_index()
for year, sales in yearly_sales.items():
    print(f"{year}: ${sales:,.2f}")
print()

# 2. Sales by Quarter
print("2. SALES BY QUARTER")
quarterly_sales = df.groupby(['Year', 'Quarter'])['SALES'].sum().reset_index()
quarterly_sales['YearQuarter'] = quarterly_sales['Year'].astype(str) + ' Q' + quarterly_sales['Quarter'].astype(str)
for _, row in quarterly_sales.iterrows():
    print(f"{row['YearQuarter']}: ${row['SALES']:,.2f}")
print()

# 3. Sales by Month
print("3. SALES BY MONTH")
monthly_sales = df.groupby('YearMonth')['SALES'].sum()
for period, sales in monthly_sales.items():
    print(f"{period}: ${sales:,.2f}")
print()

# 4. Trend Analysis
print("4. TREND ANALYSIS")
total_years = len(yearly_sales)
if total_years > 1:
    first_year_sales = yearly_sales.iloc[0]
    last_year_sales = yearly_sales.iloc[-1]
    overall_growth = ((last_year_sales - first_year_sales) / first_year_sales) * 100

    # Year-over-year changes
    yoy_changes = yearly_sales.pct_change() * 100

    print(f"Overall trend (2003-2005): {'Increasing' if overall_growth > 0 else 'Decreasing'} ({overall_growth:+.1f}%)")
    print(f"Year with highest sales: {yearly_sales.idxmax()} (${yearly_sales.max():,.2f})")
    print(f"Year with lowest sales: {yearly_sales.idxmin()} (${yearly_sales.min():,.2f})")

    print("\nYear-over-Year Changes:")
    for i in range(1, len(yearly_sales)):
        prev_year = yearly_sales.index[i-1]
        curr_year = yearly_sales.index[i]
        change_pct = yoy_changes.iloc[i]
        print(f"{prev_year} to {curr_year}: {change_pct:+.1f}%")
else:
    print("Only one year of data available")

print()

# 5. Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Sales Trends Over Time', fontsize=16, fontweight='bold')

# Yearly sales bar chart
axes[0, 0].bar(yearly_sales.index.astype(str), yearly_sales.values, color='skyblue')
axes[0, 0].set_title('Yearly Sales')
axes[0, 0].set_ylabel('Sales ($)')
axes[0, 0].tick_params(axis='x', rotation=45)

# Quarterly sales line chart
axes[0, 1].plot(quarterly_sales['YearQuarter'], quarterly_sales['SALES'], marker='o', color='orange')
axes[0, 1].set_title('Quarterly Sales Trend')
axes[0, 1].set_ylabel('Sales ($)')
axes[0, 1].tick_params(axis='x', rotation=45)

# Monthly sales line chart (main request)
axes[1, 0].plot(monthly_sales.index.astype(str), monthly_sales.values, marker='o', color='green', linewidth=2)
axes[1, 0].set_title('Monthly Sales Trend')
axes[1, 0].set_ylabel('Sales ($)')
axes[1, 0].tick_params(axis='x', rotation=45)

# Year-over-year growth
if total_years > 1:
    yoy_growth = yearly_sales.pct_change() * 100
    axes[1, 1].bar(yoy_growth.index[1:].astype(str), yoy_growth.values[1:], color='red')
    axes[1, 1].set_title('Year-over-Year Growth (%)')
    axes[1, 1].set_ylabel('Growth (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
else:
    axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor YoY growth', ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Year-over-Year Growth')

plt.tight_layout()
plt.savefig('/workspaces/Sales-trend-and-forecast-reporting-data/sales_trends_charts.png', dpi=300, bbox_inches='tight')
plt.close()

print("Charts saved to sales_trends_charts.png")

# Additional insights
print("\n=== ADDITIONAL INSIGHTS ===")
print(f"Average monthly sales: ${monthly_sales.mean():,.2f}")
print(f"Most profitable month: {monthly_sales.idxmax()} (${monthly_sales.max():,.2f})")
print(f"Least profitable month: {monthly_sales.idxmin()} (${monthly_sales.min():,.2f})")

# Seasonal analysis
monthly_avg = df.groupby(df['ORDERDATE'].dt.month)['SALES'].mean()
best_month = monthly_avg.idxmax()
worst_month = monthly_avg.idxmin()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
print(f"Best performing month (average): {month_names[best_month-1]} (${monthly_avg.max():,.2f})")
print(f"Worst performing month (average): {month_names[worst_month-1]} (${monthly_avg.min():,.2f})")