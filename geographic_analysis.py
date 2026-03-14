import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned sales data
df = pd.read_csv('/workspaces/Sales-trend-and-forecast-reporting-data/sales_data_cleaned.csv')

print("=== GEOGRAPHIC SALES ANALYSIS ===\n")

# 1. Sales by Country
print("1. SALES BY COUNTRY")
sales_by_country = df.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False)
for country, sales in sales_by_country.items():
    print(f"{country}: ${sales:,.2f}")
print()

# 2. Sales by Territory
print("2. SALES BY TERRITORY")
sales_by_territory = df.groupby('TERRITORY')['SALES'].sum().sort_values(ascending=False)
for territory, sales in sales_by_territory.items():
    print(f"{territory}: ${sales:,.2f}")
print()

# 3. Top Countries Analysis
print("3. TOP COUNTRIES ANALYSIS")
top_countries = sales_by_country.head(10)
print("Top 10 Countries by Sales:")
for i, (country, sales) in enumerate(top_countries.items(), 1):
    percentage = (sales / sales_by_country.sum() * 100)
    print(f"{i}. {country}: ${sales:,.2f} ({percentage:.1f}%)")
print()

# 4. Territory Performance Summary
print("4. TERRITORY PERFORMANCE SUMMARY")
territory_summary = pd.DataFrame({
    'Total Sales': sales_by_territory,
    'Number of Countries': df.groupby('TERRITORY')['COUNTRY'].nunique(),
    'Average Order Value': df.groupby('TERRITORY')['SALES'].mean(),
    'Percentage of Total Sales': (sales_by_territory / sales_by_territory.sum() * 100).round(1)
})
territory_summary = territory_summary.sort_values('Total Sales', ascending=False)
print(territory_summary.to_string())
print()

# 5. Geographic Insights
print("5. GEOGRAPHIC INSIGHTS")
total_sales = sales_by_country.sum()

# Market concentration
top_3_countries_pct = (sales_by_country.head(3).sum() / total_sales * 100)
top_territory = sales_by_territory.idxmax()
top_territory_sales = sales_by_territory.max()

print(f"Total Global Sales: ${total_sales:,.2f}")
print(f"Number of Countries: {df['COUNTRY'].nunique()}")
print(f"Number of Territories: {df['TERRITORY'].nunique()}")
print()
print(f"Top 3 Countries Account for: {top_3_countries_pct:.1f}% of total sales")
print(f"Best Performing Territory: {top_territory} (${top_territory_sales:,.2f})")
print()

# Country performance metrics
best_country = sales_by_country.idxmax()
worst_country = sales_by_country.idxmin()
avg_country_sales = sales_by_country.mean()

print(f"Best Performing Country: {best_country} (${sales_by_country.max():,.2f})")
print(f"Worst Performing Country: {worst_country} (${sales_by_country.min():,.2f})")
print(f"Average Sales per Country: ${avg_country_sales:,.2f}")
print()

# Territory efficiency (sales per country in territory)
territory_efficiency = sales_by_territory / df.groupby('TERRITORY')['COUNTRY'].nunique()
print("Territory Efficiency (Average Sales per Country in Territory):")
for territory, efficiency in territory_efficiency.sort_values(ascending=False).items():
    print(f"{territory}: ${efficiency:,.2f}")
print()

# 6. Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Geographic Sales Analysis', fontsize=16, fontweight='bold')

# Sales by Country (Top 10)
top_10_countries = sales_by_country.head(10)
axes[0, 0].bar(top_10_countries.index, top_10_countries.values, color='skyblue')
axes[0, 0].set_title('Top 10 Countries by Sales')
axes[0, 0].set_ylabel('Sales ($)')
axes[0, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(top_10_countries.values):
    axes[0, 0].text(i, v + max(top_10_countries.values)*0.01, '.0f', ha='center', va='bottom')

# Sales by Territory
axes[0, 1].bar(sales_by_territory.index, sales_by_territory.values, color='lightgreen')
axes[0, 1].set_title('Sales by Territory')
axes[0, 1].set_ylabel('Sales ($)')
axes[0, 1].tick_params(axis='x', rotation=45)

# Country Sales Distribution (Pie Chart)
top_5_countries = sales_by_country.head(5)
others = sales_by_country[5:].sum()
pie_data = pd.concat([top_5_countries, pd.Series({'Others': others})])
axes[1, 0].pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
axes[1, 0].set_title('Sales Distribution by Country')

# Territory Performance Comparison
territory_colors = ['gold' if x == top_territory else 'lightcoral' for x in sales_by_territory.index]
axes[1, 1].bar(sales_by_territory.index, sales_by_territory.values, color=territory_colors)
axes[1, 1].set_title('Territory Performance (Gold = Best)')
axes[1, 1].set_ylabel('Sales ($)')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/workspaces/Sales-trend-and-forecast-reporting-data/geographic_sales_charts.png', dpi=300, bbox_inches='tight')
plt.close()

print("Charts saved to geographic_sales_charts.png")

# 7. Geographic Trends Over Time
print("\n=== GEOGRAPHIC TRENDS OVER TIME ===")
df['Year'] = pd.to_datetime(df['ORDERDATE']).dt.year

# Top countries yearly performance
top_3_countries_list = sales_by_country.head(3).index.tolist()
print(f"Yearly Performance for Top 3 Countries ({', '.join(top_3_countries_list)}):")

yearly_country_sales = df.groupby(['Year', 'COUNTRY'])['SALES'].sum().unstack().fillna(0)
for country in top_3_countries_list:
    if country in yearly_country_sales.columns:
        country_sales = yearly_country_sales[country]
        print(f"\n{country}:")
        for year, sales in country_sales.items():
            print(f"  {year}: ${sales:,.2f}")
        if len(country_sales) > 1:
            growth = ((country_sales.iloc[-1] - country_sales.iloc[0]) / country_sales.iloc[0] * 100)
            print(f"  Change: {growth:+.1f}%")

# Territory trends
print(f"\nYearly Performance by Territory:")
yearly_territory_sales = df.groupby(['Year', 'TERRITORY'])['SALES'].sum().unstack().fillna(0)
print(yearly_territory_sales.to_string())