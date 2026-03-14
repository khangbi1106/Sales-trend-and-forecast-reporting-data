import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned sales data
df = pd.read_csv('/workspaces/Sales-trend-and-forecast-reporting-data/sales_data_cleaned.csv')

print("=== DEAL SIZE ANALYSIS ===\n")

# 1. Deal Size Distribution
print("1. DEAL SIZE DISTRIBUTION")
deal_size_counts = df['DEALSIZE'].value_counts().sort_index()
deal_size_revenue = df.groupby('DEALSIZE')['SALES'].sum().sort_index()

print("Number of Deals by Size:")
for size, count in deal_size_counts.items():
    percentage = (count / deal_size_counts.sum() * 100)
    print(f"{size}: {count:,} deals ({percentage:.1f}%)")
print()

print("Revenue by Deal Size:")
for size, revenue in deal_size_revenue.items():
    percentage = (revenue / deal_size_revenue.sum() * 100)
    print(f"{size}: ${revenue:,.2f} ({percentage:.1f}%)")
print()

# 2. Deal Size Performance Metrics
print("2. DEAL SIZE PERFORMANCE METRICS")
deal_size_summary = pd.DataFrame({
    'Number of Deals': deal_size_counts,
    'Total Revenue': deal_size_revenue,
    'Average Deal Value': deal_size_revenue / deal_size_counts,
    'Revenue Percentage': (deal_size_revenue / deal_size_revenue.sum() * 100).round(1),
    'Deal Percentage': (deal_size_counts / deal_size_counts.sum() * 100).round(1)
})
deal_size_summary = deal_size_summary.sort_values('Total Revenue', ascending=False)
print(deal_size_summary.to_string())
print()

# 3. Deal Size Efficiency Analysis
print("3. DEAL SIZE EFFICIENCY ANALYSIS")
total_deals = deal_size_counts.sum()
total_revenue = deal_size_revenue.sum()

print(f"Total Deals: {total_deals:,}")
print(f"Total Revenue: ${total_revenue:,.2f}")
print(f"Average Deal Value (Overall): ${total_revenue/total_deals:,.2f}")
print()

# Revenue concentration
top_deal_size = deal_size_revenue.idxmax()
top_revenue_pct = (deal_size_revenue.max() / total_revenue * 100)
top_deal_pct = (deal_size_counts[top_deal_size] / total_deals * 100)

print(f"Largest Revenue Contributor: {top_deal_size} deals")
print(f"Revenue from {top_deal_size} Deals: {top_revenue_pct:.1f}%")
print(f"{top_deal_size} Deals as % of Total: {top_deal_pct:.1f}%")
print()

# Business dependency analysis
if top_revenue_pct > 50:
    dependency = "HIGHLY dependent"
elif top_revenue_pct > 30:
    dependency = "Moderately dependent"
else:
    dependency = "Not heavily dependent"

print(f"Business Dependency on {top_deal_size} Deals: {dependency}")
print(f"Revenue Concentration Risk: {'High' if top_revenue_pct > 40 else 'Moderate' if top_revenue_pct > 20 else 'Low'}")
print()

# 4. Deal Size Trends Over Time
print("4. DEAL SIZE TRENDS OVER TIME")
df['Year'] = pd.to_datetime(df['ORDERDATE']).dt.year

yearly_deal_sizes = df.groupby(['Year', 'DEALSIZE']).agg({
    'SALES': 'sum',
    'ORDERNUMBER': 'count'
}).unstack().fillna(0)

print("Yearly Deal Size Performance:")
print("Revenue by Year and Deal Size:")
revenue_table = yearly_deal_sizes['SALES'].round(2)
print(revenue_table.to_string())
print()

print("Number of Deals by Year and Deal Size:")
deals_table = yearly_deal_sizes['ORDERNUMBER']
print(deals_table.to_string())
print()

# Growth analysis for each deal size
print("Deal Size Growth Analysis (2003-2005):")
for deal_size in deal_size_counts.index:
    if deal_size in yearly_deal_sizes['SALES'].columns:
        size_revenue = yearly_deal_sizes['SALES'][deal_size]
        if len(size_revenue) > 1 and size_revenue.sum() > 0:
            growth = ((size_revenue.iloc[-1] - size_revenue.iloc[0]) / size_revenue.iloc[0] * 100)
            print(f"{deal_size}: {growth:+.1f}% change")
print()

# 5. Deal Size by Product Line
print("5. DEAL SIZE BY PRODUCT LINE")
product_deal_analysis = df.groupby(['PRODUCTLINE', 'DEALSIZE']).agg({
    'SALES': 'sum',
    'ORDERNUMBER': 'count'
}).round(2)

print("Revenue by Product Line and Deal Size:")
revenue_by_product = product_deal_analysis['SALES'].unstack().fillna(0)
print(revenue_by_product.to_string())
print()

# 6. Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Deal Size Analysis Dashboard', fontsize=16, fontweight='bold')

# Deal Size Distribution (Count)
axes[0, 0].bar(deal_size_counts.index, deal_size_counts.values, color=['green', 'orange', 'red'])
axes[0, 0].set_title('Number of Deals by Size')
axes[0, 0].set_ylabel('Number of Deals')
for i, v in enumerate(deal_size_counts.values):
    axes[0, 0].text(i, v + max(deal_size_counts.values)*0.01, f'{v:,}', ha='center', va='bottom')

# Deal Size Revenue
axes[0, 1].bar(deal_size_revenue.index, deal_size_revenue.values, color=['green', 'orange', 'red'])
axes[0, 1].set_title('Revenue by Deal Size')
axes[0, 1].set_ylabel('Revenue ($)')
for i, v in enumerate(deal_size_revenue.values):
    axes[0, 1].text(i, v + max(deal_size_revenue.values)*0.01, '.0f', ha='center', va='bottom')

# Average Deal Value by Size
avg_deal_value = deal_size_revenue / deal_size_counts
axes[1, 0].bar(avg_deal_value.index, avg_deal_value.values, color=['green', 'orange', 'red'])
axes[1, 0].set_title('Average Deal Value by Size')
axes[1, 0].set_ylabel('Average Value ($)')
for i, v in enumerate(avg_deal_value.values):
    axes[1, 0].text(i, v + max(avg_deal_value.values)*0.01, '.0f', ha='center', va='bottom')

# Deal Size Trend Over Time (Revenue)
yearly_revenue_plot = yearly_deal_sizes['SALES'].T
yearly_revenue_plot.plot(kind='bar', ax=axes[1, 1], width=0.8)
axes[1, 1].set_title('Deal Size Revenue Trends (2003-2005)')
axes[1, 1].set_ylabel('Revenue ($)')
axes[1, 1].legend(title='Deal Size', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('/workspaces/Sales-trend-and-forecast-reporting-data/deal_size_analysis_charts.png', dpi=300, bbox_inches='tight')
plt.close()

print("Charts saved to deal_size_analysis_charts.png")

# 7. Strategic Insights
print("\n=== STRATEGIC INSIGHTS ===")

# Revenue vs Volume analysis
revenue_per_deal = deal_size_revenue / deal_size_counts
print("Revenue Efficiency (Revenue per Deal):")
for size in revenue_per_deal.sort_values(ascending=False).index:
    print(f"{size}: ${revenue_per_deal[size]:,.2f}")

print()

# Business implications
large_deal_pct = deal_size_revenue.get('Large', 0) / total_revenue * 100
medium_deal_pct = deal_size_revenue.get('Medium', 0) / total_revenue * 100
small_deal_pct = deal_size_revenue.get('Small', 0) / total_revenue * 100

print("Business Implications:")
if large_deal_pct > 60:
    print("- Business heavily dependent on large deals - focus on maintaining key accounts")
elif medium_deal_pct > 60:
    print("- Business driven by medium-sized deals - stable but growth potential in large deals")
else:
    print("- Business has diverse deal sizes - good risk distribution")

print(f"- Large deals: {large_deal_pct:.1f}% of revenue")
print(f"- Medium deals: {medium_deal_pct:.1f}% of revenue")
print(f"- Small deals: {small_deal_pct:.1f}% of revenue")

# Recommendations
print("\nRecommendations:")
if top_revenue_pct > 40:
    print("- Reduce dependency on single deal size through diversification")
if deal_size_counts['Small'] > deal_size_counts.sum() * 0.6:
    print("- Consider strategies to increase average deal size")
if deal_size_revenue['Large'] < deal_size_revenue.sum() * 0.3:
    print("- Focus on converting medium deals to large deals")