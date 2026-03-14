import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned sales data
df = pd.read_csv('/workspaces/Sales-trend-and-forecast-reporting-data/sales_data_cleaned.csv')

print("=== PRODUCT PERFORMANCE ANALYSIS ===\n")

# 1. Sales by Product Line
print("1. SALES BY PRODUCT LINE")
sales_by_productline = df.groupby('PRODUCTLINE')['SALES'].sum().sort_values(ascending=False)
for product, sales in sales_by_productline.items():
    print(f"{product}: ${sales:,.2f}")
print()

# 2. Quantity Ordered by Product Line
print("2. QUANTITY ORDERED BY PRODUCT LINE")
quantity_by_productline = df.groupby('PRODUCTLINE')['QUANTITYORDERED'].sum().sort_values(ascending=False)
for product, quantity in quantity_by_productline.items():
    print(f"{product}: {quantity:,} units")
print()

# 3. Average Sales per Unit by Product Line
print("3. AVERAGE SALES PER UNIT BY PRODUCT LINE")
avg_price_by_productline = (sales_by_productline / quantity_by_productline).sort_values(ascending=False)
for product, avg_price in avg_price_by_productline.items():
    print(f"{product}: ${avg_price:.2f}")
print()

# 4. Top Products by Sales
print("4. TOP PRODUCTS BY SALES (PRODUCT CODE)")
top_products_sales = df.groupby('PRODUCTCODE')['SALES'].sum().sort_values(ascending=False).head(10)
print("Top 10 Products by Sales:")
for product_code, sales in top_products_sales.items():
    # Get product line for context
    product_line = df[df['PRODUCTCODE'] == product_code]['PRODUCTLINE'].iloc[0]
    print(f"{product_code} ({product_line}): ${sales:,.2f}")
print()

# 5. Top Products by Quantity
print("5. TOP PRODUCTS BY QUANTITY ORDERED")
top_products_quantity = df.groupby('PRODUCTCODE')['QUANTITYORDERED'].sum().sort_values(ascending=False).head(10)
print("Top 10 Products by Quantity:")
for product_code, quantity in top_products_quantity.items():
    product_line = df[df['PRODUCTCODE'] == product_code]['PRODUCTLINE'].iloc[0]
    print(f"{product_code} ({product_line}): {quantity:,} units")
print()

# 6. Product Line Performance Summary
print("6. PRODUCT LINE PERFORMANCE SUMMARY")
performance_summary = pd.DataFrame({
    'Total Sales': sales_by_productline,
    'Total Quantity': quantity_by_productline,
    'Average Price per Unit': avg_price_by_productline,
    'Percentage of Total Sales': (sales_by_productline / sales_by_productline.sum() * 100).round(1)
})
performance_summary['Number of Orders'] = df.groupby('PRODUCTLINE')['ORDERNUMBER'].count()
performance_summary = performance_summary.sort_values('Total Sales', ascending=False)

print(performance_summary.to_string())
print()

# 7. Key Insights
print("7. KEY INSIGHTS")
total_sales = sales_by_productline.sum()
total_quantity = quantity_by_productline.sum()

print(f"Total Sales Across All Products: ${total_sales:,.2f}")
print(f"Total Quantity Across All Products: {total_quantity:,} units")
print(f"Number of Unique Products: {df['PRODUCTCODE'].nunique()}")
print(f"Number of Product Lines: {df['PRODUCTLINE'].nunique()}")
print()

# Best and worst performers
best_sales = sales_by_productline.idxmax()
worst_sales = sales_by_productline.idxmin()
best_quantity = quantity_by_productline.idxmax()
best_avg_price = avg_price_by_productline.idxmax()

print(f"Best Selling Product Line (by sales): {best_sales} (${sales_by_productline.max():,.2f})")
print(f"Worst Selling Product Line (by sales): {worst_sales} (${sales_by_productline.min():,.2f})")
print(f"Most Ordered Product Line (by quantity): {best_quantity} ({quantity_by_productline.max():,} units)")
print(f"Highest Average Price Product Line: {best_avg_price} (${avg_price_by_productline.max():.2f})")
print()

# Market concentration
top_3_sales_pct = (sales_by_productline.head(3).sum() / total_sales * 100)
print(f"Top 3 Product Lines Account for: {top_3_sales_pct:.1f}% of total sales")
print()

# 8. Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Product Performance Analysis', fontsize=16, fontweight='bold')

# Sales by Product Line
axes[0, 0].bar(sales_by_productline.index, sales_by_productline.values, color='skyblue')
axes[0, 0].set_title('Sales by Product Line')
axes[0, 0].set_ylabel('Sales ($)')
axes[0, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(sales_by_productline.values):
    axes[0, 0].text(i, v + max(sales_by_productline.values)*0.01, '.0f', ha='center', va='bottom')

# Quantity by Product Line
axes[0, 1].bar(quantity_by_productline.index, quantity_by_productline.values, color='lightgreen')
axes[0, 1].set_title('Quantity Ordered by Product Line')
axes[0, 1].set_ylabel('Quantity (units)')
axes[0, 1].tick_params(axis='x', rotation=45)

# Average Price by Product Line
axes[1, 0].bar(avg_price_by_productline.index, avg_price_by_productline.values, color='orange')
axes[1, 0].set_title('Average Price per Unit by Product Line')
axes[1, 0].set_ylabel('Average Price ($)')
axes[1, 0].tick_params(axis='x', rotation=45)

# Top 10 Products by Sales
axes[1, 1].bar(range(len(top_products_sales)), top_products_sales.values, color='purple')
axes[1, 1].set_title('Top 10 Products by Sales')
axes[1, 1].set_ylabel('Sales ($)')
axes[1, 1].set_xticks(range(len(top_products_sales)))
axes[1, 1].set_xticklabels([code.split('_')[1] if '_' in code else code for code in top_products_sales.index], rotation=45)

plt.tight_layout()
plt.savefig('/workspaces/Sales-trend-and-forecast-reporting-data/product_performance_charts.png', dpi=300, bbox_inches='tight')
plt.close()

print("Charts saved to product_performance_charts.png")

# Additional analysis: Product line trends over time
print("\n=== PRODUCT LINE TRENDS OVER TIME ===")
df['Year'] = pd.to_datetime(df['ORDERDATE']).dt.year
yearly_product_sales = df.groupby(['Year', 'PRODUCTLINE'])['SALES'].sum().unstack().fillna(0)

print("Yearly Sales by Product Line:")
print(yearly_product_sales.to_string())
print()

# Growth analysis for top product lines
top_lines = sales_by_productline.head(3).index.tolist()
print(f"Growth Analysis for Top 3 Product Lines ({', '.join(top_lines)}):")
for line in top_lines:
    if line in yearly_product_sales.columns:
        line_sales = yearly_product_sales[line]
        if len(line_sales) > 1:
            growth = ((line_sales.iloc[-1] - line_sales.iloc[0]) / line_sales.iloc[0] * 100)
            print(f"{line}: {growth:+.1f}% change from {line_sales.index[0]} to {line_sales.index[-1]}")