import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned sales data
df = pd.read_csv('/workspaces/Sales-trend-and-forecast-reporting-data/sales_data_cleaned.csv')

print("=== CUSTOMER ANALYSIS ===\n")

# 1. Top 10 Customers by Revenue
print("1. TOP 10 CUSTOMERS BY REVENUE")
customer_sales = df.groupby('CUSTOMERNAME')['SALES'].sum().sort_values(ascending=False)
top_10_customers = customer_sales.head(10)
for i, (customer, sales) in enumerate(top_10_customers.items(), 1):
    percentage = (sales / customer_sales.sum() * 100)
    print(f"{i}. {customer}: ${sales:,.2f} ({percentage:.1f}%)")
print()

# 2. Customer Purchase Analysis
print("2. CUSTOMER PURCHASE ANALYSIS")
total_customers = df['CUSTOMERNAME'].nunique()
total_sales = df['SALES'].sum()
avg_purchase_per_customer = total_sales / total_customers

print(f"Total Unique Customers: {total_customers}")
print(f"Total Sales: ${total_sales:,.2f}")
print(f"Average Purchase per Customer: ${avg_purchase_per_customer:,.2f}")
print()

# 3. Customer Order Frequency
print("3. CUSTOMER ORDER FREQUENCY")
orders_per_customer = df.groupby('CUSTOMERNAME')['ORDERNUMBER'].nunique().sort_values(ascending=False)
avg_orders_per_customer = orders_per_customer.mean()

print(f"Average Orders per Customer: {avg_orders_per_customer:.1f}")
print(f"Most Frequent Customer: {orders_per_customer.idxmax()} ({orders_per_customer.max()} orders)")
print(f"Customers with Single Order: {(orders_per_customer == 1).sum()} ({(orders_per_customer == 1).sum()/total_customers*100:.1f}%)")
print()

# 4. High-Value Customer Analysis
print("4. HIGH-VALUE CUSTOMER ANALYSIS")
# Define high-value customers as those in top 10% by sales
sales_threshold = customer_sales.quantile(0.9)
high_value_customers = customer_sales[customer_sales >= sales_threshold]

print(f"High-Value Customers (Top 10%): {len(high_value_customers)} customers")
print(f"Sales Threshold: ${sales_threshold:,.2f}")
print(f"High-Value Customers Account for: {high_value_customers.sum()/total_sales*100:.1f}% of total sales")
print()

# 5. Customer Segmentation
print("5. CUSTOMER SEGMENTATION")
# Segment customers by total purchase amount
def segment_customer(sales):
    if sales >= 100000:
        return 'Platinum'
    elif sales >= 50000:
        return 'Gold'
    elif sales >= 25000:
        return 'Silver'
    else:
        return 'Bronze'

customer_segments = customer_sales.apply(segment_customer)
segment_summary = customer_segments.value_counts().sort_index()

print("Customer Segmentation by Total Purchase:")
for segment, count in segment_summary.items():
    segment_sales = customer_sales[customer_segments == segment].sum()
    percentage = (segment_sales / total_sales * 100)
    print(f"{segment}: {count} customers (${segment_sales:,.2f} - {percentage:.1f}%)")
print()

# 6. Top Customers Detailed Analysis
print("6. TOP CUSTOMERS DETAILED ANALYSIS")
print("Top 5 Customers - Detailed Breakdown:")
for i, customer in enumerate(top_10_customers.index[:5], 1):
    customer_data = df[df['CUSTOMERNAME'] == customer]

    total_orders = customer_data['ORDERNUMBER'].nunique()
    total_quantity = customer_data['QUANTITYORDERED'].sum()
    avg_order_value = customer_data['SALES'].mean()
    favorite_product = customer_data.groupby('PRODUCTLINE')['SALES'].sum().idxmax()

    print(f"\n{i}. {customer}")
    print(f"   Total Sales: ${top_10_customers[customer]:,.2f}")
    print(f"   Total Orders: {total_orders}")
    print(f"   Total Quantity: {total_quantity:,} units")
    print(f"   Average Order Value: ${avg_order_value:,.2f}")
    print(f"   Favorite Product Line: {favorite_product}")
print()

# 7. Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Customer Analysis Dashboard', fontsize=16, fontweight='bold')

# Top 10 Customers by Revenue
axes[0, 0].barh(range(len(top_10_customers)), top_10_customers.values)
axes[0, 0].set_yticks(range(len(top_10_customers)))
axes[0, 0].set_yticklabels([name[:20] + '...' if len(name) > 20 else name for name in top_10_customers.index])
axes[0, 0].set_title('Top 10 Customers by Revenue')
axes[0, 0].set_xlabel('Sales ($)')

# Customer Segmentation Pie Chart
segment_colors = {'Platinum': 'gold', 'Gold': 'silver', 'Silver': '#CD7F32', 'Bronze': '#B87333'}
axes[0, 1].pie(segment_summary.values, labels=segment_summary.index, autopct='%1.1f%%',
               colors=[segment_colors.get(seg, 'gray') for seg in segment_summary.index])
axes[0, 1].set_title('Customer Segmentation')

# Customer Sales Distribution
axes[1, 0].hist(customer_sales.values, bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(avg_purchase_per_customer, color='red', linestyle='--', label=f'Average: ${avg_purchase_per_customer:,.0f}')
axes[1, 0].axvline(sales_threshold, color='gold', linestyle='--', label=f'Top 10%: ${sales_threshold:,.0f}')
axes[1, 0].set_title('Customer Sales Distribution')
axes[1, 0].set_xlabel('Total Sales ($)')
axes[1, 0].set_ylabel('Number of Customers')
axes[1, 0].legend()

# Orders per Customer Distribution
axes[1, 1].hist(orders_per_customer.values, bins=range(1, orders_per_customer.max()+2), edgecolor='black', alpha=0.7)
axes[1, 1].axvline(avg_orders_per_customer, color='red', linestyle='--', label=f'Average: {avg_orders_per_customer:.1f}')
axes[1, 1].set_title('Orders per Customer Distribution')
axes[1, 1].set_xlabel('Number of Orders')
axes[1, 1].set_ylabel('Number of Customers')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('/workspaces/Sales-trend-and-forecast-reporting-data/customer_analysis_charts.png', dpi=300, bbox_inches='tight')
plt.close()

print("Charts saved to customer_analysis_charts.png")

# 8. Customer Trends Over Time
print("\n=== CUSTOMER TRENDS OVER TIME ===")
df['Year'] = pd.to_datetime(df['ORDERDATE']).dt.year

# Top customers yearly performance
top_3_customers_list = top_10_customers.index[:3].tolist()
print(f"Yearly Performance for Top 3 Customers ({', '.join([c[:15] + '...' if len(c) > 15 else c for c in top_3_customers_list])}):")

yearly_customer_sales = df.groupby(['Year', 'CUSTOMERNAME'])['SALES'].sum().unstack().fillna(0)
for customer in top_3_customers_list:
    if customer in yearly_customer_sales.columns:
        customer_sales = yearly_customer_sales[customer]
        print(f"\n{customer}:")
        for year, sales in customer_sales.items():
            print(f"  {year}: ${sales:,.2f}")
        if len(customer_sales) > 1 and customer_sales.sum() > 0:
            growth = ((customer_sales.iloc[-1] - customer_sales.iloc[0]) / customer_sales.iloc[0] * 100)
            print(f"  Change: {growth:+.1f}%")

# New vs Returning Customers Analysis
print(f"\nCustomer Acquisition Trends:")
customers_by_year = df.groupby('Year')['CUSTOMERNAME'].nunique()
print("Unique Customers by Year:")
for year, count in customers_by_year.items():
    print(f"  {year}: {count} customers")