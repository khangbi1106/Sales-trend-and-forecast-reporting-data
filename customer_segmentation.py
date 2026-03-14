import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the cleaned sales data
df = pd.read_csv('/workspaces/Sales-trend-and-forecast-reporting-data/sales_data_cleaned.csv')

print("=== CUSTOMER SEGMENTATION ANALYSIS ===\n")

# 1. Prepare customer data for clustering
print("1. CUSTOMER DATA PREPARATION")
customer_data = df.groupby('CUSTOMERNAME').agg({
    'SALES': 'sum',
    'QUANTITYORDERED': 'sum',
    'ORDERNUMBER': 'nunique'  # Number of orders
}).reset_index()

customer_data.columns = ['Customer', 'Total_Sales', 'Total_Quantity', 'Num_Orders']
print(f"Customer dataset shape: {customer_data.shape}")
print(f"Total customers: {len(customer_data)}")
print()

# 2. Feature scaling for clustering
print("2. FEATURE SCALING")
features = ['Total_Sales', 'Total_Quantity', 'Num_Orders']
scaler = StandardScaler()
customer_scaled = scaler.fit_transform(customer_data[features])

print("Scaled features summary:")
for i, feature in enumerate(features):
    print(f"{feature}: Mean={customer_scaled[:, i].mean():.3f}, Std={customer_scaled[:, i].std():.3f}")
print()

# 3. Determine optimal number of clusters using Elbow method and Silhouette score
print("3. OPTIMAL CLUSTER DETERMINATION")
inertias = []
silhouette_scores = []
k_range = range(2, 8)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(customer_scaled)
    inertias.append(kmeans.inertia_)

    silhouette_scores.append(silhouette_score(customer_scaled, kmeans.labels_))

# Plot Elbow method
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.grid(True)

plt.tight_layout()
plt.savefig('/workspaces/Sales-trend-and-forecast-reporting-data/cluster_optimization.png', dpi=300, bbox_inches='tight')
plt.close()

print("Cluster optimization plots saved to cluster_optimization.png")

# Choose optimal k (based on elbow method and silhouette score)
optimal_k = 4  # Based on typical business segmentation
print(f"Selected optimal k: {optimal_k}")
print()

# 4. Perform K-means clustering
print("4. K-MEANS CLUSTERING")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customer_data['Cluster'] = kmeans.fit_predict(customer_scaled)

print("Clustering completed!")
print(f"Cluster distribution: {customer_data['Cluster'].value_counts().sort_index()}")
print()

# 5. Analyze cluster characteristics
print("5. CLUSTER ANALYSIS")
cluster_summary = customer_data.groupby('Cluster').agg({
    'Total_Sales': ['mean', 'min', 'max', 'sum'],
    'Total_Quantity': ['mean', 'min', 'max'],
    'Num_Orders': ['mean', 'min', 'max'],
    'Customer': 'count'
}).round(2)

print("Cluster Summary:")
print(cluster_summary.to_string())
print()

# 6. Interpret clusters
print("6. CLUSTER INTERPRETATION")
cluster_stats = customer_data.groupby('Cluster').agg({
    'Total_Sales': 'mean',
    'Total_Quantity': 'mean',
    'Num_Orders': 'mean',
    'Customer': 'count'
}).round(2)

# Sort clusters by sales for interpretation
cluster_stats = cluster_stats.sort_values('Total_Sales', ascending=False)

cluster_names = {}
for i, (cluster, row) in enumerate(cluster_stats.iterrows()):
    if i == 0:
        cluster_names[cluster] = "High-Value Customers"
    elif i == 1:
        cluster_names[cluster] = "Medium-Value Customers"
    elif i == 2:
        cluster_names[cluster] = "Low-Value Customers"
    else:
        cluster_names[cluster] = "Entry-Level Customers"

print("Cluster Names and Characteristics:")
for cluster, name in cluster_names.items():
    stats = cluster_stats.loc[cluster]
    print(f"\n{name} (Cluster {cluster}):")
    print(f"  Customers: {stats['Customer']}")
    print(f"  Avg Sales: ${stats['Total_Sales']:,.2f}")
    print(f"  Avg Quantity: {stats['Total_Quantity']:.0f}")
    print(f"  Avg Orders: {stats['Num_Orders']:.1f}")

print()

# 7. Create cluster visualization
print("7. CLUSTER VISUALIZATION")
plt.figure(figsize=(15, 10))

# Scatter plot: Sales vs Quantity by cluster
plt.subplot(2, 2, 1)
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
for cluster in range(optimal_k):
    cluster_data = customer_data[customer_data['Cluster'] == cluster]
    plt.scatter(cluster_data['Total_Sales'], cluster_data['Total_Quantity'],
               c=colors[cluster], label=f'Cluster {cluster} ({cluster_names[cluster]})',
               alpha=0.6, s=50)

plt.xlabel('Total Sales ($)')
plt.ylabel('Total Quantity Ordered')
plt.title('Customer Clusters: Sales vs Quantity')
plt.legend()
plt.grid(True)

# Box plot: Sales by cluster
plt.subplot(2, 2, 2)
cluster_sales = [customer_data[customer_data['Cluster'] == i]['Total_Sales'] for i in range(optimal_k)]
plt.boxplot(cluster_sales, labels=[f'Cluster {i}\n({cluster_names[i]})' for i in range(optimal_k)])
plt.ylabel('Total Sales ($)')
plt.title('Sales Distribution by Cluster')
plt.xticks(rotation=45)

# Box plot: Orders by cluster
plt.subplot(2, 2, 3)
cluster_orders = [customer_data[customer_data['Cluster'] == i]['Num_Orders'] for i in range(optimal_k)]
plt.boxplot(cluster_orders, labels=[f'Cluster {i}\n({cluster_names[i]})' for i in range(optimal_k)])
plt.ylabel('Number of Orders')
plt.title('Order Frequency by Cluster')
plt.xticks(rotation=45)

# Cluster sizes
plt.subplot(2, 2, 4)
cluster_sizes = customer_data['Cluster'].value_counts().sort_index()
bars = plt.bar(range(optimal_k), cluster_sizes.values, color=colors[:optimal_k])
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.title('Cluster Sizes')
plt.xticks(range(optimal_k), [f'Cluster {i}\n({cluster_names[i]})' for i in range(optimal_k)], rotation=45)

# Add value labels on bars
for bar, size in zip(bars, cluster_sizes.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{size}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('/workspaces/Sales-trend-and-forecast-reporting-data/customer_clusters.png', dpi=300, bbox_inches='tight')
plt.close()

print("Customer cluster visualizations saved to customer_clusters.png")

# 8. Export cluster results
customer_data['Cluster_Name'] = customer_data['Cluster'].map(cluster_names)
customer_data.to_csv('/workspaces/Sales-trend-and-forecast-reporting-data/customer_segments.csv', index=False)

print("Customer segmentation results exported to customer_segments.csv")

# 9. Strategic recommendations
print("\n=== STRATEGIC RECOMMENDATIONS ===")

for cluster, name in cluster_names.items():
    stats = cluster_stats.loc[cluster]
    customers = stats['Customer']

    print(f"\n{name} (Cluster {cluster}) - {customers} customers:")

    if 'High-Value' in name:
        print("  - Priority: VIP treatment, dedicated account management")
        print("  - Strategy: Retention focus, upselling opportunities")
    elif 'Medium-Value' in name:
        print("  - Priority: Growth potential, loyalty programs")
        print("  - Strategy: Increase order frequency and average order value")
    elif 'Low-Value' in name:
        print("  - Priority: Cost management, profitability assessment")
        print("  - Strategy: Volume incentives or minimum order requirements")
    else:  # Entry-Level
        print("  - Priority: Nurturing, conversion to higher tiers")
        print("  - Strategy: Educational marketing, product introductions")

print(f"\nOverall Strategy:")
print(f"- Focus retention efforts on top {len(customer_data[customer_data['Cluster'] == cluster_stats.index[0]])} high-value customers")
print(f"- Develop growth strategies for {len(customer_data[customer_data['Cluster'] == cluster_stats.index[1]])} medium-value customers")
print(f"- Optimize operations for {len(customer_data[customer_data['Cluster'].isin(cluster_stats.index[2:])])} lower-value customers")