def customer_segmentation(df):
    # Select features for clustering
    features_for_clustering = ['Age', 'Total Spend', 'Items Purchased', 
                              'Average Rating', 'Days Since Last Purchase']
    
    X_cluster = df[features_for_clustering].copy()
    
    # Standardize the features
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    
    # Find optimal number of clusters using elbow method
    inertias = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_cluster_scaled)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()
    
    # Perform clustering with optimal k (let's use 4)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['Customer_Segment'] = kmeans.fit_predict(X_cluster_scaled)
    
    # Analyze segments
    print(f"\nCustomer segments created with k={optimal_k}")
    print("\nSegment Analysis:")
    segment_analysis = df.groupby('Customer_Segment')[features_for_clustering].mean()
    print(segment_analysis)
    
    # Visualize segments
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Customer Segments Analysis', fontsize=14, fontweight='bold')
    
    # Segment distribution
    segment_counts = df['Customer_Segment'].value_counts().sort_index()
    axes[0,0].bar(segment_counts.index, segment_counts.values, color='lightblue')
    axes[0,0].set_title('Customer Segment Distribution')
    axes[0,0].set_xlabel('Segment')
    axes[0,0].set_ylabel('Count')
    
    # Spend vs Items by segment
    for segment in df['Customer_Segment'].unique():
        segment_data = df[df['Customer_Segment'] == segment]
        axes[0,1].scatter(segment_data['Items Purchased'], segment_data['Total Spend'], 
                         label=f'Segment {segment}', alpha=0.7)
    axes[0,1].set_title('Total Spend vs Items Purchased by Segment')
    axes[0,1].set_xlabel('Items Purchased')
    axes[0,1].set_ylabel('Total Spend')
    axes[0,1].legend()
    
    # Age distribution by segment
    df.boxplot(column='Age', by='Customer_Segment', ax=axes[1,0])
    axes[1,0].set_title('Age Distribution by Segment')
    axes[1,0].set_xlabel('Customer Segment')
    
    # Rating by segment
    df.boxplot(column='Average Rating', by='Customer_Segment', ax=axes[1,1])
    axes[1,1].set_title('Average Rating by Segment')
    axes[1,1].set_xlabel('Customer Segment')
    
    plt.tight_layout()
    plt.show()
    
    return df

customer_segmentation(df_clean)