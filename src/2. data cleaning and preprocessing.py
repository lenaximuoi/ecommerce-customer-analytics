# Create a copy to avoid modifying original
df_clean = df.copy()

# Fill missing values in 'Satisfaction Level' with the mode
df_clean["Satisfaction Level"] = df_clean["Satisfaction Level"].fillna(df_clean["Satisfaction Level"].mode()[0])

# Create additional features
print("Creating derived features...")

# Spending efficiency (spend per item)
df_clean['Spend Per Item'] = df_clean['Total Spend'] / (df_clean['Items Purchased'] + 1)  # +1 to avoid division by zero

# Customer value segments based on spend and frequency
df_clean['High Spender'] = (df_clean['Total Spend'] > df_clean['Total Spend'].quantile(0.75)).astype(int)
df_clean['Frequent Buyer'] = (df_clean['Items Purchased'] > df_clean['Items Purchased'].quantile(0.75)).astype(int)

# Recency categories
df_clean['Recency Category'] = pd.cut(df_clean['Days Since Last Purchase'],
                                      bins=[-1, 7, 30, 90, float('inf')],
                                      labels=['Very Recent', 'Recent', 'Moderate', 'Long Ago'])

# Age groups
df_clean['Age Group'] = pd.cut(df_clean['Age'],
                               bins=[0, 25, 35, 50, 100],
                               labels=['Young', 'Adult', 'Middle-aged', 'Senior'])

print(f"Dataset shape after preprocessing: {df_clean.shape}")
print("New features created: Spend Per Item, High Spender, Frequent Buyer, Recency Category, Age Group")

df_clean.head()