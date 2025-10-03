def perform_eda(df):    
    # Set up the plotting area
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('E-commerce Customer Behavior Analysis', fontsize=16, fontweight='bold')
    
    # 1. Age distribution
    axes[0,0].hist(df['Age'], bins=20, alpha=0.7, color='skyblue')
    axes[0,0].set_title('Age Distribution')
    axes[0,0].set_xlabel('Age')
    axes[0,0].set_ylabel('Frequency')
    
    # 2. Total spend distribution
    axes[0,1].hist(df['Total Spend'], bins=20, alpha=0.7, color='lightgreen')
    axes[0,1].set_title('Total Spend Distribution')
    axes[0,1].set_xlabel('Total Spend')
    axes[0,1].set_ylabel('Frequency')
    
    # 3. Gender distribution
    gender_counts = df['Gender'].value_counts()
    axes[0,2].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', colors=['lightpink', 'lightblue'])
    axes[0,2].set_title('Gender Distribution')
    
    # 4. Membership type distribution
    membership_counts = df['Membership Type'].value_counts()
    axes[1,0].bar(membership_counts.index, membership_counts.values, color='coral')
    axes[1,0].set_title('Membership Type Distribution')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 5. Satisfaction level distribution
    satisfaction_counts = df['Satisfaction Level'].value_counts()
    axes[1,1].bar(satisfaction_counts.index, satisfaction_counts.values, color='gold')
    axes[1,1].set_title('Satisfaction Level Distribution')
    
    # 6. Spend vs Items Purchased
    axes[1,2].scatter(df['Items Purchased'], df['Total Spend'], alpha=0.6, color='purple')
    axes[1,2].set_title('Total Spend vs Items Purchased')
    axes[1,2].set_xlabel('Items Purchased')
    axes[1,2].set_ylabel('Total Spend')
    
    # 7. Average Rating distribution
    axes[2,0].hist(df['Average Rating'], bins=15, alpha=0.7, color='orange')
    axes[2,0].set_title('Average Rating Distribution')
    axes[2,0].set_xlabel('Average Rating')
    axes[2,0].set_ylabel('Frequency')
    
    # 8. Days since last purchase
    axes[2,1].hist(df['Days Since Last Purchase'], bins=20, alpha=0.7, color='pink')
    axes[2,1].set_title('Days Since Last Purchase')
    axes[2,1].set_xlabel('Days')
    axes[2,1].set_ylabel('Frequency')
    
    # 9. Box plot of spending by gender
    bp = axes[2,2].boxplot([df[df['Gender']=='Male']['Total Spend'], 
                           df[df['Gender']=='Female']['Total Spend']], 
                          labels=['Male', 'Female'],
                          patch_artist=True,
                          widths=0.6)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightpink')
    axes[2,2].set_title('Total Spend by Gender')
    axes[2,2].set_xlabel('Gender')
    axes[2,2].set_ylabel('Total Spend ($)')
    axes[2,2].grid(False)

    
    plt.tight_layout()
    plt.show()
    
    # Additional insights
    print("\n📈 KEY INSIGHTS:")
    print(f"• Average customer age: {df['Age'].mean():.1f} years")
    print(f"• Average total spend: ${df['Total Spend'].mean():.2f}")
    print(f"• Average items purchased: {df['Items Purchased'].mean():.1f}")
    print(f"• Average rating: {df['Average Rating'].mean():.2f}")
    print(f"• Most common membership type: {df['Membership Type'].mode()[0]}")
    print(f"• Most common satisfaction level: {df['Satisfaction Level'].mode()[0]}")

perform_eda(df_clean)

# Correlation heatmap of numeric variables
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()