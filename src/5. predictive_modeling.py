def predictive_modeling(df):
    # Prepare data for modeling
    df_model = df.copy()

    # Encode categorical variables
    le_gender = LabelEncoder()
    le_city = LabelEncoder()
    le_membership = LabelEncoder()
    
    df_model['Gender_Encoded'] = le_gender.fit_transform(df_model['Gender'])
    df_model['City_Encoded'] = le_city.fit_transform(df_model['City'])
    df_model['Membership_Encoded'] = le_membership.fit_transform(df_model['Membership Type'])
    
    # MODEL 1: Predict Customer Satisfaction Level
    print("\n📊 MODEL 1: CUSTOMER SATISFACTION PREDICTION")
    print("-" * 40)
    
    # Check unique values in satisfaction level
    print(f"Unique Satisfaction Levels: {df_model['Satisfaction Level'].unique()}")
    print(f"Satisfaction Level counts:\n{df_model['Satisfaction Level'].value_counts()}")
    
    features_satisfaction = ['Age', 'Gender_Encoded', 'City_Encoded', 'Membership_Encoded',
                           'Total Spend', 'Items Purchased', 'Average Rating', 
                           'Discount Applied', 'Days Since Last Purchase']
    
    # Ensure all feature columns exist and have no missing values
    missing_features = [col for col in features_satisfaction if col not in df_model.columns]
    if missing_features:
        print(f"Warning: Missing feature columns: {missing_features}")
        features_satisfaction = [col for col in features_satisfaction if col in df_model.columns]
    
    X_satisfaction = df_model[features_satisfaction]
    y_satisfaction = df_model['Satisfaction Level']
    
    # Check for any remaining NaN values
    if X_satisfaction.isnull().any().any():
        print("Warning: Found NaN values in features. Filling with median/mode...")
        for col in X_satisfaction.columns:
            if X_satisfaction[col].dtype in ['int64', 'float64']:
                X_satisfaction[col].fillna(X_satisfaction[col].median(), inplace=True)
    
    if y_satisfaction.isnull().any():
        print("Warning: Found NaN values in target variable. This should not happen after cleaning.")
        y_satisfaction.fillna(y_satisfaction.mode()[0], inplace=True)
    
    # Split the data - remove stratify if there are issues
    try:
        X_train_sat, X_test_sat, y_train_sat, y_test_sat = train_test_split(
            X_satisfaction, y_satisfaction, test_size=0.2, random_state=42, stratify=y_satisfaction
        )
    except ValueError as e:
        print(f"Stratify failed: {e}. Using random split instead.")
        X_train_sat, X_test_sat, y_train_sat, y_test_sat = train_test_split(
            X_satisfaction, y_satisfaction, test_size=0.2, random_state=42
        )
    
    # Scale features
    scaler_sat = StandardScaler()
    X_train_sat_scaled = scaler_sat.fit_transform(X_train_sat)
    X_test_sat_scaled = scaler_sat.transform(X_test_sat)
    
    # Train Random Forest model
    rf_satisfaction = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_satisfaction.fit(X_train_sat_scaled, y_train_sat)
    
    # Predictions
    y_pred_sat = rf_satisfaction.predict(X_test_sat_scaled)
    
    # Evaluate
    accuracy_sat = accuracy_score(y_test_sat, y_pred_sat)
    print(f"Satisfaction Prediction Accuracy: {accuracy_sat:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test_sat, y_pred_sat))
    
    # Feature importance
    feature_importance_sat = pd.DataFrame({
        'feature': features_satisfaction,
        'importance': rf_satisfaction.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Features for Satisfaction Prediction:")
    print(feature_importance_sat.head())
    
    # MODEL 2: Predict Total Spend
    print("\n💰 MODEL 2: TOTAL SPEND PREDICTION")
    print("-" * 40)
    
    features_spend = ['Age', 'Gender_Encoded', 'City_Encoded', 'Membership_Encoded',
                     'Items Purchased', 'Average Rating', 'Discount Applied', 
                     'Days Since Last Purchase']
    
    # Ensure all feature columns exist
    missing_features = [col for col in features_spend if col not in df_model.columns]
    if missing_features:
        print(f"Warning: Missing feature columns: {missing_features}")
        features_spend = [col for col in features_spend if col in df_model.columns]
    
    X_spend = df_model[features_spend]
    y_spend = df_model['Total Spend']
    
    # Check for any remaining NaN values
    if X_spend.isnull().any().any():
        print("Warning: Found NaN values in features. Filling with median...")
        for col in X_spend.columns:
            if X_spend[col].dtype in ['int64', 'float64']:
                X_spend[col].fillna(X_spend[col].median(), inplace=True)
    
    if y_spend.isnull().any():
        print("Warning: Found NaN values in target variable. Filling with median...")
        y_spend.fillna(y_spend.median(), inplace=True)
    
    # Split the data
    X_train_spend, X_test_spend, y_train_spend, y_test_spend = train_test_split(
        X_spend, y_spend, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler_spend = StandardScaler()
    X_train_spend_scaled = scaler_spend.fit_transform(X_train_spend)
    X_test_spend_scaled = scaler_spend.transform(X_test_spend)
    
    # Train Random Forest model
    rf_spend = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_spend.fit(X_train_spend_scaled, y_train_spend)
    
    # Predictions
    y_pred_spend = rf_spend.predict(X_test_spend_scaled)
    
    # Evaluate
    mse_spend = mean_squared_error(y_test_spend, y_pred_spend)
    r2_spend = r2_score(y_test_spend, y_pred_spend)
    rmse_spend = np.sqrt(mse_spend)
    
    print(f"Total Spend Prediction RMSE: ${rmse_spend:.2f}")
    print(f"Total Spend Prediction R²: {r2_spend:.3f}")
    
    # Feature importance
    feature_importance_spend = pd.DataFrame({
        'feature': features_spend,
        'importance': rf_spend.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Features for Spend Prediction:")
    print(feature_importance_spend.head())
    
    # Visualize model performance
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Satisfaction model - confusion matrix
    cm_sat = confusion_matrix(y_test_sat, y_pred_sat)
    sns.heatmap(cm_sat, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Satisfaction Prediction - Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Spend model - actual vs predicted
    axes[1].scatter(y_test_spend, y_pred_spend, alpha=0.6)
    axes[1].plot([y_test_spend.min(), y_test_spend.max()], 
                [y_test_spend.min(), y_test_spend.max()], 'r--', lw=2)
    axes[1].set_title('Total Spend: Actual vs Predicted')
    axes[1].set_xlabel('Actual Total Spend')
    axes[1].set_ylabel('Predicted Total Spend')
    
    plt.tight_layout()
    plt.show()
    
    return rf_satisfaction, rf_spend, scaler_sat, scaler_spend

predictive_modeling(df_clean)