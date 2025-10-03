# Install required packages
%pip install pandas numpy matplotlib seaborn scikit-learn

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vS597_KVD9wqThs6lDsxukLZHE0eNbHvMQiJN66H2PhU_CRJRIBX_0wT4LLJL8vhYm3deHV-XMMoBj9/pub?gid=284931259&single=true&output=csv')

# Quick look at the data
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nBasic statistics:")
print(df.describe())

# Take a closer look at missing values
print("\nMissing values:")
print(df.isnull().sum())