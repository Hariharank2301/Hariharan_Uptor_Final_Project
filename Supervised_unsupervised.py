import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("larger_product_sales.csv")

# Encode categorical variable (Category)
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

# Features and target
X = df[['Price', 'Stock Available', 'Category']]
y = df['Sales (Last Month)']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Supervised Learning: Regression
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Unsupervised Learning: K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Price'], y=df['Sales (Last Month)'], hue=df['Cluster'], palette='viridis')
plt.title("K-Means Clustering of Product Sales")
plt.xlabel("Price")
plt.ylabel("Sales (Last Month)")
plt.show()
