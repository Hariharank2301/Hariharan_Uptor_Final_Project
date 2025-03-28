import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("rainfall.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical target variable
le = LabelEncoder()
df['weather_condition'] = le.fit_transform(df['weather_condition'])

# Features and target
X = df[['temperature', 'humidity', 'wind_speed']]
y = df['weather_condition']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Supervised Learning: Classification
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#Calculating Accuracy

print("Classification Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Unsupervised Learning: K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['temperature'], y=df['humidity'], hue=df['cluster'], palette='viridis')
plt.title("K-Means Clustering of Weather Data")
plt.show()
