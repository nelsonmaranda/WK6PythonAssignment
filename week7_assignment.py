# =========================
# Imports and Setup
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# =========================
# Task 1: Load and Explore the Dataset
# =========================

try:
    # Load Iris dataset from sklearn
    iris = load_iris(as_frame=True)
    df = iris.frame
    print("Dataset loaded successfully.\n")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nData types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())

# Clean dataset (fill or drop missing values)
if df.isnull().values.any():
    df = df.dropna()
    print("\nMissing values dropped.")
else:
    print("\nNo missing values found.")

# =========================
# Task 2: Basic Data Analysis
# =========================

print("\nBasic statistics for numerical columns:")
print(df.describe())

# Group by species and compute mean of numerical columns
grouped = df.groupby('target').mean()
print("\nMean of numerical columns grouped by species (target):")
print(grouped)

# Identify patterns or interesting findings
print("\nInteresting findings:")
for col in df.columns[:-1]:
    means = grouped[col]
    print(f"  {col}: min mean = {means.min():.2f}, max mean = {means.max():.2f}")

# =========================
# Task 3: Data Visualization
# =========================

sns.set(style="whitegrid")

# Line chart: Simulate a time-series by cumulative sum of sepal length
plt.figure(figsize=(8, 4))
df['cumulative_sepal_length'] = df['sepal length (cm)'].cumsum()
plt.plot(df.index, df['cumulative_sepal_length'], label='Cumulative Sepal Length')
plt.title('Cumulative Sepal Length Over Index')
plt.xlabel('Index')
plt.ylabel('Cumulative Sepal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart: Average petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(x=grouped.index, y=grouped['petal length (cm)'])
plt.title('Average Petal Length per Species')
plt.xlabel('Species (target)')
plt.ylabel('Average Petal Length (cm)')
plt.tight_layout()
plt.show()

# Histogram: Distribution of sepal width
plt.figure(figsize=(6, 4))
sns.histplot(df['sepal width (cm)'], bins=20, kde=True)
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Scatter plot: Sepal length vs. petal length
plt.figure(figsize=(6, 4))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', data=df, palette='deep')
plt.title('Sepal Length vs. Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.show()

# =========================
# End of Script
# =========================