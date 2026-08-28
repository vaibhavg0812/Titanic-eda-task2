"""
Task 2: Exploratory Data Analysis (EDA) - Titanic Dataset
Elevate Labs AI & ML Internship

Objective: Understand data using statistics and visualizations.
Tools: Pandas, Matplotlib, Seaborn, Plotly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

sns.set_style("whitegrid")
os.makedirs("plots", exist_ok=True)

# -------------------------------------------------
# 1. Load Data
# -------------------------------------------------
df = pd.read_csv("titanic.csv")
print("Shape of dataset:", df.shape)
print("\nColumn dtypes:\n", df.dtypes)

# -------------------------------------------------
# 2. Summary Statistics
# -------------------------------------------------
print("\n=== Summary Statistics (numeric columns) ===")
summary = df.describe(include=[np.number]).T
summary["median"] = df.median(numeric_only=True)
print(summary)

print("\n=== Missing Values ===")
print(df.isnull().sum().sort_values(ascending=False))

print("\n=== Categorical Value Counts ===")
for col in ["sex", "pclass", "embarked", "survived"]:
    print(f"\n-- {col} --")
    print(df[col].value_counts(dropna=False))

# -------------------------------------------------
# 3. Histograms for numeric features
# -------------------------------------------------
numeric_cols = ["age", "fare", "sibsp", "parch"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.flatten(), numeric_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig("plots/histograms.png", dpi=150)
plt.close()

# -------------------------------------------------
# 4. Boxplots for numeric features (spot outliers)
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(y=df["age"], ax=axes[0], color="lightgreen")
axes[0].set_title("Boxplot: Age")
sns.boxplot(y=df["fare"], ax=axes[1], color="salmon")
axes[1].set_title("Boxplot: Fare")
plt.tight_layout()
plt.savefig("plots/boxplots.png", dpi=150)
plt.close()

# Boxplot of Fare by Passenger Class (a common EDA cut)
plt.figure(figsize=(7, 5))
sns.boxplot(x="pclass", y="fare", data=df, palette="Set2")
plt.title("Fare Distribution by Passenger Class")
plt.savefig("plots/boxplot_fare_by_class.png", dpi=150)
plt.close()

# -------------------------------------------------
# 5. Correlation matrix / heatmap
# -------------------------------------------------
plt.figure(figsize=(8, 6))
corr = df[["survived", "pclass", "age", "sibsp", "parch", "fare"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix (Numeric Features)")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=150)
plt.close()

# Pairplot (feature relationships), colored by survival
pairplot_df = df[["survived", "pclass", "age", "fare"]].dropna()
sns.pairplot(pairplot_df, hue="survived", palette="husl")
plt.savefig("plots/pairplot.png", dpi=150)
plt.close()

# -------------------------------------------------
# 6. Survival patterns by category (bar charts)
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(x="sex", y="survived", data=df, ax=axes[0], palette="pastel")
axes[0].set_title("Survival Rate by Sex")
sns.barplot(x="pclass", y="survived", data=df, ax=axes[1], palette="pastel")
axes[1].set_title("Survival Rate by Passenger Class")
plt.tight_layout()
plt.savefig("plots/survival_by_category.png", dpi=150)
plt.close()

# -------------------------------------------------
# 7. Interactive Plotly chart (age vs fare, colored by survival)
# -------------------------------------------------
fig = px.scatter(
    df, x="age", y="fare", color="survived",
    hover_data=["sex", "pclass"],
    title="Age vs Fare, colored by Survival"
)
fig.write_html("plots/age_vs_fare_interactive.html")

# -------------------------------------------------
# 8. Skewness check
# -------------------------------------------------
print("\n=== Skewness of numeric columns ===")
print(df[numeric_cols].skew())

print("\nAll plots saved to the 'plots/' folder.")
