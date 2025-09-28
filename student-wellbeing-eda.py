
# Task 1: Exploratory Data Analysis (EDA) on Student Well-being dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Step 1: Load dataset
# ----------------------------
df = pd.read_csv("firsttask.csv")   
print("Dataset shape:", df.shape)
print(df.head())

# ----------------------------
# Step 2: Explore dataset
# ----------------------------
print("\nInfo:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary statistics:")
print(df.describe(include="all"))

# ----------------------------
# Step 3: Data Preprocessing
# ----------------------------
# Remove duplicates
df = df.drop_duplicates()

# Fill missing values
for col in ["Sleep_Hours", "Screen_Time", "Attendance"]:
    df[col] = df[col].fillna(df[col].mean())   # numerical → mean

# Convert categorical columns
df["Extracurricular"] = df["Extracurricular"].map({"Yes":1, "No":0})
df["Stress_Level"] = df["Stress_Level"].map({"Low":1, "Medium":2, "High":3})

# ----------------------------
# Step 4: Exploratory Data Analysis (EDA)
# ----------------------------
sns.set(style="whitegrid")

# Study Hours vs CGPA
plt.figure(figsize=(6,4))
sns.scatterplot(x="Hours_Study", y="CGPA", data=df)
plt.title("Study Hours vs CGPA")
plt.show()

# Sleep Hours vs CGPA
plt.figure(figsize=(6,4))
sns.scatterplot(x="Sleep_Hours", y="CGPA", data=df)
plt.title("Sleep Hours vs CGPA")
plt.show()

# Screen Time vs CGPA
plt.figure(figsize=(6,4))
sns.scatterplot(x="Screen_Time", y="CGPA", data=df)
plt.title("Screen Time vs CGPA")
plt.show()

# Stress Level vs CGPA
plt.figure(figsize=(6,4))
sns.boxplot(x="Stress_Level", y="CGPA", data=df)
plt.title("Stress Level vs CGPA")
plt.show()

# Extracurricular vs CGPA
plt.figure(figsize=(6,4))
sns.boxplot(x="Extracurricular", y="CGPA", data=df)
plt.title("Extracurricular vs CGPA")
plt.xticks([0,1], ["No","Yes"])
plt.show()

# ----------------------------
# Step 5: Generate Insights
# ----------------------------
print("\n--- Insights ---")
print("1. Correlation between study hours and CGPA:", df["Hours_Study"].corr(df["CGPA"]))
print("2. Correlation between sleep hours and CGPA:", df["Sleep_Hours"].corr(df["CGPA"]))
print("3. Correlation between screen time and CGPA:", df["Screen_Time"].corr(df["CGPA"]))

stress_cgpa = df.groupby("Stress_Level")["CGPA"].mean()
print("4. Avg CGPA by Stress Level:\n", stress_cgpa)

extra_cgpa = df.groupby("Extracurricular")["CGPA"].mean()
print("5. Avg CGPA by Extracurricular Participation:\n", extra_cgpa)

# ----------------------------
# Step 6: Export cleaned dataset
# ----------------------------
df.to_csv("cleaned_firsttask.csv", index=False)
print("\nCleaned dataset saved as cleaned_firsttask.csv")

