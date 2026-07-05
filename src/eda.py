import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/ai4i2020.csv")
df.columns = [c.strip() for c in df.columns]

print("Shape:", df.shape)
print(df.head())
print("\nClass balance (Machine failure):")
print(df["Machine failure"].value_counts(normalize=True))

print("\nFailure type breakdown:")
for flag in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
    print(flag, df[flag].sum())

# Correlation heatmap
plt.figure(figsize=(8, 6))
num_cols = ["Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
            "Machine failure"]
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig("eda_correlation.png")
plt.close()

# Boxplot: torque vs failure
plt.figure(figsize=(6, 5))
sns.boxplot(x="Machine failure", y="Torque [Nm]", data=df)
plt.title("Torque vs failure")
plt.savefig("eda_torque_boxplot.png")
plt.close()

# Boxplot: tool wear vs failure
plt.figure(figsize=(6, 5))
sns.boxplot(x="Machine failure", y="Tool wear [min]", data=df)
plt.title("Tool wear vs failure")
plt.savefig("eda_toolwear_boxplot.png")
plt.close()

print("\nSaved: eda_correlation.png, eda_torque_boxplot.png, eda_toolwear_boxplot.png")