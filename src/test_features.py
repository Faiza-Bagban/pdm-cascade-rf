from src.features import load_data, engineer_features, FEATURE_COLUMNS

df = load_data("data/ai4i2020.csv")
df = engineer_features(df)

print(df[FEATURE_COLUMNS].head())
print("\nFailure_Type counts:")
print(df["Failure_Type"].value_counts())