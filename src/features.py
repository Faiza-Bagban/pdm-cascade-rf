"""Feature engineering for AI4I 2020 Predictive Maintenance dataset."""
import pandas as pd

FAILURE_FLAGS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Power_W",
    "Temp_diff_K",
    "Wear_x_Torque",
    "Type_H",
    "Type_L",
    "Type_M",
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _derive_failure_type(row) -> str:
    for flag in FAILURE_FLAGS:
        if row.get(flag, 0) == 1:
            return flag
    return "No Failure"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Power_W"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * 2 * 3.14159 / 60)
    df["Temp_diff_K"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Wear_x_Torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    if set(FAILURE_FLAGS).issubset(df.columns):
        df["Failure_Type"] = df.apply(_derive_failure_type, axis=1)

    # df = pd.get_dummies(df, columns=["Type"], prefix="Type")
    df = pd.get_dummies(df, columns=["Type"], prefix="Type", dtype=int)
    for col in ["Type_H", "Type_L", "Type_M"]:
        if col not in df.columns:
            df[col] = 0

    drop_cols = [c for c in ["UDI", "Product ID"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df