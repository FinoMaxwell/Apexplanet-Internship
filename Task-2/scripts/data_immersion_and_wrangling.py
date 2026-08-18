from pathlib import Path
import pandas as pd
import os

base_dir = Path(__file__).resolve().parent.parent

data_folder = base_dir / "data"
clean_folder = base_dir / "cleaned_data"

os.makedirs(clean_folder, exist_ok=True)

file_path = data_folder / "ApexPlanet_DataAnalytics_Dataset.xlsx"

df = pd.read_excel(file_path)

print("\nDataset Preview\n")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns.tolist())

print("\nDataset Information")
print(df.info())

print("\nSummary Statistics")
print(df.describe(include="all"))

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

print("\nDuplicate Order IDs")
print(df["Order_ID"].duplicated().sum())

df["Order_Date"] = pd.to_datetime(df["Order_Date"])

text_columns = [
    "Customer_Name",
    "Gender",
    "City",
    "Product",
    "Category"
]

for column in text_columns:
    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.title()
    )

df[text_columns] = df[text_columns].replace("", pd.NA)

numeric_columns = [
    "Age",
    "Quantity",
    "Unit_Price",
    "Total_Sales"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df[column] = df[column].fillna(df[column].median())

for column in text_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[0, 18, 30, 45, 60, 100],
    labels=[
        "Teen",
        "Young Adult",
        "Adult",
        "Senior",
        "Elder"
    ]
)

df["Order_Month"] = df["Order_Date"].dt.month_name()

df["Calculated_Sales"] = (
    df["Quantity"] *
    df["Unit_Price"]
)

print("\nOutlier Report")

for column in numeric_columns:

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)

    outliers = df[
        (df[column] < lower) |
        (df[column] > upper)
    ]

    print(f"{column}: {len(outliers)}")

print("\nFinal Missing Values")
print(df.isnull().sum())

output_file = clean_folder / "Cleaned_Sales_Dataset.csv"

df.to_csv(output_file, index=False)

print("\nCleaned Dataset Saved")
print(output_file)