from pathlib import Path
import pandas as pd
import os

base_dir = Path(__file__).resolve().parent.parent

data_folder = base_dir / "data"
output_folder = base_dir / "outputs"
clean_folder = base_dir / "cleaned_data"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(clean_folder, exist_ok=True)

file_path = data_folder / "ApexPlanet_DataAnalytics_Dataset.xlsx"

if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found at:\n{file_path}")

df = pd.read_excel(file_path)

print("\nDataset Preview\n")
print(df.head())

print("\nShape:", df.shape)

print("\nDataset Info\n")
print(df.info())

print("\nSummary Statistics\n")
print(df.describe(include="all"))

data_dictionary = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isnull().sum(),
    "Unique Values": [df[col].nunique() for col in df.columns]
})

data_dictionary.to_excel(output_folder / "Data_Dictionary.xlsx", index=False)

print("\nMissing Values\n")
print(df.isnull().sum())

duplicate_rows = df.duplicated().sum()
print("\nDuplicate Rows:", duplicate_rows)

if duplicate_rows > 0:
    df = df.drop_duplicates()

duplicate_orders = df["Order_ID"].duplicated().sum()
print("Duplicate Order IDs:", duplicate_orders)

df["Order_Date"] = pd.to_datetime(df["Order_Date"])

text_columns = [
    "Customer_Name",
    "Gender",
    "City",
    "Product",
    "Category"
]

for col in text_columns:
    df[col] = df[col].fillna("").astype(str).str.strip().str.title()

df[text_columns] = df[text_columns].replace("", pd.NA)

numeric_columns = [
    "Age",
    "Quantity",
    "Unit_Price",
    "Total_Sales"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

for col in text_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[0, 18, 30, 45, 60, 100],
    labels=["Teen", "Young Adult", "Adult", "Senior", "Elder"]
)

df["Order_Month"] = df["Order_Date"].dt.month_name()
df["Order_Year"] = df["Order_Date"].dt.year
df["Day_Name"] = df["Order_Date"].dt.day_name()

df["Calculated_Sales"] = df["Quantity"] * df["Unit_Price"]

difference = (df["Calculated_Sales"] - df["Total_Sales"]).abs().sum()

print("\nSales Difference:", difference)

print("\nOutlier Report")

for col in numeric_columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = df[(df[col] < lower) | (df[col] > upper)].shape[0]

    print(f"{col}: {count}")

print("\nFinal Missing Values\n")
print(df.isnull().sum())

df.to_csv(clean_folder / "Cleaned_Sales_Dataset.csv", index=False)

print("\nFiles Created Successfully")
print("Data Dictionary :", output_folder / "Data_Dictionary.xlsx")
print("Clean Dataset   :", clean_folder / "Cleaned_Sales_Dataset.csv")