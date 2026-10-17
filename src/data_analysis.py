
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project folders ni identify cheyyadam
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "sales_and_customer_insights.csv"
RESULTS_DIR = PROJECT_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"

RESULTS_DIR.mkdir(exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# CSV dataset load cheyyadam
df = pd.read_csv(DATA_FILE)

print("\n===== CUSTOMER PURCHASE DATASET =====")
print("Dataset shape:", df.shape)

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== NUMERICAL SUMMARY =====")
print(df.describe(include="all").to_string())

# Numerical columns correlation unte graph create cheyyadam
numeric_df = df.select_dtypes(include="number")

if numeric_df.shape[1] >= 2:
    correlation = numeric_df.corr()

    plt.figure(figsize=(12, 8))
    plt.imshow(correlation, cmap="coolwarm", aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(correlation.columns)), correlation.columns,
               rotation=90)
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    plt.title("Customer Purchase Data Correlation")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

# Dataset report save cheyyadam
with open(RESULTS_DIR / "data_summary.txt", "w", encoding="utf-8") as file:
    file.write(f"Dataset shape: {df.shape}\n\n")
    file.write("Columns:\n")
    file.write("\n".join(df.columns.astype(str)))
    file.write("\n\nMissing values:\n")
    file.write(df.isnull().sum().to_string())
    file.write("\n\nSummary:\n")
    file.write(df.describe(include="all").to_string())

print("\nAnalysis completed successfully!")
print("Reports saved in:", RESULTS_DIR)
print("Graphs saved in:", GRAPHS_DIR)