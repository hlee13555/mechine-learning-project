# src/evaluate.py
"""
Produce evaluation figures and summary tables used in the report.
"""
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

(REPORTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
(REPORTS_DIR / "tables").mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "val.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

# Target distributions (val/test)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.countplot(x="is_efficient", data=val)
plt.title(f"Val Target Distribution (n={len(val)})")
plt.subplot(1,2,2)
sns.countplot(x="is_efficient", data=test)
plt.title(f"Test Target Distribution (n={len(test)})")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "figures" / "target_distribution_val_test.png")
plt.close()

# Correlation heatmap for test set
plt.figure(figsize=(10,8))
sns.heatmap(test.corr(), cmap="coolwarm", center=0)
plt.title("Testset Correlation Heatmap")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "figures" / "test_correlation_heatmap.png")
plt.close()

# Save target counts tables
for split_name, df in [("train", train), ("val", val), ("test", test)]:
    counts = df["is_efficient"].value_counts().rename_axis("label").reset_index(name="count")
    counts.to_csv(REPORTS_DIR / "tables" / f"target_counts_{split_name}.csv", index=False)

print("[eval] Saved evaluation figures & tables to reports/")
