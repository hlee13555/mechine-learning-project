import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

test = pd.read_csv("data/processed/test.csv")

# Plot 1: Class distribution
sns.countplot(x="is_efficient", data=test)
plt.title("Target Distribution (Classification)")
plt.savefig("models/plot1_class_dist.png")
plt.show()
plt.close()

# Plot 2: Correlation heatmap
sns.heatmap(test.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("models/plot2_corr_heatmap.png")
plt.show()
plt.close()

