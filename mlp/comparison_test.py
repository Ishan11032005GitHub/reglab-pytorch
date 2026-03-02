import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_CLS = "results/classification/classification_results.csv"
ES_CLS = "results/early_stopping/classification/patience_15/classification_results.csv"

BASE_REG = "results/regression/regression_results.csv"
ES_REG = "results/early_stopping/regression/patience_15/regression_results.csv"

OUT_DIR = "results/comparison_test_early_stopping"
os.makedirs(OUT_DIR, exist_ok=True)


# CLASSIFICATION – Test Accuracy
base_cls = pd.read_csv(BASE_CLS)
es_cls = pd.read_csv(ES_CLS)

plt.figure()
plt.plot(base_cls["Hidden Layers"], base_cls["Test Acc"],
         marker="o", label="Baseline")
plt.plot(es_cls["Hidden Layers"], es_cls["Test Acc"],
         marker="o", label="Early Stopping")
plt.xlabel("Hidden Layers")
plt.ylabel("Test Accuracy")
plt.title("Classification Test Accuracy Comparison (Baseline vs Early Stopping)")
plt.legend()
plt.savefig(f"{OUT_DIR}/cls_test_acc.png")
plt.close()


# REGRESSION – R² Score
base_reg = pd.read_csv(BASE_REG)
es_reg = pd.read_csv(ES_REG)

plt.figure()
plt.plot(base_reg["Hidden Layers"], base_reg["R2"],
         marker="o", label="Baseline")
plt.plot(es_reg["Hidden Layers"], es_reg["R2"],
         marker="o", label="Early Stopping")
plt.xlabel("Hidden Layers")
plt.ylabel("R² Score")
plt.title("Regression Test R² Comparison (Baseline vs Early Stopping)")
plt.legend()
plt.savefig(f"{OUT_DIR}/reg_r2.png")
plt.close()

print("Early stopping test comparison plots saved.")