import os
import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 250
DEPTHS = [1, 2, 3, 4]

BASE_CLS = "results/classification"
ES_CLS = "results/early_stopping/classification/patience_15"

BASE_REG = "results/regression"
ES_REG = "results/early_stopping/regression/patience_15"

OUT_DIR = "results/comparison_epoch_early_stopping"
os.makedirs(OUT_DIR, exist_ok=True)


# CLASSIFICATION
for d in DEPTHS:
    base_tr_acc = np.load(f"{BASE_CLS}/train_acc_depth_{d}.npy")
    base_va_acc = np.load(f"{BASE_CLS}/val_acc_depth_{d}.npy")

    es_tr_acc = np.load(f"{ES_CLS}/train_acc_depth_{d}.npy")
    es_va_acc = np.load(f"{ES_CLS}/val_acc_depth_{d}.npy")

    plt.figure()
    plt.plot(base_tr_acc, label="Baseline Train")
    plt.plot(base_va_acc, label="Baseline Val")
    plt.plot(es_tr_acc, "--", label="Early Stopping Train")
    plt.plot(es_va_acc, "--", label="Early Stopping Val")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Classification Accuracy Comparison (Depth={d})")
    plt.legend()
    plt.savefig(f"{OUT_DIR}/cls_acc_depth_{d}.png")
    plt.close()

    base_tr_loss = np.load(f"{BASE_CLS}/train_loss_depth_{d}.npy")
    base_va_loss = np.load(f"{BASE_CLS}/val_loss_depth_{d}.npy")

    es_tr_loss = np.load(f"{ES_CLS}/train_loss_depth_{d}.npy")
    es_va_loss = np.load(f"{ES_CLS}/val_loss_depth_{d}.npy")

    plt.figure()
    plt.plot(base_tr_loss, label="Baseline Train")
    plt.plot(base_va_loss, label="Baseline Val")
    plt.plot(es_tr_loss, "--", label="Early Stopping Train")
    plt.plot(es_va_loss, "--", label="Early Stopping Val")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Classification Loss Comparison (Depth={d})")
    plt.legend()
    plt.savefig(f"{OUT_DIR}/cls_loss_depth_{d}.png")
    plt.close()


# REGRESSION (MSE)
for d in DEPTHS:
    base_tr_mse = np.load(f"{BASE_REG}/train_mse_depth_{d}.npy")
    base_va_mse = np.load(f"{BASE_REG}/val_mse_depth_{d}.npy")

    es_tr_mse = np.load(f"{ES_REG}/train_mse_depth_{d}.npy")
    es_va_mse = np.load(f"{ES_REG}/val_mse_depth_{d}.npy")

    plt.figure()
    plt.plot(base_tr_mse, label="Baseline Train")
    plt.plot(base_va_mse, label="Baseline Val")
    plt.plot(es_tr_mse, "--", label="Early Stopping Train")
    plt.plot(es_va_mse, "--", label="Early Stopping Val")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title(f"Regression MSE Comparison (Depth={d})")
    plt.legend()
    plt.savefig(f"{OUT_DIR}/reg_mse_depth_{d}.png")
    plt.close()

print("Epoch-wise early stopping comparison plots saved.")