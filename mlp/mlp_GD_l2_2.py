import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.impute import SimpleImputer

torch.manual_seed(42)
np.random.seed(42)

EPOCHS = 250
HIDDEN_NEURONS = 128
KFOLDS = 5
LR = 0.001
LAMBDA = 1e-4   # 🔥 L2 REGULARIZATION STRENGTH

BASE_DIR = "mlp/results/gd_l2"

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "classification"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "regression"), exist_ok=True)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, depth):
        super().__init__()
        layers = []
        prev = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev, HIDDEN_NEURONS))
            layers.append(nn.ReLU())
            prev = HIDDEN_NEURONS
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def gd_step(model, lr):
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            p.data -= lr * p.grad
    model.zero_grad()


# 🔥 L2 PENALTY FUNCTION
def l2_penalty(model):
    l2 = 0.0
    for p in model.parameters():
        l2 += torch.sum(p ** 2)
    return l2


# =========================
# CLASSIFICATION
# =========================
print("\nCLASSIFICATION (GD + L2)")

wine = pd.read_csv("mlp/winequality-red.csv", sep=";")
wine["quality"] = (wine["quality"] >= 6).astype(int)

X = wine.drop("quality", axis=1).values
y = wine["quality"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)

cls_results = []

for depth in range(1, 5):
    epoch_tr_acc = np.zeros(EPOCHS)
    epoch_va_acc = np.zeros(EPOCHS)
    epoch_tr_loss = np.zeros(EPOCHS)
    epoch_va_loss = np.zeros(EPOCHS)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        Xt = torch.tensor(X_train[tr_idx], dtype=torch.float32)
        yt = torch.tensor(y_train[tr_idx], dtype=torch.long)
        Xv = torch.tensor(X_train[va_idx], dtype=torch.float32)
        yv = torch.tensor(y_train[va_idx], dtype=torch.long)

        model = MLP(X.shape[1], 2, depth)
        loss_fn = nn.CrossEntropyLoss()

        for e in range(EPOCHS):
            out = model(Xt)

            # 🔥 ADD L2 TO LOSS
            loss = loss_fn(out, yt) + LAMBDA * l2_penalty(model)

            model.zero_grad()
            loss.backward()
            gd_step(model, LR)

            epoch_tr_loss[e] += loss.item()
            epoch_tr_acc[e] += (out.argmax(1) == yt).float().mean().item()

            with torch.no_grad():
                vo = model(Xv)
                val_loss = loss_fn(vo, yv) + LAMBDA * l2_penalty(model)
                epoch_va_loss[e] += val_loss.item()
                epoch_va_acc[e] += accuracy_score(yv, vo.argmax(1))

    epoch_tr_acc /= KFOLDS
    epoch_va_acc /= KFOLDS
    epoch_tr_loss /= KFOLDS
    epoch_va_loss /= KFOLDS

    with torch.no_grad():
        Xtst = torch.tensor(X_test, dtype=torch.float32)
        ytst = torch.tensor(y_test, dtype=torch.long)
        test_acc = accuracy_score(ytst, model(Xtst).argmax(1))

    print(f"[Classification] Depth={depth} | Train Acc={epoch_tr_acc[-1]:.4f} | "
          f"Val Acc={epoch_va_acc[-1]:.4f} | Test Acc={test_acc:.4f}")

    cls_results.append([
        depth,
        epoch_tr_loss[-1],
        epoch_tr_acc[-1],
        epoch_va_loss[-1],
        epoch_va_acc[-1],
        test_acc
    ])

    np.save(os.path.join(BASE_DIR, "classification", f"train_acc_depth_{depth}.npy"), epoch_tr_acc)
    np.save(os.path.join(BASE_DIR, "classification", f"val_acc_depth_{depth}.npy"), epoch_va_acc)
    np.save(os.path.join(BASE_DIR, "classification", f"train_loss_depth_{depth}.npy"), epoch_tr_loss)
    np.save(os.path.join(BASE_DIR, "classification", f"val_loss_depth_{depth}.npy"), epoch_va_loss)


cls_df = pd.DataFrame(cls_results,
    columns=["Hidden Layers", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Test Acc"]
)
cls_df.to_csv(os.path.join(BASE_DIR, "classification", "classification_results.csv"), index=False)


# =========================
# REGRESSION
# =========================
print("\nREGRESSION (GD + L2)")

house = pd.read_csv("mlp/housing.csv")
house = pd.get_dummies(house, columns=["ocean_proximity"], drop_first=True)

X = house.drop("median_house_value", axis=1).values
y = house["median_house_value"].values.reshape(-1, 1)

X = SimpleImputer(strategy="mean").fit_transform(X)

y_mean, y_std = y.mean(), y.std()
y_norm = (y - y_mean) / y_std

X_train, X_test, y_train, y_test = train_test_split(
    X, y_norm, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

reg_results = []

for depth in range(1, 5):
    epoch_tr_mse = np.zeros(EPOCHS)
    epoch_va_mse = np.zeros(EPOCHS)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        Xt = torch.tensor(X_train[tr_idx], dtype=torch.float32)
        yt = torch.tensor(y_train[tr_idx], dtype=torch.float32)
        Xv = torch.tensor(X_train[va_idx], dtype=torch.float32)
        yv = torch.tensor(y_train[va_idx], dtype=torch.float32)

        model = MLP(X.shape[1], 1, depth)
        loss_fn = nn.MSELoss()

        for e in range(EPOCHS):
            out = model(Xt)

            # 🔥 ADD L2
            loss = loss_fn(out, yt) + LAMBDA * l2_penalty(model)

            model.zero_grad()
            loss.backward()
            gd_step(model, LR)

            epoch_tr_mse[e] += loss.item()

            with torch.no_grad():
                vo = model(Xv)
                val_loss = loss_fn(vo, yv) + LAMBDA * l2_penalty(model)
                epoch_va_mse[e] += val_loss.item()

    epoch_tr_mse /= KFOLDS
    epoch_va_mse /= KFOLDS

    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        preds = preds * y_std + y_mean
        y_true = y_test * y_std + y_mean
        r2 = r2_score(y_true, preds)

    print(f"[Regression] Depth={depth} | Train MSE={epoch_tr_mse[-1]:.4f} | "
          f"Val MSE={epoch_va_mse[-1]:.4f} | R2={r2:.4f}")

    reg_results.append([depth, epoch_tr_mse[-1], epoch_va_mse[-1], r2])

    np.save(os.path.join(BASE_DIR, "regression", f"train_mse_depth_{depth}.npy"), epoch_tr_mse)
    np.save(os.path.join(BASE_DIR, "regression", f"val_mse_depth_{depth}.npy"), epoch_va_mse)


reg_df = pd.DataFrame(reg_results,
    columns=["Hidden Layers", "Train MSE", "Val MSE", "R2"]
)
reg_df.to_csv(os.path.join(BASE_DIR, "regression", "regression_results.csv"), index=False)

print("\nAll GD + L2 experiments finished. Results saved in:", BASE_DIR)