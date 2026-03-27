import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
LR = 0.01

# 🔥 PROPER BASE PATH (NO MORE PATH ISSUES)
BASE_DIR = os.path.join("mlp", "results")

CLS_DIR = os.path.join(BASE_DIR, "classification")
REG_DIR = os.path.join(BASE_DIR, "regression")

os.makedirs(CLS_DIR, exist_ok=True)
os.makedirs(REG_DIR, exist_ok=True)


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


print("\nCLASSIFICATION:")

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

    epoch_train_acc = np.zeros(EPOCHS)
    epoch_val_acc = np.zeros(EPOCHS)
    epoch_train_loss = np.zeros(EPOCHS)
    epoch_val_loss = np.zeros(EPOCHS)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        Xt = torch.tensor(X_train[tr_idx], dtype=torch.float32)
        yt = torch.tensor(y_train[tr_idx], dtype=torch.long)
        Xv = torch.tensor(X_train[va_idx], dtype=torch.float32)
        yv = torch.tensor(y_train[va_idx], dtype=torch.long)

        model = MLP(X.shape[1], 2, depth)
        loss_fn = nn.CrossEntropyLoss()

        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

        for e in range(EPOCHS):
            opt.zero_grad()
            out = model(Xt)
            loss = loss_fn(out, yt)
            loss.backward()
            opt.step()

            epoch_train_loss[e] += loss.item()
            epoch_train_acc[e] += (out.argmax(1) == yt).float().mean().item()

            with torch.no_grad():
                vo = model(Xv)
                epoch_val_loss[e] += loss_fn(vo, yv).item()
                epoch_val_acc[e] += accuracy_score(yv, vo.argmax(1))

    epoch_train_acc /= KFOLDS
    epoch_val_acc /= KFOLDS
    epoch_train_loss /= KFOLDS
    epoch_val_loss /= KFOLDS

    # SAVE ARRAYS
    np.save(os.path.join(CLS_DIR, f"train_acc_depth_{depth}.npy"), epoch_train_acc)
    np.save(os.path.join(CLS_DIR, f"val_acc_depth_{depth}.npy"), epoch_val_acc)
    np.save(os.path.join(CLS_DIR, f"train_loss_depth_{depth}.npy"), epoch_train_loss)
    np.save(os.path.join(CLS_DIR, f"val_loss_depth_{depth}.npy"), epoch_val_loss)

    with torch.no_grad():
        Xtst = torch.tensor(X_test, dtype=torch.float32)
        ytst = torch.tensor(y_test, dtype=torch.long)
        o = model(Xtst)
        test_acc = accuracy_score(ytst, o.argmax(1))

    print(f"[Classification] Depth={depth} | "
          f"Train Acc={epoch_train_acc[-1]:.4f} | "
          f"Val Acc={epoch_val_acc[-1]:.4f} | "
          f"Test Acc={test_acc:.4f}")

    cls_results.append([
        depth,
        epoch_train_loss[-1],
        epoch_train_acc[-1],
        epoch_val_loss[-1],
        epoch_val_acc[-1],
        test_acc
    ])

    # PLOTS
    plt.figure()
    plt.plot(epoch_train_acc, label="Train")
    plt.plot(epoch_val_acc, label="Validation")
    plt.hlines(test_acc, 0, EPOCHS, linestyles="dashed", label="Test")
    plt.legend()
    plt.savefig(os.path.join(CLS_DIR, f"acc_depth_{depth}.png"))
    plt.close()

    plt.figure()
    plt.plot(epoch_train_loss, label="Train")
    plt.plot(epoch_val_loss, label="Validation")
    plt.legend()
    plt.savefig(os.path.join(CLS_DIR, f"loss_depth_{depth}.png"))
    plt.close()


pd.DataFrame(cls_results, columns=[
    "Hidden Layers", "Train Loss", "Train Acc",
    "Val Loss", "Val Acc", "Test Acc"
]).to_csv(os.path.join(CLS_DIR, "classification_results.csv"), index=False)


# ================= REGRESSION =================
print("\nREGRESSION:")

house = pd.read_csv("mlp/housing.csv")
house = pd.get_dummies(house, columns=["ocean_proximity"], drop_first=True)

X = house.drop("median_house_value", axis=1).values
y = house["median_house_value"].values.reshape(-1, 1)

X = SimpleImputer(strategy="mean").fit_transform(X)

y_mean, y_std = y.mean(), y.std()
y = (y - y_mean) / y_std

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

reg_results = []

for depth in range(1, 5):

    epoch_train_mse = np.zeros(EPOCHS)
    epoch_val_mse = np.zeros(EPOCHS)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        Xt = torch.tensor(X_train[tr_idx], dtype=torch.float32)
        yt = torch.tensor(y_train[tr_idx], dtype=torch.float32)
        Xv = torch.tensor(X_train[va_idx], dtype=torch.float32)
        yv = torch.tensor(y_train[va_idx], dtype=torch.float32)

        model = MLP(X.shape[1], 1, depth)
        loss_fn = nn.MSELoss()

        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

        for e in range(EPOCHS):
            opt.zero_grad()
            out = model(Xt)
            loss = loss_fn(out, yt)
            loss.backward()
            opt.step()

            epoch_train_mse[e] += loss.item()

            with torch.no_grad():
                vo = model(Xv)
                epoch_val_mse[e] += loss_fn(vo, yv).item()

    epoch_train_mse /= KFOLDS
    epoch_val_mse /= KFOLDS

    np.save(os.path.join(REG_DIR, f"train_mse_depth_{depth}.npy"), epoch_train_mse)
    np.save(os.path.join(REG_DIR, f"val_mse_depth_{depth}.npy"), epoch_val_mse)

    with torch.no_grad():
        Xtst = torch.tensor(X_test, dtype=torch.float32)
        preds = model(Xtst).detach().cpu().numpy()
        preds = preds * y_std + y_mean
        y_true = y_test * y_std + y_mean
        r2 = r2_score(y_true, preds)

    print(f"[Regression] Depth={depth} | "
          f"Train MSE={epoch_train_mse[-1]:.4f} | "
          f"Val MSE={epoch_val_mse[-1]:.4f} | "
          f"R2={r2:.4f}")

    reg_results.append([depth, epoch_train_mse[-1], epoch_val_mse[-1], r2])

    plt.figure()
    plt.plot(epoch_train_mse, label="Train")
    plt.plot(epoch_val_mse, label="Validation")
    plt.legend()
    plt.savefig(os.path.join(REG_DIR, f"mse_depth_{depth}.png"))
    plt.close()


pd.DataFrame(reg_results, columns=[
    "Hidden Layers", "Train MSE", "Val MSE", "R2"
]).to_csv(os.path.join(REG_DIR, "regression_results.csv"), index=False)