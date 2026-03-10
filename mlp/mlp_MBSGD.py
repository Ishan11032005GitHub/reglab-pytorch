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
from torch.utils.data import TensorDataset, DataLoader


torch.manual_seed(42)
np.random.seed(42)

EPOCHS = 250
HIDDEN_NEURONS = 128
KFOLDS = 5
LR = 0.001
BATCH_SIZE = 32
BASE_DIR = "results/mbsgd"
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


# CLASSIFICATION (Mini-batch SGD)
print("\nCLASSIFICATION (Mini-batch SGD)")

wine = pd.read_csv("winequality-red.csv", sep=";")
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

        train_ds = TensorDataset(Xt, yt)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        model = MLP(X.shape[1], 2, depth)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LR)


        fold_tr_loss = np.zeros(EPOCHS)
        fold_tr_acc = np.zeros(EPOCHS)
        fold_va_loss = np.zeros(EPOCHS)
        fold_va_acc = np.zeros(EPOCHS)

        for e in range(EPOCHS):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for xb, yb in train_loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * xb.shape[0]
                preds = out.argmax(1)
                correct += (preds == yb).sum().item()
                total += xb.shape[0]

            fold_tr_loss[e] = total_loss / total
            fold_tr_acc[e] = correct / total

            model.eval()
            with torch.no_grad():
                vo = model(Xv)
                fold_va_loss[e] = loss_fn(vo, yv).item()
                fold_va_acc[e] = accuracy_score(yv, vo.argmax(1))

        epoch_tr_loss += fold_tr_loss
        epoch_tr_acc += fold_tr_acc
        epoch_va_loss += fold_va_loss
        epoch_va_acc += fold_va_acc


    epoch_tr_loss /= KFOLDS
    epoch_tr_acc /= KFOLDS
    epoch_va_loss /= KFOLDS
    epoch_va_acc /= KFOLDS

    with torch.no_grad():
        Xtst = torch.tensor(X_test, dtype=torch.float32)
        ytst = torch.tensor(y_test, dtype=torch.long)
        out = model(Xtst)
        test_acc = accuracy_score(ytst, out.argmax(1))

    print(f"[Classification] Depth={depth} | "
          f"Train Acc={epoch_tr_acc[-1]:.4f} | "
          f"Val Acc={epoch_va_acc[-1]:.4f} | "
          f"Test Acc={test_acc:.4f}")

    cls_results.append([
        depth,
        float(epoch_tr_loss[-1]),
        float(epoch_tr_acc[-1]),
        float(epoch_va_loss[-1]),
        float(epoch_va_acc[-1]),
        float(test_acc)
    ])

    np.save(os.path.join(BASE_DIR, "classification", f"train_acc_depth_{depth}.npy"), epoch_tr_acc)
    np.save(os.path.join(BASE_DIR, "classification", f"val_acc_depth_{depth}.npy"), epoch_va_acc)
    np.save(os.path.join(BASE_DIR, "classification", f"train_loss_depth_{depth}.npy"), epoch_tr_loss)
    np.save(os.path.join(BASE_DIR, "classification", f"val_loss_depth_{depth}.npy"), epoch_va_loss)

    plt.figure()
    plt.plot(epoch_tr_acc, label="Train")
    plt.plot(epoch_va_acc, label="Validation")
    plt.hlines(test_acc, 0, EPOCHS-1, colors="r", linestyles="dashed", label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Mini-batch SGD Classification Accuracy (Depth={depth})")
    plt.legend()
    plt.savefig(os.path.join(BASE_DIR, "classification", f"acc_depth_{depth}.png"))
    plt.close()

    plt.figure()
    plt.plot(epoch_tr_loss, label="Train")
    plt.plot(epoch_va_loss, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Mini-batch SGD Classification Loss (Depth={depth})")
    plt.legend()
    plt.savefig(os.path.join(BASE_DIR, "classification", f"loss_depth_{depth}.png"))
    plt.close()

cls_df = pd.DataFrame(
    cls_results,
    columns=["Hidden Layers", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Test Acc"]
)
cls_df.to_csv(os.path.join(BASE_DIR, "classification", "classification_results.csv"), index=False)



# REGRESSION (Mini-batch SGD)
print("\nREGRESSION (Mini-batch SGD)")

house = pd.read_csv("housing.csv")
house = pd.get_dummies(house, columns=["ocean_proximity"], drop_first=True)

X = house.drop("median_house_value", axis=1).values
y = house["median_house_value"].values.reshape(-1, 1)

X = SimpleImputer(strategy="mean").fit_transform(X)

y_mean, y_std = y.mean(), y.std()
y_norm = (y - y_mean) / (y_std + 1e-12)

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

        train_ds = TensorDataset(Xt, yt)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        model = MLP(X.shape[1], 1, depth)
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=LR)

        fold_tr_mse = np.zeros(EPOCHS)
        fold_va_mse = np.zeros(EPOCHS)

        for e in range(EPOCHS):
            model.train()
            total_loss = 0.0
            total_samples = 0

            for xb, yb in train_loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.shape[0]
                total_samples += xb.shape[0]

            fold_tr_mse[e] = total_loss / total_samples

            model.eval()
            with torch.no_grad():
                fold_va_mse[e] = loss_fn(model(Xv), yv).item()

        epoch_tr_mse += fold_tr_mse
        epoch_va_mse += fold_va_mse

    epoch_tr_mse /= KFOLDS
    epoch_va_mse /= KFOLDS

    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).detach().cpu().numpy()
        preds = preds * (y_std + 1e-12) + y_mean
        y_true = y_test * (y_std + 1e-12) + y_mean
        r2 = r2_score(y_true, preds)

    print(f"[Regression] Depth={depth} | Train MSE={epoch_tr_mse[-1]:.4f} | "
          f"Val MSE={epoch_va_mse[-1]:.4f} | R2={r2:.4f}")

    reg_results.append([depth, float(epoch_tr_mse[-1]), float(epoch_va_mse[-1]), float(r2)])

    np.save(os.path.join(BASE_DIR, "regression", f"train_mse_depth_{depth}.npy"), epoch_tr_mse)
    np.save(os.path.join(BASE_DIR, "regression", f"val_mse_depth_{depth}.npy"), epoch_va_mse)

    plt.figure()
    plt.plot(epoch_tr_mse, label="Train")
    plt.plot(epoch_va_mse, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title(f"Mini-batch SGD Regression MSE (Depth={depth})")
    plt.legend()
    plt.savefig(os.path.join(BASE_DIR, "regression", f"mse_depth_{depth}.png"))
    plt.close()

    plt.figure()
    plt.plot(epoch_tr_mse, label="Train")
    plt.plot(epoch_va_mse, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Mini-batch SGD Regression Loss (Depth={depth})")
    plt.legend()
    plt.savefig(os.path.join(BASE_DIR, "regression", f"loss_depth_{depth}.png"))
    plt.close()

reg_df = pd.DataFrame(
    reg_results,
    columns=["Hidden Layers", "Train MSE", "Val MSE", "R2"]
)
reg_df.to_csv(os.path.join(BASE_DIR, "regression", "regression_results.csv"), index=False)

print("\nAll mini-batch SGD experiments finished. Results saved in:", BASE_DIR)