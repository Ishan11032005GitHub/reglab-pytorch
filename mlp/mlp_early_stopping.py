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
LR = 0.001
PATIENCES = [5, 10, 15]

BASE_DIR = "results/early_stopping"
os.makedirs(BASE_DIR, exist_ok=True)



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



print("\nCLASSIFICATION (Early Stopping)")

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

for patience in PATIENCES:
    print(f"\n--- Patience = {patience} ---")
    cls_dir = f"{BASE_DIR}/classification/patience_{patience}"
    os.makedirs(cls_dir, exist_ok=True)

    cls_results = []

    for depth in range(1, 5):
        fold_tr_acc, fold_va_acc = [], []
        fold_tr_loss, fold_va_loss = [], []

        max_len = 0

        for tr_idx, va_idx in kf.split(X_train):
            Xt = torch.tensor(X_train[tr_idx], dtype=torch.float32)
            yt = torch.tensor(y_train[tr_idx], dtype=torch.long)
            Xv = torch.tensor(X_train[va_idx], dtype=torch.float32)
            yv = torch.tensor(y_train[va_idx], dtype=torch.long)

            model = MLP(X.shape[1], 2, depth)
            loss_fn = nn.CrossEntropyLoss()
            opt = optim.Adam(model.parameters(), lr=LR)

            best_val = float("inf")
            wait = 0
            best_state = None

            tr_acc, va_acc = [], []
            tr_loss, va_loss = [], []

            for e in range(EPOCHS):
                opt.zero_grad()
                out = model(Xt)
                loss = loss_fn(out, yt)
                loss.backward()
                opt.step()

                tr_loss.append(loss.item())
                tr_acc.append((out.argmax(1) == yt).float().mean().item())

                with torch.no_grad():
                    vo = model(Xv)
                    vloss = loss_fn(vo, yv).item()
                    va_loss.append(vloss)
                    va_acc.append(accuracy_score(yv, vo.argmax(1)))

                if vloss < best_val:
                    best_val = vloss
                    wait = 0
                    best_state = model.state_dict()
                else:
                    wait += 1
                    if wait >= patience:
                        break

            model.load_state_dict(best_state)

            fold_tr_acc.append(tr_acc)
            fold_va_acc.append(va_acc)
            fold_tr_loss.append(tr_loss)
            fold_va_loss.append(va_loss)

            max_len = max(max_len, len(tr_acc))


        def avg_curves(curves):
            out = np.zeros(max_len)
            for c in curves:
                out[:len(c)] += c
            return out / KFOLDS

        epoch_tr_acc = avg_curves(fold_tr_acc)
        epoch_va_acc = avg_curves(fold_va_acc)
        epoch_tr_loss = avg_curves(fold_tr_loss)
        epoch_va_loss = avg_curves(fold_va_loss)

        np.save(f"{cls_dir}/train_acc_depth_{depth}.npy", epoch_tr_acc)
        np.save(f"{cls_dir}/val_acc_depth_{depth}.npy", epoch_va_acc)
        np.save(f"{cls_dir}/train_loss_depth_{depth}.npy", epoch_tr_loss)
        np.save(f"{cls_dir}/val_loss_depth_{depth}.npy", epoch_va_loss)


        plt.figure()
        plt.plot(epoch_tr_acc, label="Train")
        plt.plot(epoch_va_acc, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title(f"Classification Accuracy (Depth={depth}, Patience={patience})")
        plt.legend()
        plt.savefig(f"{cls_dir}/acc_depth_{depth}.png")
        plt.close()

        plt.figure()
        plt.plot(epoch_tr_loss, label="Train")
        plt.plot(epoch_va_loss, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Classification Loss (Depth={depth}, Patience={patience})")
        plt.legend()
        plt.savefig(f"{cls_dir}/loss_depth_{depth}.png")
        plt.close()

        test_acc = accuracy_score(
            y_test,
            model(torch.tensor(X_test, dtype=torch.float32)).argmax(1)
        )

        print(f"[Depth={depth}] Train Acc={epoch_tr_acc[-1]:.4f} | "
              f"Val Acc={epoch_va_acc[-1]:.4f} | Test Acc={test_acc:.4f}")

        cls_results.append([
            depth,
            epoch_tr_loss[-1],
            epoch_tr_acc[-1],
            epoch_va_loss[-1],
            epoch_va_acc[-1],
            test_acc
        ])

    pd.DataFrame(
        cls_results,
        columns=["Hidden Layers", "Train Loss", "Train Acc",
                 "Val Loss", "Val Acc", "Test Acc"]
    ).to_csv(f"{cls_dir}/classification_results.csv", index=False)



print("\nREGRESSION (Early Stopping)")

house = pd.read_csv("housing.csv")
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

for patience in PATIENCES:
    print(f"\n--- Patience = {patience} ---")
    reg_dir = f"{BASE_DIR}/regression/patience_{patience}"
    os.makedirs(reg_dir, exist_ok=True)

    reg_results = []

    for depth in range(1, 5):
        fold_tr, fold_va = [], []
        max_len = 0

        for tr_idx, va_idx in kf.split(X_train):
            Xt = torch.tensor(X_train[tr_idx], dtype=torch.float32)
            yt = torch.tensor(y_train[tr_idx], dtype=torch.float32)
            Xv = torch.tensor(X_train[va_idx], dtype=torch.float32)
            yv = torch.tensor(y_train[va_idx], dtype=torch.float32)

            model = MLP(X.shape[1], 1, depth)
            loss_fn = nn.MSELoss()
            opt = optim.Adam(model.parameters(), lr=LR)

            best_val = float("inf")
            wait = 0
            best_state = None

            tr_mse, va_mse = [], []

            for e in range(EPOCHS):
                opt.zero_grad()
                out = model(Xt)
                loss = loss_fn(out, yt)
                loss.backward()
                opt.step()

                tr_mse.append(loss.item())

                with torch.no_grad():
                    vloss = loss_fn(model(Xv), yv).item()
                    va_mse.append(vloss)

                if vloss < best_val:
                    best_val = vloss
                    wait = 0
                    best_state = model.state_dict()
                else:
                    wait += 1
                    if wait >= patience:
                        break

            model.load_state_dict(best_state)
            fold_tr.append(tr_mse)
            fold_va.append(va_mse)
            max_len = max(max_len, len(tr_mse))

        def avg(curves):
            out = np.zeros(max_len)
            for c in curves:
                out[:len(c)] += c
            return out / KFOLDS

        epoch_tr_mse = avg(fold_tr)
        epoch_va_mse = avg(fold_va)

        np.save(f"{reg_dir}/train_mse_depth_{depth}.npy", epoch_tr_mse)
        np.save(f"{reg_dir}/val_mse_depth_{depth}.npy", epoch_va_mse)

        plt.figure()
        plt.plot(epoch_tr_mse, label="Train")
        plt.plot(epoch_va_mse, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.title(f"Regression MSE (Depth={depth}, Patience={patience})")
        plt.legend()
        plt.savefig(f"{reg_dir}/mse_depth_{depth}.png")
        plt.close()

        preds = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
        preds = preds * y_std + y_mean
        y_true = y_test * y_std + y_mean
        r2 = r2_score(y_true, preds)

        print(f"[Depth={depth}] Train MSE={epoch_tr_mse[-1]:.4f} | "
              f"Val MSE={epoch_va_mse[-1]:.4f} | R2={r2:.4f}")

        reg_results.append([depth, epoch_tr_mse[-1], epoch_va_mse[-1], r2])

    pd.DataFrame(
        reg_results,
        columns=["Hidden Layers", "Train MSE", "Val MSE", "R2"]
    ).to_csv(f"{reg_dir}/regression_results.csv", index=False)