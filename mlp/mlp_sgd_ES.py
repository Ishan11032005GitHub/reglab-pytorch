import os
import copy
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

BASE_DIR = "results/sgd_earlystop"
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


def train_one_epoch_samplewise(model, optimizer, loss_fn, X_tensor, y_tensor):
    n = X_tensor.shape[0]
    perm = torch.randperm(n)
    total_loss = 0.0
    total_acc = 0.0
    model.train()
    for i in perm:
        xi = X_tensor[i].unsqueeze(0)
        yi = y_tensor[i].unsqueeze(0)
        optimizer.zero_grad()
        out = model(xi)
        loss = loss_fn(out, yi)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if yi.dtype == torch.long:
            total_acc += (out.argmax(1) == yi).float().item()
    avg_loss = total_loss / n
    avg_acc = total_acc / n if yi.dtype == torch.long else None
    return avg_loss, avg_acc


# CLASSIFICATION (SGD + Early Stopping)
print("\nCLASSIFICATION (SGD + Early Stopping)")

# load dataset
wine = pd.read_csv("winequality-red.csv", sep=";")
wine["quality"] = (wine["quality"] >= 6).astype(int)
X_all = wine.drop("quality", axis=1).values
y_all = wine["quality"].values

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_all, y_all, test_size=0.25, stratify=y_all, random_state=42
)

scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)

kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)

for patience in PATIENCES:
    print(f"\n--- Patience = {patience} ---")
    out_base = os.path.join(BASE_DIR, f"patience_{patience}")
    cls_dir = os.path.join(out_base, "classification")
    reg_dir = os.path.join(out_base, "regression")
    os.makedirs(cls_dir, exist_ok=True)
    os.makedirs(reg_dir, exist_ok=True)

    cls_results = []

    for depth in range(1, 5):
        epoch_tr_acc_sum = np.zeros(EPOCHS)
        epoch_va_acc_sum = np.zeros(EPOCHS)
        epoch_tr_loss_sum = np.zeros(EPOCHS)
        epoch_va_loss_sum = np.zeros(EPOCHS)

        final_test_acc = None

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_full)):
            X_tr = torch.tensor(X_train_full[tr_idx], dtype=torch.float32)
            y_tr = torch.tensor(y_train_full[tr_idx], dtype=torch.long)
            X_va = torch.tensor(X_train_full[va_idx], dtype=torch.float32)
            y_va = torch.tensor(y_train_full[va_idx], dtype=torch.long)

            model = MLP(X_all.shape[1], 2, depth)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=LR)

            fold_tr_loss = np.zeros(EPOCHS)
            fold_tr_acc = np.zeros(EPOCHS)
            fold_va_loss = np.zeros(EPOCHS)
            fold_va_acc = np.zeros(EPOCHS)

            best_val = float("inf")
            best_state = None
            best_epoch = -1
            wait = 0

            for e in range(EPOCHS):
                tr_loss_e, tr_acc_e = train_one_epoch_samplewise(model, optimizer, loss_fn, X_tr, y_tr)

                model.eval()
                with torch.no_grad():
                    vo = model(X_va)
                    va_loss_e = loss_fn(vo, y_va).item()
                    va_acc_e = accuracy_score(y_va, vo.argmax(1))

                fold_tr_loss[e] = tr_loss_e
                fold_tr_acc[e] = tr_acc_e
                fold_va_loss[e] = va_loss_e
                fold_va_acc[e] = va_acc_e

                if va_loss_e < best_val - 1e-8:
                    best_val = va_loss_e
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    best_epoch = e
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break

            if best_state is not None:
                model.load_state_dict({k: v.clone() for k, v in best_state.items()})


            last_recorded = np.argmax(fold_tr_loss != 0)
            recorded_indices = np.where(fold_tr_loss != 0)[0]
            if recorded_indices.size > 0:
                last_idx = recorded_indices[-1]
            else:
                last_idx = -1

            if best_epoch >= 0:
                fill_tr_loss = fold_tr_loss[best_epoch]
                fill_tr_acc = fold_tr_acc[best_epoch]
                fill_va_loss = fold_va_loss[best_epoch]
                fill_va_acc = fold_va_acc[best_epoch]
            else:
                fill_tr_loss = 0.0
                fill_tr_acc = 0.0
                fill_va_loss = 0.0
                fill_va_acc = 0.0


            if last_idx < EPOCHS - 1:
                fold_tr_loss[last_idx+1:] = fill_tr_loss
                fold_tr_acc[last_idx+1:] = fill_tr_acc
                fold_va_loss[last_idx+1:] = fill_va_loss
                fold_va_acc[last_idx+1:] = fill_va_acc

            epoch_tr_loss_sum += fold_tr_loss
            epoch_tr_acc_sum += fold_tr_acc
            epoch_va_loss_sum += fold_va_loss
            epoch_va_acc_sum += fold_va_acc

            model.eval()
            with torch.no_grad():
                Xtst = torch.tensor(X_test, dtype=torch.float32)
                ytst = torch.tensor(y_test, dtype=torch.long)
                out_test = model(Xtst)
                test_acc_fold = accuracy_score(ytst, out_test.argmax(1))
                final_test_acc = test_acc_fold


        epoch_tr_loss = epoch_tr_loss_sum / KFOLDS
        epoch_tr_acc = epoch_tr_acc_sum / KFOLDS
        epoch_va_loss = epoch_va_loss_sum / KFOLDS
        epoch_va_acc = epoch_va_acc_sum / KFOLDS

        print(f"[Depth={depth}] Train Acc={epoch_tr_acc[-1]:.4f} | "
              f"Val Acc={epoch_va_acc[-1]:.4f} | Test Acc={final_test_acc:.4f}")

        cls_results.append([
            depth,
            float(epoch_tr_loss[-1]),
            float(epoch_tr_acc[-1]),
            float(epoch_va_loss[-1]),
            float(epoch_va_acc[-1]),
            float(final_test_acc)
        ])


        np.save(os.path.join(cls_dir, f"train_acc_depth_{depth}.npy"), epoch_tr_acc)
        np.save(os.path.join(cls_dir, f"val_acc_depth_{depth}.npy"), epoch_va_acc)
        np.save(os.path.join(cls_dir, f"train_loss_depth_{depth}.npy"), epoch_tr_loss)
        np.save(os.path.join(cls_dir, f"val_loss_depth_{depth}.npy"), epoch_va_loss)


        plt.figure()
        plt.plot(epoch_tr_acc, label="Train")
        plt.plot(epoch_va_acc, label="Validation")
        plt.hlines(final_test_acc, 0, EPOCHS-1, colors="r", linestyles="dashed", label="Test")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title(f"SGD + EarlyStopping (pat={patience}) Classification Acc (Depth={depth})")
        plt.legend()
        plt.savefig(os.path.join(cls_dir, f"acc_depth_{depth}.png"))
        plt.close()

        plt.figure()
        plt.plot(epoch_tr_loss, label="Train")
        plt.plot(epoch_va_loss, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"SGD + EarlyStopping (pat={patience}) Classification Loss (Depth={depth})")
        plt.legend()
        plt.savefig(os.path.join(cls_dir, f"loss_depth_{depth}.png"))
        plt.close()

    cls_df = pd.DataFrame(
        cls_results,
        columns=["Hidden Layers", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Test Acc"]
    )
    cls_df.to_csv(os.path.join(cls_dir, "classification_results.csv"), index=False)



# REGRESSION (SGD + Early Stopping)
print("\nREGRESSION (SGD + Early Stopping)")

house = pd.read_csv("housing.csv")
house = pd.get_dummies(house, columns=["ocean_proximity"], drop_first=True)
X_all_reg = house.drop("median_house_value", axis=1).values
y_all_reg = house["median_house_value"].values.reshape(-1, 1)


X_all_reg = SimpleImputer(strategy="mean").fit_transform(X_all_reg)

X_train_full_r, X_test_r, y_train_full_r, y_test_r = train_test_split(
    X_all_reg, y_all_reg, test_size=0.25, random_state=42
)

scaler_r = StandardScaler()
X_train_full_r = scaler_r.fit_transform(X_train_full_r)
X_test_r = scaler_r.transform(X_test_r)

y_mean_full = y_train_full_r.mean()
y_std_full = y_train_full_r.std()
y_train_full_norm = (y_train_full_r - y_mean_full) / (y_std_full + 1e-12)
y_test_norm = (y_test_r - y_mean_full) / (y_std_full + 1e-12)

for patience in PATIENCES:
    print(f"\n--- Patience = {patience} ---")
    out_base = os.path.join(BASE_DIR, f"patience_{patience}")
    reg_dir = os.path.join(out_base, "regression")
    os.makedirs(reg_dir, exist_ok=True)

    reg_results = []

    for depth in range(1, 5):
        epoch_tr_mse_sum = np.zeros(EPOCHS)
        epoch_va_mse_sum = np.zeros(EPOCHS)
        final_r2 = None

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_full_r)):
            X_tr = torch.tensor(X_train_full_r[tr_idx], dtype=torch.float32)
            y_tr = torch.tensor(y_train_full_norm[tr_idx], dtype=torch.float32)
            X_va = torch.tensor(X_train_full_r[va_idx], dtype=torch.float32)
            y_va = torch.tensor(y_train_full_norm[va_idx], dtype=torch.float32)

            model = MLP(X_all_reg.shape[1], 1, depth)
            loss_fn = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=LR)

            fold_tr_mse = np.zeros(EPOCHS)
            fold_va_mse = np.zeros(EPOCHS)

            best_val = float("inf")
            best_state = None
            best_epoch = -1
            wait = 0

            for e in range(EPOCHS):
                n = X_tr.shape[0]
                perm = torch.randperm(n)
                epoch_loss_accum = 0.0
                model.train()
                for i in perm:
                    xi = X_tr[i].unsqueeze(0)
                    yi = y_tr[i].unsqueeze(0)
                    optimizer.zero_grad()
                    out = model(xi)
                    loss = loss_fn(out, yi)
                    loss.backward()
                    optimizer.step()
                    epoch_loss_accum += loss.item()
                tr_mse_e = epoch_loss_accum / n

                model.eval()
                with torch.no_grad():
                    va_mse_e = loss_fn(model(X_va), y_va).item()

                fold_tr_mse[e] = tr_mse_e
                fold_va_mse[e] = va_mse_e

                if va_mse_e < best_val - 1e-12:
                    best_val = va_mse_e
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    best_epoch = e
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break

            if best_state is not None:
                model.load_state_dict({k: v.clone() for k, v in best_state.items()})

            recorded_indices = np.where(fold_tr_mse != 0)[0]
            if recorded_indices.size > 0:
                last_idx = recorded_indices[-1]
            else:
                last_idx = -1

            if best_epoch >= 0:
                fill_tr = fold_tr_mse[best_epoch]
                fill_va = fold_va_mse[best_epoch]
            else:
                fill_tr = 0.0
                fill_va = 0.0

            if last_idx < EPOCHS - 1:
                fold_tr_mse[last_idx+1:] = fill_tr
                fold_va_mse[last_idx+1:] = fill_va

            epoch_tr_mse_sum += fold_tr_mse
            epoch_va_mse_sum += fold_va_mse

            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_test_r, dtype=torch.float32)).detach().cpu().numpy()
                preds = preds * (y_std_full + 1e-12) + y_mean_full
                y_true = y_test_r
                preds = preds.flatten()
                y_true = y_true.flatten()
                mask = np.isfinite(preds)

                if mask.sum() > 0:
                    r2_fold = r2_score(y_true[mask], preds[mask])
                else:
                    r2_fold = -1.0
                final_r2 = r2_fold

        epoch_tr_mse = epoch_tr_mse_sum / KFOLDS
        epoch_va_mse = epoch_va_mse_sum / KFOLDS

        print(f"[Depth={depth}] Train MSE={epoch_tr_mse[-1]:.4f} | Val MSE={epoch_va_mse[-1]:.4f} | R2={final_r2:.4f}")

        reg_results.append([depth, float(epoch_tr_mse[-1]), float(epoch_va_mse[-1]), float(final_r2)])

        np.save(os.path.join(reg_dir, f"train_mse_depth_{depth}.npy"), epoch_tr_mse)
        np.save(os.path.join(reg_dir, f"val_mse_depth_{depth}.npy"), epoch_va_mse)

        plt.figure()
        plt.plot(epoch_tr_mse, label="Train")
        plt.plot(epoch_va_mse, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.title(f"SGD + EarlyStopping (pat={patience}) Regression MSE (Depth={depth})")
        plt.legend()
        plt.savefig(os.path.join(reg_dir, f"mse_depth_{depth}.png"))
        plt.close()

        plt.figure()
        plt.plot(epoch_tr_mse, label="Train")
        plt.plot(epoch_va_mse, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"SGD + EarlyStopping (pat={patience}) Regression Loss (Depth={depth})")
        plt.legend()
        plt.savefig(os.path.join(reg_dir, f"loss_depth_{depth}.png"))
        plt.close()

    reg_df = pd.DataFrame(reg_results, columns=["Hidden Layers", "Train MSE", "Val MSE", "R2"])
    reg_df.to_csv(os.path.join(reg_dir, "regression_results.csv"), index=False)

print("\nAll SGD + Early Stopping experiments finished.")