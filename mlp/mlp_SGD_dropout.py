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
DROPOUT_RATES = [0.1, 0.2, 0.3, 0.5]

BASE_DIR = "results/sgd_dropout"
os.makedirs(BASE_DIR, exist_ok=True)



class MLP_Dropout(nn.Module):
    def __init__(self, input_dim, output_dim, depth, dropout_rate=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev, HIDDEN_NEURONS))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev = HIDDEN_NEURONS
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


wine = pd.read_csv("winequality-red.csv", sep=";")
wine["quality"] = (wine["quality"] >= 6).astype(int)
X_cls_all = wine.drop("quality", axis=1).values
y_cls_all = wine["quality"].values

house = pd.read_csv("housing.csv")
house = pd.get_dummies(house, columns=["ocean_proximity"], drop_first=True)
X_reg_all = house.drop("median_house_value", axis=1).values
y_reg_all = house["median_house_value"].values.reshape(-1, 1)


X_reg_all = SimpleImputer(strategy="mean").fit_transform(X_reg_all)

kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)

for dr in DROPOUT_RATES:
    print(f"\n=== DROPOUT RATE = {dr} ===")
    out_dir = os.path.join(BASE_DIR, f"dropout_{dr}")
    cls_dir = os.path.join(out_dir, "classification")
    reg_dir = os.path.join(out_dir, "regression")
    os.makedirs(cls_dir, exist_ok=True)
    os.makedirs(reg_dir, exist_ok=True)


    # CLASSIFICATION (SGD + Dropout)
    print("CLASSIFICATION (SGD + Dropout)")

    X_train, X_test, y_train, y_test = train_test_split(
        X_cls_all, y_cls_all, test_size=0.25, stratify=y_cls_all, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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

            model = MLP_Dropout(X_cls_all.shape[1], 2, depth, dropout_rate=dr)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=LR)

            n_samples = Xt.shape[0]

            for e in range(EPOCHS):
                perm = torch.randperm(n_samples)
                train_loss_epoch = 0.0
                train_acc_epoch = 0.0

                model.train()
                for i in perm:
                    xi = Xt[i].unsqueeze(0)
                    yi = yt[i].unsqueeze(0)
                    optimizer.zero_grad()
                    out = model(xi)
                    loss = loss_fn(out, yi)
                    loss.backward()
                    optimizer.step()

                    train_loss_epoch += loss.item()
                    train_acc_epoch += (out.argmax(1) == yi).float().item()

                epoch_tr_loss[e] += (train_loss_epoch / n_samples)
                epoch_tr_acc[e] += (train_acc_epoch / n_samples)

                with torch.no_grad():
                    model.eval()
                    vo = model(Xv)
                    epoch_va_loss[e] += loss_fn(vo, yv).item()
                    epoch_va_acc[e] += accuracy_score(yv, vo.argmax(1))


        epoch_tr_acc /= KFOLDS
        epoch_va_acc /= KFOLDS
        epoch_tr_loss /= KFOLDS
        epoch_va_loss /= KFOLDS

        with torch.no_grad():
            model.eval()
            Xtst = torch.tensor(X_test, dtype=torch.float32)
            ytst = torch.tensor(y_test, dtype=torch.long)
            out_test = model(Xtst)
            test_acc = accuracy_score(ytst, out_test.argmax(1))

        print(f"[Depth={depth}] Train Acc={epoch_tr_acc[-1]:.4f} | "
              f"Val Acc={epoch_va_acc[-1]:.4f} | Test Acc={test_acc:.4f}")

        cls_results.append([
            depth,
            float(epoch_tr_loss[-1]),
            float(epoch_tr_acc[-1]),
            float(epoch_va_loss[-1]),
            float(epoch_va_acc[-1]),
            float(test_acc)
        ])


        np.save(os.path.join(cls_dir, f"train_acc_depth_{depth}.npy"), epoch_tr_acc)
        np.save(os.path.join(cls_dir, f"val_acc_depth_{depth}.npy"), epoch_va_acc)
        np.save(os.path.join(cls_dir, f"train_loss_depth_{depth}.npy"), epoch_tr_loss)
        np.save(os.path.join(cls_dir, f"val_loss_depth_{depth}.npy"), epoch_va_loss)

        plt.figure()
        plt.plot(epoch_tr_acc, label="Train")
        plt.plot(epoch_va_acc, label="Validation")
        plt.hlines(test_acc, 0, EPOCHS-1, colors="r", linestyles="dashed", label="Test")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title(f"SGD + Dropout={dr} Classification Accuracy (Depth={depth})")
        plt.legend()
        plt.savefig(os.path.join(cls_dir, f"acc_depth_{depth}.png"))
        plt.close()

        plt.figure()
        plt.plot(epoch_tr_loss, label="Train")
        plt.plot(epoch_va_loss, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"SGD + Dropout={dr} Classification Loss (Depth={depth})")
        plt.legend()
        plt.savefig(os.path.join(cls_dir, f"loss_depth_{depth}.png"))
        plt.close()

    pd.DataFrame(
        cls_results,
        columns=["Hidden Layers", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Test Acc"]
    ).to_csv(os.path.join(cls_dir, "classification_results.csv"), index=False)


    # REGRESSION (SGD + Dropout)
    print("REGRESSION (SGD + Dropout)")


    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg_all, y_reg_all, test_size=0.25, random_state=42
    )

    y_mean, y_std = y_train_r.mean(), y_train_r.std()
    y_train_norm = (y_train_r - y_mean) / (y_std + 1e-12)
    y_test_norm = (y_test_r - y_mean) / (y_std + 1e-12)

    scaler_r = StandardScaler()
    X_train_r = scaler_r.fit_transform(X_train_r)
    X_test_r = scaler_r.transform(X_test_r)

    reg_results = []

    for depth in range(1, 5):
        epoch_tr_mse = np.zeros(EPOCHS)
        epoch_va_mse = np.zeros(EPOCHS)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_r)):
            Xt = torch.tensor(X_train_r[tr_idx], dtype=torch.float32)
            yt = torch.tensor(y_train_norm[tr_idx], dtype=torch.float32)
            Xv = torch.tensor(X_train_r[va_idx], dtype=torch.float32)
            yv = torch.tensor(y_train_norm[va_idx], dtype=torch.float32)

            model = MLP_Dropout(X_reg_all.shape[1], 1, depth, dropout_rate=dr)
            loss_fn = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=LR)

            n_samples = Xt.shape[0]

            for e in range(EPOCHS):
                perm = torch.randperm(n_samples)
                epoch_loss_accum = 0.0

                model.train()
                for i in perm:
                    xi = Xt[i].unsqueeze(0)
                    yi = yt[i].unsqueeze(0)
                    optimizer.zero_grad()
                    out = model(xi)
                    loss = loss_fn(out, yi)
                    loss.backward()
                    optimizer.step()
                    epoch_loss_accum += loss.item()

                epoch_tr_mse[e] += (epoch_loss_accum / n_samples)

                with torch.no_grad():
                    model.eval()
                    epoch_va_mse[e] += loss_fn(model(Xv), yv).item()

        epoch_tr_mse /= KFOLDS
        epoch_va_mse /= KFOLDS


        with torch.no_grad():
            model.eval()
            preds = model(torch.tensor(X_test_r, dtype=torch.float32)).detach().cpu().numpy()
            preds = preds * (y_std + 1e-12) + y_mean
            y_true = y_test_r
            r2 = r2_score(y_true, preds)

        print(f"[Depth={depth}] Train MSE={epoch_tr_mse[-1]:.4f} | "
              f"Val MSE={epoch_va_mse[-1]:.4f} | R2={r2:.4f}")

        reg_results.append([depth, float(epoch_tr_mse[-1]), float(epoch_va_mse[-1]), float(r2)])


        np.save(os.path.join(reg_dir, f"train_mse_depth_{depth}.npy"), epoch_tr_mse)
        np.save(os.path.join(reg_dir, f"val_mse_depth_{depth}.npy"), epoch_va_mse)

        plt.figure()
        plt.plot(epoch_tr_mse, label="Train")
        plt.plot(epoch_va_mse, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.title(f"SGD + Dropout={dr} Regression MSE (Depth={depth})")
        plt.legend()
        plt.savefig(os.path.join(reg_dir, f"mse_depth_{depth}.png"))
        plt.close()

        plt.figure()
        plt.plot(epoch_tr_mse, label="Train")
        plt.plot(epoch_va_mse, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"SGD + Dropout={dr} Regression Loss (Depth={depth})")
        plt.legend()
        plt.savefig(os.path.join(reg_dir, f"loss_depth_{depth}.png"))
        plt.close()

    pd.DataFrame(
        reg_results,
        columns=["Hidden Layers", "Train MSE", "Val MSE", "R2"]
    ).to_csv(os.path.join(reg_dir, "regression_results.csv"), index=False)

    print(f"Saved results for dropout={dr} in {out_dir}")

print("\nAll SGD + Dropout experiments finished.")