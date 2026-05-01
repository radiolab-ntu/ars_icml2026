import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import StratifiedShuffleSplit


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random seed set to {seed}")


class LinearClassifier(nn.Module):
    def __init__(self, feat_dim):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, 1)
    def forward(self, features):
        return self.fc(features)

class NonLinearClassifier(nn.Module):
    def __init__(self, feat_dim, hidden=512, p_drop=0.4):
        super(NonLinearClassifier, self).__init__()
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden, momentum=0.05)
        self.drop = nn.Dropout(p_drop)
        self.fc3 = nn.Linear(hidden, 1)
    def forward(self, features):
        if self.training:
            features = features + 0.008 * torch.randn_like(features)
        x = self.fc1(features)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop(x)
        return self.fc3(x)

class TorchSklearnWrapper:
    def __init__(self, model, device=None, lr=1e-2, batch_size=128, weight_decay=5e-2):
        self.model = model
        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9, weight_decay=1e-2)
        self.batch_size = batch_size

    def fit(self, X, y):
        self.model.train()
        X_tensor = torch.from_numpy(np.asarray(X)).float()
        y_tensor = torch.from_numpy(np.asarray(y)).float()
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(features).view(-1)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.from_numpy(np.asarray(X)).float().to(self.device)
        loader = DataLoader(X_tensor, batch_size=self.batch_size, shuffle=False)
        probs = []
        with torch.no_grad():
            for features in loader:
                logits = self.model(features).view(-1)
                prob = torch.sigmoid(logits).cpu().numpy()
                probs.extend(prob.tolist())
        probs = np.asarray(probs)
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)

def build_classifier_wrapper(feat_dim, classifier_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if classifier_name.upper() == "MLP":
        net = NonLinearClassifier(feat_dim)
        return TorchSklearnWrapper(net, device=device, lr=1e-2)
    else:
        net = LinearClassifier(feat_dim)
        return TorchSklearnWrapper(net, device=device, lr=1e-2)

def plot_training_curve(loss_curve, val_curve, classifier_name, save_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(loss_curve, label="Train Loss")
    if val_curve:
        plt.plot(val_curve, label="Val Loss", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Curve ({classifier_name})")
    plt.legend()
    save_dir = os.path.join(save_dir, "curves", classifier_name)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"classifier_curve.png"))
    plt.close()


def main(args):
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    layer_dirs = sorted([
        os.path.join(args.emb_root, d)
        for d in os.listdir(args.emb_root)
        if os.path.isdir(os.path.join(args.emb_root, d)) and d.endswith("_original_only")
    ])

    if not layer_dirs:
        print(f"[ERROR] can not find input {args.emb_root}")
        return

    # 载入标签
    y_all = np.load(os.path.join(args.label_dir, "label.npy"))

    for layer_dir in layer_dirs:
        layer_name = os.path.basename(layer_dir)
        print(f"\n[INFO] === Processing {layer_name} ===")

        X_orig = np.load(os.path.join(layer_dir, "embeddings.npy"))
        X_proj = np.load(os.path.join(layer_dir, "projected.npy"))
        y = y_all.copy() 
        assert len(X_orig) == len(X_proj) == len(y), f"{layer_name}: data does not match"

        layer_out_dir = os.path.join(args.output_dir, layer_name)
        os.makedirs(layer_out_dir, exist_ok=True)
        print(f"[INFO] Results will be saved to: {layer_out_dir}")

        idx = np.arange(len(y))
        np.random.shuffle(idx)

        X_orig = X_orig[idx]
        X_proj = X_proj[idx]
        y = y[idx]

        for emb_type, X in zip(["orig", "proj"], [X_orig, X_proj]):
            print(f"[INFO] Training classifier on {layer_name} - {emb_type} embeddings")

            split_idx = int(len(X) * 0.75)
            val_idx = split_idx - 100

            X_train, y_train = X[:val_idx], y[:val_idx]
            X_val, y_val     = X[val_idx:split_idx], y[val_idx:split_idx]
            X_test, y_test   = X[split_idx:], y[split_idx:]


            clf = build_classifier_wrapper(X.shape[1], "MLP")
            optimizer = clf.optimizer
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-4
            )

            loss_curve, val_curve = [], []
            for epoch in range(args.max_epochs):
                clf.fit(X_train, y_train)
                y_train_prob = clf.predict_proba(X_train)[:, 1]
                y_val_prob = clf.predict_proba(X_val)[:, 1]

                train_loss = log_loss(y_train, y_train_prob)
                val_loss = log_loss(y_val, y_val_prob)
                loss_curve.append(train_loss)
                val_curve.append(val_loss)
                scheduler.step(val_loss)
                print(f"[Epoch {epoch+1}] {emb_type} train={train_loss:.6f}, val={val_loss:.6f}, lr={optimizer.param_groups[0]['lr']:.2e}")

            y_test_prob = clf.predict_proba(X_test)[:, 1]
            y_test_pred = (y_test_prob > 0.5).astype(int)
            test_acc = accuracy_score(y_test, y_test_pred)
            test_auroc = roc_auc_score(y_test, y_test_prob)

            result = {
                "layer": layer_name,
                "embedding_type": emb_type,
                "test_acc": test_acc,
                "test_auroc": test_auroc,
                "val_loss": val_loss,
                "classifier": "MLP",
                "epochs": len(loss_curve),
            }

            result_path = os.path.join(layer_out_dir, f"results_{emb_type}.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            plot_training_curve(loss_curve, val_curve, f"MLP_{emb_type}", layer_out_dir)

            if emb_type == "orig":
                extra_path = os.path.join(args.output_dir, "original")
            else:
                extra_path = os.path.join(args.output_dir, "projected")
            os.makedirs(extra_path, exist_ok=True)

            curve_filename = f"layer_{layer_name}_MLP_{emb_type}_curve.png"
            plt.figure(figsize=(6, 4))
            plt.plot(loss_curve, label="Train Loss")
            if val_curve:
                plt.plot(val_curve, label="Val Loss", linestyle="--")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Training Curve ({emb_type})")
            plt.legend()
            plt.savefig(os.path.join(extra_path, curve_filename))
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_root", type=str, required=True, help="Root directory containing per-layer *_original_only subdirectories")
    parser.add_argument("--label_dir", type=str, required=True, help="Directory containing label.npy")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save classifier results and plots")
    parser.add_argument("--classifier", type=str, choices=["LR", "MLP"], default="MLP")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=100)
    args = parser.parse_args()
    main(args)
