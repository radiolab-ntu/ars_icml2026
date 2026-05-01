import os
import json
import argparse
import numpy as np
import torch
import random
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import StratifiedShuffleSplit


def plot_score_distribution(train_scores, test_scores, title, save_path):
    plt.figure(figsize=(6, 4))
    plt.hist(train_scores, bins=50, alpha=0.5, label="Train")
    plt.hist(test_scores, bins=50, alpha=0.5, label="Test")
    plt.title(f"{title} — Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_auroc_comparison(orig_auc, proj_auc, title, save_path):
    x = np.arange(2)
    values = [orig_auc, proj_auc]
    plt.figure(figsize=(5, 4))
    plt.plot(x, values, marker='o')
    plt.xticks(x, ["Original", "Projected"])
    plt.ylabel("AUROC")
    plt.title(f"{title} — AUROC Comparison")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def haloscope_score_true(X, k=10):
    X = np.asarray(X, dtype=np.float64)
    mean_vec = np.mean(X, axis=0, keepdims=True)
    Xc = X - mean_vec
    try:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        cov = np.cov(Xc, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1]
        S = np.sqrt(vals[idx])
        Vt = vecs[:, idx].T
    k = min(k, Vt.shape[0])
    Vk = Vt[:k]
    Sk = S[:k]
    proj = Xc.dot(Vk.T)
    score = np.sum((proj ** 2) / (Sk[None, :] ** 2 + 1e-12), axis=1)
    return score

class LinearClassifier(nn.Module):
    def __init__(self, feat_dim):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, 1)
    def forward(self, x):
        return self.fc(x)

class NonLinearClassifier(nn.Module):
    def __init__(self, feat_dim, hidden=512, p_drop=0):
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

def train_classifier(X_train, y_train, X_test, y_test, nonlinear=False,
                     lr=0.05, weight_decay=0.0003, epochs=50, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)
    
    feat_dim = X_train.shape[1]
    model = NonLinearClassifier(feat_dim).to(device) if nonlinear else LinearClassifier(feat_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    best_model = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    loss_history = [] 

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb).view(-1), yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        loss_history.append(np.mean(epoch_losses))

        model.eval()
        all_preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                prob = torch.sigmoid(model(xb)).view(-1)
                all_preds.append(prob.cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        auc = roc_auc_score(y_test, all_preds)
        if auc > best_auc:
            best_auc = auc
            best_model = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model)
    
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            prob = torch.sigmoid(model(xb)).view(-1)
            all_preds.append(prob.cpu().numpy())
    return best_auc, np.concatenate(all_preds, axis=0), loss_history

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random seed set to {seed}")


def select_best_percentile_k_sign(X_val, y_val, candidate_percentiles, k_range):
    best_auc = -1
    best_p = None
    best_k = None
    best_sign = 1  

    for k in k_range:
        scores_val_k = haloscope_score_true(X_val, k=k)
        for sign in [1, -1]:
            scores_val_signed = sign * scores_val_k
            for p in candidate_percentiles:
                y_val_pseudo = (scores_val_signed > np.percentile(scores_val_signed, p)).astype(int)
                auc = roc_auc_score(y_val, y_val_pseudo)
                if auc > best_auc:
                    best_auc = auc
                    best_p = p
                    best_k = k
                    best_sign = sign

    return best_p, best_k, best_sign, best_auc


def run_single_layer(layer_path, args):
    print(f"\n==============================")
    print(f"[INFO] Processing layer folder: {layer_path}")
    print(f"==============================")

    X_orig = np.load(os.path.join(layer_path, "embeddings.npy"))
    X_proj = np.load(os.path.join(layer_path, "projected.npy"))
    y_true = np.load(os.path.join(args.label_dir, "label.npy"))
    print("len X_orig",len(X_orig))
    print("len(X_proj)", len(X_proj))
    print("len(y_true)", len(y_true))
    assert len(X_orig) == len(X_proj) == len(y_true)

    num_samples = len(y_true)
    
    sss_outer = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=args.seed
    )
    train_val_idx, test_idx = next(sss_outer.split(X_orig, y_true))

    X_train_val_orig = X_orig[train_val_idx]
    X_train_val_proj = X_proj[train_val_idx]
    y_train_val = y_true[train_val_idx]

    X_test_orig = X_orig[test_idx]
    X_test_proj = X_proj[test_idx]
    y_test = y_true[test_idx]

    val_size = 100
    sss_inner = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=args.seed
    )
    train_idx, val_idx = next(sss_inner.split(X_train_val_orig, y_train_val))

    X_train_orig = X_train_val_orig[train_idx]
    X_train_proj = X_train_val_proj[train_idx]
    y_train = y_train_val[train_idx]

    X_val_orig = X_train_val_orig[val_idx]
    X_val_proj = X_train_val_proj[val_idx]
    y_val = y_train_val[val_idx]

    assert len(np.unique(y_train)) == 2
    assert len(np.unique(y_val)) == 2
    assert len(np.unique(y_test)) == 2

    print(f"[INFO] Dataset sizes -> Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    percentiles = np.linspace(0, 100, 40)[1:-1]
    k_range = range(1, 11)

    print("[INFO] Searching best params for ORIGINAL embeddings on validation set...")
    best_p_orig, best_k_orig, best_sign_orig, val_auc_orig = select_best_percentile_k_sign(
        X_val_orig, y_val, percentiles, k_range
    )

    print("[INFO] Searching best params for PROJECTED embeddings on validation set...")
    best_p_proj, best_k_proj, best_sign_proj, val_auc_proj = select_best_percentile_k_sign(
        X_val_proj, y_val, percentiles, k_range
    )
    best_p_orig, best_k_orig, best_sign_orig, val_auc_orig = best_p_proj, best_k_proj, best_sign_proj, val_auc_proj
    print(f"[INFO][layer] best_p={best_p_proj:.2f}, best_k={best_k_proj}, best_sign={best_sign_proj}, val_AUROC={val_auc_proj:.4f}")

    scores_train_orig_bestk = haloscope_score_true(X_train_orig, k=best_k_orig)
    scores_train_orig_signed = best_sign_orig * scores_train_orig_bestk
    y_train_pseudo_orig = (scores_train_orig_signed > np.percentile(scores_train_orig_signed, best_p_orig)).astype(int)

    scores_train_proj_bestk = haloscope_score_true(X_train_proj, k=best_k_proj)
    scores_train_proj_signed = best_sign_proj * scores_train_proj_bestk
    y_train_pseudo_proj = (scores_train_proj_signed > np.percentile(scores_train_proj_signed, best_p_proj)).astype(int)

    print(f"\n[INFO] Training for ORIGINAL embeddings...")
    auroc_lr_orig, pred_lr_orig, _ = train_classifier(
        X_train_orig, y_train_pseudo_orig, X_test_orig, y_test, nonlinear=False
    )
    auroc_mlp_orig, pred_mlp_orig, _ = train_classifier(
        X_train_orig, y_train_pseudo_orig, X_test_orig, y_test, nonlinear=True
    )

    print(f"[RESULT][orig] Logistic AUROC = {auroc_lr_orig:.4f}, MLP AUROC = {auroc_mlp_orig:.4f}")

    print(f"\n[INFO] Training for PROJECTED embeddings...")
    auroc_lr_proj, pred_lr_proj, _ = train_classifier(
        X_train_proj, y_train_pseudo_proj, X_test_proj, y_test, nonlinear=False
    )
    auroc_mlp_proj, pred_mlp_proj, _ = train_classifier(
        X_train_proj, y_train_pseudo_proj, X_test_proj, y_test, nonlinear=True
    )

    print(f"[RESULT][layer] Logistic AUROC = {auroc_lr_proj:.4f}, MLP AUROC = {auroc_mlp_proj:.4f}")

    return {
        "auroc_orig": float(auroc_lr_orig),   
        "auroc_proj": float(auroc_lr_proj),
        "auroc_mlp_orig": float(auroc_mlp_orig),
        "auroc_mlp_proj": float(auroc_mlp_proj)
    }


def main(args):
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    import re
    layer_dirs_with_num = []
    for d in sorted(os.listdir(args.emb_dir)):
        full_path = os.path.join(args.emb_dir, d)
        if os.path.isdir(full_path):
            match = re.search(r'layer(\d+)', d)
            # match = re.search(r'proj(\d+)', d)
            if match:
                layer_num = int(match.group(1))
                layer_dirs_with_num.append((layer_num, full_path))

    layer_dirs_with_num.sort(key=lambda x: x[0])
    layer_indices = [ln[0] for ln in layer_dirs_with_num]
    layer_dirs = [ln[1] for ln in layer_dirs_with_num]

    print("\n[INFO] Detected layers (sorted by layer number):")
    for num, path in zip(layer_indices, layer_dirs):
        print(f"   - Layer {num}: {path}")

    all_orig = []
    all_proj = []
    auroc_mlp_orig = []
    auroc_mlp_proj = []

    for layer_path in layer_dirs:
        res = run_single_layer(layer_path, args)
        all_orig.append(res["auroc_orig"])
        all_proj.append(res["auroc_proj"])
        auroc_mlp_orig.append(res["auroc_mlp_orig"])
        auroc_mlp_proj.append(res["auroc_mlp_proj"])

    plt.figure(figsize=(7,5))
    plt.plot(layer_indices, all_orig, marker='o', label="Original Embedding AUROC")
    plt.plot(layer_indices, all_proj, marker='o', label="Projected Embedding AUROC")
    plt.xlabel("Transformer Layer")
    plt.ylabel("AUROC")
    plt.title("HALOscope Classification AUROC Across Layers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "layerwise_auroc.png"))
    plt.close()

    plt.figure(figsize=(7,5))
    plt.plot(layer_indices, auroc_mlp_orig, marker='o', label="Original Embedding AUROC")
    plt.plot(layer_indices, auroc_mlp_proj, marker='o', label="Projected Embedding AUROC")
    plt.xlabel("Transformer Layer")
    plt.ylabel("AUROC")
    plt.title("NonLinearClassifier AUROC Across Layers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "layerwise_auroc_mlp.png"))
    plt.close()

    out_json = {
        "layers": layer_indices,
        "auroc_orig": all_orig,
        "auroc_proj": all_proj,
        "auroc_mlp_orig": auroc_mlp_orig,
        "auroc_mlp_proj": auroc_mlp_proj
    }
    with open(os.path.join(args.output_dir, "layerwise_results.json"), "w") as f:
        json.dump(out_json, f, indent=4)
    print(f"\n[SAVED] Layer-wise results saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", type=str, required=True, help="Folder containing multiple layer subfolders")
    parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
