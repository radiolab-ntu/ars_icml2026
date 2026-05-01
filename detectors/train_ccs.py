import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def plot_auroc_comparison(auroc_orig, auroc_proj, save_path=None):
    import re
    import numpy as np

    def extract_layer_index(layer_name):
        m = re.search(r"layer(\d+)", layer_name)
        return int(m.group(1)) if m else None

    all_names = set(list(auroc_orig.keys()) + list(auroc_proj.keys()))
    name_idx_pairs = []
    bad = []
    for name in all_names:
        idx = extract_layer_index(name)
        if idx is None:
            bad.append(name)
        else:
            name_idx_pairs.append((name, idx))
    if bad:
        print(f"[WARN] can not find the input：{bad}")

    name_idx_pairs.sort(key=lambda x: x[1])
    if not name_idx_pairs:
        print("[ERROR] can not find files")
        return

    indices = [idx for (_, idx) in name_idx_pairs]
    labels = [name for (name, _) in name_idx_pairs]
    orig_values = [auroc_orig.get(name, np.nan) for name in labels]
    proj_values = [auroc_proj.get(name, np.nan) for name in labels]


    plt.figure(figsize=(9, 5))
    x = np.arange(len(indices))  
    plt.plot(x, orig_values, marker='o', label="Original Embeddings", linewidth=2)
    plt.plot(x, proj_values, marker='s', label="Projected Embeddings", linewidth=2)

    plt.xlabel("Layer (sorted by index)")
    plt.ylabel("AUROC")
    plt.title("Layer-wise AUROC Comparison: Original vs Projected Embeddings")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    xtick_labels = [f"layer{idx}" for idx in indices]
    plt.xticks(x, xtick_labels, rotation=45, ha='right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[Saved] AUROC comparsion -> {save_path}")
    plt.show()


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random seed set to {seed}")

class CCS(nn.Module):
    def __init__(self, neg, pos, nepochs=1000, ntries=10, lr=1e-3, batch_size=-1, device="cuda",
                 linear=True, weight_decay=0.01, var_normalize=False, verbose=False):
        super().__init__()
        self.device = device
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.batch_size = batch_size if batch_size > 0 else len(neg)
        self.verbose = verbose
        self.var_normalize = var_normalize

        self.neg = torch.tensor(neg, dtype=torch.float32).to(device)
        self.pos = torch.tensor(pos, dtype=torch.float32).to(device)

        input_dim = self.neg.shape[1]
        if linear:
            self.model = nn.Linear(input_dim, 1, bias=True).to(device)
        else:
            self.model = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, 1)
            ).to(device)

        self.criterion = nn.BCEWithLogitsLoss()

    def repeated_train(self):
        best_loss = float("inf")
        best_state = None
        for t in range(self.ntries):
            if self.verbose:
                print(f"[CCS] Training try {t+1}/{self.ntries}")
            self.model.apply(self._init_weights)
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)

            for epoch in range(self.nepochs):
                idx = torch.randperm(len(self.neg))
                for start in range(0, len(self.neg), self.batch_size):
                    end = start + self.batch_size
                    batch_idx = idx[start:end]
                    neg_batch = self.neg[batch_idx]
                    pos_batch = self.pos[batch_idx]

                    x = pos_batch - neg_batch
                    y = torch.ones(len(x), 1, device=self.device)

                    if self.var_normalize:
                        x = (x - x.mean(0)) / (x.std(0) + 1e-12)

                    optimizer.zero_grad()
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                    loss.backward()
                    optimizer.step()

            final_loss = self.criterion(self.model(self.pos - self.neg), torch.ones(len(self.neg), 1, device=self.device)).item()
            if final_loss < best_loss:
                best_loss = final_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        self.model.load_state_dict(best_state)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def main(args):
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    layer_dirs = sorted([
        os.path.join(args.emb_root, d)
        for d in os.listdir(args.emb_root)
        if os.path.isdir(os.path.join(args.emb_root, d)) and d.endswith("_original_only")
    ])

    if not layer_dirs:
        print(f"[ERROR] there are no files {args.emb_root}")
        return

    y = np.load(os.path.join(args.label_dir, "label.npy"))
    
    auroc_orig = {}
    auroc_proj = {}

    for layer_dir in layer_dirs:
        layer_name = os.path.basename(layer_dir)
        print(f"\n[INFO] === Processing {layer_name} ===")

        X_orig = np.load(os.path.join(layer_dir, "embeddings.npy"))
        X_proj = np.load(os.path.join(layer_dir, "projected.npy"))
        assert len(X_orig) == len(X_proj) == len(y), f"{layer_name}: datas not match"

        layer_out_dir = os.path.join(args.output_dir, layer_name)
        os.makedirs(layer_out_dir, exist_ok=True)
        print(f"[INFO] Results will be saved to: {layer_out_dir}")

        idx = np.random.permutation(len(X_orig))
        X_orig = X_orig[idx]
        X_proj = X_proj[idx]
        y_shuffled = y[idx]

        for emb_type, X in zip(["orig", "proj"], [X_orig, X_proj]):
            print(f"[INFO] Evaluating CCS on {layer_name} - {emb_type} embeddings")

            split_idx = int(len(X) * 0.75)
            X_train, y_train = X[:split_idx], y_shuffled[:split_idx]
            X_test,  y_test  = X[split_idx:], y_shuffled[split_idx:]

            pos_train = X_train[y_train == 1]
            neg_train = X_train[y_train == 0]
            min_train = min(len(pos_train), len(neg_train))
            pos_train, neg_train = pos_train[:min_train], neg_train[:min_train]

            ccs = CCS(
                neg_train, pos_train,
                nepochs=args.nepochs,
                ntries=args.ntries,
                lr=args.lr,
                batch_size=args.ccs_batch_size,
                verbose=args.verbose,
                device=args.ccs_device,
                linear=args.linear,
                weight_decay=args.weight_decay,
                var_normalize=args.var_normalize
            )
            ccs.repeated_train()

            device = args.ccs_device

            x_test = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_test_np = y_test.copy()

            with torch.no_grad():
                logits = ccs.model(x_test).detach().cpu().numpy().flatten()
                probs = 1 / (1 + np.exp(-logits))

            ccs_auroc = roc_auc_score(y_test_np, probs)
            print(f"[CCS] AUROC={ccs_auroc:.6f}")

            if emb_type == "orig":
                auroc_orig[layer_name] = ccs_auroc
            else:
                auroc_proj[layer_name] = ccs_auroc

            result = {
                "layer": layer_name,
                "embedding_type": emb_type,
                "ccs_accuracy": float(ccs_auroc),
                "classifier": "CCS",
                "epochs": args.nepochs
            }

            result_path = os.path.join(layer_out_dir, f"results_{emb_type}.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
    
    plot_path = os.path.join(args.output_dir, "layerwise_auroc_comparison.png")
    plot_auroc_comparison(auroc_orig, auroc_proj, save_path=plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_root", type=str, required=True, help="Root directory containing per-layer *_original_only subdirectories")
    parser.add_argument("--label_dir", type=str, required=True, help="Directory containing label.npy")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--ntries", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ccs_batch_size", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ccs_device", type=str, default="cuda")
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--var_normalize", action="store_true")
    args = parser.parse_args()
    main(args)
