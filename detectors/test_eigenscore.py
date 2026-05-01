import os
import json
import argparse
import numpy as np
import torch
import random
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import re


def plot_auroc_layers(layers, auroc_orig_list, auroc_proj_list, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(layers, auroc_orig_list, marker='o', label='Original')
    plt.plot(layers, auroc_proj_list, marker='o', label='Projected')
    plt.xlabel("Layer")
    plt.ylabel("AUROC")
    plt.title("EigenScore AUROC across Transformer Layers")
    plt.xticks(layers)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compute_eigenscore_multi(X_list, alpha=1e-3):
    scores = []

    for X in X_list:
        X = np.asarray(X, dtype=np.float64)
        G, D = X.shape

        if G == 1:
            scores.append(0.0)
            continue

        Xc = X - X.mean(axis=0, keepdims=True)

        Gram = (Xc @ Xc.T) / (G - 1)

        Gram = Gram + alpha * np.eye(G)

        try:
            s = np.linalg.eigvalsh(Gram)
        except np.linalg.LinAlgError:
            s = np.linalg.svd(Gram, compute_uv=False)

        s = np.clip(s, a_min=1e-12, a_max=None)
        scores.append(np.mean(np.log10(s)))

    return np.array(scores)


def compute_auroc(scores, labels):
    return roc_auc_score(labels, -scores)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random seed set to {seed}")


def main(args):
    print("[INFO] Entered main()", flush=True)
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    layer_dirs = []
    for d in os.listdir(args.emb_dir):
        full_path = os.path.join(args.emb_dir, d)
        if os.path.isdir(full_path) and re.search(r'^layer\d+', d):
            layer_dirs.append(full_path)


    layer_dirs.sort(key=lambda x: int(re.search(r'layer(\d+)', x).group(1)))

    print("Found layer_dirs:", layer_dirs)

    y_true = np.load(os.path.join(args.label_dir, "label.npy"))

    auroc_orig_list = []
    auroc_proj_list = []
    layers = []
    
    num_sample = len(y_true) 
    idx = np.arange(num_sample)
    np.random.shuffle(idx)
    split = int(0.75 * num_sample)
    test_idx = idx[split:] 


    for layer_dir in layer_dirs:
        layer_num = int(re.search(r'layer(\d+)', layer_dir).group(1))
        layers.append(layer_num)

        X_orig_all = np.load(os.path.join(layer_dir, "embeddings.npy"))
        X_proj_all = np.load(os.path.join(layer_dir, "projected.npy"))

        num_sample = len(y_true)  
        dim = X_orig_all.shape[1]
        num_generation = X_orig_all.shape[0] // num_sample

        print("X_orig_all:", X_orig_all.shape[0])
        print("X_proj_all:", X_proj_all.shape[0])
        print("num_sample:", num_sample)

        assert X_orig_all.shape[0] == num_sample * num_generation
        assert X_proj_all.shape[0] == num_sample * num_generation

        X_orig_list = [
            X_orig_all[i * num_generation:(i + 1) * num_generation]
            for i in range(num_sample)
        ]

        X_proj_list = [
            X_proj_all[i * num_generation:(i + 1) * num_generation]
            for i in range(num_sample)
        ]  

        y_test = y_true[test_idx]  
        X_test_orig_list = [X_orig_list[i] for i in test_idx] 
        X_test_proj_list = [X_proj_list[i] for i in test_idx] 

        scores_test_orig = compute_eigenscore_multi(X_test_orig_list)
        scores_test_proj = compute_eigenscore_multi(X_test_proj_list)

        auroc_orig = compute_auroc(scores_test_orig, y_test)
        auroc_proj = compute_auroc(scores_test_proj, y_test)

        print(f"[Layer {layer_num}] AUROC Original={auroc_orig:.6f}, "
              f"Projected={auroc_proj:.6f}")

        auroc_orig_list.append(auroc_orig)
        auroc_proj_list.append(auroc_proj)

    results = {
        "layers": layers,
        "orig": auroc_orig_list,
        "proj": auroc_proj_list
    }

    out_path = os.path.join(args.output_dir, "results_eigenscore_layers.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"[SAVED] Results saved to {out_path}")

    plot_auroc_layers(
        layers,
        auroc_orig_list,
        auroc_proj_list,
        os.path.join(args.output_dir, "eigenscore_auroc_layers.png")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", type=str, required=True)
    parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
