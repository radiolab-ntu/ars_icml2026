import os
import argparse
import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ICMEmbeddingDataset(Dataset):
    def __init__(self, embeddings, meta, samples_per_id=7):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.meta = meta
        self.samples_per_id = samples_per_id

        id_to_indices = defaultdict(list)
        for idx, (orig_id, _, _) in enumerate(self.meta):
            id_to_indices[orig_id].append(idx)

        self.final_indices = []
        for orig_id, indices in id_to_indices.items():
            if samples_per_id is None:
                self.final_indices.extend(indices)
            else:
                self.final_indices.extend(indices[:samples_per_id])

        self.consistent_idx = [i for i in self.final_indices if bool(self.meta[i][2])]
        self.inconsistent_idx = [i for i in self.final_indices if not bool(self.meta[i][2])]

    def __len__(self):
        return len(self.final_indices)

    def __getitem__(self, idx):
        real_idx = self.final_indices[idx]
        emb = self.embeddings[real_idx]
        orig_id, ans, is_c = self.meta[real_idx]
        return emb, int(orig_id), bool(is_c)

class LinearMap(nn.Module):
    def __init__(self, input_dim, output_dim=None):
        super().__init__()
        self.output_dim = output_dim or input_dim
        self.W = nn.Linear(input_dim, self.output_dim, bias=False)

    def forward(self, x):
        return self.W(x)

def icm_sameid_ars_loss(
    h,
    orig_ids,
    is_consistent,
    temperature=0.1,
    lambda_reg=1e-4,
    W=None,
    debug=False
):
    batch_size = h.size(0)
    h = nn.functional.normalize(h, dim=1)
    sim_matrix = torch.matmul(h, h.T) / temperature

    loss_list = []
    for i in range(batch_size):
        oid_i = orig_ids[i]
        label_i = is_consistent[i]

        same_id_mask = (orig_ids == oid_i)

        pos_mask = same_id_mask & (is_consistent == label_i)
        pos_mask[i] = False

        neg_mask = same_id_mask & (is_consistent != label_i)
        other_id_mask = (orig_ids != oid_i)

        pos_sims = sim_matrix[i][pos_mask]
        neg_sims = sim_matrix[i][neg_mask]
        other_sims = sim_matrix[i][other_id_mask]

        if pos_sims.numel() == 0:
            continue

        numerator = torch.logsumexp(pos_sims, dim=0)
        all_sims = torch.cat([pos_sims, neg_sims, other_sims])
        denominator = torch.logsumexp(all_sims, dim=0)

        loss_list.append(denominator - numerator)

    if len(loss_list) == 0:
        loss = torch.zeros(1, device=h.device, dtype=h.dtype, requires_grad=True)[0]
    else:
        loss = torch.stack(loss_list).mean()

    if W is not None:
        loss = loss + lambda_reg * torch.norm(W.weight, p="fro") ** 2

    if debug:
        print("sim_matrix:\n", sim_matrix)
        print("loss per batch element:", [l.item() for l in loss_list])

    return loss

def train_icm_layer(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = np.load(args.meta_file, allow_pickle=True)

    for emb_file in sorted(os.listdir(args.embeddings_dir)):
        if not emb_file.endswith("_embeddings.npy"):
            continue

        layer_path = os.path.join(args.embeddings_dir, emb_file)
        embeddings = np.load(layer_path)
        prefix = os.path.splitext(emb_file)[0]
        print(f"[Layer] Training on {prefix} embeddings: {embeddings.shape}")

        dataset = ICMEmbeddingDataset(embeddings, meta)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        model = LinearMap(embeddings.shape[1], output_dim=128).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        save_subdir = os.path.join(args.save_dir, prefix)
        os.makedirs(save_subdir, exist_ok=True)

        config_dict = vars(args).copy()
        config_dict.pop("func", None) 

        with open(os.path.join(save_subdir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

        total_steps = len(loader) * args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-6
        )

        step_count = 0

        loss_history = []
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0

            for h, orig_ids, is_consistent in loader:
                h = h.to(device)
                orig_ids = orig_ids.to(device)
                is_consistent = is_consistent.to(device)

                optimizer.zero_grad()
                h_proj = model(h)
                loss = icm_sameid_ars_loss(
                    h_proj,
                    orig_ids,
                    is_consistent,
                    temperature=args.temperature,
                    lambda_reg=args.lambda_reg,
                    W=model.W,
                    debug=args.debug
                )
                loss.backward()
                optimizer.step()

                scheduler.step() 
                step_count += 1

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            loss_history.append(avg_loss)
            print(f"[{prefix}] Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f}")

        torch.save(model.state_dict(), os.path.join(save_subdir, "linear_map.pt"))
        np.save(os.path.join(save_subdir, "loss.npy"), np.array(loss_history))

        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(loss_history) + 1), loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_subdir, "training_curve.png"))
        plt.close()

        model.eval()
        edited = []
        with torch.no_grad():
            for i in range(0, embeddings.shape[0], 1024):
                batch = torch.from_numpy(embeddings[i:i+1024]).float().to(device)
                edited.append(model(batch).cpu().numpy())
        edited_embeddings = np.concatenate(edited, axis=0)
        np.save(os.path.join(save_subdir, "tilde_embeddings.npy"), edited_embeddings)

def process_single_layer(layer_prefix, orig_dir, proj_dir, save_dir, meta_file):
    orig_emb_path = os.path.join(orig_dir, f"{layer_prefix}.npy")
    proj_emb_path = os.path.join(proj_dir, layer_prefix, "tilde_embeddings.npy")

    if not os.path.exists(orig_emb_path) or not os.path.exists(proj_emb_path):
        print(f"[WARN] can not find {layer_prefix}")
        return

    embeddings = np.load(orig_emb_path)
    projected = np.load(proj_emb_path)
    meta = np.load(meta_file, allow_pickle=True)

    seen = set()
    indices = []
    for i, (oid, _, _) in enumerate(meta):
        if oid not in seen:
            seen.add(oid)
            indices.append(i)

    subdir = os.path.join(save_dir, f"{layer_prefix}_original_only")
    os.makedirs(subdir, exist_ok=True)

    np.save(os.path.join(subdir, "embeddings.npy"), embeddings[indices])
    np.save(os.path.join(subdir, "projected.npy"), projected[indices])
    np.save(os.path.join(subdir, "meta.npy"), meta[indices])


def extract_original_only(args):
    os.makedirs(args.save_dir, exist_ok=True)

    layers = sorted(
        f.replace(".npy", "")
        for f in os.listdir(args.original_root)
        if f.endswith(".npy") and "_layer" in f
    )

    for layer in layers:
        process_single_layer(
            layer,
            args.original_root,
            args.projected_root,
            args.save_dir,
            args.meta_file
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train")
    p_train.add_argument("--embeddings_dir", type=str, required=True)
    p_train.add_argument("--meta_file", type=str, required=True)
    p_train.add_argument("--save_dir", type=str, default="icm_model")
    p_train.add_argument("--batch_size", type=int, default=32)
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight_decay", type=float, default=1e-4)
    p_train.add_argument("--temperature", type=float, default=0.1)
    p_train.add_argument("--lambda_reg", type=float, default=1e-3)
    p_train.add_argument("--debug", action="store_true")
    p_train.set_defaults(func=train_icm_layer)

    p_ext = subparsers.add_parser("extract")
    p_ext.add_argument("--original_root", type=str, required=True)
    p_ext.add_argument("--projected_root", type=str, required=True)
    p_ext.add_argument("--meta_file", type=str, required=True)
    p_ext.add_argument("--save_dir", type=str, default="original_only_all_layers")
    p_ext.set_defaults(func=extract_original_only)

    args = parser.parse_args()
    args.func(args)
