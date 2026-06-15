import os
import json
import random
import argparse
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer


def normalize_text(s: str) -> str:
    if s is None:
        return "no_op"
    return s.replace("_", " ").lower().strip()

def load_all_episodes(data_dir: str) -> List[Dict[str, Any]]:
    episodes = []
    for fname in os.listdir(data_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(data_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "episodes" in data:
                episodes.extend(data["episodes"])
            elif isinstance(data, list):
                episodes.extend(data)
            else:
                raise ValueError(f"Unrecognized json structure in {path}")
    return episodes

def gather_all_actions(episodes: List[Dict[str, Any]]):
    actions = set()
    for ep in episodes:
        for a in ep.get("human_task_seq", []):
            actions.add(normalize_text(a))
        for a in ep.get("robot_vocab", []):
            actions.add(normalize_text(a))
        for lab in ep.get("oracle_labels", []):
            if isinstance(lab, dict) and lab.get("best_robot_action"):
                actions.add(normalize_text(lab["best_robot_action"]))
    actions.add("no_op")
    return sorted(actions)

def ensure_sbert_embeddings(actions: List[str], data_dir: str, sbert_model_name: str, batch_size=64):
    os.makedirs(data_dir, exist_ok=True)
    map_path = os.path.join(data_dir, "action2idx.json")
    emb_path = os.path.join(data_dir, "action_embs.npy")
    if os.path.exists(map_path) and os.path.exists(emb_path):
        with open(map_path, "r", encoding="utf-8") as f:
            action2idx = json.load(f)
        emb_matrix = np.load(emb_path)
        return action2idx, emb_matrix

    print("Encoding actions with SBERT (first-time run). Model:", sbert_model_name)
    sbert = SentenceTransformer(sbert_model_name)
    emb_list = []
    for i in range(0, len(actions), batch_size):
        batch = actions[i:i+batch_size]
        emb = sbert.encode(batch, convert_to_numpy=True, show_progress_bar=True)
        emb_list.append(emb)
    emb_matrix = np.vstack(emb_list).astype(np.float32)
    action2idx = {a: idx for idx, a in enumerate(actions)}
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(action2idx, f, indent=2, ensure_ascii=False)
    np.save(emb_path, emb_matrix)
    print(f"Saved action2idx ({len(action2idx)}) and embeddings to {data_dir}")
    return action2idx, emb_matrix


# Stratified split by (scene, label_count)
def label_count(ep):
    labs = ep.get("oracle_labels", [])
    cnt = 0
    for lab in labs:
        if not isinstance(lab, dict):
            continue
        if normalize_text(lab.get("best_robot_action", "no_op")) == "no_op":
            continue
        if int(lab.get("human_step_idx", -1)) == -1:
            continue
        cnt += 1
    return cnt

def stratified_split(episodes, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    groups = defaultdict(list)
    for ep in episodes:
        key = (ep.get("scene", "unknown"), label_count(ep))
        groups[key].append(ep)
    train, val, test = [], [], []
    for k, items in groups.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if n > 2 and n_val == 0:
            n_val = 1
        n_test = n - n_train - n_val
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train+n_val])
        test.extend(items[n_train+n_val:])
    random.shuffle(train); random.shuffle(val); random.shuffle(test)
    return train, val, test


# Dataset Settings
class InterventionExample:
    def __init__(self, human_ids, human_len, human_step_idx, candidate_ids, target_idx):
        self.human_ids = human_ids
        self.human_len = human_len
        self.human_step_idx = human_step_idx
        self.candidate_ids = candidate_ids
        self.target_idx = target_idx

class InterventionDataset(Dataset):
    def __init__(self, episodes, action2idx, max_human_len=12):
        self.examples = []
        self.action2idx = action2idx
        for ep in episodes:
            human = ep['human_task_seq']
            cand = ep['robot_vocab']
            for lab in ep['oracle_labels']:
                human_ids = [action2idx.get(h, action2idx['no_op']) for h in human]
                cand_ids = [action2idx.get(c, action2idx['no_op']) for c in cand]
                target_idx = 0
                if lab.get('best_robot_action') in cand:
                    target_idx = cand.index(lab['best_robot_action'])
                self.examples.append({
                    "human_ids": human_ids,
                    "human_raw": list(human),
                    "human_len": len(human_ids),
                    "human_step_idx": int(lab.get('human_step_idx', 0)),
                    "candidate_ids": cand_ids,
                    "candidate_raw": list(cand),
                    "target_idx": int(target_idx)
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]



# Produce embeddings tensors from precomputed action_embs
def collate_fn(batch, emb_matrix_np, pad_action_str=""):
    B = len(batch)
    S_max = max(x["human_len"] for x in batch)
    C_max = max(len(x["candidate_ids"]) for x in batch)

    D = emb_matrix_np.shape[1]
    human_embs = np.zeros((B, S_max, D), dtype=np.float32)
    human_mask = np.zeros((B, S_max), dtype=np.bool_)
    cand_embs = np.zeros((B, C_max, D), dtype=np.float32)
    cand_mask = np.zeros((B, C_max), dtype=np.bool_)
    human_step_idx = torch.zeros((B,), dtype=torch.long)
    target_idx = torch.zeros((B,), dtype=torch.long)

    batch_raw_human_seqs = []
    batch_raw_candidate_seqs = []

    for i, ex in enumerate(batch):
        # humans
        for s, hid in enumerate(ex["human_ids"]):
            human_embs[i, s, :] = emb_matrix_np[int(hid)]
            human_mask[i, s] = True
        human_step_idx[i] = int(ex.get("human_step_idx", 0))
        # candidates
        cand_ids = ex["candidate_ids"]
        for c, cid in enumerate(cand_ids):
            cand_embs[i, c, :] = emb_matrix_np[int(cid)]
            cand_mask[i, c] = True
        target_idx[i] = int(ex.get("target_idx", 0))


        batch_raw_human_seqs.append(ex.get("human_raw", [""]*S_max) + [""]*(S_max - len(ex.get("human_raw", []))))
        cand_raw = list(ex.get("candidate_raw", []))
        cand_raw = cand_raw + [pad_action_str] * (C_max - len(cand_raw))
        batch_raw_candidate_seqs.append(cand_raw)


    batch_out = {
        "human_embs": torch.from_numpy(human_embs),   # (B,S,D)
        "human_mask": torch.from_numpy(human_mask),   # (B,S)
        "human_step_idx": human_step_idx,             # (B,)
        "cand_embs": torch.from_numpy(cand_embs),     # (B,C,D)
        "cand_mask": torch.from_numpy(cand_mask),     # (B,C)
        "target_idx": target_idx,                     # (B,)
        "batch_raw_human_seqs": batch_raw_human_seqs,
        "batch_raw_candidate_seqs": batch_raw_candidate_seqs
    }
    return batch_out


def tokenize_action(a: str):
    if not a:
        return set()
    return set(a.split('_'))

def make_match_mask_per_sample(batch_raw_humans, batch_raw_cands, scope="all_steps"):
    B = len(batch_raw_humans)
    C = len(batch_raw_cands[0]) if B>0 else 0
    match = torch.zeros((B, C), dtype=torch.float32)
    for b in range(B):
        human_tokens = set()
        for s_tok in batch_raw_humans[b]:
            human_tokens |= tokenize_action(s_tok)
        for c in range(C):
            cand_tokens = tokenize_action(batch_raw_cands[b][c])
            if len(cand_tokens & human_tokens) > 0:
                match[b, c] = 1.0
    return match  # (B, C)



# Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:L, :].unsqueeze(0).to(x.device)

class TransformerCrossAttentionMHA(nn.Module):
    def __init__(self, emb_dim, model_dim=256, n_layers=2, n_heads=4, ffn_dim=None, dropout=0.1):
        super().__init__()
        assert model_dim % n_heads == 0, "model_dim must be divisible by n_heads"
        self.model_dim = model_dim
        self.h_proj = nn.Linear(emb_dim, model_dim) if emb_dim != model_dim else nn.Identity()
        self.pos = PositionalEncoding(model_dim, max_len=64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads,
                                                   dim_feedforward=ffn_dim or model_dim*4,
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # MultiheadAttention for candidate queries
        self.mha = nn.MultiheadAttention(embed_dim=model_dim, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.cand_q = nn.Linear(emb_dim, model_dim)

        self.post_fc = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1)
        )

    def forward(self, human_embs, human_mask, human_step_idx, cand_embs, cand_mask):
        device = human_embs.device
        B, S, E = human_embs.shape
        _, C, _ = cand_embs.shape

        # encode human seq
        h = self.h_proj(human_embs)
        h = self.pos(h)
        src_key_padding_mask = ~human_mask.to(device)
        enc = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        # project candidates
        q = self.cand_q(cand_embs.to(device))


        logits = torch.einsum('bsd,bcd->bsc', enc, q)  # (B, S, C)
        logits = logits.masked_fill(~human_mask.to(device).unsqueeze(-1), float('-1e9'))
        logits = logits.masked_fill(~cand_mask.to(device).unsqueeze(1), float('-1e9'))

        return logits


# Metrics
def selection_acc(logits, target_idx):
    preds = torch.argmax(logits, dim=1)
    return (preds == target_idx).float().mean().item()

def topk_acc(logits, target_idx, k=3):
    k = min(k, logits.size(1))
    topk = torch.topk(logits, k=k, dim=1).indices
    hits = 0
    for i in range(logits.size(0)):
        if target_idx[i].item() in topk[i].cpu().numpy().tolist():
            hits += 1
    return hits / logits.size(0)

def mrr(logits, target_idx):
    rr = []
    for i in range(logits.size(0)):
        sorted_idx = torch.argsort(-logits[i]).cpu().numpy().tolist()
        t = target_idx[i].item()
        try:
            pos = sorted_idx.index(t)
            rr.append(1.0 / (pos + 1))
        except ValueError:
            rr.append(0.0)
    return sum(rr) / len(rr)

def safe_flat_oracles(human_step_idx, target_idx, cand_mask, batch_raw_cands, no_op_token="no_op"):
    B = human_step_idx.size(0)
    device = human_step_idx.device
    C = cand_mask.size(1)
    
    flat_oracles = []
    new_hidx = human_step_idx.clone().cpu().numpy()
    new_tidx = target_idx.clone().cpu().numpy()

    for b in range(B):
        hidx = int(new_hidx[b])
        tidx = int(new_tidx[b])
        # if human_step_idx == -1 -> no-op case
        if hidx < 0:
            try:
                noidx = batch_raw_cands[b].index(no_op_token)
            except ValueError:
                found = False
                for j, s in enumerate(batch_raw_cands[b]):
                    if s.lower() == no_op_token:
                        noidx = j
                        found = True
                        break
                if not found:
                    noidx = 0
            
            flat = 0 * C + noidx
            flat_oracles.append(flat)
        else:
            flat = hidx * C + tidx
            flat_oracles.append(flat)
    return torch.tensor(flat_oracles, dtype=torch.long, device=device)


# Train and Eval
def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        human_embs = batch["human_embs"].to(device)
        human_mask = batch["human_mask"].to(device)
        cand_embs = batch["cand_embs"].to(device)
        cand_mask = batch["cand_mask"].to(device)
        human_step_idx = batch["human_step_idx"].to(device)
        target_idx = batch["target_idx"].to(device)


        logits_s_c = model(human_embs, human_mask, human_step_idx, cand_embs, cand_mask)
        B, S, C = logits_s_c.shape
        logits_flat = logits_s_c.view(B, S*C)

        flat_oracles = safe_flat_oracles(batch["human_step_idx"], batch["target_idx"],
                                 batch["cand_mask"], batch["batch_raw_candidate_seqs"],
                                 no_op_token="no_op").cuda()

        max_valid_idx = S * C - 1
        invalid = (flat_oracles < 0) | (flat_oracles > max_valid_idx)
        if invalid.any():
            print(f"flat_oracles invalid. The correct scope [0, {max_valid_idx}], but exist {flat_oracles[invalid]}")

        eps = 0.12
        p_target = torch.zeros_like(logits_flat, dtype=torch.float32, device=logits_flat.device)
        p_target.scatter_(1, flat_oracles.unsqueeze(1), 1.0 - eps)

        batch_raw_humans = batch["batch_raw_human_seqs"]
        batch_raw_cands = batch["batch_raw_candidate_seqs"]
        match_bc = make_match_mask_per_sample(batch_raw_humans, batch_raw_cands).device
        match_bsc = match_bc.unsqueeze(1).repeat(1, S, 1)   # (B, S, C)
        match_flat = match_bsc.view(B, S*C)                 # (B, S*C)

        for b in range(B):
            match_flat[b, flat_oracles[b]] = 0.0

        match_counts = match_flat.sum(dim=1)
        for b in range(B):
            cnt = match_counts[b].item()
            if cnt > 0.0:
                print(p_target[b].device, match_flat[b].device)
                p_target[b] += eps * (match_flat[b] / cnt)
            else:
                p_target[b, flat_oracles[b]] += eps


        p_target = p_target / p_target.sum(dim=1, keepdim=True).clamp(min=1e-12)

        log_probs = torch.log_softmax(logits_flat, dim=1)
        kl_loss = torch.nn.functional.kl_div(log_probs, p_target, reduction='batchmean')

        ce_loss = torch.nn.functional.cross_entropy(logits_flat, flat_oracles)
        loss = 0.7 * ce_loss + 0.3 * kl_loss


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * human_embs.size(0)
        total_n += human_embs.size(0)
    return total_loss / max(1, total_n)

def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_n = 0
    sel_sum = 0.0
    top3_sum = 0.0
    mrr_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            human_embs = batch["human_embs"].to(device)
            human_mask = batch["human_mask"].to(device)
            human_step_idx = batch["human_step_idx"].to(device)
            cand_embs = batch["cand_embs"].to(device)
            cand_mask = batch["cand_mask"].to(device)
            target_idx = batch["target_idx"].to(device)

            logits_s_c = model(human_embs, human_mask, human_step_idx, cand_embs, cand_mask)  # (B,S,C)
            B, S, C = logits_s_c.shape
 
            logits_flat = logits_s_c.view(B, S*C)

            hidx = human_step_idx.clone()
            neg_mask = (hidx < 0)
            if neg_mask.any():
                hidx[neg_mask] = 0

            flat_target = (hidx * C + target_idx).long()
            loss = F.cross_entropy(logits_flat, flat_target)

            n = human_embs.size(0)
            total_loss += loss.item() * n
            total_n += n
            sel_sum += selection_acc(logits_flat, target_idx) * n
            top3_sum += topk_acc(logits_flat, target_idx, k=3) * n
            mrr_sum += mrr(logits_flat, target_idx) * n
    if total_n == 0:
        return 0.0, 0.0, 0.0, 0.0
    return total_loss/total_n, sel_sum/total_n, top3_sum/total_n, mrr_sum/total_n



def main(args):
    episodes = load_all_episodes(args.data_dir)
    print(f"Loaded {len(episodes)} episodes from {args.data_dir}")

    actions = gather_all_actions(episodes)
    print(f"Unique actions: {len(actions)} -> encoding with SBERT model {args.sbert_model}")
    action2idx, emb_matrix = ensure_sbert_embeddings(actions, args.data_dir, args.sbert_model, batch_size=args.sbert_batch_size)
    emb_matrix = emb_matrix.astype(np.float32)
    vocab_size, emb_dim = emb_matrix.shape
    print(f"Emb matrix loaded: vocab={vocab_size}, emb_dim={emb_dim}")

    train_eps, val_eps, test_eps = stratified_split(episodes, train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
    print(f"Split: train {len(train_eps)}, val {len(val_eps)}, test {len(test_eps)}")

    train_ds = InterventionDataset(train_eps, action2idx)
    val_ds = InterventionDataset(val_eps, action2idx)
    test_ds = InterventionDataset(test_eps, action2idx)
    print(f"Examples: train {len(train_ds)} val {len(val_ds)} test {len(test_ds)}")

    collate = lambda b: collate_fn(b, emb_matrix)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = TransformerCrossAttentionMHA(emb_dim, model_dim=args.model_dim, n_layers=args.n_layers, n_heads=args.n_heads, ffn_dim=args.ffn_dim, dropout=args.dropout)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_sel, val_top3, val_mrr = eval_one_epoch(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} sel={val_sel:.4f} top3={val_top3:.4f} mrr={val_mrr:.4f}")
        if val_sel > best_val:
            best_val = val_sel
            save_obj = {
                "model_state_dict": model.state_dict(),
                "action2idx": action2idx,
                "emb_dim": emb_dim,
                "args": vars(args)
            }
            torch.save(save_obj, args.save_path)
            print(f"Saved best model to {args.save_path} (val_sel={val_sel:.4f})")

    ckpt = torch.load(args.save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_sel, test_top3, test_mrr = eval_one_epoch(model, test_loader, device)
    print(f"Test -> loss={test_loss:.4f} sel={test_sel:.4f} top3={test_top3:.4f} mrr={test_mrr:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data", help="directory containing episode jsons and where embeddings are cached")
    parser.add_argument("--sbert_model", type=str, default="./all-MiniLM-L6-v2", help="sentence-transformers model")
    parser.add_argument("--sbert_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--ffn_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--save_path", type=str, default="best_sbert_transformer_mha.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
