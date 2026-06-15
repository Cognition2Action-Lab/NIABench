import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


from mht3 import (
    TransformerCrossAttentionMHA,
    normalize_text,
    collate_fn
)

def load_inference_components(model_path):
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    args = ckpt['args']
    action2idx = ckpt['action2idx']
    emb_dim = ckpt['emb_dim']
    
    model = TransformerCrossAttentionMHA(
        emb_dim=emb_dim,
        model_dim=args['model_dim'],
        n_layers=args['n_layers'],
        n_heads=args['n_heads'],
        ffn_dim=args['ffn_dim'],
        dropout=args['dropout']
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    data_dir = args['data_dir']
    emb_matrix_path = os.path.join(data_dir, 'action_embs.npy')
    if os.path.exists(emb_matrix_path):
        emb_matrix = np.load(emb_matrix_path)
    else:
        print("Action embedding cache not found, regenerating...")
        sbert = SentenceTransformer(args['sbert_model'])
        actions = list(action2idx.keys())
        emb_matrix = sbert.encode(actions, convert_to_numpy=True)
        np.save(emb_matrix_path, emb_matrix)
    
    sbert_model = SentenceTransformer(args['sbert_model'])
    
    return model, action2idx, emb_matrix, sbert_model, args

def prepare_input(human_task_seq, candidate_actions, action2idx, sbert_model, emb_matrix):
    human_normalized = [normalize_text(t) for t in human_task_seq]
    human_ids = [action2idx.get(t, action2idx['no_op']) for t in human_normalized]
    human_len = len(human_ids)
    
    candidate_normalized = [normalize_text(act) for act in candidate_actions]
    candidate_ids = []
    for act in candidate_normalized:
        if act in action2idx:
            candidate_ids.append(action2idx[act])
        else:
            print(f"Unseen action detected: {act}, encoding...")
            emb = sbert_model.encode([act], convert_to_numpy=True)[0]
            emb_matrix = np.vstack([emb_matrix, emb])
            new_idx = len(action2idx)
            action2idx[act] = new_idx
            candidate_ids.append(new_idx)

    sample = {
        "human_ids": human_ids,
        "human_len": human_len,
        "human_step_idx": max(0, human_len - 1),
        "candidate_ids": candidate_ids,
        "target_idx": 0
    }
    
    batch = collate_fn([sample], emb_matrix)

    return batch

def predict_best_action(human_task_seq, candidate_actions, model, action2idx, 
                       emb_matrix, sbert_model, device):
    batch = prepare_input(
        human_task_seq, candidate_actions, 
        action2idx, sbert_model, emb_matrix
    )
    
    input_data = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            input_data[k] = v.to(device)
        else:
            input_data[k] = v
    
    with torch.no_grad():
        logits_s_c = model(
            input_data['human_embs'],
            input_data['human_mask'],
            input_data['human_step_idx'],
            input_data['cand_embs'],
            input_data['cand_mask']
        )
    print(logits_s_c.shape)
    B, S, C = logits_s_c.shape

    logits_flat = logits_s_c.view(B, S*C)
    best_flat = torch.argmax(logits_flat, dim=1)
    pred_step = best_flat // C
    pred_cand = best_flat % C

    best_step = human_task_seq[pred_step]
    best_action = candidate_actions[pred_cand]


    topk_k = min(10, logits_flat.size(1))
    probs = torch.softmax(logits_flat, dim=1)
    topk_vals, topk_idxs = torch.topk(logits_flat, k=topk_k, dim=1)
    topk_probs = torch.gather(probs, 1, topk_idxs)

    for i in range(topk_k):
        flat_idx = int(topk_idxs[0, i].item())
        logit_val = float(topk_vals[0, i].item())
        prob_val = float(topk_probs[0, i].item())
        step_idx = flat_idx // C
        cand_idx = flat_idx % C
        step_str = human_task_seq[step_idx] if 0 <= step_idx < len(human_task_seq) else f"step_{step_idx}"
        action_str = candidate_actions[cand_idx] if 0 <= cand_idx < len(candidate_actions) else f"cand_{cand_idx}"
        print(f"Top{i+1}: step_idx={step_idx} ({step_str}), cand_idx={cand_idx} ({action_str}), logit={logit_val:.4f}, prob={prob_val:.4f}")
    
    return best_step, best_action

def main():
    model_path = "best_sbert_transformer_mha.pt"
    candidate_actions_path = "robot_instructions.json"
    
    with open(candidate_actions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        candidate_actions = data.get('kitchen', [])
    print(f"{len(candidate_actions)} candidate instructions have been loaded.")

    model, action2idx, emb_matrix, sbert_model, args = load_inference_components(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    

    human_task_seq = [
        "find_tomato", 
        "bring_tomato_to_sink", 
        "wash_tomato", 
        "bring_tomato_to_couter", 
        "find_knife", 
        "bring_knife_to_couter", 
        "cut_tomato", 
        "place_tomato_on_plate"
    ]
    
    print(f"human_task_seq: {human_task_seq}")
    
    best_step, best_action = predict_best_action(
        human_task_seq, candidate_actions,
        model, action2idx, emb_matrix, sbert_model, device
    )
    
    print(f"best intervention: {best_step}")
    print(f"best robot action: {best_action}")

if __name__ == "__main__":
    main()
