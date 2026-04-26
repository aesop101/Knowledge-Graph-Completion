import torch
import torch.nn.functional as F
import numpy as np
import json
import random
import sys
from src.modules.imp import calculate_score
sys.modules['np.core.numeric'] = np._core.numeric

def evaluate(mode, dataset_path, ise, imp, few, entity_embeddings, relation_embeddings, batch_size, NBHD, get_collate_fn):
    ise.eval()
    imp.eval()
    device = next(ise.parameters()).device

    test_tasks = json.load(open(f'{dataset_path}/{mode}_tasks.json'))
    rel2candidates = json.load(open(f'{dataset_path}/rel2candidates_all.json'))
    e1rel_e2 = json.load(open(f'{dataset_path}/e1rel_e2.json'))
    ent2ids = json.load(open(f'{dataset_path}/ent2ids.json'))

    hits10_all, hits5_all, hits1_all, mrr_all = [], [], [], []
    collate_fn = get_collate_fn(NBHD, entity_embeddings, relation_embeddings)

    def encode_pairs_with_ise(pairs_list):
        pairs_tensor = torch.tensor(pairs_list)
        dummy_batch = [(pairs_tensor, pairs_tensor, pairs_tensor)]
        data, _, _ = collate_fn(dummy_batch)

        emb_h, emb_t, e_h_emb, e_t_emb, mask_h, mask_t = [d.to(device) for d in data]
        p = ise(emb_h, emb_t, e_h_emb, e_t_emb, mask_h, mask_t)
        return p

    with torch.no_grad():
        for query_rel in test_tasks.keys():
            candidates = [x for x in rel2candidates[query_rel] if ent2ids.get(x) in NBHD]
            all_triples = [x for x in test_tasks[query_rel]
                           if ent2ids.get(x[0]) in NBHD and ent2ids.get(x[2]) in NBHD]

            if len(candidates) <= 0 or len(all_triples) <= few:
                continue

            support_triples = all_triples[:few]
            support_pairs = [[ent2ids[t[0]], ent2ids[t[2]]] for t in support_triples]
            P_support = encode_pairs_with_ise(support_pairs) 
            
            t_dict = {}
            for triple in all_triples[few:]:
                t_dict.setdefault(triple[0], []).append(triple)

            for head, list_t in t_dict.items():
                negative_pairs = []
                true_tails = e1rel_e2.get(head + query_rel, [])

                for ent in candidates:
                    if ent not in true_tails:
                        negative_pairs.append([ent2ids[head], ent2ids[ent]])

                random.shuffle(negative_pairs)
                sampled_negatives = negative_pairs[:batch_size]

                for triple in list_t:
                    true_pair = [ent2ids[triple[0]], ent2ids[triple[2]]]
                    eval_pairs = [true_pair] + sampled_negatives

                    P_eval = encode_pairs_with_ise(eval_pairs) 
                    P_support_exp = P_support.unsqueeze(0).expand(P_eval.size(0), -1, -1)
                    scores = calculate_score(P_eval.unsqueeze(1), P_support_exp).cpu().numpy()

                    ranking = np.argsort(-scores, kind='stable')
                    rank = np.where(ranking == 0)[0][0] + 1

                    mrr_all.append(1.0 / rank)
                    hits1_all.append(1.0 if rank <= 1 else 0.0)
                    hits5_all.append(1.0 if rank <= 5 else 0.0)
                    hits10_all.append(1.0 if rank <= 10 else 0.0)

    return {
        "MRR": np.mean(mrr_all),
        "Hits@1": np.mean(hits1_all),
        "Hits@5": np.mean(hits5_all),
        "Hits@10": np.mean(hits10_all)
    }