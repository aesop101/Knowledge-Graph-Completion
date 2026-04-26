import torch
import torch.nn as nn
import math
import sys
import numpy as np
sys.modules['np.core.numeric'] = np._core.numeric

from torch.utils.data import Dataset
import random
import os

import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from src.dataloader import KGCFewShotDataset, get_collate_fn
from src.utils import set_seed, adjust_learning_rate
from src.modules.ise_TE import ISE_TE
from src.modules.ise_bilinear import ISE_Bilinear
from src.modules.imp import IMP
from evaluate import evaluate

def train_main(dataset_dir, few, ise, imp, nbhd, entity_embeddings, relation_embeddings, lamda, max_batches=100000, batch_size=4):
    device = next(ise.parameters()).device
    dataset = KGCFewShotDataset(
        f'{dataset_dir}/support_5k_F{few}.pt',
        f'{dataset_dir}/query_5k_F{few}.pt',
        f'{dataset_dir}/false_5k_F{few}.pt'
    )
    collate_fn = get_collate_fn(nbhd, entity_embeddings, relation_embeddings)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(list(ise.parameters()) + list(imp.parameters()), lr=5e-5)

    batch_nums = 0

    for sup_data, qry_data, fls_data in dataloader:
        ise.train()
        imp.train()
        batch_nums += 1

        sup = [d.to(device) for d in sup_data]
        qry = [d.to(device) for d in qry_data]
        fls = [d.to(device) for d in fls_data]

        P_sup = ise(*sup)
        P_qry = ise(*qry)
        P_fls = ise(*fls)

        B = P_qry.size(0)
        P_sup_expanded = P_sup.unsqueeze(0).expand(B, -1, -1) 
        pos_score = imp(P_qry.unsqueeze(1), P_sup_expanded) 
        neg_score = imp(P_fls.unsqueeze(1), P_sup_expanded) 

        loss = F.relu(lamda + neg_score - pos_score).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ise.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(imp.parameters(), 5.0)
        optimizer.step()
        adjust_learning_rate(optimizer, batch_nums, 5e-5, 100, max_batches)

        if batch_nums % 40 == 0:
            print(f"\nBatch: {batch_nums}, Loss: {loss.item():.4f}")
            print(f"--- Intermediate Evaluation at Batch {batch_nums} ---")
            ise.eval()
            imp.eval()
            with torch.no_grad():
                result = evaluate(
                    mode='dev',
                    dataset_path=dataset_dir,
                    ise=ise,
                    imp=imp,
                    few=few,
                    entity_embeddings=entity_embeddings,
                    relation_embeddings=relation_embeddings,
                    batch_size=batch_size,
                    NBHD=nbhd, 
                    get_collate_fn=get_collate_fn
                )
                hits10, hits5, hits1, mrr = result['Hits@10'], result['Hits@5'], result['Hits@1'], result['MRR']
                print(f"Dev Evaluation Results: Hits@10={float(hits10):.4f}, Hits@5={float(hits5):.4f}, Hits@1={float(hits1):.4f}, MRR={float(mrr):.4f}")
            
            ise.train()
            imp.train()

        if batch_nums >= max_batches:
            break