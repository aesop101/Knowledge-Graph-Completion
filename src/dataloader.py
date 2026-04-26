import torch
import numpy as np
from torch.utils.data import Dataset

class KGCFewShotDataset(Dataset):
    def __init__(self, support_path, query_path, false_path):
        self.support = torch.load(support_path)
        self.query = torch.load(query_path)
        self.false = torch.load(false_path)

    def __len__(self):
        return self.support.shape[0]

    def __getitem__(self, idx):
        return self.support[idx], self.query[idx], self.false[idx]

def get_collate_fn(nbhd, entity_embeddings, relation_embeddings):
    def collate_fn(batch):
        support, query, false = batch[0]
        def process_triples(triples):
            N = triples.size(0)
            e_h_ids = triples[:, 0].long().numpy()
            e_t_ids = triples[:, 1].long().numpy()

            N_h_list = [nbhd.get(int(eid), np.zeros((0, 2), dtype=int)) for eid in e_h_ids]
            N_t_list = [nbhd.get(int(eid), np.zeros((0, 2), dtype=int)) for eid in e_t_ids]

            max_Nh = max([len(n) for n in N_h_list]) if N_h_list else 0
            max_Nt = max([len(n) for n in N_t_list]) if N_t_list else 0

            emb_size = entity_embeddings.size(-1)
            emb_h = torch.zeros(N, max_Nh, 2 * emb_size)
            emb_t = torch.zeros(N, max_Nt, 2 * emb_size)

            mask_h = torch.ones(N, max_Nh + 1, dtype=torch.bool)
            mask_t = torch.ones(N, max_Nt + 1, dtype=torch.bool)

            for i in range(N):
                nh = N_h_list[i]
                if len(nh) > 0:
                    r_ids = torch.from_numpy(nh[:, 0]).long()
                    e_ids = torch.from_numpy(nh[:, 1]).long()
                    r_emb = relation_embeddings[r_ids]
                    e_emb = entity_embeddings[e_ids]
                    emb_h[i, :len(nh)] = torch.cat((e_emb, r_emb), dim=-1)

                nt = N_t_list[i]
                if len(nt) > 0:
                    r_ids = torch.from_numpy(nt[:, 0]).long()
                    e_ids = torch.from_numpy(nt[:, 1]).long()
                    r_emb = relation_embeddings[r_ids]
                    e_emb = entity_embeddings[e_ids]
                    emb_t[i, :len(nt)] = torch.cat((e_emb, r_emb), dim=-1)

                mask_h[i, 0] = False
                mask_h[i, 1:len(nh)+1] = False

                mask_t[i, 0] = False
                mask_t[i, 1:len(nt)+1] = False

            e_h_emb = entity_embeddings[torch.from_numpy(e_h_ids)]
            e_t_emb = entity_embeddings[torch.from_numpy(e_t_ids)]

            return emb_h, emb_t, e_h_emb, e_t_emb, mask_h, mask_t

        sup_data = process_triples(support)
        qry_data = process_triples(query)
        fls_data = process_triples(false)

        return sup_data, qry_data, fls_data
    return collate_fn
