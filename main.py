import torch
import numpy as np
import pickle

import sys
sys.modules['np.core.numeric'] = np._core.numeric
from src.utils import set_seed
from src.dataloader import get_collate_fn
from src.modules.ise import ISE
from src.modules.ise_TE import ISE_TE
from src.modules.ise_bilinear import ISE_Bilinear
from src.modules.imp import IMP
from train import train_main
from evaluate import evaluate
def main():
    dataset_dir = "C:\\Users\Infobell\\Desktop\\major_2\\data\\Food"
    embed_dir = "C:\\Users\\Infobell\\Desktop\\major_2\\data\\NELL_processed"
    few = 3
    batch_size = 4
    max_batches = 1000
    seed = 2025

    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading embeddings...")
    embed_dir = embed_dir if embed_dir else dataset_dir

    try:
        entity_emb_np = np.loadtxt(f'{embed_dir}\\entity2vec.TransE')
        rel_emb_np = np.loadtxt(f'{embed_dir}\\relation2vec.TransE')

        entity_embeddings = torch.tensor(entity_emb_np, dtype=torch.float32).to(device)
        relation_embeddings = torch.tensor(rel_emb_np, dtype=torch.float32).to(device)
    except Exception as e:
        print(f"Failed to load embeddings: {e}")
        return

    print("Loading NBHD...")
    try:
        with open(f'{dataset_dir}\\NBHD.pickle', 'rb') as f:
            nbhd = pickle.load(f)
    except Exception as e:
        print(f"Failed to load NBHD: {e}")
        return

    input_size = entity_embeddings.size(-1)
    # ise = ISE(input_size=input_size, num_head=4, hidden_size=64, num_layers=1).to(device)
    # ise = ISE_Bilinear(input_size=input_size, num_head=4, hidden_size=64, num_layers=1).to(device)
    ise = ISE_TE(input_size=input_size, num_head=4, hidden_size=64, num_layers=1).to(device)
    imp = IMP(input_size=input_size, num_head=2, hidden_size=64, num_layers=1).to(device)


    print("Training started...")
    train_main(
        dataset_dir=dataset_dir,
        few=few,
        ise=ise,
        imp=imp,
        nbhd=nbhd,
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
        lamda=5.0,
        max_batches=max_batches
    )

    print("Evaluating on dev...")
    result = evaluate(
        mode='dev',
        dataset_path=dataset_dir,
        ise=ise,
        imp=imp,
        few=few,
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
        batch_size=batch_size,
        NBHD=nbhd, get_collate_fn=get_collate_fn
    )
    hits10, hits5, hits1, mrr = result['Hits@10'], result['Hits@5'], result['Hits@1'], result['MRR']
    print(f"Dev Evaluation Results: Hits@10={float(hits10):.4f}, Hits@5={float(hits5):.4f}, Hits@1={float(hits1):.4f}, MRR={float(mrr):.4f}")

if __name__ == '__main__':
    main()
