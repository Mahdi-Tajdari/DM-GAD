"""
OC-Mamba: مثل OCGNN
- train: فقط روی subset از نودهای نرمال (80%)
- val: بقیه نودهای نرمال (20%)  
- test: همه نودها (نرمال + anomaly) → AUC
"""

import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
import random

from utils import load_data, random_drop_edges, info_nce_loss
from model import GCN_mamba_Net

# تنظیمات ورودی
# تنظیمات ورودی
parser = argparse.ArgumentParser(description='Unsupervised Graph Anomaly Detection with Contrastive Mamba')
parser.add_argument('--dataset', type=str, default='BlogCatalog')
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--mamba_dropout', type=float, default=0.5)
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--layer_num', type=int, default=3)
parser.add_argument('--alpha', type=float, default=0.8)
parser.add_argument('--graph_weight', type=float, default=0.7)
parser.add_argument('--bias', action='store_true')
parser.add_argument('--drop_rate1', type=float, default=0.3)
parser.add_argument('--drop_rate2', type=float, default=0.4)
parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--score_alpha', type=float, default=0.3, help='Weight for combining structural and attribute scores (0 to 1).')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')



parser.add_argument('--train_ratio', type=float, default=0.8,
                    help='fraction of normal nodes used for training')
parser.add_argument('--dt_rank', type=int, default=4)
parser.add_argument('--d_state', type=int, default=64)
parser.add_argument('--d_conv', type=int, default=4)
parser.add_argument('--expand', type=int, default=2)
args = parser.parse_args()

# reproducibility
seed = args.seed
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
print(f'--- {args.dataset} | score_alpha={args.score_alpha} | seed={args.seed} ---')

# ==================== Data ====================
pyg_data, adj_dense, ano_label = load_data(args.dataset)
pyg_data = pyg_data.to(device)
adj_t = adj_dense.to(device)
y = pyg_data.y.cpu().numpy()

# ==================== Split مثل OCGNN ====================
# همه نودهای نرمال
normal_idx = torch.where(pyg_data.y == 0)[0]
n_normal   = len(normal_idx)

# shuffle و split
perm       = torch.randperm(n_normal, generator=torch.Generator().manual_seed(seed))
n_train    = int(n_normal * args.train_ratio)
train_idx  = normal_idx[perm[:n_train]]   # 80% نرمال → train
val_idx    = normal_idx[perm[n_train:]]   # 20% نرمال → val

print(f'Normal nodes: {n_normal} | Train: {len(train_idx)} | Val: {len(val_idx)}')
print(f'Anomaly nodes: {(pyg_data.y==1).sum().item()} | Test: all {len(y)} nodes')
print('-' * 70)

# adj های مربوط به train nodes
train_mask_full = torch.zeros(len(y), dtype=torch.bool)
train_mask_full[train_idx] = True
train_adj = adj_t[train_mask_full][:, train_mask_full]

# ==================== Model ====================
model     = GCN_mamba_Net(pyg_data, args).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train():
    model.train()
    optimizer.zero_grad()

    # دو view با edge drop روی train subgraph
    view1_adj = random_drop_edges(train_adj, args.drop_rate1)
    view2_adj = random_drop_edges(train_adj, args.drop_rate2)

    x_train = pyg_data.x[train_mask_full]
    emb1, _ = model(x_train, view1_adj)
    emb2, _ = model(x_train, view2_adj)

    loss = info_nce_loss(emb1, emb2, args.temperature)
    loss.backward()

    grad_norm = sum(p.grad.data.norm(2).item()**2
                    for p in model.parameters() if p.grad is not None) ** 0.5
    optimizer.step()
    return loss.item(), emb1, emb2, grad_norm


def evaluate():
    model.eval()
    with torch.no_grad():
        # inference روی کل گراف
        final_emb, x_proj = model(pyg_data.x, adj_t)

        if torch.isnan(final_emb).any() or torch.isinf(final_emb).any():
            return 0.0, None, None, None

        recon             = adj_t @ final_emb
        structural_scores = (recon - final_emb).norm(dim=1)
        attribute_scores  = (final_emb - x_proj).norm(dim=1)
        scores            = args.score_alpha * structural_scores \
                            - (1 - args.score_alpha) * attribute_scores
        scores            = torch.nan_to_num(scores, nan=0.0)

        y_scores = scores.cpu().numpy()
        try:
            auc = roc_auc_score(y, y_scores)
        except ValueError:
            auc = 0.0

    return auc, scores, final_emb, structural_scores


# ==================== Loop ====================
best_auc = 0.0
anomaly_mask = (pyg_data.y == 1).to(device)

for epoch in range(args.epochs):
    loss, emb1, emb2, grad_norm = train()

    if epoch % 10 == 0:
        auc, scores, final_emb, structural_scores = evaluate()
        if auc > best_auc:
            best_auc = auc

        print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | AUC: {auc:.4f} | '
              f'Best: {best_auc:.4f} | GradNorm: {grad_norm:.4f}')

        if scores is not None:
            with torch.no_grad():
                pos_sim = torch.mm(F.normalize(emb1),
                                   F.normalize(emb2).t()).diag().mean().item()
                print(f'  [Contrastive] pos_sim={pos_sim:.4f}')
                n_score = scores[~anomaly_mask].mean().item()
                a_score = scores[anomaly_mask].mean().item()
                print(f'  [Score] Normal={n_score:.4f} | Anomaly={a_score:.4f}')

print('=' * 70)
print(f'Dataset: {args.dataset} | Best AUC: {best_auc:.4f}')
print(f'Train setup: {len(train_idx)}/{n_normal} normal nodes '
      f'({args.train_ratio*100:.0f}% of normal, '
      f'{len(train_idx)/len(y)*100:.1f}% of all nodes)')
