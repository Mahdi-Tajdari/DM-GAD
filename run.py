import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
import random

# ایمپورت توابع از فایل‌های دیگر
from utils import load_data, random_drop_edges, info_nce_loss
from model import GCN_mamba_Net

# تنظیمات ورودی
parser = argparse.ArgumentParser(description='Unsupervised Graph Anomaly Detection with Contrastive Mamba')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--mamba_dropout', type=float, default=0.5)
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--layer_num', type=int, default=5)
parser.add_argument('--alpha', type=float, default=0.8)
parser.add_argument('--graph_weight', type=float, default=0.9)
parser.add_argument('--bias', action='store_true')
parser.add_argument('--drop_rate1', type=float, default=0.1)
parser.add_argument('--drop_rate2', type=float, default=0.15)
parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--score_alpha', type=float, default=0.5, help='Weight for combining structural and attribute scores (0 to 1).')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

# پارامترهای داخلی Mamba
parser.add_argument('--d_inner', type=int, default=512)
parser.add_argument('--dt_rank', type=int, default=4)
parser.add_argument('--d_state', type=int, default=64)
parser.add_argument('--d_conv', type=int, default=4)
parser.add_argument('--expand', type=int, default=2)
args = parser.parse_args()

# تنظیم random seed
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# تنظیم دستگاه پردازشی
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
print(f'--- Running on {args.dataset} with score_alpha={args.score_alpha} and seed={args.seed} ---')

# بارگذاری داده‌ها
pyg_data, adj_dense, ano_label = load_data(args.dataset)
pyg_data = pyg_data.to(device)
adj_t = adj_dense.to(device)
normal_mask = (pyg_data.y == 0)

# تعریف مدل
model = GCN_mamba_Net(pyg_data, args).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train():
    model.train()
    optimizer.zero_grad()
    
    view1_adj = random_drop_edges(adj_t, args.drop_rate1)
    view2_adj = random_drop_edges(adj_t, args.drop_rate2)
    sub_mask = normal_mask.to(device)
    
    # مدل حالا دو مقدار برمی‌گرداند
    emb1, _ = model(pyg_data.x[sub_mask], view1_adj[sub_mask][:, sub_mask])
    emb2, _ = model(pyg_data.x[sub_mask], view2_adj[sub_mask][:, sub_mask])
    
    loss = info_nce_loss(emb1, emb2, args.temperature)
    loss.backward()
    
    total_grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_grad_norm += param_norm.item() ** 2
    total_grad_norm = total_grad_norm ** 0.5

    optimizer.step()
    return loss.item(), emb1, emb2, total_grad_norm

def evaluate():
    model.eval()
    with torch.no_grad():
        final_emb, x_proj = model(pyg_data.x, adj_t)
        y_true = pyg_data.y.cpu().numpy()

        # محاسبه امتیاز و AUC
        if torch.isnan(final_emb).any() or torch.isinf(final_emb).any():
            return 0.0, None, None, None, None
        
        recon = adj_t @ final_emb
        structural_scores = (recon - final_emb).norm(dim=1)
        attribute_scores = (final_emb - x_proj).norm(dim=1)
        scores = args.score_alpha * structural_scores - (1 - args.score_alpha) * attribute_scores
        
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores = torch.nan_to_num(scores, nan=0.0)
        
        y_scores = scores.cpu().numpy()
        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc = 0.0
            
    return auc, scores, final_emb, structural_scores, attribute_scores

# حلقه آموزش
best_auc = 0.0
for epoch in range(args.epochs):
    loss, emb1, emb2, grad_norm = train()
    
    if epoch % 10 == 0:
        auc, scores, final_emb, structural_scores, attribute_scores = evaluate()
        
        # ذخیره بالاترین AUC
        if auc > best_auc:
            best_auc = auc

        print("-" * 80)
        print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Current AUC: {auc:.4f} | Best AUC: {best_auc:.4f} | Grad Norm: {grad_norm:.4f}')

        if final_emb is None:
            print("!!! ارزیابی به دلیل وجود NaN متوقف شد. !!!")
            continue

        with torch.no_grad():
            pos_sim = torch.mm(F.normalize(emb1), F.normalize(emb2).t()).diag().mean().item()
            print(f'  [Contrastive Health]: Positive Pair Similarity: {pos_sim:.4f}')
            
            emb_norm_train = emb1.norm(dim=1).mean().item()
            emb_norm_eval = final_emb.norm(dim=1).mean().item()
            print(f'  [Embedding Health]: Avg Norm (Train): {emb_norm_train:.4f} | Avg Norm (Eval): {emb_norm_eval:.4f}')

            if scores is not None and len(scores) > 0:
                normal_scores_final = scores[normal_mask].cpu().numpy()
                anomaly_scores_final = scores[~normal_mask].cpu().numpy()
                
                if len(normal_scores_final) > 0 and len(anomaly_scores_final) > 0:
                    print(f'  [Anomaly Score (Final)]: Mean(N): {np.mean(normal_scores_final):.4f}, Mean(A): {np.mean(anomaly_scores_final):.4f}')
            else:
                print("  [Scores]: تعداد نمونه برای مقایسه امتیازها کافی نیست.")
        print("-" * 80)

# نمایش نتیجه نهایی
print("-" * 50)
print(f'Final Result - Dataset: {args.dataset}')
print(f'Highest AUC Achieved: {best_auc:.4f}')
print("-" * 50)

