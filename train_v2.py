"""
双塔语义召回模型 - 深度优化版
优化：增加数据量 + 增强模型 + 难负例挖掘
"""

import os
import json
import random
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FastDataset(Dataset):
    def __init__(self, samples, user_hist, seq_len=30):
        n = len(samples)
        self.users = np.zeros(n, dtype=np.int64)
        self.items = np.zeros(n, dtype=np.int64)
        self.hists = np.zeros((n, seq_len), dtype=np.int64)
        
        for i, s in enumerate(samples):
            self.users[i] = s['user_id']
            self.items[i] = s['item_id']
            h = user_hist.get(s['user_id'], [])[-seq_len:]
            h = [0] * (seq_len - len(h)) + h
            self.hists[i] = h
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.hists[idx]


class EnhancedModel(nn.Module):
    """增强版模型"""
    def __init__(self, n_users, n_items, dim=128):
        super().__init__()
        self.n_items = n_items
        
        # 嵌入层
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.hist_emb = nn.Embedding(n_items, dim)
        
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        nn.init.normal_(self.hist_emb.weight, 0, 0.01)
        
        # 用户塔 - 更深的MLP
        self.user_tower = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        # 商品塔 - 更深的MLP
        self.item_tower = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    
    def get_user(self, u, h):
        return self.user_tower(torch.cat([self.user_emb(u), self.hist_emb(h).mean(1)], -1))
    
    def get_item(self, i):
        return self.item_tower(self.item_emb(i))


def main():
    print("=" * 60)
    print("双塔模型 - 深度优化版")
    print("=" * 60)
    print(f"时间: {datetime.now()}")
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # ========== 数据加载（采样50%）==========
    print("\n【加载数据】")
    behaviors, user_cnt, item_cnt = [], Counter(), Counter()
    
    with open('/home/z/my-project/dssm_recall/data/UserBehavior.csv') as f:
        for line in tqdm(f, desc="读取"):
            if random.random() > 0.5:  # 采样50%
                continue
            p = line.strip().split(',')
            if len(p) == 5:
                behaviors.append((p[0], p[1], p[3]))
                user_cnt[p[0]] += 1
                item_cnt[p[1]] += 1
    
    print(f"采样记录: {len(behaviors):,}")
    
    # ========== 预处理 ==========
    print("\n【预处理】")
    valid_users = {u for u, c in user_cnt.items() if c >= 2}
    valid_items = {i for i, c in item_cnt.items() if c >= 2}
    
    user_enc = {u: i+1 for i, u in enumerate(valid_users)}
    item_enc = {i: j+1 for j, i in enumerate(valid_items)}
    
    n_users, n_items = len(valid_users) + 1, len(valid_items) + 1
    print(f"用户: {n_users:,}, 商品: {n_items:,}")
    
    samples = []
    for u, i, b in behaviors:
        if u in valid_users and i in valid_items and b in ['buy', 'cart', 'fav']:
            samples.append({'user_id': user_enc[u], 'item_id': item_enc[i]})
    
    random.shuffle(samples)
    train = samples[:int(len(samples)*0.9)]
    test = samples[int(len(samples)*0.9):]
    print(f"训练: {len(train):,}, 测试: {len(test):,}")
    
    user_hist = defaultdict(list)
    for s in train:
        user_hist[s['user_id']].append(s['item_id'])
    
    # ========== 训练 ==========
    print("\n【训练】")
    dataset = FastDataset(train, user_hist, seq_len=30)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    model = EnhancedModel(n_users, n_items, dim=128).to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    log = []
    start = datetime.now()
    epochs = 15
    temperature = 0.05
    
    # 预计算所有商品向量用于难负例
    all_item_ids = torch.arange(1, n_items, device=device)
    
    for epoch in range(epochs):
        model.train()
        loss_sum, pos_sum, neg_sum, n = 0, 0, 0, 0
        
        # 每3轮更新一次商品向量
        if epoch % 3 == 0:
            with torch.no_grad():
                all_item_vecs = F.normalize(model.get_item(all_item_ids), dim=-1)
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for u, i, h in pbar:
            u, i, h = u.to(device), i.to(device), h.to(device)
            bs = u.size(0)
            
            # 归一化的向量
            uv = F.normalize(model.get_user(u, h), dim=-1)
            iv = F.normalize(model.get_item(i), dim=-1)
            
            # 正样本相似度
            pos_sim = (uv * iv).sum(-1)
            
            # 负样本策略
            neg_sims = []
            
            # 1. In-Batch负采样 (4个)
            for k in range(1, 5):
                neg_i = torch.roll(i, k)
                neg_v = F.normalize(model.get_item(neg_i), dim=-1)
                neg_sims.append((uv * neg_v).sum(-1))
            
            # 2. 随机负采样 (4个)
            rand_i = torch.randint(1, n_items, (bs, 4), device=device)
            rand_v = F.normalize(model.get_item(rand_i.view(-1)), dim=-1).view(bs, 4, -1)
            rand_sim = (uv.unsqueeze(1) * rand_v).sum(-1)
            neg_sims.extend([rand_sim[:, k] for k in range(4)])
            
            # 3. 难负例 (2个) - 从最后几轮开始
            if epoch >= 5:
                with torch.no_grad():
                    # 批量计算相似度
                    batch_sim = torch.mm(uv, all_item_vecs.T)  # [bs, n_items]
                    # 获取Top-K难负例
                    hard_indices = batch_sim.topk(10, dim=1).indices  # [bs, 10]
                    # 随机选择2个
                    hard_sel = torch.randint(0, 10, (bs, 2), device=device)
                    hard_i = torch.gather(hard_indices, 1, hard_sel) + 1  # [bs, 2]
                
                hard_v = F.normalize(model.get_item(hard_i.view(-1)), dim=-1).view(bs, 2, -1)
                hard_sim = (uv.unsqueeze(1) * hard_v).sum(-1)  # [bs, 2]
                neg_sims.extend([hard_sim[:, 0], hard_sim[:, 1]])
            
            # 合并负样本
            all_neg = torch.stack(neg_sims, dim=1)  # [bs, num_neg]
            
            # InfoNCE损失
            logits = torch.cat([pos_sim.unsqueeze(1), all_neg], dim=1) / temperature
            labels = torch.zeros(bs, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            loss_sum += loss.item()
            pos_sum += pos_sim.mean().item()
            neg_sum += all_neg.mean().item()
            n += 1
        
        t = (datetime.now() - start).total_seconds()
        print(f"Epoch {epoch+1}: Loss={loss_sum/n:.4f}, PosSim={pos_sum/n:.3f}, NegSim={neg_sum/n:.3f}, 用时={t:.0f}s")
        log.append({
            'epoch': epoch+1, 
            'loss': round(loss_sum/n, 4),
            'pos_sim': round(pos_sum/n, 4),
            'neg_sim': round(neg_sum/n, 4),
            'time': int(t)
        })
    
    total_time = (datetime.now() - start).total_seconds()
    print(f"\n训练时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    
    # ========== 评估 ==========
    print("\n【评估】")
    model.eval()
    
    with torch.no_grad():
        all_vec = F.normalize(model.get_item(all_item_ids), -1)
    
    user_test = defaultdict(list)
    for s in test:
        user_test[s['user_id']].append(s['item_id'])
    
    # 多K值评估
    recalls = {10: [], 50: [], 100: []}
    
    for uid, items in tqdm(user_test.items(), desc="评估"):
        h = user_hist.get(uid, [])[-30:]
        h = [0] * (30 - len(h)) + h
        
        with torch.no_grad():
            uv = F.normalize(model.get_user(
                torch.tensor([uid], device=device),
                torch.tensor([h], device=device)
            ), -1)
            sim = torch.mm(uv, all_vec.T).squeeze()
            
            # 排除历史
            for it in user_hist.get(uid, []):
                if 0 < it < n_items:
                    sim[it-1] = -1e9
            
            for k in recalls.keys():
                topk = sim.topk(k).indices.cpu().numpy() + 1
                hit = len(set(topk) & set(items))
                recalls[k].append(hit / len(items) if items else 0)
    
    results = {
        'Recall@10': round(np.mean(recalls[10]), 4),
        'Recall@50': round(np.mean(recalls[50]), 4),
        'Recall@100': round(np.mean(recalls[100]), 4),
        'total_time_seconds': int(total_time),
        'total_time_minutes': round(total_time / 60, 2),
        'train_samples': len(train),
        'test_samples': len(test),
        'n_users': n_users,
        'n_items': n_items,
        'epochs': epochs,
        'temperature': temperature,
        'model_dim': 128
    }
    
    print(f"Recall@10: {results['Recall@10']:.4f}")
    print(f"Recall@50: {results['Recall@50']:.4f}")
    print(f"Recall@100: {results['Recall@100']:.4f}")
    
    # ========== 保存 ==========
    os.makedirs('/home/z/my-project/dssm_recall/checkpoints', exist_ok=True)
    os.makedirs('/home/z/my-project/dssm_recall/logs', exist_ok=True)
    
    torch.save(model.state_dict(), '/home/z/my-project/dssm_recall/checkpoints/model_v2.pt')
    
    with open('/home/z/my-project/dssm_recall/logs/train_log_v2.json', 'w') as f:
        json.dump(log, f, indent=2)
    with open('/home/z/my-project/dssm_recall/logs/results_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n完成!")
    return results


if __name__ == "__main__":
    main()
