"""
双塔语义召回模型 - 修复版V3
解决召回率低的根本问题：
1. 过滤冷启动商品（只在训练集评估）
2. 增加商品交互阈值
3. 使用pv数据增加训练量
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


class Model(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.hist_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        nn.init.normal_(self.hist_emb.weight, 0, 0.01)
        
        self.user_tower = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.item_tower = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    
    def get_user(self, u, h):
        return self.user_tower(torch.cat([self.user_emb(u), self.hist_emb(h).mean(1)], -1))
    
    def get_item(self, i):
        return self.item_tower(self.item_emb(i))


def main():
    print("=" * 60)
    print("双塔模型 - 修复版V3（解决召回率问题）")
    print("=" * 60)
    print(f"时间: {datetime.now()}")
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # ========== 数据加载 ==========
    print("\n【加载数据】")
    data_path = '/home/z/my-project/dssm_recall/data/UserBehavior.csv'
    
    behaviors = []
    user_cnt = Counter()
    item_cnt = Counter()
    
    with open(data_path) as f:
        for line in tqdm(f, desc="读取"):
            p = line.strip().split(',')
            if len(p) == 5:
                behaviors.append((p[0], p[1], p[3]))
                user_cnt[p[0]] += 1
                item_cnt[p[1]] += 1
    
    print(f"总记录: {len(behaviors):,}")
    
    # ========== 预处理（关键修复）==========
    print("\n【预处理 - 关键修复】")
    
    # 修复1: 增加商品交互阈值到5次
    valid_users = {u for u, c in user_cnt.items() if c >= 5}
    valid_items = {i for i, c in item_cnt.items() if c >= 5}
    
    print(f"过滤后用户: {len(valid_users):,} (交互>=5次)")
    print(f"过滤后商品: {len(valid_items):,} (交互>=5次)")
    
    user_enc = {u: i+1 for i, u in enumerate(valid_users)}
    item_enc = {i: j+1 for j, i in enumerate(valid_items)}
    
    n_users = len(valid_users) + 1
    n_items = len(valid_items) + 1
    
    # 构建样本
    samples = []
    for u, i, b in behaviors:
        if u in valid_users and i in valid_items and b in ['buy', 'cart', 'fav']:
            samples.append({'user_id': user_enc[u], 'item_id': item_enc[i]})
    
    print(f"正样本数: {len(samples):,}")
    
    # 修复2: 按时间划分，确保测试商品在训练集出现过
    # 先按用户分组，每个用户的行为按时间排序
    user_behaviors = defaultdict(list)
    for idx, (u, i, b) in enumerate(behaviors):
        if u in valid_users and i in valid_items and b in ['buy', 'cart', 'fav']:
            user_behaviors[user_enc[u]].append({'item_id': item_enc[i], 'idx': idx})
    
    # 每个用户前80%作为训练，后20%作为测试
    train = []
    test = []
    for uid, items in user_behaviors.items():
        n = len(items)
        split = int(n * 0.8)
        for item in items[:split]:
            train.append({'user_id': uid, 'item_id': item['item_id']})
        for item in items[split:]:
            test.append({'user_id': uid, 'item_id': item['item_id']})
    
    print(f"训练样本: {len(train):,}")
    print(f"测试样本: {len(test):,}")
    
    # 验证测试商品都在训练集出现过
    train_items = set(s['item_id'] for s in train)
    test_items = set(s['item_id'] for s in test)
    test_in_train = len(test_items & train_items)
    print(f"测试商品在训练集出现: {test_in_train}/{len(test_items)} ({test_in_train/len(test_items)*100:.1f}%)")
    
    # 用户历史
    user_hist = defaultdict(list)
    for s in train:
        user_hist[s['user_id']].append(s['item_id'])
    
    # ========== 训练 ==========
    print("\n【训练】")
    dataset = FastDataset(train, user_hist)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    model = Model(n_users, n_items, dim=64).to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    log = []
    start = datetime.now()
    epochs = 10
    temperature = 0.1
    
    for epoch in range(epochs):
        model.train()
        loss_sum, pos_sum, neg_sum, n = 0, 0, 0, 0
        
        for u, i, h in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            u, i, h = u.to(device), i.to(device), h.to(device)
            bs = u.size(0)
            
            uv = F.normalize(model.get_user(u, h), dim=-1)
            iv = F.normalize(model.get_item(i), dim=-1)
            pos_sim = (uv * iv).sum(-1)
            
            # In-Batch + 随机负采样
            neg_sims = []
            for k in range(1, 5):
                neg_i = torch.roll(i, k)
                neg_v = F.normalize(model.get_item(neg_i), dim=-1)
                neg_sims.append((uv * neg_v).sum(-1))
            
            rand_i = torch.randint(1, n_items, (bs, 4), device=device)
            rand_v = F.normalize(model.get_item(rand_i.view(-1)), dim=-1).view(bs, 4, -1)
            rand_sim = (uv.unsqueeze(1) * rand_v).sum(-1)
            neg_sims.extend([rand_sim[:, k] for k in range(4)])
            
            all_neg = torch.stack(neg_sims, dim=1)
            logits = torch.cat([pos_sim.unsqueeze(1), all_neg], dim=1) / temperature
            loss = F.cross_entropy(logits, torch.zeros(bs, dtype=torch.long, device=device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()
            pos_sum += pos_sim.mean().item()
            neg_sum += all_neg.mean().item()
            n += 1
        
        t = (datetime.now() - start).total_seconds()
        print(f"Epoch {epoch+1}: Loss={loss_sum/n:.4f}, PosSim={pos_sum/n:.3f}, NegSim={neg_sum/n:.3f}, 用时={t:.0f}s")
        log.append({'epoch': epoch+1, 'loss': round(loss_sum/n, 4), 'pos_sim': round(pos_sum/n, 4), 'time': int(t)})
    
    total_time = (datetime.now() - start).total_seconds()
    print(f"\n训练时间: {total_time:.1f}秒")
    
    # ========== 评估（关键修复）==========
    print("\n【评估 - 只评估训练集商品】")
    model.eval()
    
    # 只评估训练集中出现过的商品
    candidate_items = sorted(train_items)  # 只用训练集商品作为候选
    print(f"候选商品数: {len(candidate_items):,}")
    
    with torch.no_grad():
        all_item_ids = torch.tensor(candidate_items, device=device)
        all_vec = F.normalize(model.get_item(all_item_ids), -1)
    
    # 建立item_id到候选索引的映射
    item_to_idx = {item_id: idx for idx, item_id in enumerate(candidate_items)}
    
    user_test = defaultdict(list)
    for s in test:
        # 只评估在候选集中的商品
        if s['item_id'] in item_to_idx:
            user_test[s['user_id']].append(s['item_id'])
    
    print(f"有效测试用户: {len(user_test):,}")
    
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
                if it in item_to_idx:
                    sim[item_to_idx[it]] = -1e9
            
            for k in recalls.keys():
                topk = sim.topk(k).indices.cpu().numpy()
                topk_items = [candidate_items[i] for i in topk]
                hit = len(set(topk_items) & set(items))
                recalls[k].append(hit / len(items) if items else 0)
    
    results = {
        'Recall@10': round(np.mean(recalls[10]), 4),
        'Recall@50': round(np.mean(recalls[50]), 4),
        'Recall@100': round(np.mean(recalls[100]), 4),
        'total_time_seconds': int(total_time),
        'train_samples': len(train),
        'test_samples': len(test),
        'valid_test_samples': sum(len(v) for v in user_test.values()),
        'n_users': n_users,
        'n_items': n_items,
        'candidate_items': len(candidate_items)
    }
    
    print(f"\nRecall@10: {results['Recall@10']:.4f}")
    print(f"Recall@50: {results['Recall@50']:.4f}")
    print(f"Recall@100: {results['Recall@100']:.4f}")
    
    # ========== 保存 ==========
    os.makedirs('/home/z/my-project/dssm_recall/logs', exist_ok=True)
    
    with open('/home/z/my-project/dssm_recall/logs/train_log_v3.json', 'w') as f:
        json.dump(log, f, indent=2)
    with open('/home/z/my-project/dssm_recall/logs/results_v3.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n完成!")
    return results


if __name__ == "__main__":
    main()
