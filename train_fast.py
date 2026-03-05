"""
双塔语义召回模型 - 高效版
目标：3分钟内完成训练
优化：数据采样 + 简化模型 + 完全向量化
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
    def __init__(self, samples, user_history, seq_len=20):
        n = len(samples)
        self.users = np.zeros(n, dtype=np.int64)
        self.items = np.zeros(n, dtype=np.int64)
        self.hists = np.zeros((n, seq_len), dtype=np.int64)
        
        for i, s in enumerate(samples):
            self.users[i] = s['user_id']
            self.items[i] = s['item_id']
            h = user_history.get(s['user_id'], [])[-seq_len:]
            h = [0] * (seq_len - len(h)) + h
            self.hists[i] = h
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.hists[idx]


class SimpleModel(nn.Module):
    def __init__(self, n_users, n_items, dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.hist_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        nn.init.normal_(self.hist_emb.weight, 0, 0.01)
    
    def get_user(self, users, hists):
        u = self.user_emb(users)
        h = self.hist_emb(hists).mean(dim=1)
        return u + h
    
    def get_item(self, items):
        return self.item_emb(items)


def main():
    print("=" * 50)
    print("双塔模型 - 高效版")
    print("=" * 50)
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # ========== 数据加载（采样20%）==========
    print("\n加载数据...")
    data_path = '/home/z/my-project/dssm_recall/data/UserBehavior.csv'
    
    behaviors = []
    user_cnt = Counter()
    item_cnt = Counter()
    
    with open(data_path) as f:
        for line in tqdm(f, desc="读取"):
            if random.random() > 0.2:  # 采样20%
                continue
            p = line.strip().split(',')
            if len(p) == 5:
                behaviors.append((p[0], p[1], p[3]))
                user_cnt[p[0]] += 1
                item_cnt[p[1]] += 1
    
    print(f"采样后记录: {len(behaviors):,}")
    
    # ========== 预处理 ==========
    print("预处理...")
    valid_users = {u for u, c in user_cnt.items() if c >= 2}
    valid_items = {i for i, c in item_cnt.items() if c >= 2}
    
    user_enc = {u: i+1 for i, u in enumerate(valid_users)}
    item_enc = {i: j+1 for j, i in enumerate(valid_items)}
    
    n_users = len(valid_users) + 1
    n_items = len(valid_items) + 1
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
    print("\n训练...")
    dataset = FastDataset(train, user_hist)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    model = SimpleModel(n_users, n_items, dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    start = datetime.now()
    epochs = 5
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n = 0
        
        for users, items, hists in tqdm(loader, desc=f"Epoch {epoch+1}"):
            users = users.to(device)
            items = items.to(device)
            hists = hists.to(device)
            bs = users.size(0)
            
            # 正样本
            u_vec = F.normalize(model.get_user(users, hists), dim=-1)
            i_vec = F.normalize(model.get_item(items), dim=-1)
            pos_sim = (u_vec * i_vec).sum(-1)
            
            # 负样本（In-Batch）
            neg_items = torch.roll(items, 1)
            neg_vec = F.normalize(model.get_item(neg_items), dim=-1)
            neg_sim = (u_vec * neg_vec).sum(-1)
            
            # 损失
            logits = torch.stack([pos_sim, neg_sim], dim=1) * 10  # temperature=0.1
            labels = torch.zeros(bs, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n += 1
        
        elapsed = (datetime.now() - start).total_seconds()
        print(f"Epoch {epoch+1}: Loss={total_loss/n:.4f}, 用时={elapsed:.0f}s")
    
    total_time = (datetime.now() - start).total_seconds()
    print(f"\n训练时间: {total_time:.1f}秒")
    
    # ========== 评估 ==========
    print("\n评估...")
    model.eval()
    
    with torch.no_grad():
        all_items = torch.arange(1, n_items, device=device)
        all_vecs = F.normalize(model.get_item(all_items), dim=-1)
    
    # 按用户分组测试
    user_test = defaultdict(list)
    for s in test:
        user_test[s['user_id']].append(s['item_id'])
    
    recalls = []
    for uid, test_items in tqdm(user_test.items(), desc="评估"):
        h = user_hist.get(uid, [])[-20:]
        h = [0] * (20 - len(h)) + h
        
        with torch.no_grad():
            u = torch.tensor([uid], device=device)
            h = torch.tensor([h], device=device)
            u_vec = F.normalize(model.get_user(u, h), dim=-1)
            sim = torch.mm(u_vec, all_vecs.T).squeeze()
            
            # 排除历史
            for item in user_hist.get(uid, []):
                if 0 < item < n_items:
                    sim[item-1] = -1e9
            
            top100 = sim.topk(100).indices.cpu().numpy() + 1
            hit = len(set(top100) & set(test_items))
            recalls.append(hit / len(test_items) if test_items else 0)
    
    recall = np.mean(recalls)
    print(f"Recall@100: {recall:.4f}")
    
    # ========== 保存 ==========
    os.makedirs('/home/z/my-project/dssm_recall/checkpoints', exist_ok=True)
    os.makedirs('/home/z/my-project/dssm_recall/logs', exist_ok=True)
    
    torch.save(model.state_dict(), '/home/z/my-project/dssm_recall/checkpoints/model.pt')
    
    results = {
        'Recall@100': round(recall, 4),
        'total_time_seconds': int(total_time),
        'train_samples': len(train),
        'test_samples': len(test)
    }
    
    with open('/home/z/my-project/dssm_recall/logs/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n完成!")
    return results


if __name__ == "__main__":
    main()
