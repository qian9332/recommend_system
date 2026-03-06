"""
双塔语义召回模型 - 平衡版
目标：5分钟内完成训练，召回率尽可能高
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
    def __init__(self, samples, user_hist, seq_len=20):
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
        
        self.user_mlp = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.item_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    
    def get_user(self, u, h):
        return self.user_mlp(torch.cat([self.user_emb(u), self.hist_emb(h).mean(1)], -1))
    
    def get_item(self, i):
        return self.item_mlp(self.item_emb(i))


def main():
    print("=" * 60)
    print("双塔模型 - 平衡版")
    print("=" * 60)
    print(f"时间: {datetime.now()}")
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 数据加载 - 采样30%
    print("\n【加载数据】")
    behaviors, user_cnt, item_cnt = [], Counter(), Counter()
    
    with open('/home/z/my-project/dssm_recall/data/UserBehavior.csv') as f:
        for line in tqdm(f, desc="读取"):
            if random.random() > 0.3:
                continue
            p = line.strip().split(',')
            if len(p) == 5:
                behaviors.append((p[0], p[1], p[3]))
                user_cnt[p[0]] += 1
                item_cnt[p[1]] += 1
    
    print(f"采样记录: {len(behaviors):,}")
    
    # 预处理
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
    
    # 训练
    print("\n【训练】")
    dataset = FastDataset(train, user_hist)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    model = Model(n_users, n_items, dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    log = []
    start = datetime.now()
    
    for epoch in range(8):
        model.train()
        loss_sum, n = 0, 0
        
        for u, i, h in tqdm(loader, desc=f"Epoch {epoch+1}/8"):
            u, i, h = u.to(device), i.to(device), h.to(device)
            bs = u.size(0)
            
            uv = F.normalize(model.get_user(u, h), -1)
            iv = F.normalize(model.get_item(i), -1)
            pos = (uv * iv).sum(-1)
            
            # In-Batch + 随机负采样
            neg = torch.cat([
                (uv * F.normalize(model.get_item(torch.roll(i, k)), -1)).sum(-1).unsqueeze(1)
                for k in range(1, 5)
            ] + [
                (uv.unsqueeze(1) * F.normalize(model.get_item(
                    torch.randint(1, n_items, (bs, 4), device=device).view(-1)
                ).view(bs, 4, -1), -1)).sum(-1)
            ], 1)
            
            logits = torch.cat([pos.unsqueeze(1), neg], 1) / 0.1
            loss = F.cross_entropy(logits, torch.zeros(bs, dtype=torch.long, device=device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n += 1
        
        t = (datetime.now() - start).total_seconds()
        print(f"Epoch {epoch+1}: Loss={loss_sum/n:.4f}, 用时={t:.0f}s")
        log.append({'epoch': epoch+1, 'loss': round(loss_sum/n, 4), 'time': int(t)})
    
    total_time = (datetime.now() - start).total_seconds()
    print(f"\n训练时间: {total_time:.1f}秒")
    
    # 评估
    print("\n【评估】")
    model.eval()
    
    with torch.no_grad():
        all_vec = F.normalize(model.get_item(torch.arange(1, n_items, device=device)), -1)
    
    user_test = defaultdict(list)
    for s in test:
        user_test[s['user_id']].append(s['item_id'])
    
    recalls = []
    for uid, items in tqdm(user_test.items(), desc="评估"):
        h = user_hist.get(uid, [])[-20:]
        h = [0] * (20 - len(h)) + h
        
        with torch.no_grad():
            uv = F.normalize(model.get_user(
                torch.tensor([uid], device=device),
                torch.tensor([h], device=device)
            ), -1)
            sim = torch.mm(uv, all_vec.T).squeeze()
            
            for it in user_hist.get(uid, []):
                if 0 < it < n_items:
                    sim[it-1] = -1e9
            
            top100 = sim.topk(100).indices.cpu().numpy() + 1
            recalls.append(len(set(top100) & set(items)) / len(items) if items else 0)
    
    recall = np.mean(recalls)
    print(f"Recall@100: {recall:.4f}")
    
    # 保存
    os.makedirs('/home/z/my-project/dssm_recall/checkpoints', exist_ok=True)
    os.makedirs('/home/z/my-project/dssm_recall/logs', exist_ok=True)
    
    torch.save(model.state_dict(), '/home/z/my-project/dssm_recall/checkpoints/model.pt')
    
    results = {
        'Recall@100': round(recall, 4),
        'total_time_seconds': int(total_time),
        'train_samples': len(train),
        'test_samples': len(test)
    }
    
    with open('/home/z/my-project/dssm_recall/logs/train_log.json', 'w') as f:
        json.dump(log, f, indent=2)
    with open('/home/z/my-project/dssm_recall/logs/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n完成!")
    return results


if __name__ == "__main__":
    main()
