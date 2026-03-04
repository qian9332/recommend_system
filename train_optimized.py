"""
双塔语义召回模型 - 高效优化版
实现四种负采样策略 + 分阶段混合采样
优化: 批量TopK + 预计算历史序列（移除正例mask避免OOM）
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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==================== 数据处理 ====================

class TaobaoDataset(Dataset):
    """优化版数据集：预计算历史序列"""
    
    def __init__(self, samples, user_history, max_seq_len=50):
        self.samples = samples
        self.max_seq_len = max_seq_len
        
        # 优化：预计算所有历史序列
        print("预计算历史序列...")
        self.precomputed_history = {}
        for s in tqdm(samples, desc="预处理"):
            uid = s['user_id']
            if uid not in self.precomputed_history:
                h = user_history.get(uid, [])[-max_seq_len:]
                h = [0] * (max_seq_len - len(h)) + h
                self.precomputed_history[uid] = np.array(h, dtype=np.int64)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'user_id': s['user_id'],
            'item_id': s['item_id'],
            'history': self.precomputed_history[s['user_id']]
        }


# ==================== 模型定义 ====================

class TwoTowerModel(nn.Module):
    """双塔模型"""
    
    def __init__(self, num_users, num_items, embed_dim=64):
        super().__init__()
        self.num_items = num_items
        
        # 嵌入层
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.history_embedding = nn.Embedding(num_items, embed_dim)
        
        # 初始化
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.history_embedding.weight)
        
        # 用户塔
        self.user_tower = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # 商品塔
        self.item_tower = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
    
    def get_user_vector(self, user_ids, history):
        user_emb = self.user_embedding(user_ids)
        hist_emb = self.history_embedding(history)
        hist_emb = hist_emb.mean(dim=1)
        user_vec = self.user_tower(torch.cat([user_emb, hist_emb], dim=-1))
        return user_vec
    
    def get_item_vector(self, item_ids):
        item_emb = self.item_embedding(item_ids)
        item_vec = self.item_tower(item_emb)
        return item_vec
    
    def forward(self, user_ids, item_ids, history):
        user_vec = self.get_user_vector(user_ids, history)
        item_vec = self.get_item_vector(item_ids)
        return user_vec, item_vec


# ==================== 负采样器（优化版）====================

class NegativeSampler:
    """优化版负采样器：批量TopK"""
    
    def __init__(self, num_items, user_inc_items, user_positive_items):
        self.num_items = num_items
        self.user_inc_items = user_inc_items
        self.user_positive_items = user_positive_items  # 保留字典形式，避免OOM
    
    def random_negative(self, batch_size, num_neg, device):
        return torch.randint(1, self.num_items, (batch_size, num_neg), device=device)
    
    def inc_negative(self, user_ids, num_neg, device):
        """INC负采样"""
        batch_size = user_ids.size(0)
        neg_items = []
        
        for uid in user_ids.cpu().numpy():
            inc_list = self.user_inc_items.get(int(uid), [])
            if len(inc_list) >= num_neg:
                selected = random.sample(inc_list, num_neg)
            else:
                selected = list(inc_list)
                while len(selected) < num_neg:
                    rand_item = random.randint(1, self.num_items - 1)
                    if rand_item not in selected:
                        selected.append(rand_item)
            neg_items.append(selected)
        
        return torch.tensor(neg_items, device=device, dtype=torch.long)
    
    def inbatch_negative(self, item_ids, num_neg):
        """In-Batch负采样"""
        batch_size = item_ids.size(0)
        device = item_ids.device
        neg_items = []
        
        for i in range(batch_size):
            others = torch.cat([item_ids[:i], item_ids[i+1:]])
            if len(others) >= num_neg:
                indices = torch.randperm(len(others), device=device)[:num_neg]
                neg = others[indices]
            else:
                random_neg = torch.randint(1, self.num_items, (num_neg - len(others),), device=device)
                neg = torch.cat([others, random_neg])
            neg_items.append(neg)
        
        return torch.stack(neg_items)
    
    def hard_negative_optimized(self, user_vectors, all_item_vectors, user_ids, num_neg, device):
        """优化版难负例挖掘：批量TopK"""
        batch_size = user_vectors.size(0)
        
        # 归一化
        user_vec_norm = F.normalize(user_vectors, dim=-1)
        item_vec_norm = F.normalize(all_item_vectors, dim=-1)
        
        # 批量计算相似度 [B, num_items]
        similarities = torch.mm(user_vec_norm, item_vec_norm.T)
        
        # 优化：批量TopK，一次完成 [B, K]
        top_k = similarities.topk(num_neg * 2, dim=1).indices
        
        # 过滤正例（使用字典，避免OOM）
        neg_items = []
        for i in range(batch_size):
            uid = int(user_ids[i].item())
            positive_items = self.user_positive_items.get(uid, set())
            top_k_items = top_k[i] + 1  # +1因为0是padding
            
            # 过滤正例
            valid_neg = []
            for item_id in top_k_items.cpu().numpy():
                if item_id not in positive_items and item_id < self.num_items:
                    valid_neg.append(item_id)
                    if len(valid_neg) >= num_neg:
                        break
            
            # 不够则补充随机
            while len(valid_neg) < num_neg:
                rand_item = random.randint(1, self.num_items - 1)
                if rand_item not in positive_items and rand_item not in valid_neg:
                    valid_neg.append(rand_item)
            
            neg_items.append(valid_neg[:num_neg])
        
        return torch.tensor(neg_items, device=device, dtype=torch.long)


# ==================== 训练器 ====================

class Trainer:
    """训练器"""
    
    def __init__(self, model, neg_sampler, device, temperature=0.05):
        self.model = model
        self.neg_sampler = neg_sampler
        self.device = device
        self.temperature = temperature
        
        # 优化：减少难负例阶段比例
        self.stage_config = {
            'stage1': {'ratio': 0.2, 'main_strategy': 'inc', 'random_ratio': 0.1},
            'stage2': {'ratio': 0.7, 'main_strategy': 'inbatch', 'random_ratio': 0.05},
            'stage3': {'ratio': 0.1, 'main_strategy': 'hard', 'random_ratio': 0.05}
        }
    
    def get_current_stage(self, progress):
        if progress < 0.2:
            return 'stage1'
        elif progress < 0.9:
            return 'stage2'
        else:
            return 'stage3'
    
    def compute_loss(self, user_vec, pos_item_vec, neg_item_vec):
        user_vec = F.normalize(user_vec, dim=-1)
        pos_item_vec = F.normalize(pos_item_vec, dim=-1)
        neg_item_vec = F.normalize(neg_item_vec, dim=-1)
        
        pos_sim = torch.sum(user_vec * pos_item_vec, dim=-1) / self.temperature
        neg_sim = torch.bmm(neg_item_vec, user_vec.unsqueeze(-1)).squeeze(-1) / self.temperature
        
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(user_vec.size(0), dtype=torch.long, device=self.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss, pos_sim.mean().item(), neg_sim.mean().item()
    
    def train_epoch(self, dataloader, optimizer, epoch, total_epochs, all_item_vectors, scheduler=None):
        self.model.train()
        
        progress = epoch / total_epochs
        stage = self.get_current_stage(progress)
        config = self.stage_config[stage]
        
        stage_names = {'stage1': 'INC阶段', 'stage2': 'In-Batch阶段', 'stage3': '难负例阶段'}
        
        total_loss = 0
        total_pos_sim = 0
        total_neg_sim = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [{stage_names[stage]}]")
        
        for batch in pbar:
            user_ids = batch['user_id'].to(self.device)
            item_ids = batch['item_id'].to(self.device)
            history = batch['history'].to(self.device)
            
            batch_size = user_ids.size(0)
            num_neg = 8
            
            user_vec, pos_item_vec = self.model(user_ids, item_ids, history)
            
            main_num_neg = int(num_neg * (1 - config['random_ratio']))
            random_num_neg = num_neg - main_num_neg
            
            if config['main_strategy'] == 'inc':
                main_neg = self.neg_sampler.inc_negative(user_ids, main_num_neg, self.device)
            elif config['main_strategy'] == 'inbatch':
                main_neg = self.neg_sampler.inbatch_negative(item_ids, main_num_neg)
            else:
                main_neg = self.neg_sampler.hard_negative_optimized(
                    user_vec.detach(), all_item_vectors, user_ids, main_num_neg, self.device
                )
            
            random_neg = self.neg_sampler.random_negative(batch_size, random_num_neg, self.device)
            neg_item_ids = torch.cat([main_neg, random_neg], dim=1)
            
            neg_item_vec = self.model.get_item_vector(neg_item_ids.view(-1))
            neg_item_vec = neg_item_vec.view(batch_size, num_neg, -1)
            
            loss, pos_sim, neg_sim = self.compute_loss(user_vec, pos_item_vec, neg_item_vec)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_pos_sim += pos_sim
            total_neg_sim += neg_sim
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pos': f'{pos_sim:.2f}',
                'neg': f'{neg_sim:.2f}'
            })
        
        if scheduler:
            scheduler.step()
        
        if num_batches == 0:
            return {'loss': 0, 'pos_sim': 0, 'neg_sim': 0, 'stage': stage_names[stage]}
        
        return {
            'loss': total_loss / num_batches,
            'pos_sim': total_pos_sim / num_batches,
            'neg_sim': total_neg_sim / num_batches,
            'stage': stage_names[stage]
        }


# ==================== 评估 ====================

def evaluate(model, test_data, user_history, num_items, device, k_list=[10, 50, 100]):
    model.eval()
    
    with torch.no_grad():
        all_item_ids = torch.arange(1, num_items, device=device)
        all_item_vec = model.get_item_vector(all_item_ids)
        all_item_vec = F.normalize(all_item_vec, dim=-1)
    
    user_test_items = defaultdict(list)
    for s in test_data:
        user_test_items[s['user_id']].append(s['item_id'])
    
    recalls = {k: [] for k in k_list}
    
    for user_id, test_items in tqdm(user_test_items.items(), desc="评估中"):
        if not test_items:
            continue
        
        with torch.no_grad():
            hist = user_history.get(user_id, [])[-50:]
            hist = [0] * (50 - len(hist)) + hist
            
            user_tensor = torch.tensor([user_id], device=device)
            hist_tensor = torch.tensor([hist], device=device)
            
            user_vec = model.get_user_vector(user_tensor, hist_tensor)
            user_vec = F.normalize(user_vec, dim=-1)
            
            sim = torch.mm(user_vec, all_item_vec.T).squeeze()
            
            for item in user_history.get(user_id, []):
                if 0 < item < num_items:
                    sim[item - 1] = -float('inf')
            
            for k in k_list:
                top_k = sim.topk(k).indices.cpu().numpy() + 1
                hit = len(set(top_k) & set(test_items))
                recalls[k].append(hit / len(test_items) if test_items else 0)
    
    results = {}
    for k in k_list:
        recall = np.mean(recalls[k]) if recalls[k] else 0
        results[f'Recall@{k}'] = round(recall, 4)
    
    return results


# ==================== 主函数 ====================

def main(sample_ratio=1.0, epochs=20, temperature=0.05, lr=0.001):
    """主函数"""
    print("=" * 70)
    print("双塔语义召回模型训练 - 高效优化版")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据采样比例: {sample_ratio*100:.0f}%")
    print(f"训练轮数: {epochs}")
    print(f"温度系数: {temperature}")
    
    set_seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 数据加载
    print("\n" + "=" * 70)
    print("【数据加载】")
    print("=" * 70)
    
    data_path = '/home/z/my-project/dssm_recall/data/UserBehavior.csv'
    
    behaviors = []
    user_counter = Counter()
    item_counter = Counter()
    behavior_counter = Counter()
    
    with open(data_path, 'r') as f:
        for line in tqdm(f, desc="读取数据"):
            parts = line.strip().split(',')
            if len(parts) == 5:
                if random.random() > sample_ratio:
                    continue
                    
                user_id, item_id, category_id, behavior_type, timestamp = parts
                behaviors.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'behavior_type': behavior_type,
                    'timestamp': int(timestamp)
                })
                user_counter[user_id] += 1
                item_counter[item_id] += 1
                behavior_counter[behavior_type] += 1
    
    total = len(behaviors)
    print(f"\n总记录: {total:,}")
    print(f"用户数: {len(user_counter):,}")
    print(f"商品数: {len(item_counter):,}")
    print(f"\n行为分布:")
    for b, c in behavior_counter.most_common():
        print(f"  {b}: {c:,} ({c/total*100:.1f}%)")
    
    # 数据预处理
    print("\n" + "=" * 70)
    print("【数据预处理】")
    print("=" * 70)
    
    valid_users = {u for u, c in user_counter.items() if c >= 2}
    valid_items = {i for i, c in item_counter.items() if c >= 2}
    
    print(f"过滤后: 用户 {len(valid_users):,}, 商品 {len(valid_items):,}")
    
    user_encoder = {u: i+1 for i, u in enumerate(valid_users)}
    item_encoder = {i: j+1 for j, i in enumerate(valid_items)}
    
    num_users = len(valid_users) + 1
    num_items = len(valid_items) + 1
    
    print(f"编码后: 用户 {num_users}, 商品 {num_items}")
    
    samples = []
    for b in behaviors:
        if b['user_id'] in valid_users and b['item_id'] in valid_items:
            if b['behavior_type'] in ['buy', 'cart', 'fav']:
                samples.append({
                    'user_id': user_encoder[b['user_id']],
                    'item_id': item_encoder[b['item_id']],
                    'timestamp': b['timestamp']
                })
    
    samples.sort(key=lambda x: x['timestamp'])
    n = len(samples)
    train_data = samples[:int(n*0.8)]
    test_data = samples[int(n*0.9):]
    
    print(f"训练样本: {len(train_data):,}")
    print(f"测试样本: {len(test_data):,}")
    
    user_history = defaultdict(list)
    user_positive_items = defaultdict(set)
    for s in train_data:
        user_history[s['user_id']].append(s['item_id'])
        user_positive_items[s['user_id']].add(s['item_id'])
    
    print("\n构建INC样本...")
    user_inc_items = defaultdict(list)
    for b in tqdm(behaviors, desc="INC构建"):
        if b['behavior_type'] == 'pv' and b['user_id'] in valid_users and b['item_id'] in valid_items:
            uid = user_encoder[b['user_id']]
            iid = item_encoder[b['item_id']]
            if iid not in user_positive_items[uid]:
                user_inc_items[uid].append(iid)
    
    total_inc = sum(len(v) for v in user_inc_items.values())
    print(f"INC样本总数: {total_inc:,}")
    
    # 模型训练
    print("\n" + "=" * 70)
    print("【模型训练】")
    print("=" * 70)
    
    train_dataset = TaobaoDataset(train_data, user_history)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    
    model = TwoTowerModel(num_users, num_items, embed_dim=64).to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    neg_sampler = NegativeSampler(num_items, dict(user_inc_items), dict(user_positive_items))
    trainer = Trainer(model, neg_sampler, device, temperature=temperature)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
    
    with torch.no_grad():
        all_item_ids = torch.arange(1, num_items, device=device)
        all_item_vectors = model.get_item_vector(all_item_ids)
    
    history = []
    start_time = datetime.now()
    
    for epoch in range(epochs):
        if epoch > 0:
            with torch.no_grad():
                all_item_vectors = model.get_item_vector(all_item_ids)
        
        metrics = trainer.train_epoch(train_loader, optimizer, epoch, epochs, all_item_vectors, scheduler)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"Epoch {epoch+1}: Loss={metrics['loss']:.4f}, PosSim={metrics['pos_sim']:.4f}, NegSim={metrics['neg_sim']:.4f}, 已用时={elapsed:.0f}秒")
        
        history.append({
            'epoch': epoch + 1,
            'loss': round(metrics['loss'], 4),
            'pos_sim': round(metrics['pos_sim'], 4),
            'neg_sim': round(metrics['neg_sim'], 4),
            'stage': metrics['stage'],
            'elapsed_seconds': int(elapsed)
        })
    
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n总训练时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    
    # 评估
    print("\n" + "=" * 70)
    print("【模型评估】")
    print("=" * 70)
    
    results = evaluate(model, test_data, user_history, num_items, device)
    
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    
    results['total_time_seconds'] = int(total_time)
    results['total_time_minutes'] = round(total_time / 60, 2)
    
    # 保存
    os.makedirs('/home/z/my-project/dssm_recall/checkpoints', exist_ok=True)
    os.makedirs('/home/z/my-project/dssm_recall/logs', exist_ok=True)
    
    torch.save(model.state_dict(), '/home/z/my-project/dssm_recall/checkpoints/model_optimized.pt')
    
    with open('/home/z/my-project/dssm_recall/logs/training_history_optimized.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    with open('/home/z/my-project/dssm_recall/logs/evaluation_results_optimized.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    
    return history, results


if __name__ == "__main__":
    main(sample_ratio=1.0, epochs=20, temperature=0.05, lr=0.001)
