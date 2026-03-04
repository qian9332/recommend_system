"""
双塔语义召回模型 - 修复版
实现四种负采样策略 + 分阶段混合采样
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
    def __init__(self, samples, user_history, max_seq_len=50):
        self.samples = samples
        self.user_history = user_history
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        h = self.user_history.get(s['user_id'], [])[-self.max_seq_len:]
        h = [0] * (self.max_seq_len - len(h)) + h
        return {
            'user_id': s['user_id'],
            'item_id': s['item_id'],
            'history': np.array(h, dtype=np.int64)
        }


# ==================== 模型定义 ====================

class TwoTowerModel(nn.Module):
    """双塔模型"""
    
    def __init__(self, num_users, num_items, embed_dim=64):
        super().__init__()
        self.num_items = num_items
        
        # 嵌入层 - 使用Xavier初始化
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.history_embedding = nn.Embedding(num_items, embed_dim)
        
        # 初始化
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.history_embedding.weight)
        
        # 用户塔
        self.user_tower = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 商品塔
        self.item_tower = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def get_user_vector(self, user_ids, history):
        """获取用户向量"""
        user_emb = self.user_embedding(user_ids)  # [B, D]
        hist_emb = self.history_embedding(history)  # [B, L, D]
        hist_emb = hist_emb.mean(dim=1)  # [B, D]
        
        user_vec = self.user_tower(torch.cat([user_emb, hist_emb], dim=-1))
        return user_vec
    
    def get_item_vector(self, item_ids):
        """获取商品向量"""
        item_emb = self.item_embedding(item_ids)
        item_vec = self.item_tower(item_emb)
        return item_vec
    
    def forward(self, user_ids, item_ids, history):
        user_vec = self.get_user_vector(user_ids, history)
        item_vec = self.get_item_vector(item_ids)
        return user_vec, item_vec


# ==================== 负采样器 ====================

class NegativeSampler:
    """负采样器 - 实现四种策略"""
    
    def __init__(self, num_items, user_inc_items, user_positive_items):
        self.num_items = num_items
        self.user_inc_items = user_inc_items  # {user_id: [item_ids]}
        self.user_positive_items = user_positive_items  # {user_id: set(item_ids)}
    
    def random_negative(self, batch_size, num_neg, device):
        """随机负采样"""
        return torch.randint(1, self.num_items, (batch_size, num_neg), device=device)
    
    def inc_negative(self, user_ids, num_neg, device):
        """INC负采样（曝光未点击）"""
        batch_size = user_ids.size(0)
        neg_items = []
        
        for uid in user_ids.cpu().numpy():
            inc_list = self.user_inc_items.get(int(uid), [])
            if len(inc_list) >= num_neg:
                selected = random.sample(inc_list, num_neg)
            else:
                # 不够则用随机补充
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
                # 不够则用随机补充
                random_neg = torch.randint(1, self.num_items, (num_neg - len(others),), device=device)
                neg = torch.cat([others, random_neg])
            neg_items.append(neg)
        
        return torch.stack(neg_items)
    
    def hard_negative(self, user_vectors, all_item_vectors, user_ids, num_neg, device):
        """难负例挖掘"""
        batch_size = user_vectors.size(0)
        
        # 归一化
        user_vec_norm = F.normalize(user_vectors, dim=-1)
        item_vec_norm = F.normalize(all_item_vectors, dim=-1)
        
        # 计算相似度
        similarities = torch.mm(user_vec_norm, item_vec_norm.T)  # [B, num_items]
        
        neg_items = []
        for i in range(batch_size):
            uid = int(user_ids[i].item())
            positive_items = self.user_positive_items.get(uid, set())
            
            # 获取Top-K相似商品
            top_k = similarities[i].topk(num_neg * 3).indices.cpu().numpy()
            
            # 排除正例
            hard_neg = []
            for item_idx in top_k:
                item_id = item_idx + 1  # 因为0是padding
                if item_id not in positive_items and item_id < self.num_items:
                    hard_neg.append(item_id)
                if len(hard_neg) >= num_neg:
                    break
            
            # 不够则补充随机
            while len(hard_neg) < num_neg:
                rand_item = random.randint(1, self.num_items - 1)
                if rand_item not in positive_items and rand_item not in hard_neg:
                    hard_neg.append(rand_item)
            
            neg_items.append(hard_neg[:num_neg])
        
        return torch.tensor(neg_items, device=device, dtype=torch.long)


# ==================== 训练器 ====================

class Trainer:
    """训练器 - 分阶段混合采样"""
    
    def __init__(self, model, neg_sampler, device, temperature=0.1):
        self.model = model
        self.neg_sampler = neg_sampler
        self.device = device
        self.temperature = temperature
        
        # 分阶段配置
        self.stage_config = {
            'stage1': {'ratio': 0.2, 'main_strategy': 'inc', 'random_ratio': 0.1},
            'stage2': {'ratio': 0.6, 'main_strategy': 'inbatch', 'random_ratio': 0.05},
            'stage3': {'ratio': 0.2, 'main_strategy': 'hard', 'random_ratio': 0.05}
        }
    
    def get_current_stage(self, progress):
        if progress < 0.2:
            return 'stage1'
        elif progress < 0.8:
            return 'stage2'
        else:
            return 'stage3'
    
    def compute_loss(self, user_vec, pos_item_vec, neg_item_vec):
        """计算InfoNCE损失"""
        # 归一化
        user_vec = F.normalize(user_vec, dim=-1)
        pos_item_vec = F.normalize(pos_item_vec, dim=-1)
        neg_item_vec = F.normalize(neg_item_vec, dim=-1)
        
        # 正样本相似度
        pos_sim = torch.sum(user_vec * pos_item_vec, dim=-1) / self.temperature
        
        # 负样本相似度
        neg_sim = torch.bmm(neg_item_vec, user_vec.unsqueeze(-1)).squeeze(-1) / self.temperature
        
        # InfoNCE损失
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(user_vec.size(0), dtype=torch.long, device=self.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss, pos_sim.mean().item(), neg_sim.mean().item()
    
    def train_epoch(self, dataloader, optimizer, epoch, total_epochs, all_item_vectors):
        self.model.train()
        
        progress = epoch / total_epochs
        stage = self.get_current_stage(progress)
        config = self.stage_config[stage]
        
        stage_names = {'stage1': 'INC阶段', 'stage2': 'In-Batch阶段', 'stage3': '难负例阶段'}
        print(f"\n当前阶段: {stage_names[stage]} (进度: {progress*100:.0f}%)")
        
        total_loss = 0
        total_pos_sim = 0
        total_neg_sim = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            user_ids = batch['user_id'].to(self.device)
            item_ids = batch['item_id'].to(self.device)
            history = batch['history'].to(self.device)
            
            batch_size = user_ids.size(0)
            num_neg = 8
            
            # 前向传播
            user_vec, pos_item_vec = self.model(user_ids, item_ids, history)
            
            # 获取主策略负样本
            main_num_neg = int(num_neg * (1 - config['random_ratio']))
            random_num_neg = num_neg - main_num_neg
            
            if config['main_strategy'] == 'inc':
                main_neg = self.neg_sampler.inc_negative(user_ids, main_num_neg, self.device)
            elif config['main_strategy'] == 'inbatch':
                main_neg = self.neg_sampler.inbatch_negative(item_ids, main_num_neg)
            else:  # hard
                main_neg = self.neg_sampler.hard_negative(
                    user_vec.detach(), all_item_vectors, user_ids, main_num_neg, self.device
                )
            
            # 随机负样本
            random_neg = self.neg_sampler.random_negative(batch_size, random_num_neg, self.device)
            
            # 合并负样本
            neg_item_ids = torch.cat([main_neg, random_neg], dim=1)
            
            # 获取负样本向量
            neg_item_vec = self.model.get_item_vector(neg_item_ids.view(-1))
            neg_item_vec = neg_item_vec.view(batch_size, num_neg, -1)
            
            # 计算损失
            loss, pos_sim, neg_sim = self.compute_loss(user_vec, pos_item_vec, neg_item_vec)
            
            # 检查数值是否正常
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 检测到异常Loss值，跳过此batch")
                continue
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_pos_sim += pos_sim
            total_neg_sim += neg_sim
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pos_sim': f'{pos_sim:.4f}',
                'neg_sim': f'{neg_sim:.4f}'
            })
        
        if num_batches == 0:
            return {'loss': 0, 'pos_sim': 0, 'neg_sim': 0, 'stage': stage_names[stage]}
        
        return {
            'loss': total_loss / num_batches,
            'pos_sim': total_pos_sim / num_batches,
            'neg_sim': total_neg_sim / num_batches,
            'stage': stage_names[stage]
        }


# ==================== 评估 ====================

def evaluate(model, test_data, user_history, num_items, device):
    """评估模型"""
    print("\n" + "=" * 60)
    print("评估模型")
    print("=" * 60)
    
    model.eval()
    
    # 获取所有商品向量
    with torch.no_grad():
        all_item_ids = torch.arange(1, num_items, device=device)
        all_item_vec = model.get_item_vector(all_item_ids)
        all_item_vec = F.normalize(all_item_vec, dim=-1)
    
    # 按用户分组测试数据
    user_test_items = defaultdict(list)
    for s in test_data:
        user_test_items[s['user_id']].append(s['item_id'])
    
    recalls = []
    
    for user_id, test_items in tqdm(user_test_items.items(), desc="评估中"):
        if not test_items:
            continue
        
        with torch.no_grad():
            # 获取用户向量
            hist = user_history.get(user_id, [])[-50:]
            hist = [0] * (50 - len(hist)) + hist
            
            user_tensor = torch.tensor([user_id], device=device)
            hist_tensor = torch.tensor([hist], device=device)
            
            user_vec = model.get_user_vector(user_tensor, hist_tensor)
            user_vec = F.normalize(user_vec, dim=-1)
            
            # 计算相似度
            sim = torch.mm(user_vec, all_item_vec.T).squeeze()
            
            # 排除历史商品
            for item in user_history.get(user_id, []):
                if 0 < item < num_items:
                    sim[item - 1] = -float('inf')
            
            # Top-100
            top_k = sim.topk(100).indices.cpu().numpy() + 1
            hit = len(set(top_k) & set(test_items))
            recalls.append(hit / len(test_items) if test_items else 0)
    
    recall = np.mean(recalls) if recalls else 0
    print(f"\nRecall@100: {recall:.4f}")
    
    return {'Recall@100': round(recall, 4)}


# ==================== 主函数 ====================

def main(sample_ratio=0.1):
    """主函数
    
    Args:
        sample_ratio: 数据采样比例，用于快速验证
    """
    print("=" * 70)
    print("双塔语义召回模型训练 - 修复版")
    print("实现: 四种负采样策略 + 分阶段混合采样")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据采样比例: {sample_ratio*100:.0f}%")
    
    set_seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # ==================== 数据加载 ====================
    print("\n" + "=" * 70)
    print("【数据加载与分析】")
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
    
    # ==================== 数据预处理 ====================
    print("\n" + "=" * 70)
    print("【数据预处理】")
    print("=" * 70)
    
    # 过滤冷启动
    valid_users = {u for u, c in user_counter.items() if c >= 2}
    valid_items = {i for i, c in item_counter.items() if c >= 2}
    
    print(f"过滤后: 用户 {len(valid_users):,}, 商品 {len(valid_items):,}")
    
    # 编码
    user_encoder = {u: i+1 for i, u in enumerate(valid_users)}
    item_encoder = {i: j+1 for j, i in enumerate(valid_items)}
    
    num_users = len(valid_users) + 1
    num_items = len(valid_items) + 1
    
    print(f"编码后: 用户 {num_users}, 商品 {num_items}")
    
    # 构建样本
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
    
    # 用户历史
    user_history = defaultdict(list)
    user_positive_items = defaultdict(set)
    for s in train_data:
        user_history[s['user_id']].append(s['item_id'])
        user_positive_items[s['user_id']].add(s['item_id'])
    
    # INC样本
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
    
    # ==================== 模型训练 ====================
    print("\n" + "=" * 70)
    print("【模型训练】")
    print("=" * 70)
    
    # 创建数据集
    train_dataset = TaobaoDataset(train_data, user_history)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # 创建模型
    model = TwoTowerModel(num_users, num_items, embed_dim=64).to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建负采样器
    neg_sampler = NegativeSampler(num_items, dict(user_inc_items), dict(user_positive_items))
    
    # 创建训练器
    trainer = Trainer(model, neg_sampler, device, temperature=0.1)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 获取所有商品向量
    with torch.no_grad():
        all_item_ids = torch.arange(1, num_items, device=device)
        all_item_vectors = model.get_item_vector(all_item_ids)
    
    # 训练
    epochs = 10
    history = []
    
    print(f"\n训练轮数: {epochs}")
    print(f"分阶段混合采样:")
    print(f"  - 阶段1 (0-20%): INC负采样为主")
    print(f"  - 阶段2 (20-80%): In-Batch负采样为主")
    print(f"  - 阶段3 (80-100%): 难负例挖掘为主")
    
    for epoch in range(epochs):
        # 更新商品向量
        if epoch > 0:
            with torch.no_grad():
                all_item_vectors = model.get_item_vector(all_item_ids)
        
        metrics = trainer.train_epoch(train_loader, optimizer, epoch, epochs, all_item_vectors)
        
        print(f"Epoch {epoch+1}: Loss={metrics['loss']:.4f}, PosSim={metrics['pos_sim']:.4f}, NegSim={metrics['neg_sim']:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'loss': round(metrics['loss'], 4),
            'pos_sim': round(metrics['pos_sim'], 4),
            'neg_sim': round(metrics['neg_sim'], 4),
            'stage': metrics['stage']
        })
    
    # ==================== 模型评估 ====================
    results = evaluate(model, test_data, user_history, num_items, device)
    
    # ==================== 保存 ====================
    os.makedirs('/home/z/my-project/dssm_recall/checkpoints', exist_ok=True)
    os.makedirs('/home/z/my-project/dssm_recall/logs', exist_ok=True)
    
    torch.save(model.state_dict(), '/home/z/my-project/dssm_recall/checkpoints/model.pt')
    
    with open('/home/z/my-project/dssm_recall/logs/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    with open('/home/z/my-project/dssm_recall/logs/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    
    print("\n训练历史:")
    for h in history:
        print(f"  Epoch {h['epoch']} [{h['stage']:10}]: Loss={h['loss']:.4f}, PosSim={h['pos_sim']:.4f}, NegSim={h['neg_sim']:.4f}")
    
    return history, results


if __name__ == "__main__":
    # 使用10%数据进行快速验证
    main(sample_ratio=0.1)
