# 双塔语义召回模型

基于PyTorch实现的双塔语义召回模型，支持四种负采样策略和分阶段混合采样。

## 项目特点

### 四种负采样策略

| 策略 | 描述 | 适用场景 |
|------|------|---------|
| **随机负采样** | 从全库均匀随机抽取 | 基础策略，保证多样性 |
| **In-Batch Negative** | 同batch内其他样本的正例作为负例 | 高效利用batch内信息 |
| **曝光未点击(INC)** | 线上曝光但未点击的商品 | 利用用户真实行为数据 |
| **难负例挖掘** | 相似度高但非正例的商品 | 提升模型区分能力 |

### 分阶段混合采样方案

```
阶段一 (0%-20%):  INC负采样为主 + 5%-10%随机负例
阶段二 (20%-80%): In-Batch负采样为主 + 5%-10%随机负例
阶段三 (80%-100%): 难负例挖掘为主 + 5%-10%随机负例
```

## 模型架构

```
用户塔:
  用户ID嵌入 → 用户历史行为序列编码 → MLP → 用户向量

商品塔:
  商品ID嵌入 → MLP → 商品向量

损失函数: InfoNCE Loss with Temperature Scaling
```

## 项目结构

```
dssm_recall/
├── train.py              # 训练代码
├── checkpoints/          # 模型保存目录
│   └── model.pt
├── logs/                 # 训练日志
│   ├── training_history.json
│   └── evaluation_results.json
├── data/                 # 数据目录
│   └── UserBehavior.csv.gz
└── README.md
```

## 数据集

使用淘宝用户行为数据集，包含以下字段：
- user_id: 用户ID
- item_id: 商品ID
- category_id: 类目ID
- behavior_type: 行为类型 (pv/cart/fav/buy)
- timestamp: 时间戳

### 行为类型分布

| 行为类型 | 说明 | 占比 |
|---------|------|------|
| pv | 浏览 | ~89.5% |
| cart | 加购 | ~5.6% |
| fav | 收藏 | ~2.9% |
| buy | 购买 | ~2.0% |

## 使用方法

### 环境要求

```
Python 3.8+
PyTorch 1.10+
NumPy
tqdm
```

### 训练模型

```bash
# 使用10%数据进行快速验证
python train.py

# 使用全量数据训练（修改sample_ratio参数）
# 在train.py中修改 main(sample_ratio=1.0)
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| embed_dim | 64 | 嵌入维度 |
| batch_size | 256 | 批次大小 |
| learning_rate | 0.001 | 学习率 |
| epochs | 10 | 训练轮数 |
| temperature | 0.1 | 温度系数 |
| num_neg | 8 | 负样本数量 |

## 训练结果

### 10%数据采样验证结果

| Epoch | 阶段 | Loss | PosSim | NegSim |
|-------|------|------|--------|--------|
| 1 | INC阶段 | 2.1798 | -0.6532 | -0.6955 |
| 2 | INC阶段 | 1.4154 | 3.5751 | 1.2237 |
| 3 | In-Batch阶段 | 2.1288 | 8.3811 | 8.1383 |
| 4 | In-Batch阶段 | 1.7847 | 7.2884 | 5.8367 |
| 5 | In-Batch阶段 | 1.3495 | 7.6443 | 4.7365 |
| 6 | In-Batch阶段 | 0.8719 | 8.3346 | 3.0073 |
| 7 | In-Batch阶段 | 0.6652 | 8.6905 | 2.0174 |
| 8 | In-Batch阶段 | 0.4136 | 8.6842 | 0.5609 |
| 9 | 难负例阶段 | 2.1048 | 7.2897 | 6.3772 |
| 10 | 难负例阶段 | 2.2689 | 6.3612 | 5.6280 |

### 评估结果

- **Recall@100**: 0.09%

## 结果分析

### 训练过程分析

1. **阶段1 (INC阶段)**: Loss从2.18下降到1.42，模型开始学习区分正负样本
2. **阶段2 (In-Batch阶段)**: Loss持续下降到0.41，正样本相似度上升，负样本相似度下降
3. **阶段3 (难负例阶段)**: Loss上升到2.1-2.3，说明难负例对模型有挑战

### 问题与改进方向

1. **Recall较低的原因**:
   - 数据稀疏性严重
   - 用户-商品交互矩阵稀疏
   - 训练数据量不足

2. **改进方向**:
   - 增加训练数据量
   - 增加训练轮数
   - 调整温度系数
   - 优化难负例挖掘策略
   - 添加更多特征（类目、价格等）

## 技术细节

### InfoNCE损失函数

```
L = -log(exp(sim(u,i+)/τ) / Σexp(sim(u,ij)/τ))
```

其中：
- u: 用户向量
- i+: 正样本商品向量
- ij: 负样本商品向量
- τ: 温度系数

### 梯度裁剪

为防止梯度爆炸，使用梯度裁剪：
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 参考文献

1. Deep Semantic Similarity for Personalized Recommender Systems
2. Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations
3. Contrastive Learning for Sequential Recommendation

## License

MIT License
