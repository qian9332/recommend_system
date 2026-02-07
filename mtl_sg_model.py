"""# -*- coding: utf-8 -*-
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# ---------------------------
# Scene 编码模块
# ---------------------------
class SceneEncoder(nn.Module):
    """
    将场景相关信号编码为一个连续向量 x_scene。
    """
    def __init__(self, num_resource_ids: int, resource_emb_dim: int = 32, realtime_dim: int = 8, scene_dim: int = 128):
        super().__init__()
        self.resource_emb = nn.Embedding(num_resource_ids, resource_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(resource_emb_dim + realtime_dim + 0, scene_dim),
            nn.ReLU(),
            nn.Linear(scene_dim, scene_dim),
            nn.ReLU()
        )
        self.hist_proj = nn.Linear(0, 0) if False else None
        self.scene_dim = scene_dim

    def forward(self, hist_scene_dist: torch.Tensor, resource_id: torch.LongTensor, realtime_feats: torch.Tensor):
        emb = self.resource_emb(resource_id)
        if hist_scene_dist is None:
            hist_scene_dist = torch.zeros(emb.size(0), 0, device=emb.device)
        x = torch.cat([emb, realtime_feats, hist_scene_dist], dim=1)
        scene_vec = self.mlp(x)
        return scene_vec


class SimpleFM(nn.Module):
    def __init__(self, embedding_dims: List[int], emb_out_dim: int = 128):
        super().__init__()
        self.num_fields = len(embedding_dims)
        self.emb_tables = nn.ModuleList([nn.Embedding(n, emb_out_dim) for n in embedding_dims])
        self.output_dim = emb_out_dim * self.num_fields
        self.project = nn.Sequential(nn.Linear(self.output_dim, emb_out_dim), nn.ReLU())

    def forward(self, ids: List[torch.LongTensor]):
        embs = [table(id_) for table, id_ in zip(self.emb_tables, ids)]
        concat = torch.cat(embs, dim=1)
        out = self.project(concat)
        return out


class BCNLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.W = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.U = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        tmp = torch.mul(x @ self.W, x)
        out = x + F.relu(tmp @ self.U + self.bias)
        return out


class Expert(nn.Module):
    def __init__(self, input_dim: int, bcn_layers: int = 2, hidden_dim: int = 128, output_dims: dict = None):
        super().__init__()
        self.bcn_layers = nn.ModuleList([BCNLayer(input_dim) for _ in range(bcn_layers)])
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.output_dim = hidden_dim
        self.output_heads = nn.ModuleDict()
        if output_dims:
            for task, dim in output_dims.items():
                self.output_heads[task] = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        out = x
        for l in self.bcn_layers:
            out = l(out)
        out = self.mlp(out)
        outputs = {}
        for task, head in self.output_heads.items():
            outputs[task] = head(out)
        return outputs, out


class GateNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)
        return probs, logits

    def prior_kl_loss(self, g_s: torch.Tensor, p_hist: torch.Tensor, exclude_idx: Optional[List[int]] = None, eps=1e-8):
        if p_hist is None:
            return torch.tensor(0.0, device=g_s.device)
        if exclude_idx:
            mask = torch.ones_like(g_s)
            mask[:, exclude_idx] = 0.0
            g_masked = g_s * mask
            p_masked = p_hist * mask
            g_norm = g_masked.sum(dim=1, keepdim=True).clamp(min=eps)
            p_norm = p_masked.sum(dim=1, keepdim=True).clamp(min=eps)
            g = g_masked / g_norm
            p = p_masked / p_norm
        else:
            g = g_s
            p = p_hist
        kl = (g * (torch.log(g + eps) - torch.log(p + eps))).sum(dim=1).mean()
        return kl


class MTL_SG(nn.Module):
    def __init__(self, num_experts: int, expert_output_tasks: dict, embedding_dims: List[int], resource_vocab_size: int,
                 scene_dim: int = 128, user_pref_dim: int = 64, context_dim: int = 16, expert_bcn_layers: int = 2):
        super().__init__()
        self.shared_bottom = SimpleFM(embedding_dims, emb_out_dim=128)
        self.scene_encoder = SceneEncoder(num_resource_ids=resource_vocab_size, scene_dim=scene_dim,
                                          realtime_dim=8, resource_emb_dim=32)
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_dim=128, bcn_layers=expert_bcn_layers,
                                             hidden_dim=128, output_dims=expert_output_tasks) for _ in range(num_experts)])
        gate_input_dim = scene_dim + user_pref_dim + context_dim + 128
        self.gate = GateNetwork(gate_input_dim, num_experts)

    def forward(self, ids_fields: List[torch.LongTensor], hist_scene_dist: torch.Tensor, resource_id: torch.LongTensor,
                realtime_feats: torch.Tensor, user_short_pref: torch.Tensor, context_feats: torch.Tensor,
                p_hist: Optional[torch.Tensor] = None, prior_exclude_idx: Optional[List[int]] = None, topk: Optional[int] = None):
        B = ids_fields[0].size(0)
        shared_repr = self.shared_bottom(ids_fields)
        scene_vec = self.scene_encoder(hist_scene_dist, resource_id, realtime_feats)
        gate_input = torch.cat([scene_vec, user_short_pref, context_feats, shared_repr], dim=1)
        g_s, gate_logits = self.gate(gate_input)

        if topk is not None:
            topk_vals, topk_idx = torch.topk(g_s, topk, dim=1)
            mask = torch.zeros_like(g_s)
            mask.scatter_(1, topk_idx, topk_vals)
            g_s = mask / (mask.sum(dim=1, keepdim=True) + 1e-9)

        all_expert_hidden = []
        all_expert_task_outputs = []
        for expert in self.experts:
            task_outputs, hidden = expert(shared_repr)
            all_expert_task_outputs.append(task_outputs)
            all_expert_hidden.append(hidden)

        expert_hiddens = torch.stack(all_expert_hidden, dim=1)
        g_s_expanded = g_s.unsqueeze(-1)
        mixed_hidden = (expert_hiddens * g_s_expanded).sum(dim=1)

        example_outputs = all_expert_task_outputs[0]
        final_outputs = {}
        for task, head in self.experts[0].output_heads.items():
            final_outputs[task] = head(mixed_hidden)

        prior_loss = torch.tensor(0.0, device=g_s.device)
        if p_hist is not None:
            prior_loss = self.gate.prior_kl_loss(g_s, p_hist, exclude_idx=prior_exclude_idx)

        return final_outputs, g_s, prior_loss
