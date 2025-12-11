import os
import sys
# sys.path.append('/root/diffusion/diffusers_dpm/models')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from diffusers_dpm.models.quantizer.net2quan import QuantizableLayer as Q, quantize_jit, QuanSoftmax

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, seq_len, quantize, frequency_embedding_size=256):
        super().__init__()
        self.seq_len = seq_len
        self.mlp = nn.Sequential(
            Q.Conv2d(frequency_embedding_size, hidden_size, kernel_size=1, quantize=quantize),
            Q.nn_GELU(approximate='tanh', quantize=quantize),
            Q.Conv2d(hidden_size, hidden_size, kernel_size=1, quantize=quantize),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.unsqueeze(-1).unsqueeze(-1)
        t_freq = t_freq.expand(*t_freq.shape[:-1], self.seq_len)
        t_emb = self.mlp(t_freq)
        return t_emb

    @staticmethod
    def timestep_embedding(timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


# --- 新增：任务嵌入模块 -------------------------------------------------------
class TaskEmbedder(nn.Module):
    def __init__(self, num_tasks, hidden_size, seq_len, quantize, mlp_ratio=4):
        super().__init__()
        self.seq_len = seq_len
        self.label_embed = Q.Embedding11(num_tasks, hidden_size, quantize=quantize)
        self.mlp = nn.Sequential(
            Q.Conv2d(hidden_size, hidden_size * mlp_ratio, kernel_size=1, quantize=quantize),
            Q.nn_GELU(approximate='tanh', quantize=quantize),
            Q.Conv2d(hidden_size * mlp_ratio, hidden_size, kernel_size=1, quantize=quantize),
        )

    def forward(self, task_ids):
        # task_ids: [B]，long dtype
        emb = self.label_embed(task_ids)                    # [B, hidden, 1, 1]
        emb = emb.expand(-1, -1, 1, self.seq_len)           # [B, hidden, 1, T]
        return self.mlp(emb)
    

class AdaLayerNormZero(nn.Module):
    def __init__(self, hidden_size, cond_dim, quantize):
        super().__init__()
        self.norm = Q.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, quantize=quantize)
        self.cond_gelu = Q.nn_GELU(approximate='tanh', quantize=quantize)
        self.cond_scale = Q.Conv2d(cond_dim, hidden_size, kernel_size=1, quantize=quantize)
        self.cond_shift = Q.Conv2d(cond_dim, hidden_size, kernel_size=1, quantize=quantize)
        self.cond_gate = Q.Conv2d(cond_dim, hidden_size, kernel_size=1, quantize=quantize)
        self.scale_emmul = Q.QuanEMMul(quantize=quantize)
        self.quantize = quantize
        # 初始化
        nn.init.zeros_(self.cond_scale.weight)
        nn.init.zeros_(self.cond_scale.bias)
        nn.init.zeros_(self.cond_shift.weight)
        nn.init.zeros_(self.cond_shift.bias)
        nn.init.zeros_(self.cond_gate.weight)
        nn.init.zeros_(self.cond_gate.bias)
        
    def forward(self, x, condition):
        condition = self.cond_gelu(condition)
        scale = self.cond_scale(condition)
        shift = self.cond_shift(condition)
        gate = self.cond_gate(condition)
        # cond_params = self.cond_proj(condition)
        # scale, shift, gate = cond_params.chunk(3, dim=-3)
        scale = scale  
        _, C, H, W = x.shape
        x = rearrange(x, "b (n d) h w -> b n d (h w)", n=1)
        x = x.transpose(-1, -2)
        x = self.norm(x)
        x = x.transpose(-1, -2)
        x = rearrange(x, "b n d (h w)-> b (n d) h w", h=H, w=W)
        xs = self.scale_emmul(scale, x)
        if not self.quantize or self.scale_emmul.quan_b.initialized_alpha==0:
            xq = x
        else:
            xq = quantize_jit(x, self.scale_emmul.quan_b.s, -128, 127)
        x = xs + xq + shift
        # x = scale * x + x + shift 
        return x, gate   # scale=scale+1  以1为中心


class MultiheadAttention_own(nn.Module):
    def __init__(self, embed_dim, quantize, wo_quant_softmax=True , is_causal=False, dropout_level=0.0, n_heads=4):
        super().__init__()
        self.q_linear = Q.Conv2d(embed_dim, embed_dim, kernel_size=1, quantize=quantize)
        self.k_linear = Q.Conv2d(embed_dim, embed_dim, kernel_size=1, quantize=quantize)
        self.v_linear = Q.Conv2d(embed_dim, embed_dim, kernel_size=1, quantize=quantize)
        
        self.mha = Q.MHAttention(is_causal, dropout_level, n_heads, _MHAttention=None, quantize=(quantize and wo_quant_softmax))

    def forward(self, x, y, z):
        # q, k, v = self.qkv_linear(x).chunk(3, dim=2)
        q = self.q_linear(x)
        k = self.k_linear(y)
        v = self.v_linear(z)
        return self.mha(q, k, v)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, time_cond_dim, mlp_multiplier, quantize):
        super().__init__()
        self.hidden_size = hidden_size
        
        # AdaLN for Self-Attention
        self.adaln_attn = AdaLayerNormZero(hidden_size, time_cond_dim, quantize = quantize)
        self.attn_emmul = Q.QuanEMMul(quantize=quantize)
        self.self_attn = MultiheadAttention_own(hidden_size, n_heads = num_heads, quantize=quantize)
        
        # Cross-Attention with Observations
        self.cross_attn = MultiheadAttention_own(hidden_size, n_heads = num_heads, quantize=quantize)
        self.norm_cross = Q.LayerNorm(hidden_size, elementwise_affine = True, quantize=quantize)
        
        # AdaLN for MLP
        self.adaln_mlp = AdaLayerNormZero(hidden_size, time_cond_dim, quantize = quantize)
        self.mlp_emmul = Q.QuanEMMul(quantize=quantize)
        self.mlp = nn.Sequential(
            Q.Conv2d(hidden_size, hidden_size * mlp_multiplier, kernel_size=1, quantize=quantize),
            Q.nn_GELU(approximate='tanh', quantize=quantize),
            Q.Conv2d(hidden_size * mlp_multiplier, hidden_size, kernel_size=1, quantize=quantize)
        )
        
    def forward(self, x, time_cond, obs_cond):      # x: [B, d_model, 1, T] time_emb: [B, d_model, 1, 1] obs_emb: [B, d_model, 5, 5]
        # Self-Attention with AdaLN
        res, gate_attn = self.adaln_attn(x, time_cond)
        res = self.self_attn(res, res, res)
        res = rearrange(res, "b d h w -> b d 1 (h w)")
        x = x + self.attn_emmul(res, gate_attn)

        # Cross-Attention with Observations
        _, C, H, W = x.shape
        res = rearrange(x, "b (n d) h w -> b n d (h w)", n=1)
        res = res.transpose(-1, -2)
        res = self.norm_cross(res)
        # print('res21',torch.mean(abs(res)),torch.mean(abs(obs_cond)))
        res = res.transpose(-1, -2)
        res = rearrange(res, "b n d (h w)-> b (n d) h w", h=H, w=W)
        res = self.cross_attn(res, obs_cond, obs_cond)
        # print('res22',torch.mean(abs(res)))
        res = rearrange(res, "b d h w -> b d 1 (h w)")
        x = x + res
        # print('x3',torch.mean(abs(x)))
        # MLP with AdaLN
        res, gate_mlp = self.adaln_mlp(x, time_cond)
        res = self.mlp(res)
        x = x + self.mlp_emmul(res, gate_mlp)
        
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, output_dim, quantize):
        super().__init__()
        self.norm = Q.LayerNorm(hidden_size, quantize=quantize)
        self.linear = Q.Conv2d(hidden_size, output_dim, kernel_size=1, quantize=quantize)
        
        # 初始化为零
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        _, C, H, W = x.shape
        x = rearrange(x, "b (n d) h w -> b n d (h w)", n=1)
        x = x.transpose(-1, -2)
        x = self.norm(x)
        x = x.transpose(-1, -2)
        x = rearrange(x, "b n d (h w)-> b (n d) h w", h=H, w=W)
        x = self.linear(x)
        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, 
                 quantize, 
                 action_dim,
                 seq_len,
                 d_model,
                 n_heads,
                 n_layers,
                 obs_cond_dim,   # fusion_output_dim*n_obs_steps
                 mlp_multiplier,
                 num_tasks,
                 ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.quantize = quantize
        self.num_tasks = num_tasks
        # Action Embedding
        self.action_embed = Q.Conv2d(action_dim, d_model, kernel_size=1, quantize=quantize)
        
        # Positional Embedding
        # self.pos_embed = nn.Parameter(torch.randn(1, 1, seq_len, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, d_model, 1, seq_len) * 0.02)
        self.quant_pos_embed_input = Q.Quantizer4input(quantize = False)

        # Timestep Embedding
        self.time_embed = TimestepEmbedder(d_model, seq_len=seq_len, quantize=quantize)
        if num_tasks is not None:
            self.task_embed = TaskEmbedder(num_tasks, d_model, seq_len=seq_len, quantize=quantize)  # <---
            
        
        # Observation Condition Projection
        # self.global_proj = Q.Conv2d(obs_cond_dim, d_model, kernel_size=1, quantize=quantize)
        self.obs_projs = nn.ModuleList([
            Q.Conv2d(obs_cond_dim, d_model, kernel_size=1, quantize=quantize)
            for _ in range(n_layers)
        ])
        
        # DiT Blocks
        self.dit_blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=d_model,
                num_heads=n_heads,
                time_cond_dim=d_model,
                mlp_multiplier=mlp_multiplier,
                quantize=quantize
            ) for _ in range(n_layers)
        ])
        
        # Final Layer
        self.final_layer = FinalLayer(d_model, action_dim, quantize=quantize)
        
    def forward(self, x, timesteps, task_ids=None, global_cond=None, local_cond=None):     # global_cond: [B, output_dim*T, 5, 5]
        # print('dit', x.shape, timesteps, global_cond.shape)
        B, T, D = x.shape
        assert local_cond == None, "local_cond not None"
        assert global_cond is not None, "global_cond must not None"
        assert self.seq_len == T, "self.seq_len not T"
        # x = x.unsqueeze(1)                          # x: [B, 1, T, D]
        x = rearrange(x, "b t d -> b d 1 t") # x: [B, D, 1, T]
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        assert len(timesteps.shape) == 1
        timesteps = timesteps.to(x.device)
        if self.num_tasks is not None:
            task_ids = task_ids.to(x.device)

        # Action Embedding + Position Embedding
        self.pos_embed,_,_ = self.quant_pos_embed_input(self.pos_embed)
        x = self.action_embed(x) + self.pos_embed[:, :, :, :]   # x: [B, 1, T, D]
        # x = x.transpose(-1, -2)
        # x = rearrange(x, "b n d (h w)-> b (n d) h w", h=1, w=T) # x: [B, D, 1, T]
        
        # Time Conditioning
        time_emb = self.time_embed(timesteps)           # time_emb: [B, d_model, 1, 1]  
        if self.num_tasks is not None:
            task_emb = self.task_embed(task_ids)                                       # [B, d_model, 1, T]
            time_emb = time_emb + task_emb                                            # <--- 组合条件

        # # Observation Conditioning
        # global_cond = self.global_proj(global_cond)

        # DiT Forward
        for i, block in enumerate(self.dit_blocks):
            # print('global_cond', i, torch.max(global_cond), torch.mean(global_cond))
            obs_emb = self.obs_projs[i](global_cond)
            # print('obs_emb', i, torch.max(obs_emb), torch.mean(obs_emb))
            # # print('cross_attn.v_linear_quan_scale', block.cross_attn.v_linear.quan_a.s)
            x = block(x, time_emb, obs_emb)                 # x: [B, d_model, 1, T] time_emb: [B, d_model, 1, 1] obs_emb: [B, d_model, 5, 5]
            
        # Final projection
        output = self.final_layer(x)                        # x: [B, d_model, 1, T]
        output = rearrange(output, "b d 1 l -> b l d")
        # if self.quantize and self.action_embed.quan_a.initialized_alpha>0:
        #     output = quantize_jit(output, 0.015625, -128, 127)
        return output


class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, channel, multiplier, quantize):
        super().__init__()
        assert height == width, "Height and width must be equal"
        
        self.height = height
        self.width = width
        self.channel = channel
        self.multiplier = multiplier
        self.quantize = quantize

        self.softmax = Q.QuanSoftmax(dim=-1, quantize = False)
        # 预计算坐标网格，归一化到[-1, 1]
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1, 1, width),
            np.linspace(-1, 1, height)
        )
        
        # 第一步：用width大小的卷积做加权求和，padding=multiplier-1
        # 输出大小：[B, C, 2*m-1, 2*m-1]
        self.conv_x = Q.Conv2d(channel, channel, kernel_size=width, 
                               padding=multiplier-1, bias=False, groups=channel, quantize=quantize)
        self.conv_y = Q.Conv2d(channel, channel, kernel_size=width, 
                               padding=multiplier-1, bias=False, groups=channel, quantize=quantize)
        
        # 第二步：用multiplier大小的全1卷积实现repeat效果
        # 输入：[B, C, 2*m-1, 2*m-1]，输出：[B, C, m, m]
        self.repeat_conv_x = Q.Conv2d(channel, channel, kernel_size=multiplier, 
                                      bias=False, groups=channel, quantize=quantize)
        self.repeat_conv_y = Q.Conv2d(channel, channel, kernel_size=multiplier, 
                                      bias=False, groups=channel, quantize=quantize)
        
        # 初始化卷积核权重
        self._init_conv_weights(pos_x, pos_y)
        
    def _init_conv_weights(self, pos_x, pos_y):
        """初始化卷积核权重"""
        with torch.no_grad():
            # 为每个channel设置权重
            for c in range(self.channel):
                # 第一步：设置坐标权重用于加权求和
                self.conv_x.weight.data[c, 0] = torch.tensor(pos_x, dtype=torch.float32)
                self.conv_y.weight.data[c, 0] = torch.tensor(pos_y, dtype=torch.float32)
                
                # 第二步：设置repeat卷积权重为全1，实现repeat效果
                self.repeat_conv_x.weight.data[c, 0] = torch.ones(self.multiplier, self.multiplier, dtype=torch.float32)
                self.repeat_conv_y.weight.data[c, 0] = torch.ones(self.multiplier, self.multiplier, dtype=torch.float32)
            if hasattr(self.repeat_conv_x, "quan_w"):
                self.repeat_conv_x.quan_w.s.data = torch.nn.Parameter(torch.ones(self.channel)) 
                self.repeat_conv_y.quan_w.s.data = torch.nn.Parameter(torch.ones(self.channel)) 
                self.repeat_conv_x.quan_w.initialized_alpha.fill_(1) 
                self.repeat_conv_x.quan_w.initialized_alpha.fill_(1)
        # 冻结权重
        if hasattr(self.repeat_conv_x, "quan_w"):
            self.conv_x.quan_w.s.requires_grad = False
            self.conv_y.quan_w.s.requires_grad = False
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False
        self.repeat_conv_x.weight.requires_grad = False
        self.repeat_conv_y.weight.requires_grad = False
        
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.height and W == self.width, f'{H} != {self.height} or {W} != {self.width}'
        
        # 应用softmax得到attention map
        x = rearrange(x, "b (n d) h w -> b n d (h w)", n=1) # [B, 1, C, H*W]]
        attention = self.softmax(x)
        attention = rearrange(attention, "b n d (h w)-> b (n d) h w", h=H, w=W)   # [B, C, H, W]
        
        # 第一步：通过padding的width卷积得到[B, C, 2*m-1, 2*m-1]
        # 由于padding=multiplier-1，输出大小为：H + 2*(m-1) - H + 1 = 2*m-1
        weighted_x = self.conv_x(attention)  # [B, C, 2*m-1, 2*m-1]
        weighted_y = self.conv_y(attention)  # [B, C, 2*m-1, 2*m-1]
        
        # 第二步：通过multiplier×multiplier的全1卷积实现repeat效果
        # 输入：[B, C, 2*m-1, 2*m-1]，卷积核：m×m，输出：[B, C, m, m]
        result_x = self.repeat_conv_x(weighted_x)  # [B, C, m, m]
        result_y = self.repeat_conv_y(weighted_y)  # [B, C, m, m]
        
        # 在channel维度上concat
        result = torch.cat([result_x, result_y], dim=1)  # [B, C*2, m, m]
        
        return result

class CNNVisualEncoder(nn.Module):
    def __init__(self, quantize, input_channels, visual_output_dim, visual_hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads 
        base_dim = int(visual_hidden_dim/16)
        # 更深更宽的CNN Backbone
        self.feature_extractor = nn.Sequential(
            Q.Conv2d(input_channels, base_dim, kernel_size=3, stride=1, padding=0, quantize = quantize),   # [B, 3, 145, 145] -> [B, 64, 143, 143]
            Q.nn_GELU(approximate='tanh', quantize = quantize),
            Q.Conv2d(base_dim, base_dim*2, kernel_size=3, stride=2, padding=0, quantize = quantize),              # [B, 64, 143, 143] -> [B, 128, 71, 71]
            Q.nn_GELU(approximate='tanh', quantize=quantize),
            Q.Conv2d(base_dim*2, base_dim*4, kernel_size=3, stride=2, padding=0, quantize = quantize),             # [B, 128, 71, 71] -> [B, 256, 35, 35]
            Q.nn_GELU(approximate='tanh', quantize=quantize),
            Q.Conv2d(base_dim*4, base_dim*8, kernel_size=3, stride=2, padding=0, quantize = quantize),             # [B, 256, 35, 35] -> [B, 512, 17, 17]
            Q.nn_GELU(approximate='tanh', quantize=quantize),
            Q.Conv2d(base_dim*8, base_dim*16, kernel_size=3, stride=1, padding=0, quantize = quantize),            # [B, 512, 17, 17] -> [B, 1024, 15, 15]
            Q.nn_GELU(approximate='tanh', quantize=quantize),
        )

        # SpatialSoftmax提取位置信息
        self.spatial_softmax = SpatialSoftmax(height=15, width=15, channel=visual_hidden_dim, multiplier=5, quantize=quantize)        # Shape: [B, 1024, 15, 15] -> 返回[B, 1, 1024*2, 25]
        
        self.quant_pooling_input = Q.Quantizer4input(quantize = quantize)
        # 全局特征提取: 池化 -> 降维 -> Flatten -> 大型MLP
        self.global_pool = nn.AvgPool2d(kernel_size=3, stride=3)                        # Shape: [B, 1024, 15, 15] -> [B, 1024, 5, 5]
        
        # 特征融合 - self attn
        self.self_attn = MultiheadAttention_own(visual_hidden_dim*3, n_heads = num_heads, quantize=False, wo_quant_softmax=False)

        # 特征融合 - 更大的MLP
        self.feature_fusion = nn.Sequential(
            Q.Conv2d(visual_hidden_dim*3, visual_hidden_dim*2, kernel_size=1, quantize=quantize),  # 多头融合后的特征
            Q.nn_GELU(approximate='tanh', quantize=quantize),
            # Q.Conv2d(2048, 1536, kernel_size=1, quantize=quantize),
            # Q.nn_GELU(approximate='tanh', quantize=quantize),
            Q.Conv2d(visual_hidden_dim*2, visual_output_dim, kernel_size=1, quantize=quantize),
        )
        
        
    def forward(self, x):
        # 深度CNN特征提取
        features = self.feature_extractor(x) # [B*T, 1024, 15, 15]
        # 提取位置信息
        spatial_features_im = self.spatial_softmax(features)  # [B*T, 2048, 5, 5]
        # spatial_features_tk = rearrange(spatial_features_im, "b (n d) h w -> b n d (h w)", n=1)  # [B*T, 1, 2048, 25]

        # 提取全局特征
        features,_,_ = self.quant_pooling_input(features)
        global_features_im = self.global_pool(features)       # [B*T, 1024, 5, 5]
        # global_features_tk = rearrange(global_features_im, "b (n d) h w -> b n d (h w)", n=1)  # [B*T, 1, 1024, 25]
        
        # 分成4份
        spatial_im_chunks = torch.chunk(spatial_features_im, self.num_heads, dim=-3)
        global_im_chunks = torch.chunk(global_features_im, self.num_heads, dim=-3)
        
        # 每个头分别concat spatial和global特征
        head_features = []
        for i in range(self.num_heads):
            head_feature = torch.cat([global_im_chunks[i], spatial_im_chunks[i]], dim=-3)  # [B*T, 768, 5, 5]
            head_features.append(head_feature)
        
        # 合并所有头的特征
        combined = torch.cat(head_features, dim=-3)  # [B*T, 3072, 5, 5]
        
        output = self.feature_fusion(self.self_attn(combined, combined, combined))  # [B*T, output_dim, 5, 5]
        
        return output

class MultiModalFusion(nn.Module):
    def __init__(self, quantize, visual_output_dim, pose_dim, num_heads, fusion_output_dim, pose_embed_dim):
        super().__init__()
        
        self.num_heads = num_heads
        # # 更大的视觉投影
        # self.visual_proj = nn.Sequential(
        #     Q.Conv2d(visual_dim, 1536, kernel_size=1, quantize=quantize),
        #     Q.nn_GELU(approximate='tanh', quantize=quantize),
        #     nn.Dropout(0.1),
        #     Q.Conv2d(1536, 1536, kernel_size=1, quantize=quantize),
        # )
        
        # 位置编码
        self.pose_proj = nn.Sequential(
            Q.Conv2d(pose_dim, int(pose_embed_dim/4), kernel_size=1, quantize=quantize),
            Q.nn_GELU(approximate='tanh', quantize=quantize),
            # Q.Conv2d(64, 128, kernel_size=1, quantize=quantize),
            # Q.nn_GELU(approximate='tanh', quantize=quantize),
            Q.Conv2d(int(pose_embed_dim/4), pose_embed_dim, kernel_size=1, padding=4, quantize=quantize),
        )
        
        self.pose_repeat_conv = Q.Conv2d(pose_embed_dim, pose_embed_dim, kernel_size=5, bias=False, groups=pose_embed_dim, quantize=False)
        with torch.no_grad():
            self.pose_repeat_conv.weight.data = torch.ones_like(self.pose_repeat_conv.weight.data)
        self.pose_repeat_conv.weight.requires_grad = False

        self.self_attn1 = MultiheadAttention_own(visual_output_dim + pose_embed_dim, n_heads = num_heads, quantize=False, wo_quant_softmax=False)

        # self.self_attn2 = MultiheadAttention_own(visual_dim + 512, n_heads = num_heads, quantize=quantize)

        self.final_proj = Q.Conv2d(visual_output_dim + pose_embed_dim, fusion_output_dim, kernel_size=1, quantize=quantize)
        # # 大型融合MLP
        # self.fusion_mlp = nn.Sequential(
        #     Q.Conv2d(visual_dim + 512, 2048, kernel_size=1, quantize=quantize),
        #     Q.nn_GELU(approximate='tanh', quantize=quantize),
        #     nn.Dropout(0.1),
        #     Q.Conv2d(2048, 2048, kernel_size=1, quantize=quantize),
        #     Q.nn_GELU(approximate='tanh', quantize=quantize),
        #     nn.Dropout(0.1),
        #     Q.Conv2d(2048, 1536, kernel_size=1, quantize=quantize),
        #     Q.nn_GELU(approximate='tanh', quantize=quantize),
        #     Q.Conv2d(1536, output_dim, kernel_size=1, quantize=quantize)
        # )
        
    def forward(self, visual_feat, pose_feat):              # [B*T, visual_dim, 5, 5] [B*T, 2]
        pose_feat = rearrange(pose_feat, "bt d -> bt d 1 1")   
        # 投影到相同维度
        # visual_emb = self.visual_proj(visual_feat)          # [B*T, 768, 5, 5]
        pose_emb = self.pose_proj(pose_feat)                # [B*T, 768, 5, 5]
        pose_emb = self.pose_repeat_conv(pose_emb)
        # # 拼接
        # fused = torch.cat([visual_feat, pose_emb], dim=-3)  # [B*T, 2048, 5, 5]
        # 分成4份
        visual_feat_chunks = torch.chunk(visual_feat, self.num_heads, dim=-3)
        pose_emb_chunks = torch.chunk(pose_emb, self.num_heads, dim=-3)
        # 每个头分别concat spatial和global特征
        head_features = []
        for i in range(self.num_heads):
            head_feature = torch.cat([visual_feat_chunks[i], pose_emb_chunks[i]], dim=-3)  # [B*T, 512, 5, 5]
            head_features.append(head_feature)
        combined = torch.cat(head_features, dim=-3)
        
        attn1 = self.self_attn1(combined, combined, combined)
        # attn2 = self.self_attn2(attn1,attn1,attn1)
        output = self.final_proj(attn1)
        # # 大型融合MLP
        # output = self.fusion_mlp(head_features)  # [B*T, output_dim, 5, 5]
        
        return output
    

class ModernObsEncoder(nn.Module):
    def __init__(self, 
                 quantize, 
                 pose_dim,
                 visual_output_dim,
                 fusion_output_dim,
                 visual_hidden_dim,
                 pose_embed_dim,
                 obs_n_head,
                 ):
        super().__init__()
        self.fusion_output_dim = fusion_output_dim
        
        # 大型CNN视觉编码器
        self.visual_encoder = CNNVisualEncoder(
            input_channels=3,
            visual_output_dim=visual_output_dim, 
            visual_hidden_dim=visual_hidden_dim,
            num_heads=obs_n_head,
            quantize=quantize
        )
        
        # 大型多模态融合
        self.multimodal_fusion = MultiModalFusion(
            visual_output_dim=visual_output_dim,
            pose_dim=pose_dim,
            num_heads=obs_n_head,
            fusion_output_dim=fusion_output_dim, 
            pose_embed_dim=pose_embed_dim,
            quantize=quantize
        )
        self.out_norm = Q.LayerNorm(fusion_output_dim, elementwise_affine=True, quantize=quantize)
        
    def forward(self, obs_dict):
        images = obs_dict['image']
        pose_feat = obs_dict['agent_pos']   
        # print('obs images', torch.max(images), torch.mean(images))
        # print('obs pose_feat', torch.max(pose_feat), torch.mean(pose_feat))
        # print('obs', images.shape, pose_feat.shape )

        visual_feat = self.visual_encoder(images)   # [B*T, output_dim, 5, 5]
        fused_feat = self.multimodal_fusion(visual_feat, pose_feat)
        # # print('obs output', torch.max(fused_feat), torch.mean(fused_feat))
        
        B, C, H, W = fused_feat.shape
        y = rearrange(fused_feat, "b (n d) h w -> b n d (h w)", n=1)
        y = y.transpose(-1, -2)           # [B,1,HW,C]
        y = self.out_norm(y)
        y = y.transpose(-1, -2)
        fused_feat = rearrange(y, "b n d (h w) -> b (n d) h w", h=H, w=W)

        return fused_feat
    
    def output_shape(self):
        return [self.fusion_output_dim]

