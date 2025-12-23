import sys
import os
from pathlib import Path
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from semseg.models.layers import DropPath
from einops import rearrange # 必须使用 einops

# ================= 路径修补 =================
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[4]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from diffusers_dpm.models.quantizer.net2quan import QuantizableLayer as Q
    from diffusers_dpm.models.quantizer.net2quan import QuanMMHead, QuanSoftmax, QuanConv, QuanLinear
except ImportError as e:
    print(f"导入失败，请检查路径。当前 sys.path: {sys.path}")
    raise e
# ===========================================

class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio, quantize=False): 
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio 
        self.scale = (dim // head) ** -0.5
        
        # Linear 底层是 Conv2d，输入必须是 [B, C, H, W]
        self.q = Q.Linear(dim, dim, quantize=quantize)
        self.k = Q.Linear(dim, dim, quantize=quantize)      # 原来是用一个层映射kv，然后5D分开，由于硬件不支持5D，因此这里做了修改，改为分别映射k和v。导入原来的权重时，把原来权重拆开赋值即可。
        self.v = Q.Linear(dim, dim, quantize=quantize)
        self.proj = Q.Linear(dim, dim, quantize=quantize)

        self.matmul_qk = QuanMMHead(quan_input_a=quantize, quan_input_b=quantize)
        self.matmul_av = QuanMMHead(quan_input_a=quantize, quan_input_b=quantize)
        self.softmax = Q.QuanSoftmax(dim=-1, quantize=quantize)

        if sr_ratio > 1:
            self.sr = Q.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, quantize=quantize)
            self.norm = Q.LayerNorm(dim, quantize=quantize)

    def forward(self, x: Tensor, H, W, metric: Tensor=None) -> Tensor:
        # Block 输入 x: [B, 1, N, C]
        
        # 1. 恢复成图像 [B, C, H, W] 准备计算 Q
        x_img = rearrange(x, 'b 1 (h w) c -> b c h w', h=H, w=W)
        
        if metric is None:
            # Q Linear 输入: [B, C, H, W] -> 输出: [B, C, H, W]
            q = self.q(x_img)
        else:
            # Metric 也要转成 [B, C, H, W]
            metric_img = rearrange(metric, 'b 1 (h w) c -> b c h w', h=H, w=W)
            q = self.q(metric_img)

        # 准备 Attention Q: [B, C, H, W] -> [B, Heads, N, Dim]
        q = rearrange(q, 'b (head dim) h w -> b head (h w) dim', head=self.head)

        # 2. SR 路径 (计算 K, V 输入)
        if self.sr_ratio > 1:
            # SR Conv: [B, C, H, W] -> [B, C, H', W']
            x_sr = self.sr(x_img)
            
            # LayerNorm 需要 Token: [B, 1, N', C]
            x_sr_norm = rearrange(x_sr, 'b c h w -> b 1 (h w) c')
            x_sr_norm = self.norm(x_sr_norm)
            
            # 【重点】计算完 Norm 后，必须恢复成 SR 后的图像尺寸 [B, C, H', W']
            # H' = H / sr, W' = W / sr
            # 只有这样，喂给 K/V Linear 的才是真正的 4D 图像
            x_for_kv = rearrange(x_sr_norm, 'b 1 (h w) c -> b c h w', h=H//self.sr_ratio, w=W//self.sr_ratio)
        else:
            # 【重点】无 SR，直接使用原图 x_img [B, C, H, W]
            # 绝对不进行任何 reshape 到 1 N 的操作
            x_for_kv = x_img

        # 3. 计算 K, V (输入输出全是 B C H W)
        k = self.k(x_for_kv) # Output: [B, C, H', W']
        v = self.v(x_for_kv) # Output: [B, C, H', W']

        # 4. 准备 MatMul
        # K -> [B, Heads, Dim, N']
        k = rearrange(k, 'b (head dim) h w -> b head dim (h w)', head=self.head)
        
        # V -> [B, Heads, N', Dim]
        v = rearrange(v, 'b (head dim) h w -> b head (h w) dim', head=self.head)
        
        # 5. Attention 计算
        # q: [B, H, N, D], k: [B, H, D, N']
        attn = (self.matmul_qk(q, k) * self.scale) 
        attn = self.softmax(attn)
        
        # x: [B, H, N, D]
        x = self.matmul_av(attn, v)
        
        # 6. Output Projection
        # [B, H, N, D] -> [B, C, H, W] (恢复图像格式给 Proj Linear)
        x = rearrange(x, 'b head (h w) dim -> b (head dim) h w', h=H, w=W)
        
        # Proj Linear: [B, C, H, W] -> [B, C, H, W]
        x = self.proj(x)
        
        # 恢复 Token 模式 [B, 1, N, C] 给 Block 输出
        x = rearrange(x, 'b c h w -> b 1 (h w) c')
        return x


class DWConv(nn.Module):
    def __init__(self, dim, quantize=False):
        super().__init__()
        self.dwconv = Q.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, quantize=quantize
        )

    def forward(self, x: Tensor, H, W) -> Tensor:
        # 输入 x: [B, 1, N, C]
        
        # 恢复图像 [B, C, H, W]
        x = rearrange(x, 'b 1 (h w) c -> b c h w', h=H, w=W)
        x = self.dwconv(x)
        
        # 变回 Token [B, 1, N, C] (因为 DWConv 在 MLP 中间，后面接 Act 还是 4D 无所谓，但为了统一接口返回 Token)
        # 或者为了配合 MLP 的后续 FC2，这里返回图像格式最省事，看 MLP 怎么写
        # 这里我们按标准返回 Token
        x = rearrange(x, 'b c h w -> b 1 (h w) c')
        return x


class MLP(nn.Module):
    def __init__(self, c1, c2, quantize=False): 
        super().__init__()
        self.fc1 = Q.Linear(c1, c2, quantize=quantize)
        self.dwconv = DWConv(c2, quantize=quantize)
        self.fc2 = Q.Linear(c2, c1, quantize=quantize)
        self.act = Q.nn_GELU(quantize=quantize)
        
    def forward(self, x: Tensor, H, W) -> Tensor:
        # 输入 x: [B, 1, N, C]
        
        # 1. 变为 Image [B, C, H, W] 给 fc1
        x = rearrange(x, 'b 1 (h w) c -> b c h w', h=H, w=W)
        
        # 2. FC1: [B, C1, H, W] -> [B, C2, H, W]
        x = self.fc1(x) 
        
        # 3. DWConv (DWConv 内部我修改为接收 Image 格式会更高效，但为了兼容上面写的接口，这里先转 token 再转回来)
        # 为了效率，建议修改 DWConv 接收 Image。
        # 这里假设 DWConv 接收 Token 并返回 Token (如上面定义)
        # 所以我们需要先转 Token
        x = rearrange(x, 'b c h w -> b 1 (h w) c')
        x = self.dwconv(x, H, W) # Output: [B, 1, N, C2]
        
        # 转回 Image 给 Act 和 FC2
        x = rearrange(x, 'b 1 (h w) c -> b c h w', h=H, w=W)
        
        # 4. Act
        x = self.act(x)
        
        # 5. FC2: [B, C2, H, W] -> [B, C1, H, W]
        x = self.fc2(x)
        
        # 6. 恢复 [B, 1, N, C]
        x = rearrange(x, 'b c h w -> b 1 (h w) c')
        
        return x

# 优化版 DWConv (配合 MLP 优化)
class DWConvOptimized(nn.Module):
    def __init__(self, dim, quantize=False):
        super().__init__()
        self.dwconv = Q.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, quantize=quantize
        )
    def forward(self, x: Tensor) -> Tensor:
        # 直接接收 [B, C, H, W]
        return self.dwconv(x)

# 优化版 MLP (全程 B C H W)
class MLPOptimized(nn.Module):
    def __init__(self, c1, c2, quantize=False): 
        super().__init__()
        self.fc1 = Q.Linear(c1, c2, quantize=quantize)
        self.dwconv = DWConvOptimized(c2, quantize=quantize)
        self.fc2 = Q.Linear(c2, c1, quantize=quantize)
        self.act = Q.nn_GELU(quantize=quantize)
        
    def forward(self, x: Tensor, H, W) -> Tensor:
        # 输入 x: [B, 1, N, C]
        
        # 全程 Image 格式处理
        x = rearrange(x, 'b 1 (h w) c -> b c h w', h=H, w=W)
        x = self.fc1(x) 
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        x = rearrange(x, 'b c h w -> b 1 (h w) c')
        return x

class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0, quantize=False):
        super().__init__()
        self.proj = Q.Conv2d(
            c1, c2, kernel_size=patch_size, stride=stride, padding=padding, quantize=quantize
        )
        self.norm = Q.LayerNorm(c2, quantize=quantize)
    # B C H W  /  B H W C 
    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 3, H_in, W_in]
        x = self.proj(x) # -> [B, C, H, W]
        _, _, H, W = x.shape
        
        # [B, C, H, W] -> [B, 1, N, C]
        x = rearrange(x, 'b c h w -> b 1 (h w) c')
        
        x = self.norm(x)
        return x, H, W

class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0., is_fan=False, quantize=False):
        super().__init__()
        self.norm1 = Q.LayerNorm(dim, quantize=quantize)
        self.attn = Attention(dim, head, sr_ratio, quantize=quantize)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = Q.LayerNorm(dim, quantize=quantize)
        # 使用优化版 MLP，减少 reshape 次数
        self.mlp = MLPOptimized(dim, int(dim*4), quantize=quantize)

    def forward(self, x: Tensor, H, W, metric: Tensor=None) -> Tensor:
        # x 始终保持 [B, 1, N, C]
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, metric))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

cmnext_settings = {
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
}

class CMNeXt(nn.Module):
    def __init__(self, model_name: str = 'B0', modals: list = ['rgb', 'depth', 'event', 'lidar'], with_events=False, quantize=False) -> None:
        super().__init__()
        assert model_name in cmnext_settings.keys(), f"Model name should be in {list(cmnext_settings.keys())}"
        embed_dims, depths = cmnext_settings[model_name]
        
        self.modals = modals[1:] if len(modals)>1 else []  
        self.num_modals = len(self.modals)
        self.with_events = with_events
        drop_path_rate = 0.1
        self.channels = embed_dims

        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4, 7//2, quantize=quantize)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 3//2, quantize=quantize)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 3//2, quantize=quantize)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 3//2, quantize=quantize)
   
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur+i], quantize=quantize) for i in range(depths[0])]) 
        self.norm1 = Q.LayerNorm(embed_dims[0], quantize=quantize)
        
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur+i], quantize=quantize) for i in range(depths[1])])
        self.norm2 = Q.LayerNorm(embed_dims[1], quantize=quantize)

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur+i], quantize=quantize) for i in range(depths[2])])
        self.norm3 = Q.LayerNorm(embed_dims[2], quantize=quantize)

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur+i], quantize=quantize) for i in range(depths[3])])
        self.norm4 = Q.LayerNorm(embed_dims[3], quantize=quantize)


    def forward(self, x: list, x_ext: list=None, metric: Tensor=None) -> list:
        x_cam = x[0]
        metric_ = None
        x_ext = None
        B = x_cam.shape[0]
        outs = []

        # stage 1
        x_cam, H, W = self.patch_embed1(x_cam) # -> [B, 1, N, C]
        if metric is not None:
             metric = torch.nn.functional.interpolate(input=metric, size=(H, W), mode='bilinear', align_corners=False)
             # Metric [B, 1, N, C]
             metric_ = rearrange(metric, 'b c h w -> b 1 (h w) c')
        for blk in self.block1:
            x_cam = blk(x_cam, H, W, metric_)
        
        # Output: [B, 1, N, C] -> Norm -> [B, C, H, W]
        x1_cam = self.norm1(x_cam)
        x1_cam = rearrange(x1_cam, 'b 1 (h w) c -> b c h w', h=H, w=W)
        outs.append(x1_cam)

        # stage 2
        x_cam, H, W = self.patch_embed2(x1_cam)
        if metric is not None:
             metric = torch.nn.functional.interpolate(input=metric, size=(H, W), mode='bilinear', align_corners=False)
             metric_ = rearrange(metric, 'b c h w -> b 1 (h w) c')
        for blk in self.block2:
            x_cam = blk(x_cam, H, W, metric_)
        x2_cam = self.norm2(x_cam)
        x2_cam = rearrange(x2_cam, 'b 1 (h w) c -> b c h w', h=H, w=W)
        outs.append(x2_cam)

        # stage 3
        x_cam, H, W = self.patch_embed3(x2_cam)
        if metric is not None:
             metric = torch.nn.functional.interpolate(input=metric, size=(H, W), mode='bilinear', align_corners=False)
             metric_ = rearrange(metric, 'b c h w -> b 1 (h w) c')
        for blk in self.block3:
            x_cam = blk(x_cam, H, W, metric_)
        x3_cam = self.norm3(x_cam)
        x3_cam = rearrange(x3_cam, 'b 1 (h w) c -> b c h w', h=H, w=W)
        outs.append(x3_cam)

        # stage 4
        x_cam, H, W = self.patch_embed4(x3_cam)
        if metric is not None:
             metric = torch.nn.functional.interpolate(input=metric, size=(H, W), mode='bilinear', align_corners=False)
             metric_ = rearrange(metric, 'b c h w -> b 1 (h w) c')
        for blk in self.block4:
            x_cam = blk(x_cam, H, W, metric_)
        x4_cam = self.norm4(x_cam)
        x4_cam = rearrange(x4_cam, 'b 1 (h w) c -> b c h w', h=H, w=W)
        outs.append(x4_cam)

        return outs


if __name__ == '__main__':
    modals = ['img', 'depth', 'event', 'lidar']
    x = [torch.zeros(1, 3, 256, 256), torch.ones(1, 3, 256, 256), torch.ones(1, 3, 256, 256)*2, torch.ones(1, 3, 256, 256) *3]
    # 实例化时可以控制 quantize 开关
    model = CMNeXt('B2', modals, quantize=True) 
    outs = model(x)
    for i, y in enumerate(outs):
        print(f"Stage {i} Output shape: {y.shape}")