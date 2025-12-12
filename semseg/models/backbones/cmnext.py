import sys
import os
from pathlib import Path
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from semseg.models.layers import DropPath

# ================= 路径修补 =================
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[4]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from models.quantizer.net2quan import QuantizableLayer as Q
    from models.quantizer.net2quan import QuanMMHead, QuanSoftmax, QuanConv, QuanLinear
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
        
        # Q.Linear 本质是 Conv2d 1x1，需要 [B, C, H, W] 输入
        self.q = Q.Linear(dim, dim, quantize=quantize)
        self.kv = Q.Linear(dim, dim*2, quantize=quantize)
        self.proj = Q.Linear(dim, dim, quantize=quantize)

        self.matmul_qk = QuanMMHead(quan_input_a=quantize, quan_input_b=quantize)
        self.matmul_av = QuanMMHead(quan_input_a=quantize, quan_input_b=quantize)
        self.softmax = Q.QuanSoftmax(dim=-1, quantize=quantize)

        if sr_ratio > 1:
            # 这里的 stride 必须显式指定
            self.sr = Q.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, quantize=quantize)
            self.norm = Q.LayerNorm(dim, quantize=quantize)

    def forward(self, x: Tensor, H, W, metric: Tensor=None) -> Tensor:
        B, N, C = x.shape
        
        # 【维度修正 1】将 [B, N, C] 转为 [B, C, H, W] 以适应 Q.Linear (Conv2d)
        x_4d = x.permute(0, 2, 1).reshape(B, C, H, W)
        
        # 计算 Q
        if metric is None:
            # 输入 4D -> Q.Linear -> 输出 4D -> 展平回 [B, N, C] 以继续后面的 reshape
            q = self.q(x_4d).flatten(2).transpose(1, 2)
        else:
            # metric 也需要转 4D
            metric_4d = metric.permute(0, 2, 1).reshape(B, C, H, W)
            q = self.q(metric_4d).flatten(2).transpose(1, 2)
        
        # 此时 q 又是 [B, N, C]，按照原逻辑处理 heads
        q = q.reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        # 处理 SR (Spatial Reduction)
        if self.sr_ratio > 1:
            # x_4d 已经是 [B, C, H, W]，直接喂给 self.sr (Conv2d)
            x_sr = self.sr(x_4d) 
            # 变回 [B, N_sr, C] 给 LayerNorm 和 KV 计算
            x_sr = x_sr.flatten(2).transpose(1, 2)
            x_sr = self.norm(x_sr)
            
            # 为了进 self.kv (Q.Linear)，还需要变回 4D
            # 注意：sr 之后的 H, W 变小了，不能用原来的 H, W
            # 简单的做法是利用 x_sr 的 shape 自动推导，或者再次 reshape
            B, N_sr, C = x_sr.shape
            # H_sr = H // self.sr_ratio (大致如此，但用 view 还原更安全)
            # 这里为了简单，我们重新 view 成 (B, C, N_sr, 1) 或者 (B, C, H_sr, W_sr)
            # 因为 Q.Linear 是 1x1 卷积，空间维度由 H*W 还是 N*1 其实不影响计算结果
            x_sr_4d = x_sr.permute(0, 2, 1).unsqueeze(-1) # [B, C, N_sr, 1]
        else:
            x_sr_4d = x_4d # [B, C, H, W]

        # 计算 K, V
        # 输入 [B, C, H, W] 或 [B, C, N, 1] -> 输出对应的 4D -> 展平
        kv = self.kv(x_sr_4d).flatten(2).transpose(1, 2)
        k, v = kv.reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        # Attention 计算 (输入已经是量化友好的 4D 形式: B, Heads, N, Dim)
        attn = (self.matmul_qk(q, k.transpose(-2, -1)) * self.scale)
        attn = self.softmax(attn)
        x = self.matmul_av(attn, v).transpose(1, 2).reshape(B, N, C)
        
        # 【维度修正 2】最后的投影 Output Projection
        # 再次转 4D 给 self.proj
        x_4d_out = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.proj(x_4d_out).flatten(2).transpose(1, 2)
        
        return x


class DWConv(nn.Module):
    def __init__(self, dim, quantize=False):
        super().__init__()
        # Conv2d 参数全用关键字指定
        self.dwconv = Q.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, quantize=quantize
        )

    def forward(self, x: Tensor, H, W) -> Tensor:
        # DWConv 原代码本来就处理了 4D 转换，所以这里不需要大改，保留原逻辑即可
        # x: [B, N, C]
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W) # 变为 [B, C, H, W]
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)    # 变回 [B, N, C]


class MLP(nn.Module):
    def __init__(self, c1, c2, quantize=False): 
        super().__init__()
        self.fc1 = Q.Linear(c1, c2, quantize=quantize)
        self.dwconv = DWConv(c2, quantize=quantize)
        self.fc2 = Q.Linear(c2, c1, quantize=quantize)
        self.act = Q.nn_GELU(quantize=quantize)
        
    def forward(self, x: Tensor, H, W) -> Tensor:
        # 【维度修正 3】MLP 全程使用 4D 计算会更高效，避免反复 transpose
        B, N, C = x.shape
        x_4d = x.permute(0, 2, 1).reshape(B, C, H, W)
        
        # fc1: [B, C, H, W] -> [B, C2, H, W]
        x_4d = self.fc1(x_4d) 
        
        # dwconv: 它内部有 transpose，我们得改一下调用方式或者让它接受 4D
        # 既然 DWConv 类已经写死了接收 [B, N, C]，我们就先把 4D 转回去喂给它，
        # 或者为了性能，把 DWConv 里的 reshape 拆出来。
        # 为了代码改动最小化，我们这里还是切回 [B, N, C] 喂给 dwconv
        
        x_temp = x_4d.flatten(2).transpose(1, 2) # [B, N, C2]
        x_temp = self.dwconv(x_temp, H, W)       # [B, N, C2]
        x_4d = x_temp.permute(0, 2, 1).reshape(B, -1, H, W) # [B, C2, H, W]
        
        # act: 4D 没问题
        x_4d = self.act(x_4d)
        
        # fc2: [B, C2, H, W] -> [B, C1, H, W]
        x_4d = self.fc2(x_4d)
        
        return x_4d.flatten(2).transpose(1, 2)


class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0, quantize=False):
        super().__init__()
        self.proj = Q.Conv2d(
            c1, c2, kernel_size=patch_size, stride=stride, padding=padding, quantize=quantize
        )
        self.norm = Q.LayerNorm(c2, quantize=quantize)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0., is_fan=False, quantize=False):
        super().__init__()
        self.norm1 = Q.LayerNorm(dim, quantize=quantize)
        self.attn = Attention(dim, head, sr_ratio, quantize=quantize)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = Q.LayerNorm(dim, quantize=quantize)
        self.mlp = MLP(dim, int(dim*4), quantize=quantize)

    def forward(self, x: Tensor, H, W, metric: Tensor=None) -> Tensor:
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
        x_cam, H, W = self.patch_embed1(x_cam)
        if metric is not None:
            metric = torch.nn.functional.interpolate(input=metric, size=(H, W), mode='bilinear', align_corners=False)
            metric_ = metric.flatten(2).transpose(1, 2).repeat(1, 1, x_cam.shape[-1])
        for blk in self.block1:
            x_cam = blk(x_cam, H, W, metric_)
        x1_cam = self.norm1(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x1_cam)

        # stage 2
        x_cam, H, W = self.patch_embed2(x1_cam)
        if metric is not None:
            metric = torch.nn.functional.interpolate(input=metric, size=(H, W), mode='bilinear', align_corners=False)
            metric_ = metric.flatten(2).transpose(1, 2).repeat(1, 1, x_cam.shape[-1])
        for blk in self.block2:
            x_cam = blk(x_cam, H, W, metric_)
        x2_cam = self.norm2(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x2_cam)

        # stage 3
        x_cam, H, W = self.patch_embed3(x2_cam)
        if metric is not None:
            metric = torch.nn.functional.interpolate(input=metric, size=(H, W), mode='bilinear', align_corners=False)
            metric_ = metric.flatten(2).transpose(1, 2).repeat(1, 1, x_cam.shape[-1])
        for blk in self.block3:
            x_cam = blk(x_cam, H, W, metric_)
        x3_cam = self.norm3(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x3_cam)

        # stage 4
        x_cam, H, W = self.patch_embed4(x3_cam)
        if metric is not None:
            metric = torch.nn.functional.interpolate(input=metric, size=(H, W), mode='bilinear', align_corners=False)
            metric_ = metric.flatten(2).transpose(1, 2).repeat(1, 1, x_cam.shape[-1])
        for blk in self.block4:
            x_cam = blk(x_cam, H, W, metric_)
        x4_cam = self.norm4(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x4_cam)

        return outs


if __name__ == '__main__':
    modals = ['img', 'depth', 'event', 'lidar']
    x = [torch.zeros(1, 3, 256, 256), torch.ones(1, 3, 256, 256), torch.ones(1, 3, 256, 256)*2, torch.ones(1, 3, 256, 256) *3]
    # 实例化时可以控制 quantize 开关
    model = CMNeXt('B2', modals, quantize=True) 
    outs = model(x)
    for y in outs:
        print(f"Output shape: {y.shape}")