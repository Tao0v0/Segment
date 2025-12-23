import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import bilinear_sampler, coords_grid

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class AvgPool2dAsConv(nn.Module):
    """AvgPool2d implemented via depthwise Conv2d (parameter-free).

    This is mathematically equivalent to average pooling when padding=0 and ceil_mode=False.
    """

    def __init__(self, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.kernel_size
        c = x.shape[1]
        weight = x.new_ones((c, 1, k, k))
        y = F.conv2d(x, weight, stride=self.stride, padding=0, groups=c)
        return y * (1.0 / float(k * k))


_avg_pool2x2_s2 = AvgPool2dAsConv(kernel_size=2, stride=2)


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = _avg_pool2x2_s2(corr)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):     #  corr_fn = CorrBlock(fmap1, fmap2, num_levels=4, radius=4)   调用corr函数    corr_feat = corr_fn(coords) 调用call
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)  #(N,2,H,W) -> (N,H,W,2)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels): # 4
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)       #假如r是4，那么就是[-4,-3,-2,-1,0,1,2,3,4] 固定的
            dy = torch.linspace(-r, r, 2*r+1)       #假如r是4，那么就是[-4,-3,-2,-1,0,1,2,3,4] 固定的
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device) # 产出(K,K) ,stack成(K,K,2)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)       # 展开
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())
