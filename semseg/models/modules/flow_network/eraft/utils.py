import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()   # 把序列反过来(yy,xx) -> (xx,yy) ，meshgrid会生成两个列表，列表yy表示每个网格点的y坐标值，xx表示每个坐标的x坐标值
    return coords[None].repeat(batch, 1, 1, 1)      # coords(2,H,W), coords[None]相当于coords.unsqueeze(0)，。reapt按第0维复制batch次，后面不复制，


def upflowX(flow, mode='bilinear', X=8):
    new_size = (X * flow.shape[2], X * flow.shape[3])
    return  X * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
