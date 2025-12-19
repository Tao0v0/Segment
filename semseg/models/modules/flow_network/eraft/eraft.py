import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock
from .extractor import BasicEncoder
from .corr import CorrBlock
from .utils import coords_grid, upflowX
from argparse import Namespace
from .image_utils import ImagePadder

try:
    from torch.cuda.amp import autocast
    # autocast = torch.amp.autocast
except:
    pass

    # # dummy autocast for PyTorch < 1.6
    # class autocast:
    #     def __init__(self, enabled):
    #         pass
    #     def __enter__(self):
    #         pass
    #     def __exit__(self, *args):
    #         pass


def get_args():
    # This is an adapter function that converts the arguments given in out config file to the format, which the ERAFT
    # expects.
    args = Namespace(small=False,
                     dropout=False,
                     mixed_precision=False,
                     clip=1.0)
    return args



class ERAFT(nn.Module):
    def __init__(self, n_first_channels):
        # args:
        super(ERAFT, self).__init__()
        args = get_args()       # 当输入python eraft.py --lr 0.0005 --batch_size 4 时，会返回指定的参数 lr 和 batchsize
        self.args = args
        self.raft_type = 'large'
        self.image_padder = ImagePadder(min_size=32)

        if self.raft_type == 'large':
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4
            self.iters = 12

            # feature network, context network, and update block
            self.fnet = BasicEncoder(dims=(64, 64, 96, 128, 256), norm_fn='instance', dropout=0,
                                        n_first_channels=n_first_channels)
            self.cnet = BasicEncoder(dims=(64, 64, 96, 128, 256), norm_fn='batch', dropout=0,
                                        n_first_channels=n_first_channels)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim, raft_type=self.raft_type)
        elif self.raft_type == 'small':
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
            self.iters = 12

            self.fnet = BasicEncoder(dims=(32, 32, 64, 96, 128), norm_fn='instance', dropout=0,
                                        n_first_channels=n_first_channels, raft_type=self.raft_type)
            self.cnet = BasicEncoder(dims=(32, 32, 64, 96, 160), norm_fn='none', dropout=0,
                                        n_first_channels=n_first_channels, raft_type=self.raft_type)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim, raft_type=self.raft_type)
        self.iters = 12
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)  # 输出形状：[B, 2, ht, wd]  ，第一个通道装的每个特网格的x坐标，第二个通道装的y坐标
        coords1 = coords_grid(N, H//8, W//8).to(img.device)  

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)   # 含义：对每个低分辨率像素 (H,W) 的每个 8×8 子像素位置，都准备了 9 个权重（对应 3×3 的邻域点）。
        mask = torch.softmax(mask, dim=2)       # 在“9 个邻域权重”上做 softmax，确保每组权重非负且总和为 1（凸组合）

        up_flow = F.unfold(8 * flow, [3,3], padding=1)  # 先解释 为什么乘 8：低分辨率上的光流是以特征网格为单位（步长=8个原图像素）；要变回原图单位，需要把位移放大 8 倍。
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)     # F.unfold(x, [3,3], padding=1)：对 x ∈ [N, C, H, W] 提取滑动 3×3 块（步长=1），输出形状 [N, C*9, H*W]。
                                                        # 这里 x = 8*flow, C=2，所以得到 [N, 18, H*W]，其中每个位置收集了3×3 邻域的 2 通道光流。
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, event1, event2, flow_init=None, upsample=True):
        """ Estimate optical flow between pair of frames """

        # Pad Image (for flawless up&downsampling)
        event1 = self.image_padder.pad(event1)
        event2 = self.image_padder.pad(event2)

        event1 = event1.contiguous()
        event2 = event2.contiguous()

        # event1 = F.interpolate(event1, scale_factor=0.5, mode='bilinear', align_corners=False)
        # event2 = F.interpolate(event2, scale_factor=0.5, mode='bilinear', align_corners=False)

        hdim = self.hidden_dim  # 96
        cdim = self.context_dim # 64

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([event1, event2])
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(event2)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)   # 在dim =1 上，按照hdim 和 cdim长度进行切分 net：[B, hdim, H, W]作为GRU的更新块  inp：[B, cdim, H, W]作为GRU的上下文输入
            net = torch.tanh(net)
            inp = torch.relu(inp)

        # Initialize Grids. First channel: x, 2nd channel: y. Image is just used to get the shape
        coords0, coords1 = self.initialize_flow(event1)    # 没有对event做改变，只是用event1的形状（N,C,H,W）生成两张相同的坐标网格，1/8分辨率，每个坐标shape(N,2,H/8,W/8)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(self.iters):       # self/iters = 12
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume   把(N,2,H,W)的网格坐标变为 一张特征相关图 （B,L*K*K,H,W)  K = 2r+1

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions   up_mask 不是None,要执行 upsample_flow
            if up_mask is None:      
                if self.raft_type == 'small':
                    flow_up = upflowX(coords1 - coords0, X=8)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                # flow_up = upflowX(flow_up, X=2)

            # flow_predictions.append(flow_up)
            flow_predictions.append(self.image_padder.unpad(flow_up))
        # print("Flow predictions: ", flow_predictions[-1].mean())

        # return flow_predictions
        return coords1 - coords0, flow_predictions
