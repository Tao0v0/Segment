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
    # autocast = torch.cuda.amp.autocast
    autocast = torch.amp.autocast
    
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


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
        args = get_args()
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
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

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

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast('cuda', enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([event1, event2])
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast('cuda', enabled=self.args.mixed_precision):
            cnet = self.cnet(event2)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        # Initialize Grids. First channel: x, 2nd channel: y. Image is just used to get the shape
        coords0, coords1 = self.initialize_flow(event1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(self.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast('cuda', enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
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
