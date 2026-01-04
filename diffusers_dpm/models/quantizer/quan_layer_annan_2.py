import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
# from .lsq_annan import *
from .lsq_ymr import *
import warnings
from typing import Union, Callable, Optional
from torch import Tensor

log_onnx = False
save_path = '/root/diffusion/diffusers_dpm/'
train_dtype = torch.float32
def process_sample_dir(sample_dir): # 很粗糙，随便写写
    global save_path
    save_path = save_path + sample_dir
    print(f"Processing sample directory: {sample_dir}")
    print(f"Finally saving path: {save_path}")
    import os
    if not os.path.exists(save_path+'check_res'):
        os.makedirs(save_path+'check_res')
        
class QuanELU(nn.ELU):
    """docstring for QuanConv"""

    def __init__(self, alpha: float = 1., inplace: bool = False, quan_input=True, nbit_a=8, mode='lsq', N=1, C=1):
        super(QuanELU, self).__init__(
            alpha, inplace)
        self.quan_input = quan_input
        self.mode = mode
        if self.quan_input:
            self.nbit_a = nbit_a
            if self.mode == 'lsq':
                lsq_a  = LsqQuantizer4input(
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False)
                self.quan_a = lsq_a
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')
        
    def forward(self, input, scale_x=None):
        if self.quan_input:
            input, scale_x, _ = self.quan_a(input)
        return F.elu(input, self.alpha, self.inplace)
    
class QuanSigmoid(nn.Sigmoid):
    """docstring for QuanConv"""

    def __init__(self, quan_input=True, nbit_a=8, mode='lsq', N=1, C=1):
        super(QuanSigmoid, self).__init__()
        self.quan_input = quan_input
        self.mode = mode
        # PWL sigmoid approximation (7 seg points -> 8 segments)
        self.seg_point = torch.tensor([-4.375, -2.0, -1.0, 0.0, 1.0, 2.0, 4.375])
        self.coeff = torch.tensor([0.0, 0.046875, 0.15625, 0.234375, 0.234375, 0.15625, 0.046875, 0.0])
        self.intercept = torch.tensor([0.0, 0.203125, 0.421875, 0.5, 0.5, 0.578125, 0.796875, 1.0])
        if self.quan_input:
            self.nbit_a = nbit_a
            if self.mode == 'lsq':
                lsq_a  = LsqQuantizer4input(
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False)
                self.quan_a = lsq_a
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')
             
    def forward(self, input, scale_x=None):
        if not self.quan_input:
            return super().forward(input)

        input, scale_x, if_init = self.quan_a(input)

        if if_init:
            return super().forward(input)

        input = input / scale_x
        device = input.device

        scale = scale_x
        decimal_bit = -torch.log2(scale).int().item()

        seg_point_scale = self.seg_point.to(device)
        self.coeff = self.coeff.to(device)
        self.intercept = self.intercept.to(device)

        intercept_scale = round_to_nearest_bits_torch(self.intercept, 6) / scale
        coeff_scale = round_to_nearest_bits_torch(self.coeff, 6)

        seg_point_scale = round_to_nearest_bits_torch(seg_point_scale, decimal_bit)
        seg_point_scale = seg_point_scale / scale

        pwl_func = torch.zeros_like(input, device=device)
        mask = input.lt(seg_point_scale[0])
        pwl_func = torch.where(mask, intercept_scale[0] + coeff_scale[0] * input, pwl_func)
        for i in range(1, len(seg_point_scale)):
            mask = input.ge(seg_point_scale[i - 1]) & input.lt(seg_point_scale[i])
            pwl_func = torch.where(mask, intercept_scale[i] + coeff_scale[i] * input, pwl_func)
        mask = input.ge(seg_point_scale[-1])
        pwl_func = torch.where(mask, intercept_scale[-1] + coeff_scale[-1] * input, pwl_func)

        pwl_out = torch.clamp(pwl_func * scale, 0.0, 1.0)

        # Avoid emitting sigmoid during inference/export; keep STE behavior in training.
        if self.training and torch.is_grad_enabled():
            fp_out = super().forward(input * scale)
            return (pwl_out - fp_out).detach() + fp_out
        return pwl_out
    
class QuanGELU(nn.GELU):
    """docstring for QuanConv"""

    def __init__(self, quan_input=True, nbit_a=8, mode='lsq', N=1, C=1):
        super(QuanGELU, self).__init__()
        self.quan_input = quan_input
        self.mode = mode
        self.seg_point = torch.tensor([-3.015625, -2.203125, -0.890625, -0.421875, 0.03125, 0.625, 2.796875])
        self.coeff = torch.tensor([-0.0, -0.03125, -0.109375, 0.046875, 0.359375, 0.75, 1.078125, 1.0])
        self.intercept = torch.tensor([-0.0, -0.09375, -0.265625, -0.125, 0.0, -0.015625, -0.21875, -0.0])
        if self.quan_input:
            self.nbit_a = nbit_a
            if self.mode == 'lsq':
                lsq_a  = LsqQuantizer4input(
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False)
                self.quan_a = lsq_a
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')
        
    def forward(self, input, scale_x=None):
        if self.quan_input:
            input, scale_x, if_init = self.quan_a(input)
            quan_input = input
        if if_init:
            return super().forward(input)
        input = input/scale_x
        #=============================================
        device = input.device
        # fp func
        func = nn.GELU(approximate='tanh')      # 
        # pwl func
        scale = scale_x
        decimal_bit = -torch.log2(scale).int().item() 
        seg_point_scale = self.seg_point.to(device)
        self.coeff = self.coeff.to(device)
        self.intercept = self.intercept.to(device)
        
        intercept_scale = round_to_nearest_bits_torch(self.intercept, 6) / scale
        coeff_scale = round_to_nearest_bits_torch(self.coeff, 6)

        seg_point_scale = round_to_nearest_bits_torch(seg_point_scale, decimal_bit)
        seg_point_scale = seg_point_scale / scale
        pwl_func = torch.zeros_like(input, device = device)
        mask = input.lt(seg_point_scale[0])
        pwl_func = torch.where(mask, intercept_scale[0] + coeff_scale[0] * input, pwl_func)
        for i in range(1, len(seg_point_scale)):
            mask = input.ge(seg_point_scale[i-1]) & input.lt(seg_point_scale[i])
            pwl_func = torch.where(mask, intercept_scale[i] + coeff_scale[i] * input, pwl_func)
        mask = input.ge(seg_point_scale[-1])
        pwl_func = torch.where(mask, intercept_scale[-1] + coeff_scale[-1] * input, pwl_func)
            
        return (pwl_func * scale - func(input *scale)).detach() + func(input * scale) 
        # return F.gelu(input)

class QuanTanh(nn.Tanh):
    """docstring for QuanConv"""

    def __init__(self, quan_input=True, nbit_a=8, mode='lsq', N=1, C=1):
        super().__init__()
        self.quan_input = quan_input
        self.mode = mode
        # PWL tanh approximation (7 seg points -> 8 segments)
        self.seg_point = torch.tensor([-2.125, -1.25, -0.625, 0.0, 0.625, 1.25, 2.125])
        self.coeff = torch.tensor([0.0, 0.140625, 0.46875, 0.890625, 0.890625, 0.46875, 0.140625, 0.0])
        self.intercept = torch.tensor([-1.0, -0.671875, -0.265625, 0.0, 0.0, 0.265625, 0.671875, 1.0])
        if self.quan_input:
            self.nbit_a = nbit_a
            if self.mode == 'lsq':
                lsq_a = LsqQuantizer4input(bit=self.nbit_a, all_positive=False, per_channel=False)
                self.quan_a = lsq_a
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')

    def forward(self, input, scale_x=None):
        if not self.quan_input:
            return super().forward(input)

        input, scale_x, if_init = self.quan_a(input)

        # During LSQ init, fall back to exact tanh.
        if if_init:
            return super().forward(input)

        input = input / scale_x
        device = input.device

        scale = scale_x
        decimal_bit = -torch.log2(scale).int().item()

        seg_point_scale = self.seg_point.to(device)
        self.coeff = self.coeff.to(device)
        self.intercept = self.intercept.to(device)

        intercept_scale = round_to_nearest_bits_torch(self.intercept, 6) / scale
        coeff_scale = round_to_nearest_bits_torch(self.coeff, 6)

        seg_point_scale = round_to_nearest_bits_torch(seg_point_scale, decimal_bit)
        seg_point_scale = seg_point_scale / scale

        pwl_func = torch.zeros_like(input, device=device)
        mask = input.lt(seg_point_scale[0])
        pwl_func = torch.where(mask, intercept_scale[0] + coeff_scale[0] * input, pwl_func)
        for i in range(1, len(seg_point_scale)):
            mask = input.ge(seg_point_scale[i - 1]) & input.lt(seg_point_scale[i])
            pwl_func = torch.where(mask, intercept_scale[i] + coeff_scale[i] * input, pwl_func)
        mask = input.ge(seg_point_scale[-1])
        pwl_func = torch.where(mask, intercept_scale[-1] + coeff_scale[-1] * input, pwl_func)

        pwl_out = torch.clamp(pwl_func * scale, -1.0, 1.0)

        # Avoid emitting tanh during inference/export; keep STE behavior in training.
        if self.training and torch.is_grad_enabled():
            fp_out = super().forward(input * scale)
            return (pwl_out - fp_out).detach() + fp_out
        return pwl_out

def round_to_nearest_bits_torch(x, decimal_bits):
    """

    :param x: floating input
    :param decimal_bits: bits that the input should reserve
    :return: the formatted input with specific decimal bits
    """
    scaled_value = x * (2 ** decimal_bits)
    rounded_value = torch.round(scaled_value)  # very important
    result = rounded_value / (2 ** decimal_bits)
    y = result
    y_grad = x
    return (y - y_grad).detach() + y_grad

class QuanGELU_pwl(nn.Module):
    def __init__(self) -> None:
        super(QuanGELU_pwl, self).__init__()
        self.seg_point = torch.tensor([-3.015625, -2.203125, -0.890625, -0.421875, 0.03125, 0.625, 2.796875])
        self.coeff = torch.tensor([-0.0, -0.03125, -0.109375, 0.046875, 0.359375, 0.75, 1.078125, 1.0])
        self.intercept = torch.tensor([-0.0, -0.09375, -0.265625, -0.125, 0.0, -0.015625, -0.21875, -0.0])

    def forward(self, input, scale) -> torch.Tensor:
        device = input.device
        
        self.seg_point = self.seg_point.to(device)
        self.coeff = self.coeff.to(device)
        self.intercept = self.intercept.to(device)
        # self.intercept = self.coeff.to(device)
        # fp func
        func = nn.GELU(approximate='tanh')
        # pwl func
        decimal_bit = -torch.log2(scale).int().item() 
        seg_point_scale = self.seg_point
        intercept_scale = round_to_nearest_bits_torch(self.intercept, 6) / scale
        coeff_scale = round_to_nearest_bits_torch(self.coeff, 6)

        seg_point_scale = round_to_nearest_bits_torch(seg_point_scale, decimal_bit)
        seg_point_scale = seg_point_scale / scale
        pwl_func = torch.zeros_like(input, device = device)
        mask = input.lt(seg_point_scale[0])
        pwl_func = torch.where(mask, intercept_scale[0] + coeff_scale[0] * input, pwl_func)
        for i in range(1, len(seg_point_scale)):
            mask = input.ge(seg_point_scale[i-1]) & input.lt(seg_point_scale[i])
            pwl_func = torch.where(mask, intercept_scale[i] + coeff_scale[i] * input, pwl_func)
        mask = input.ge(seg_point_scale[-1])
        pwl_func = torch.where(mask, intercept_scale[-1] + coeff_scale[-1] * input, pwl_func)
        return (pwl_func * scale - func(input *scale)).detach() + func(input * scale), seg_point_scale

class QuanSoftmax(nn.Softmax):
    """对齐硬件 CModel 的 Softmax
       - 强制 quan_input=True
       - exp_s_out 作为类成员（暂时默认 0.0039）
       - 用原始exp的梯度反向传播
       - ymr
    """

    def __init__(self, dim, quan_input=True, nbit_a=8, mode='lsq', N=1, C=1, head=1, init_scale=None):
        super().__init__(dim)
        self.quan_input = quan_input
        self.mode = mode
        if self.quan_input:
            self.nbit_a = nbit_a
            if self.mode == 'lsq':
                lsq_a  = LsqQuantizer4input(
                                bit=self.nbit_a, 
                                all_positive=False,
                                per_channel=False,
                                init_scale = init_scale)
                self.quan_a = lsq_a
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')

        # self.quan_exp_new = LsqQuantizer4input(bit=8, all_positive=True, per_channel=False, init_scale = init_scale)
        # self.quan_exp_new.s.requires_grad = False
        # # self.quan_exp_new.initialized_alpha = 1
        # # self.quan_softmax_new = LsqQuantizer4input(bit=self.nbit_a, all_positive=False, per_channel=False)
        # # self.q_exp = pwl_exp_gqa
        self.register_buffer('exp_s_out', torch.tensor(0.0039), persistent=False)
        # 预计算并注册原始PWL模板参数（未缩放），并做一次静态量化到模板精度
        self._register_pwl_parameters()


    def _register_pwl_parameters(self):
        """预计算并注册PWL参数为buffer，避免重复创建"""
        # 原始参数
        seg_point = torch.tensor([-5.5, -3.3125, -2.375, -1.5625, -1.375, -0.75, -0.3125])
        coeff = torch.tensor([0.0, 0.015625, 0.0625, 0.140625, 0.234375, 0.359375, 0.59375, 0.859375])
        intercept = torch.tensor([0.0, 0.078125, 0.234375, 0.421875, 0.578125, 0.734375, 0.90625, 1.0])
        
        # 应用floor_to_nearest_bits
        seg_point = self._floor_to_nearest_bits_static(seg_point, 4)
        coeff = self._floor_to_nearest_bits_static(coeff, 6)
        intercept = self._floor_to_nearest_bits_static(intercept, 6)
        
        # 注册为 buffer，随 .to(device) 同步移动；作为常量不需要持久化到 state_dict 则设 persistent=False
        self.register_buffer('seg_point_tmpl', seg_point, persistent=False)
        self.register_buffer('coeff_tmpl', coeff, persistent=False)
        self.register_buffer('intercept_tmpl', intercept, persistent=False)


    def forward(self, input, scale_x=None, mask = None):
        if self.quan_input:
            input, scale_x, if_init = self.quan_a(input)    # 第一步伪量化，获得可以学习的scale_x，需要考虑这部分到底要不要计算梯度。
        else:
            if_init = False

        if mask is not None:            # 目前还没用过
            ori_input = input + mask    # 第二步对mask处理做一个硬件的适配，保持之前mask和input的梯度。
            if if_init:
                input = ori_input
            else:
                # inv_mask = 1 - mask
                # mask = inv_mask/-10000
                # scale_max = scale_x * -128
                # input = input * mask + inv_mask * scale_max
                hw_mask = 1 - mask / -10000
                mask_safe = (hw_mask - mask).detach() + mask
                input = (input * mask_safe + (1 - mask_safe) * scale_x * -128 - ori_input).detach() + ori_input
        
        sm_s_in = scale_x.detach()  # 想了一下后面不让scale接收梯度了。# TODO detach也不一定对
        sm_in_max, _ = torch.max(input, dim=-1, keepdim=True)
        # # attn_exp = self._pwl_exp_gqa(((input - attn_max) / sm_s_in), sm_s_in) # TODO
        # attn_exp = self._pwl_exp_gqa(input_in, sm_s_in)
        # attn_exp, exp_s_out,_ = self.quan_exp_new(attn_exp)
        if if_init:
            return super().forward(input)
        attn_exp = self._spu_exp(input - sm_in_max, sm_s_in)  # 这里相当于exp+量化，此时输出域是整数域。

        if mask is not None:
            attn_exp = attn_exp * mask
        # attn_exp_sum = torch.sum(attn_exp, dim=-1, keepdim=True)
        sum_exp = torch.sum(attn_exp, dim=-1) # 对应CModel，sum不保持最后一维
        reci_sum_exp = LN_hw_long_divider(torch.ones_like(sum_exp), (sum_exp), denom_decimal_bits=0, output_decimal_bits=20)
        attn_hw  = reci_sum_exp.unsqueeze(-1) * attn_exp

        attn_exact  = attn_exp / (sum_exp).unsqueeze(-1)  # 倒数用长除法实现，这里仅提供梯度
        attn = (attn_hw - attn_exact).detach() + attn_exact 

        x_out = floor_to_nearest_bits_torch(attn, 12)
        # x_out_cmodel = SoftMax_CModel_torch(input/scale_x, scale_x, self.exp_s_out*scale_x, 1.0)    # TODO 原代码的scale变换有问题，为不改动源代码，这里暂时加一个乘法。
        # sm_s_out = 1.0
        # x_out = x_out / sm_s_out
        # x_out = torch.clamp(x_out, -128, 127)
        # x_out = torch.round(x_out)
        return x_out

    def _floor_to_nearest_bits_static(self, x, decimal_bits):
        """静态版本的floor_to_nearest_bits，用于初始化"""
        return torch.floor(x * (2 ** decimal_bits)) / (2 ** decimal_bits)
    
    def _pwl_exp_gqa(self, input_in, sm_s_in):
        with torch.no_grad():                       # 我的想法是这部分不计算梯度，而是使用原版exp的梯度，不过这里的值一定会有误差，误差包括线性拟合和量化误差。
            input_in_q_s_in = input_in / sm_s_in
            pwl_func_q_s_in = self._spu_exp_pwl(input_in_q_s_in, sm_s_in) 
            pwl_func = pwl_func_q_s_in * sm_s_in
        exp_exact = torch.exp(input_in)
        return (pwl_func - exp_exact).detach() + exp_exact
    
    def _spu_exp(self, input_in, sm_s_in):
        pwl_func = self._pwl_exp_gqa(input_in, sm_s_in) # 这里硬件cmodel实现乘了一个sm_s_in我觉得不对
        pwl_func_exp_s_out = pwl_func / self.exp_s_out
        pwl_func_q_exp_s_out = torch.clamp(RoundPass.apply(pwl_func_exp_s_out) , 0, 255)
        pwl_func = pwl_func_q_exp_s_out* self.exp_s_out
        return pwl_func 

    def _spu_exp_pwl(self, input, sm_s_in):
        """
        使用 bucketize 的 PWL 实现（仅数值前向，不需要梯度）
        input: ((attn_score - attn_max) / sm_s_in)
        sm_s_in: 标量(或0维张量)
        seg_point, coeff, intercept: 1D张量，已为常量模板（未缩放）
        """
        device = input.device
        dtype = input.dtype
        sm_s_in = sm_s_in.to(device=device, dtype=dtype)

        seg_point = self.seg_point_tmpl.to(device=device, dtype=dtype)
        coeff     = self.coeff_tmpl.to(device=device, dtype=dtype)
        intercept = self.intercept_tmpl.to(device=device, dtype=dtype)

        # 除以 sm_s_in 之后再做量化
        seg_point_scale = self._floor_to_nearest_bits_static((seg_point / sm_s_in), 0)  # 0位
        intercept_scale = self._floor_to_nearest_bits_static((intercept / sm_s_in), 6)  # 6位小数
        coeff_scale = coeff  # coeff已做6bit量化

        # 使用bucketize更快
        indices = torch.bucketize(input, seg_point_scale, right=True)
        pwl_output = coeff_scale[indices] * input + intercept_scale[indices]
        pwl_output = torch.clamp(pwl_output, 0, 511)

        return pwl_output

def floor_to_nearest_bits_torch(x, decimal_bits):
    """

    :param x: floating input
    :param decimal_bits: bits that the input should reserve
    :return: the formatted input with specific decimal bits
    """
    scaled_value = x * (2 ** decimal_bits)
    rounded_value = torch.floor(scaled_value)  # very important
    result = rounded_value / (2 ** decimal_bits)
    y = result
    y_grad = x
    return (y - y_grad).detach() + y_grad

def LN_hw_long_divider(numer, denom, denom_decimal_bits, output_decimal_bits):
    if torch.any(denom == 0):
        denom[denom == 0] = torch.tensor(1.0).to(numer.device)
        # denom = torch.ones_like(denom).to(numer.device)
    denom = denom.expand_as(numer)
    # Create a tensor for the quotient
    quotient = torch.empty_like(numer)

    # Handle positive and negative values
    positive_mask = (numer >= 0) & (denom >= 0)
    negative_mask = (numer < 0) & (denom < 0)
    pos_neg_mask = (numer >= 0) & (denom < 0)
    neg_pos_mask = (numer < 0) & (denom >= 0)

    quotient[positive_mask] = numer[positive_mask] / denom[positive_mask]
    quotient[negative_mask] = -numer[negative_mask] / -denom[negative_mask]
    quotient[pos_neg_mask] = numer[pos_neg_mask] / -denom[pos_neg_mask]
    quotient[neg_pos_mask] = -numer[neg_pos_mask] / denom[neg_pos_mask]

    # Apply rounding function (assuming round_to_nearest_bits_torch is defined elsewhere)
    quotient = floor_to_nearest_bits_torch(quotient, output_decimal_bits)
    quotient[positive_mask] = quotient[positive_mask]
    quotient[negative_mask] = quotient[negative_mask]
    quotient[pos_neg_mask]  = -quotient[pos_neg_mask]
    quotient[neg_pos_mask]  = -quotient[neg_pos_mask]
    return quotient


class QuanMMHead(nn.Module):
    """docstring for QuanConv"""

    def __init__(self, quan_input_a=True, quan_input_b=True, nbit_a=8, mode='lsq', N_a=1, C=1, N_b=1, head=None, init_scale = None):
        super(QuanMMHead, self).__init__()
        self.quan_input_a = quan_input_a
        self.quan_input_b = quan_input_b
        self.mode = mode
        if self.quan_input_a:
            self.nbit_a = nbit_a
            if self.mode == 'lsq':
                lsq_a  = LsqQuantizer4input(
                                bit=self.nbit_a, 
                                all_positive=False,
                                per_channel=False,
                                init_scale = init_scale)
                self.quan_a = lsq_a
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')
        if self.quan_input_b:
            self.nbit_a = nbit_a
            if self.mode == 'lsq':
                lsq_b  = LsqQuantizer4input(
                                bit=self.nbit_a, 
                                all_positive=False,
                                per_channel=False,
                                init_scale = init_scale)
                self.quan_b = lsq_b
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')
        
    def forward(self, input_a, input_b, scale_a=None, scale_b=None):
        if self.quan_input_a:
            input_a, scale_a, _ = self.quan_a(input_a)
        if self.quan_input_b:
            input_b, scale_b, _ = self.quan_b(input_b)
        # return input_a @ input_b
        return torch.matmul(input_a, input_b)

class QuanEMMul(nn.Module):
    """Element-wise metrix multiplication"""

    def __init__(self, quan_input_a=True, quan_input_b=True, nbit_a_a=8, nbit_a_b=8, mode='lsq', N_a=1, C=1, N_b=1, head=None, init_scale = None):
        super(QuanEMMul, self).__init__()
        self.quan_input_a = quan_input_a
        self.quan_input_b = quan_input_b
        self.mode = mode
        if self.quan_input_a:
            self.nbit_a_a = nbit_a_a
            if self.mode == 'lsq':
                lsq_a  = LsqQuantizer4input(
                                bit=self.nbit_a_a, 
                                all_positive=False,
                                per_channel=False,
                                init_scale = init_scale)
                self.quan_a = lsq_a
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')
        if self.quan_input_b:
            self.nbit_a_b = nbit_a_b
            if self.mode == 'lsq':
                lsq_b  = LsqQuantizer4input(
                                bit=self.nbit_a_b, 
                                all_positive=False,
                                per_channel=False,
                                init_scale = init_scale)
                self.quan_b = lsq_b
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')
        
    def forward(self, input_a, input_b, scale_a=None, scale_b=None):
        if self.quan_input_a:
            input_a, scale_a, _ = self.quan_a(input_a)
        if self.quan_input_b:
            input_b, scale_b, _ = self.quan_b(input_b)
        return input_a * input_b
    
class QuanResidual(nn.Module):
    """docstring for QuanConv"""

    def __init__(self, quan_input_res=True, quan_input_out=True, nbit_a=16, mode='lsq', N=1, C=1): # TODO
        super(QuanResidual, self).__init__()
        self.quan_input_res = quan_input_res
        self.quan_input_out = quan_input_out
        self.mode = mode
        if self.quan_input_res:
            self.nbit_a = nbit_a
            if self.mode == 'lsq':
                lsq_a  = LsqQuantizer4input(
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False)
                self.quan_res = lsq_a
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')
        if self.quan_input_out:
            self.nbit_a = nbit_a
            if self.mode == 'lsq':
                lsq_b  = LsqQuantizer4input(
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False)
                self.quan_out = lsq_b
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')
        
    def forward(self, input_res, input_out, scale_a=None, scale_b=None):
        if self.quan_input_res:
            input_res, scale_res, _ = self.quan_res(input_res)
        if self.quan_input_out:
            input_out, scale_out, _ = self.quan_out(input_out)
        return input_res + input_out

class QuanResize(nn.Module):
    """docstring for QuanConv"""

    def __init__(self, quan_input=True, nbit_a=8, mode='lsq', N=1, C=1):
        super(QuanResize, self).__init__()
        self.quan_input = quan_input
        self.mode = mode
        if self.quan_input:
            self.nbit_a = nbit_a
            if self.mode == 'lsq':
                lsq_a  = LsqQuantizer4input(
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False)
                self.quan_a = lsq_a
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')
            
    def forward(self, input,
            size=None,
            scale_factor=None,
            mode='nearest',
            align_corners=None,
            warning=True):
        if warning:
            if size is not None and align_corners:
                input_h, input_w = tuple(int(x) for x in input.shape[2:])
                output_h, output_w = tuple(int(x) for x in size)
                if output_h > input_h or output_w > output_h:
                    if ((output_h > 1 and output_w > 1 and input_h > 1
                        and input_w > 1) and (output_h - 1) % (input_h - 1)
                            and (output_w - 1) % (input_w - 1)):
                        warnings.warn(
                            f'When align_corners={align_corners}, '
                            'the output would more aligned if '
                            f'input size {(input_h, input_w)} is `x+1` and '
                            f'out size {(output_h, output_w)} is `nx+1`')
        if isinstance(size, torch.Size):
            size = tuple(int(x) for x in size)
        if self.quan_input:
            input, scale_x, _ = self.quan_a(input)
        return F.interpolate(input, size, scale_factor, mode, align_corners)

class QuanLinear(nn.Linear):
    """docstring for QuanConv"""

    def __init__(self, in_features, out_features, bias=True, quan_input=True, mode='lsq', nbit_w=8, nbit_a=8, N=1, C=1, init_scale = None):
        super(QuanLinear, self).__init__(
            in_features, out_features, bias)
        self.mode = mode
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        # self.w = torch.zeros(self.weight.shape)
        # self.b = torch.zeros(self.weight.shape[0])
        # self.ilog = False
        if self.mode == 'lsq':
            lsq_w  = LsqQuantizer4weight(
                            bit=self.nbit_w,
                            all_positive=False,
                            per_channel=True,
                            per_channel_num=self.weight.shape[0],
                            init_scale = init_scale)
            self.quan_w = lsq_w
            
        self.quan_input = quan_input
        if self.quan_input:
            if self.mode == 'lsq':
                lsq_a  = LsqQuantizer4input(
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False,
                                # init_scale = init_scale
                                )
                self.quan_a = lsq_a
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')

    # @weak_script_method
    def forward(self, x, scale_x=None, scale_for_w=None):
        # return F.linear(x, self.weight, self.bias) 
        if self.quan_input:
            x, scale_x, _ = self.quan_a(x)
        # quantize input
        w = self.weight
        b = self.bias
        
        weight_integer, weight_scaling_factor, if_init = self.quan_w(w)
        if if_init:
            return super().forward(x)
        if b is not None:
            if self.mode == 'lsq':
                bias_integer = SymmetricQuantFunction.apply(b, 16, scale_x * weight_scaling_factor.squeeze()) * scale_x * weight_scaling_factor.squeeze()
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')
        else:
            bias_integer = None
        output = F.linear(x, weight_integer, bias_integer)
        return output
       
class QuanConv(nn.Conv2d):
    """docstring for QuanConv"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, norm=False, act=False, dilation=1, groups=1, 
                 bias=True, quan_input=True, mode_w='lsq', mode_a='lsq', vsq_mode='normal', nbit_w=8, nbit_a=8, init_scale = None): # vsq_mode: normal, pw, 3d
        super(QuanConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.mode_w = mode_w
        self.mode_a = mode_a
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        self.norm = norm
        self.act = act
        # self.w = torch.zeros(self.weight.shape)
        # self.b = torch.zeros(self.weight.shape[0])
        # self.ilog = False
        if self.mode_w == 'lsq':
            lsq_w  = LsqQuantizer4weight(
                            bit=self.nbit_w,
                            all_positive=False,
                            per_channel=True,
                            per_channel_num=self.weight.shape[0],
                            init_scale = init_scale)
            self.quan_w = lsq_w
        else:
            raise NotImplementedError('Not implemented other quantization technique yet')
        self.quan_input = quan_input
        if self.quan_input:
            self.nbit_a = nbit_a
            if self.mode_a == 'lsq' or self.kernel_size[0] != 1:
                lsq_a  = LsqQuantizer4input(
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False,
                                # init_scale = init_scale
                                )
                self.quan_a = lsq_a
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')
        if self.norm:
            self.bn_weight = torch.nn.parameter.Parameter(torch.ones(out_channels))
            self.bn_bias = torch.nn.parameter.Parameter(torch.zeros(out_channels))
            self.register_buffer('bn_running_mean', torch.zeros(out_channels))
            self.register_buffer('bn_running_var', torch.ones(out_channels))
            self.bn_running_mean = torch.zeros(out_channels)
            self.bn_running_var = torch.ones(out_channels)
            self.momentum = 0.1

    # @weak_script_method
    def forward(self, x, scale_x=None, scale_for_w=None):
        # print(self.layer_name)
        if self.quan_input:
            x, scale_x, _ = self.quan_a(x)
        
        # quantize input
        # x = input
        if self.training and self.norm:
            output = F.conv2d(x , self.weight, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)
            mean_bn = output.mean([0, 2,3], keepdim=False).squeeze(0)
            var_bn = output.var([0, 2,3], keepdim=False).squeeze(0)
            output1 = F.batch_norm(output, self.bn_running_mean, self.bn_running_var, weight=self.bn_weight,
                                   bias=self.bn_bias, training=self.training, momentum=0.1, eps=1e-05)
        if not self.training and self.norm:
            mean_bn = self.bn_running_mean
            var_bn = self.bn_running_var
            
        if self.norm:
            tmp = self.bn_weight / torch.sqrt(var_bn + 1e-5)
            w = tmp.view(tmp.size()[0], 1, 1, 1) * self.weight
            b = self.bias
            if self.bias:
                b = tmp*(self.bias - mean_bn) + self.bn_bias
            else:
                b = tmp * (0 - mean_bn) + self.bn_bias
        else:
            w = self.weight
            b = self.bias

        weight_integer, weight_scaling_factor, if_init = self.quan_w(w)
        if if_init:
            return super().forward(x)
        # print(weight_scaling_factor.shape)
        # sys.exit()
        if b is not None:
            if self.mode_a == 'lsq':
                bias_integer = SymmetricQuantFunction.apply(b, 16, scale_x * weight_scaling_factor.squeeze()) * scale_x * weight_scaling_factor.squeeze()
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')
        else:
            bias_integer = None
        # if (bias_integer != None):
        #     bias_integer = bias_integer.to(torch.float32)
        x = F.conv2d(x, weight_integer.to(train_dtype), bias_integer.to(train_dtype) if bias_integer is not None else bias_integer, self.stride, self.padding, self.dilation, self.groups) 


        if self.training and self.norm:
            output1 = output1 - x.detach() + x
            
        if self.act:
            return F.relu(x)
        else:
            return x
     
def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize floating point input tensor to integers with the given scaling factor and zeropoint.

    Parameters:
    ----------
    input: floating point input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """
    # reshape scale and zeropoint for convolutional weights and activations
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        scale = scale.view(-1)
        zero_point = zero_point.view(-1)
    if inplace:
        input.mul_(1. / scale).add_(zero_point).round_()
        return input
    return torch.round(1. / scale * input + zero_point)

class SymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale=None):
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        Note that the current implementation of SymmetricQuantFunction requires pre-calculated scaling factor.
        specified_scale: pre-calculated scaling factor for the tensor x
        """
        n = 2 ** (k - 1) - 1

        if specified_scale is not None:
            scale = specified_scale
        else:
            raise ValueError("The SymmetricQuantFunction requires a pre-calculated scaling factor")

        zero_point = torch.tensor(0., device = x.device)

        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)

        new_quant_x = torch.clamp(new_quant_x, -n - 1, n)

        ctx.scale = scale
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):

        scale = ctx.scale
        if len(grad_output.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
        # reshape scale and zeropoint for linear weights
        elif len(grad_output.shape) == 2:
            scale = scale.view(-1, 1)
        else:
            scale = scale.view(-1)

        return grad_output.clone() / scale, None, None, None
    
class QuanLayerNorm(nn.LayerNorm): # 跟softmax一样的也是整条的
    """docstring for QuanConv"""

    def __init__(self, normalized_shape, quan_input=True, nbit_a=8, mode='lsq', N=1, C=1, elementwise_affine = False, eps=1e-6, per_channel = False, init_scale = None):
        super(QuanLayerNorm, self).__init__(normalized_shape, eps = eps, elementwise_affine = elementwise_affine)
        self.quan_input = quan_input
        self.mode = mode
        if self.quan_input:
            self.nbit_a = nbit_a
            if self.mode == 'lsq':
                if per_channel == False:
                    lsq_a  = LsqQuantizer4input(
                                    bit=self.nbit_a,
                                    all_positive=False,
                                    per_channel=False,
                                    init_scale = init_scale)
                self.quan_a = lsq_a
            else:
                raise NotImplementedError('Not implemented other quantization technique yet')
        self.eps = eps
    def forward(self, input, scale_x=None):
        # original_input = input.clone().detach()
        if self.quan_input:
            input, scale_x, _ = self.quan_a(input)
        #     # 打印量化前后的统计信息
        # with torch.no_grad():
        #     # 计算量化误差
        #     diff = (original_input - input).abs()
        #     print(f"Quantization Error - Max: {diff.max():.6f}, Mean: {diff.mean():.6f}")
        return F.layer_norm(input, self.normalized_shape, weight = None, bias = None, eps = self.eps)
