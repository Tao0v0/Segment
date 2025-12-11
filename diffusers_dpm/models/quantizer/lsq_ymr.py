import torch
import csv
from abc import ABC, abstractmethod

train_dtype = torch.float32
set_step = 1000
set_select_index = 20
set_save_csv = False
def set_lsq_init_param(init_step, init_select_index, init_save_csv):
    global set_step, set_select_index, set_save_csv
    assert init_step == 1000
    assert init_select_index == 20
    assert set_save_csv == False
    set_step = init_step
    set_select_index = init_select_index
    set_save_csv = init_save_csv
    # print(set_step, set_select_index, set_save_csv)
class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

class RoundPass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.round()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# class TruncPass(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         return input.trunc()
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output

class Clip(torch.autograd.Function):        # 这个似乎已经自带梯度了
    @staticmethod
    def forward(ctx, input, eps):
        return torch.clamp(input, min=eps)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
    

def batch_frexp(inputs, bit=8):
    output_m, output_e = torch.frexp(inputs)
    # output_m = torch.where(torch.isnan(output_m), 0, output_m)
    output_m = torch.nan_to_num(output_m, nan=0.0)
    output_m = torch.round(output_m * (2 ** bit)).to(torch.int16)
    output_e = float(bit) - output_e
    return output_m, output_e

# ==================== JIT编译的前向函数 ====================
# @torch.jit.script
def quantize_jit(x: torch.Tensor, scale: torch.Tensor, 
                          thd_neg: float, thd_pos: float):
    x = x / scale
    x = torch.clamp(x, thd_neg, thd_pos)
    x = RoundPass.apply(x)
    return x * scale

# @torch.jit.script  
def quantize_jit_vectorized(x: torch.Tensor, scale: torch.Tensor,
                                     thd_neg: float, thd_pos: float):
    scale = scale.view(-1, 1, 1, 1).contiguous()
    x = x / scale    
    x = torch.clamp(x, thd_neg, thd_pos)
    x = RoundPass.apply(x)
    return x * scale

# # ==================== JIT编译的反向函数，有问题 ====================
# @torch.jit.script
# def compute_backward_jit(grad_output: torch.Tensor, 
#                                   scale: torch.Tensor,
#                                   scaled: torch.Tensor,
#                                   in_range_mask: torch.Tensor,
#                                   thd_neg: float, 
#                                   thd_pos: float):
#     grad_input = grad_output * in_range_mask / scale
#     quantized_scaled = torch.clamp(scaled, thd_neg, thd_pos).round()
#     n_in_range = torch.sum(in_range_mask).float()
    
#     if n_in_range > 0.0:
#         term1 = quantized_scaled
#         term2 = -scaled
#         grad_scale = torch.sum(grad_output * (term1 + term2)).view(1)
#         grad_scale = grad_scale / torch.sqrt(n_in_range)
#     else:
#         grad_scale = torch.zeros_like(scale)
#     return grad_input, grad_scale

# @torch.jit.script
# def compute_backward_vectorized_jit(grad_output: torch.Tensor,
#                                              scale: torch.Tensor,
#                                              scaled: torch.Tensor,
#                                              in_range_mask: torch.Tensor,
#                                              thd_neg: float,
#                                              thd_pos: float):
#     scale_expanded = scale.view(-1, 1, 1, 1)
#     grad_input = grad_output * in_range_mask / scale_expanded
    
#     quantized_scaled = torch.clamp(scaled, thd_neg, thd_pos).round()
#     term1 = torch.sum(grad_output * quantized_scaled, dim=(1, 2, 3))
#     term2 = torch.sum(grad_output * (-scaled * in_range_mask), dim=(1, 2, 3))
    
#     n_in_range = torch.sum(in_range_mask, dim=(1, 2, 3)).float()
#     valid_channels = n_in_range > 0.0
#     grad_scale = torch.zeros_like(n_in_range)
#     grad_scale[valid_channels] = (term1[valid_channels] + term2[valid_channels]) / torch.sqrt(n_in_range[valid_channels])
    
#     return grad_input, grad_scale

# # ==================== 主要的量化类，反向传播有问题 ====================
# class JITLSQQuantization(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, scale, thd_neg, thd_pos, per_channel=False):
#         if per_channel:
#             output, scaled, in_range_mask = quantize_jit_vectorized(input, scale, thd_neg, thd_pos)
#         else:
#             output, scaled, in_range_mask = quantize_jit(input, scale, thd_neg, thd_pos)
        
#         ctx.save_for_backward(scale, scaled, in_range_mask)
#         ctx.thd_neg = thd_neg
#         ctx.thd_pos = thd_pos
#         ctx.per_channel = per_channel
#         return output
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         scale, scaled, in_range_mask = ctx.saved_tensors
#         thd_neg, thd_pos = ctx.thd_neg, ctx.thd_pos
#         per_channel = ctx.per_channel
        
#         if per_channel:
#             grad_input, grad_scale = compute_backward_vectorized_jit(
#                 grad_output, scale, scaled, in_range_mask, thd_neg, thd_pos)
#         else:
#             grad_input, grad_scale = compute_backward_jit(
#                 grad_output, scale, scaled, in_range_mask, thd_neg, thd_pos)
                
#         return grad_input, grad_scale, None, None, None
    
class BaseLsqQuantizer(torch.nn.Module, ABC):
    def __init__(self, bit, all_positive=False, per_channel=False, per_channel_num=1, head = None, init_scale=None):
        super().__init__()
        self.bit = bit
        assert bit != 1, 'not support bit==1 until uncomment the part of _apply_quantization' 
        self.per_channel = per_channel
        self.all_positive = all_positive
        self.init_scale = init_scale
        self.per_channel_num = per_channel_num
        # Set thresholds based on bit width and sign
        if all_positive:
            self.thd_neg = 0
            self.thd_pos = 1 if bit == 1 else 2 ** bit - 1
        else:
            self.thd_neg = -1 if bit == 1 else -2 ** (bit - 1)
            self.thd_pos = 1 if bit == 1 else 2 ** (bit - 1) - 1

        # Initialize scale parameter
        if per_channel:
            if per_channel_num is None:
                raise ValueError("per_channel_num must be specified when per_channel is True")
            self.s = torch.nn.Parameter(torch.ones(per_channel_num)/(2 ** (bit - 1) - 1))
        else:
            self.s = torch.nn.Parameter(torch.ones(1)/(2 ** (bit - 1) - 1))

        self.register_buffer('initialized_alpha', torch.zeros(1))

    @abstractmethod
    def init_from(self, x, *args, **kwargs):
        pass
    
    def _calc_ptq_scale(self, x):
        if self.per_channel:
            if len(x.shape) == 4:
                init_val_max = x.detach().abs().amax(dim=(1,2,3))
            if len(x.shape) == 2:
                init_val_max = x.detach().abs().amax(dim=(1))
        else:
            init_val_max = x.detach().abs().max()
        scale = init_val_max / self.thd_pos
        return scale
    
    # def _apply_quantization(self, x, scale):
    #     x = self._get_x_integer(x, scale)
    #     return x * scale
    
    # def _get_x_integer(self, x, scale):
    #     x = x / scale
    #     if self.bit == 1 and not self.all_positive:
    #         x = torch.sign(x)
    #     else:
    #         x = torch.clamp(x, self.thd_neg, self.thd_pos)
    #         x = RoundPass.apply(x)
    #     return x

class CalculateInputScale(torch.autograd.Function):
    """
        s_scale = GradScale.apply(Clip.apply(self.s, self.eps), s_grad_scale)
        pow = torch.round(torch.log2(s_scale.detach()))
        clip_val = torch.exp2(pow)
        scale = (clip_val - s_scale).detach() + s_scale
    """
    # 这些不知道为什么用jit不会加速，甚至会更慢。
    @staticmethod
    def forward(ctx, s, eps, s_grad_scale):
        ctx.s_grad_scale = s_grad_scale
        s_clipped = torch.clamp(s, min=eps)
        pow = torch.round(torch.log2(s_clipped))
        scale = torch.exp2(pow)
        return scale
    
    @staticmethod
    def backward(ctx, grad_output):
        s_grad_scale = ctx.s_grad_scale
        grad_s = grad_output * s_grad_scale
        return grad_s, None, None, None
    
class LsqQuantizer4input(BaseLsqQuantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_vals = []
        self.register_buffer('eps', torch.tensor(2**-16), persistent=False)

    def init_from(self, x, modified_init=True):
        if modified_init:
            if self.per_channel:
                init_val_max = x.detach().abs().amax(dim=(0,1)) / self.thd_pos
                init_val_lsq = (2 if not self.all_positive else 4) * x.detach().abs().mean(dim=(0,1)) / (self.thd_pos ** 0.5)
            else:
                init_val_max = x.detach().abs().max() / self.thd_pos
                init_val_lsq = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
            # init_val = init_val_max if self.bit >= 8 and init_val_max < init_val_lsq else init_val_lsq
            if self.bit >= 8:
                init_val = torch.where(init_val_max < init_val_lsq, init_val_max, init_val_lsq)
            else:
                init_val = init_val_lsq
        else:
            init_val = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)

        self.init_vals.append(init_val.item())
        # print('self.initialized_alpha == 0', len(self.init_vals), set_step)
        if len(self.init_vals) == set_step:
            init_scale = 1 if self.init_scale is None else self.init_scale
            init_vals_sorted = sorted(self.init_vals)
            self.s.data.copy_(init_vals_sorted[set_select_index] * init_scale)
            self.initialized_alpha.fill_(1)
            if set_save_csv:
                self.init_vals.append(self.s.data.item())
                self.save_to_file()
            # del self.init_vals

    def forward(self, x, modified_init=True):
        if self.initialized_alpha == 0:
            self.init_from(x, modified_init=modified_init)
            return x, torch.tensor(1.0, device = x.device, dtype=x.dtype), True
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        """
        s_scale = GradScale.apply(Clip.apply(self.s, self.eps), s_grad_scale)
        pow = torch.round(torch.log2(s_scale.detach()))
        clip_val = torch.exp2(pow)
        scale = (clip_val - s_scale).detach() + s_scale
        """
        scale = CalculateInputScale.apply(self.s, self.eps, s_grad_scale)

        # scale_m, scale_e = batch_frexp(s_scale.detach(), bit=self.bit)
        # scale = (scale_m / torch.pow(2, scale_e)).type(train_dtype)
        # scale = (scale - s_scale).detach() + s_scale

        # x = self._apply_quantization(x, scale)
        # x = JITLSQQuantization.apply(x, scale, self.thd_neg, self.thd_pos, self.per_channel)
        x = quantize_jit(x, scale, self.thd_neg, self.thd_pos)

        return x, scale, False

    def save_to_file(self):
        with open('init_vals.csv', 'a', newline='') as f:
            csv.writer(f).writerow(self.init_vals)

class CalculateWeightScale(torch.autograd.Function):
    """
        s_scale = GradScale.apply(Clip.apply(alpha, self.eps), s_grad_scale)
        scale_m, scale_e = batch_frexp(s_scale.detach(), bit=self.bit)
        scale = (scale_m / torch.exp2(scale_e)).type(train_dtype)
        scale_new = (scale - s_scale).detach() + s_scale
    """
    @staticmethod
    def forward(ctx, alpha, eps, s_grad_scale, bit):
        ctx.s_grad_scale = s_grad_scale
        s_scale = torch.clamp(alpha, min=eps)
        scale_m, scale_e = batch_frexp(s_scale, bit=bit)
        scale = (scale_m / torch.exp2(scale_e)).type(train_dtype)
        return scale
    
    @staticmethod
    def backward(ctx, grad_output):
        s_grad_scale = ctx.s_grad_scale
        alpha_grad = grad_output * s_grad_scale
        return alpha_grad, None, None, None
    
class LsqQuantizer4weight(BaseLsqQuantizer):
    def __init__(self, *args, learnable=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.learnable = learnable
        self.init_time = 0
        self.register_buffer('eps', torch.tensor(2^-16), persistent=False)

    def init_from(self, x):
        if self.initialized_alpha == 0:
            if self.per_channel:
                init_val = self._compute_per_channel_init(x)
            else:
                init_val = x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)
            
            init_scale = 1 if self.init_scale is None else self.init_scale
            self.s.data.copy_(init_val * init_scale)
        self.init_time +=1
        if self.init_time == set_step:
            self.initialized_alpha.fill_(1)
            del self.init_time

    def _compute_per_channel_init(self, x):
        if len(x.shape) == 1:  # bias
            return x.detach().abs() / self.thd_pos
        
        factor = 4 if self.all_positive else 2
        if len(x.shape) == 2:  # linear weight
            return factor * x.detach().abs().mean(dim=-1) / (self.thd_pos ** 0.5)
        elif len(x.shape) == 4:  # conv weight
            return factor * x.detach().abs().mean(dim=-1).mean(dim=-1).mean(dim=-1) / (self.thd_pos ** 0.5)
        
    def forward(self, x, ilog=False):
        if self.initialized_alpha < 1:
            self.init_from(x)
            return x, torch.tensor(1.0, device = x.device, dtype=x.dtype), True
        # alpha = self._get_alpha_shape(self.s, x).to(x.device)
        alpha = self.s.to(x.device)
        s_grad_scale = self._compute_grad_scale(x)
        """
        s_scale = GradScale.apply(Clip.apply(alpha, self.eps), s_grad_scale)
        scale_m, scale_e = batch_frexp(s_scale.detach(), bit=self.bit)
        scale = (scale_m / torch.exp2(scale_e)).type(train_dtype)
        scale_new = (scale - s_scale).detach() + s_scale
        """
        scale_new = CalculateWeightScale.apply(alpha, self.eps, s_grad_scale, self.bit)
        # scale_new = self._reshape_scale(scale_new, x.shape)   #放在JITLSQQuantization里了

        # x = self._apply_quantization(x, scale_new)
        # x = JITLSQQuantization.apply(x, scale_new, self.thd_neg, self.thd_pos, self.per_channel)
        x = quantize_jit_vectorized(x, scale_new, self.thd_neg, self.thd_pos)
        return x, scale_new, False

    def _get_alpha_shape(self, s, x):   # remove in next version
        if not self.per_channel:
            return s
        return s if len(x.shape) == 1 else torch.unsqueeze(s, dim=-1)

    def _compute_grad_scale(self, x):
        if not self.per_channel:
            return 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        
        if len(x.shape) == 1:
            return 1.0 / (self.thd_pos ** 0.5)
        elif len(x.shape) == 2:
            return 1.0 / ((self.thd_pos * x.shape[-1]) ** 0.5)
        elif len(x.shape) == 4:
            return 1.0 / ((self.thd_pos * x.shape[-3] * x.shape[-2] * x.shape[-1]) ** 0.5)

    def _reshape_scale(self, scale, shape):
        if len(shape) == 2:
            return scale.view(scale.shape[0], 1)
        elif len(shape) == 4:
            return scale.view(scale.shape[0], 1, 1, 1)
        return scale