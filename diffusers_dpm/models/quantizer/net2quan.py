from torch import nn
import torch
# from diffusers.models.attention_processor import Attention
# from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, FP32SiLU, SwiGLU
from einops import rearrange
import torch.nn.functional as F
from .quan_layer_annan_2 import QuanLayerNorm, LsqQuantizer4input, QuanMMHead, QuanSoftmax, QuanSigmoid, QuanGELU, QuanEMMul, LsqQuantizer4weight, LsqQuantizer4input, SymmetricQuantFunction
from .quan_layer_annan_2 import QuanConv, QuanLinear
# from .NCQuan_layer import QuanConv, QuanLinear, set_ncq_init_param, set_ncqlora_init_param, NCQuantizer4input
from .quan_layer_annan_2 import set_lsq_init_param, quantize_jit
import numpy as np
nbit_w=8
def set_nbit_w(set_nbit_w):
    assert set_nbit_w<=8, f'set_nbit_w {set_nbit_w} needs to be less than 8 '
    global nbit_w
    nbit_w = set_nbit_w

class MHAttention_own_quan(nn.Module):   
    def __init__(self, is_causal=False, dropout_level=0.0, n_heads=4, quantize = True):
        super().__init__()
        self.is_causal = is_causal
        self.dropout_level = dropout_level
        self.n_heads = n_heads
        assert is_causal == False , 'is_causal not False'
        assert dropout_level == 0 , 'dropout_level not 0'
        
        # self.mul_qk = torch.bmm
        # self.mul_av = torch.bmm
        # self.softmax = new_masked_softmax
        if quantize:
            self.mul_qk = QuanMMHead(quan_input_a=True, quan_input_b=True)
            self.mul_av = QuanMMHead(quan_input_a=True, quan_input_b=True)
            self.softmax = QuanSoftmax(dim=-1)      # 这个很耗时
            # self.softmax = nn.Softmax(dim=-1)
        else:
            self.mul_qk = torch.matmul
            self.mul_av = torch.matmul
            # self.softmax = new_masked_softmax
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):
        assert attn_mask == None
        assert q.size(-3) == k.size(-3)
        assert k.size(-2)*k.size(-1) == v.size(-2)*v.size(-1)

        q, k, v = [rearrange(x, "b (n d) h w-> b n d (h w)", n=self.n_heads) for x in [q, k, v]]   # (d n)=c (h w) = l "b (n d) l 1-> b n d l"
        q, k, v = [x.transpose(-1, -2).contiguous() for x in [q, k, v]]   # (d n)=c 
        
        self.scale = q.shape[-1] ** -0.5

        # q, k, v = [rearrange(x, "b n l d-> (b n) l d", n=self.n_heads).contiguous() for x in [q, k, v]]   # (d n)=c (h w) = l "b (n d) l 1-> b n d l"
        # 计算注意力得分
        if attn_mask is not None:
            attn_mask = attn_mask.repeat_interleave(self.n_heads, dim=0)
        attn = (self.mul_qk(q , k.transpose(-1, -2)) * self.scale)     # 这步必须是float32
        # attn = self.softmax(attn, mask = attn_mask)
        attn = self.softmax(attn)   # nn.Softmax(dim=-1)
        # 加权求和 V
        out = self.mul_av(attn, v)
        
        # out = rearrange(out, '(b n) l d -> b n l d', h=h, w=w, n=self.n_heads) # (h w) = l 'b n d l -> b (n d) l 1'
        # 将 hidden_states 变回 (batch_size, seq_len, dim) 的形状
        w = h = int(np.sqrt(out.size(-2)))
        out = out.transpose(-1, -2) 
        out = rearrange(out, 'b n d (h w) -> b (n d) h w', h=h, w=w, n=self.n_heads).contiguous() # (h w) = l 'b n d l -> b (n d) l 1'

        return out
    
class MHAttention_own(nn.Module):   
    def __init__(self, is_causal=False, dropout_level=0.0, n_heads=4, cross_attention = False):
        super().__init__()
        self.is_causal = is_causal
        self.dropout_level = dropout_level
        self.n_heads = n_heads
        assert is_causal == False , 'is_causal not False'
        assert dropout_level == 0 , 'dropout_level not 0'

    def forward(self, q, k, v, attn_mask=None, **cond_kwargs):
        assert q.size(-3) == k.size(-3)
        assert k.size(-2)*k.size(-1) == v.size(-2)*v.size(-1)

        # q, k, v = [rearrange(x, "b (n d) l 1-> (b n) l d", n=self.n_heads).contiguous() for x in [q, k, v]]   # (d n)=c
        q, k, v = [rearrange(x, "b (n d) h w-> (b n) d (h w)", n=self.n_heads) for x in [q, k, v]]   # (d n)=c (h w) = l "b (n d) l 1-> b n d l"
        q, k, v = [x.transpose(-1, -2).contiguous() for x in [q, k, v]]   # (d n)=c 
        self.scale = q.shape[-1] ** -0.5
        # 计算注意力得分
        if attn_mask is not None:
            attn_mask = attn_mask.repeat_interleave(self.n_heads, dim=0)
        attn = (torch.bmm(q , k.transpose(-1, -2)) * self.scale)     # 这步必须是float32
        attn = new_masked_softmax(attn, mask = attn_mask)
        # 加权求和 V
        out = torch.bmm(attn, v)
        # 将 hidden_states 变回 (batch_size, seq_len, dim) 的形状
        w = h = int(np.sqrt(out.size(-2)))
        out = out.transpose(-1, -2)
        out = rearrange(out, '(b n) d (h w) -> b (n d) h w', h=h, w=w, n=self.n_heads).contiguous() # (h w) = l 'b n d l -> b (n d) l 1'

        return out
    
class Attention_own_quan(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        out_bias: bool = True,
    ) -> None:
        super().__init__()
        self.query_dim = query_dim
        self.inner_dim = dim_head * heads
        self.inner_kv_dim = self.inner_dim 
        self.num_heads = heads
        self.dim_head = dim_head
        self.out_dim = query_dim
        self.scale = dim_head ** -0.5
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.to_q = Linear_Quan(query_dim, self.inner_dim, bias=bias)
        self.to_k = Linear_Quan(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_v = Linear_Quan(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        # hidden_states, encoder_hidden_states
        self.quan_hs_before_attn = LsqQuantizer4input(bit=8, all_positive=False, per_channel=False,)
        if cross_attention_dim is not None: 
            self.quan_encoder_hidden_states_before_attn = LsqQuantizer4input(bit=8, all_positive=False, per_channel=False,)
        self.mul_qk = QuanMMHead(quan_input_a=True, quan_input_b=True, init_scale = 10)
        self.mul_av = QuanMMHead(quan_input_a=True, quan_input_b=True, init_scale = 10)
        self.upcast_attention = upcast_attention
        
        self.to_out = nn.ModuleList([
            Linear_Quan(query_dim, self.out_dim, bias=out_bias, init_scale=4),
            nn.Dropout(dropout)
        ])

        self.softmax = QuanSoftmax(dim=-1, init_scale=2)

    def forward(self, hs, encoder_hidden_states=None, attention_mask=None, **cond_kwargs):   #cross_attention_kwargs必须是空
        # B, N, C = hs.shape
        hs, scale_hs,_ = self.quan_hs_before_attn(hs)
        if encoder_hidden_states is None:
            encoder_hidden_states = hs
            scale_encoder_hidden_states = scale_hs
        else:
            encoder_hidden_states, scale_encoder_hidden_states,_ = self.quan_encoder_hidden_states_before_attn(encoder_hidden_states)
        
        # 生成 q, k, v
        q = self.to_q(hs, **cond_kwargs)
        k = self.to_k(encoder_hidden_states, **cond_kwargs)
        v = self.to_v(encoder_hidden_states, **cond_kwargs)
        # 将 q, k, v 转换为 (batch_size * num_heads, seq_len, head_dim) 的形状
        q, k, v = [rearrange(x, "b s (h d) -> (b h) s d", h=self.num_heads) for x in [q, k, v]]
        # q = head_to_batch_dim(q, self.num_heads)
        # k = head_to_batch_dim(k, self.num_heads)
        # v = head_to_batch_dim(v, self.num_heads)

        # 计算注意力得分
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(self.num_heads, dim=0)

        # attn = (self.mul_qk(q.to(torch.float32), k.transpose(-1, -2).to(torch.float32)) * self.scale).to(torch.float16)
        # attn = (torch.bmm(q, k.transpose(-1, -2)) * self.scale)
        attn = (self.mul_qk(q, k.transpose(-1, -2)) * self.scale)     # 这步必须是float32
        attn = self.softmax(attn, attention_mask = attention_mask)
        # attn = new_masked_softmax(attn, mask = attention_mask)
        # 加权求和 V
        # hidden_states = torch.bmm(attn, v)
        hidden_states = self.mul_av(attn, v)
        # 将 hidden_states 变回 (batch_size, seq_len, dim) 的形状
        hidden_states = rearrange(hidden_states, "(b h) s d -> b s (h d)", h=self.num_heads)
        # hidden_states = batch_to_head_dim(hidden_states, self.num_heads)
        # 投影输出并应用 dropout
        hidden_states = self.to_out[0](hidden_states, **cond_kwargs)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

# region 未量化Attention_own    
class Attention_own(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        out_bias: bool = True,
    ) -> None:
        super().__init__()
        self.query_dim = query_dim
        self.inner_dim = dim_head * heads
        self.inner_kv_dim = self.inner_dim 
        self.num_heads = heads
        self.dim_head = dim_head
        self.out_dim = query_dim
        self.scale = dim_head ** -0.5
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        # self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        # self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        # self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_q = Linear_Quan(query_dim, self.inner_dim, bias=bias)
        self.to_k = Linear_Quan(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_v = Linear_Quan(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        
        self.upcast_attention = upcast_attention

        self.to_out = nn.ModuleList([
            # nn.Linear(query_dim, self.out_dim, bias=out_bias),
            Linear_Quan(query_dim, self.out_dim, bias=out_bias),
            nn.Dropout(dropout)
        ])

        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cond_kwargs):
        B, N, C = hidden_states.shape
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # 生成 q, k, v
        q = self.to_q(hidden_states, **cond_kwargs)
        k = self.to_k(encoder_hidden_states, **cond_kwargs)
        v = self.to_v(encoder_hidden_states, **cond_kwargs)
        # 将 q, k, v 转换为 (batch_size * num_heads, seq_len, head_dim) 的形状
        q, k, v = [rearrange(x, "b s (h d) -> (b h) s d", h=self.num_heads) for x in [q, k, v]]
        # q = head_to_batch_dim(q, self.num_heads)
        # k = head_to_batch_dim(k, self.num_heads)
        # v = head_to_batch_dim(v, self.num_heads)

        # 计算注意力得分
        if attention_mask is not None:
            # target_length = encoder_hidden_states.shape[1]
            # current_length = attention_mask.shape[-1]
            # if current_length !=target_length:
            #     attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
            attention_mask = attention_mask.repeat_interleave(self.num_heads, dim=0)
        
        #     beta = 1
        #     baddbmm_input = attention_mask
        # else:
        #     baddbmm_input = torch.empty(
        #         q.shape[0], q.shape[1], k.shape[1], dtype=q.dtype, device=q.device
        #     )
        #     beta = 0
        # attn = torch.baddbmm(
        #     baddbmm_input,
        #     q,
        #     k.transpose(-1, -2),
        #     beta=beta,
        #     alpha=self.scale,
        # )
        # del baddbmm_input
        # attn = self.softmax(attn)

        # attn = (torch.bmm(q.to(torch.float32), k.transpose(-1, -2).to(torch.float32)) * self.scale).to(torch.float16)
        attn = (torch.bmm(q, k.transpose(-1, -2)) * self.scale)     # 这步必须是float32
        # attn = self.softmax(attn, attention_mask = attention_mask)
        attn = new_masked_softmax(attn, mask = attention_mask)
        # 加权求和 V
        hidden_states = torch.bmm(attn, v)
        # 将 hidden_states 变回 (batch_size, seq_len, dim) 的形状
        hidden_states = rearrange(hidden_states, "(b h) s d -> b s (h d)", h=self.num_heads)
        # hidden_states = batch_to_head_dim(hidden_states, self.num_heads)
        # 投影输出并应用 dropout
        hidden_states = self.to_out[0](hidden_states, **cond_kwargs)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
    
def new_masked_softmax(x, mask, dim=-1):
    """
    新的mask方法：使用乘法和修改分母
    """
    if mask is not None:
        mask = (1-mask/-10000)
        # 应用mask到输入
        x = x * mask + (1-mask) * torch.min(x)
    
    # 找最大值
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    # 计算exp并再次应用mask
    exp_x = torch.exp(x - x_max) 

    if mask is not None:
        # 应用mask到输入
        exp_x = exp_x * mask
    # 计算分母
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    
    # 计算softmax
    return exp_x / sum_exp_x

# def head_to_batch_dim(tensor: torch.Tensor, num_heads) -> torch.Tensor:
#     """
#     Reshape the tensor from [batch_size, seq_len, dim] to [batch_size * heads, seq_len, dim // heads]
#     """
#     batch_size, seq_len, dim = tensor.shape
#     dim_head = dim // num_heads
#     tensor = tensor.view(batch_size, seq_len, num_heads, dim_head)
#     # [batch_size, num_heads, seq_len, head_dim] -> [batch_size * num_heads, seq_len, head_dim]
#     tensor = tensor.permute(0, 2, 1, 3).contiguous().reshape(batch_size * num_heads, seq_len, dim_head)
#     return tensor

# def batch_to_head_dim(tensor: torch.Tensor, num_heads) -> torch.Tensor:
#     """
#     Reshape the tensor from [batch_size * heads, seq_len, dim // heads] back to [batch_size, seq_len, dim]
#     """
#     batch_size_heads, seq_len, dim_head = tensor.shape
#     batch_size = batch_size_heads // num_heads
#     # [batch_size * num_heads, seq_len, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
#     tensor = tensor.view(batch_size, num_heads, seq_len, dim_head)
#     # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, dim]
#     tensor = tensor.permute(0, 2, 1, 3).contiguous().reshape(batch_size, seq_len, num_heads * dim_head)
#     return tensor


# endregion    
class Conv2d_Quan(QuanConv):       # nn.Conv2d QuanConv
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        groups = 1,
        init_scale = None,
        mode_w = 'lsq', 
        mode_a = 'lsq',
        **kwargs
    ):  #! padding = 'same', groups=
        # super().__init__(in_channels, out_channels, kernel_size, groups = groups, **kwargs)
        super().__init__(in_channels, out_channels, kernel_size, groups = groups, init_scale = init_scale, nbit_w=nbit_w, mode_w = mode_w, mode_a = mode_a, **kwargs)
        assert groups == 1 or groups == in_channels == out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

  
class LayerNorm_Quan(QuanLayerNorm): # nn.LayerNorm QuanLayerNorm, QuanLayerNorm_eff    # 暂时不考虑layernorm的权重系数
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        init_scale = None,
        **kwargs
    ):
        # super().__init__(normalized_shape, eps = eps, elementwise_affine = elementwise_affine, **kwargs)
        super().__init__(normalized_shape, eps = eps, elementwise_affine = elementwise_affine, per_channel = False, init_scale = init_scale, **kwargs)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.mul = QuanEMMul(quan_input_a=True, quan_input_b=True, init_scale = init_scale) # nbit_a_b=nbit_w,
        # assert elementwise_affine == False, "elementwise_affine can't be True"
    def forward(self, x):
        normalized = super().forward(x)
        if self.elementwise_affine:
            # 应用weight和bias进行仿射变换
            normalized = self.mul(normalized , self.weight) + self.bias
            # normalized = normalized * self.weight + self.bias
        return normalized
    
class Linear_Quan(QuanConv):   # nn.Conv2d nn.Linear QuanLinear QuanConv
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_scale = None,
        mode_w = 'lsq', 
        mode_a = 'lsq',
        **kwargs
    ):
        # # super().__init__(in_features, out_features, **kwargs)
        # super().__init__(in_features, out_features, init_scale = init_scale, nbit_w=nbit_w, mode_w = mode_w, mode_a = mode_a,**kwargs)
        # self.in_features = in_features
        # self.out_features = out_features
        super().__init__(in_features, out_features, kernel_size=1, init_scale = init_scale, nbit_w=nbit_w, mode_w = mode_w, mode_a = mode_a,**kwargs)

    # def forward(self, x):
    #     # assert x.dim() == 3, 'Linear_Quan, x.dim() != 3'
    #     if x.dim() == 3:
    #         x = x.permute(0, 2, 1).contiguous().unsqueeze(-1)   # (batch_size, in_channels, height*width, 1)
    #         output = super().forward(x).permute(0, 2, 1, 3).contiguous().squeeze(-1)
    #     elif x.dim() == 2:
    #         x = x.unsqueeze(-1).unsqueeze(-1)   # (batch_size, in_channels, height*width, 1)
    #         output = super().forward(x).squeeze(-1).squeeze(-1)
    #     return output

class Embedding_Quan(Conv2d_Quan):  # Conv2d_Quan Linear_Quan
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_scale = None,
        **kwargs
    ):
        super().__init__(num_embeddings, embedding_dim, kernel_size=1, init_scale = init_scale, **kwargs) 
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, indices):     # indices: [B]
        one_hot = F.one_hot(indices, num_classes=self.num_embeddings).unsqueeze(-1).unsqueeze(-1)   # [B, indices, 1, 1]
        out = super().forward(one_hot)
        return out
    
class Parameter_Quan(nn.Parameter):     #! 我的想法是这个部分全都放在外面改
    def __new__(
        cls, data=None, requires_grad=True
        # **kwargs
    ):
        return super().__new__(cls, data, requires_grad)
    # # def forward(self, x):
    # #     #! self.scale_shift_table[None] 有这么用的，不能这么定义
    # #     return self.param(x)
    
    # # GatedSelfAttentionDense有个特殊的操作.tanh(),但是暂时用不上

class nn_GELU_Quan(QuanGELU):    # nn.GELU QuanGELU
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()   # 有一些GELU带近似，似乎影响不大，我给删了 # **kwargs
    
class diff_GELU_Quan(nn.Module):    #GELU
    #     #这个相比nn的似乎还多了一个线性层
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True, **kwargs):
        super().__init__()
        # self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.proj = Linear_Quan(dim_in, dim_out, bias=bias, init_scale=8)
        self.act = nn_GELU_Quan(**kwargs)
        self.approximate = approximate

    def forward(self, hidden_states, **kwargs):
        hidden_states = self.proj(hidden_states, **kwargs)
        hidden_states = self.act(hidden_states) # mps: gelu is not implemented for float16
        return hidden_states
    
class SiLU_Quan(QuanSigmoid):    # nn.SiLU nn.Sigmoid QuanSigmoid
    def __init__(
        self,
        **kwargs
    ):
        # print(**kwargs)
        super().__init__(**kwargs)  
    #     self.mul = QuanEMMul(quan_input_a=True, quan_input_b=True, init_scale = 3)
    # #     self.act = nn.SiLU()
    # #     # print("SiLU_Quan",kwargs)
    # def forward(self, x):
    #     # return super().forward(x)
    #     # return super().forward(x) * x
    #     return self.mul(super().forward(x) , x)
    
class Attention_Quan(Attention_own_quan):    # Attention Attention_own Attention_own_quan
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
    #     self.attn = Attention(**kwargs)
    #     # print("Attention_Quan", kwargs)
    #     # self.attn = Attention_own_quan(**kwargs)
    # def forward(self, hs, encoder_hidden_states=None, attention_mask=None):
    #     return self.attn(hs, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask)


    
class GEGLU(nn.Module):
    def __init__():
        super().__init__()
        raise NotImplementedError("GEGLU cannot be Quan.")

class ApproximateGELU(nn.Module):
    def __init__():
        super().__init__()
        raise NotImplementedError("ApproximateGELU cannot be Quan.")
    
class SwiGLU(nn.Module):
    def __init__():
        super().__init__()
        raise NotImplementedError("SwiGLU cannot be Quan.")
    

class IdentityWithThreeOutputs(nn.Identity):
    def forward(self, x):
        return x, None, None
        
class QuantizableLayer:
    
    @staticmethod
    def Conv2d(in_channels, out_channels, kernel_size, quantize, quan_kwargs={}, **kwargs):
        if quantize:
            return Conv2d_Quan(in_channels, out_channels, kernel_size, **{**kwargs, **quan_kwargs} )
        else:
            return nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)

    @staticmethod
    def Linear(in_features, out_features, quantize, quan_kwargs={}, **kwargs):
        if quantize:
            return Linear_Quan(in_features, out_features, **{**kwargs, **quan_kwargs} )
        else:
            # return nn.Linear(in_features, out_features, **kwargs)
            return nn.Conv2d(in_features, out_features, kernel_size=1, **kwargs)
    
    @staticmethod
    def Embedding11(num_embeddings, embedding_dim, quantize, quan_kwargs={}, **kwargs):
        if quantize:
            return Embedding_Quan(num_embeddings, embedding_dim, **{**kwargs, **quan_kwargs} )
        else:
            return nn.Embedding(num_embeddings, embedding_dim, **kwargs).unsqueeze(-1).unsqueeze(-1)    # 和量化版本同步

    @staticmethod
    def nn_GELU(quantize, quan_kwargs={}, **kwargs):
        """创建nn.GELU层，不是diffusers中的GELU"""
        if quantize:
            return nn_GELU_Quan(**{**kwargs, **quan_kwargs})
        else:
            return nn.GELU(**kwargs)
    
    # @staticmethod # 忘了怎么写了，先放着
    # def diffusers_GELU(quantize=True, quan_kwargs={}, **kwargs):
    #     if quantize:
    #         return net2quan.diff_GELU_Quan(**{**kwargs, **quan_kwargs})
    #     else:
    #         return diffusers.GELU(**kwargs)

    @staticmethod
    def LayerNorm(normalized_shape, quantize, quan_kwargs={}, **kwargs):
        if quantize:
            return LayerNorm_Quan(normalized_shape, **{**kwargs, **quan_kwargs} )
        else:
            return nn.LayerNorm(normalized_shape, **kwargs)

    @staticmethod
    def MHAttention(is_causal, dropout_level, n_heads, _MHAttention, quantize, quan_kwargs={}, **kwargs):
        if quantize:
            return MHAttention_own_quan(is_causal, dropout_level, n_heads, quantize = quantize, **{**kwargs, **quan_kwargs} )
        else:
            if _MHAttention == None:
                return MHAttention_own_quan(is_causal, dropout_level, n_heads, quantize = quantize, **{**kwargs, **quan_kwargs} )
            else:
                return _MHAttention(is_causal, dropout_level, n_heads, **kwargs)


    @staticmethod
    def Quantizer4input(quantize, **kwargs):
        if quantize:
            return LsqQuantizer4input(**kwargs)
        else:
            return IdentityWithThreeOutputs()
    
    @staticmethod
    def QuanEMMul(quantize, quan_input_a=True, quan_input_b=True, **kwargs):
        if quantize:
            return QuanEMMul(quan_input_a, quan_input_b, **kwargs)
        else:
            return torch.mul
    
    @staticmethod
    def QuanSoftmax(quantize, dim = -1,  **kwargs):
        if quantize:
            return QuanSoftmax(dim = dim, **kwargs)
        else:
            return nn.Softmax(dim=dim)
        
    # 写一个upsample
    # 写一个 tahn