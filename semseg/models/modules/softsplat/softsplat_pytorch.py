import torch

def softsplat_soft_torch(
    tenIn: torch.Tensor,       # [B, C_in, H, W]
    tenFlow: torch.Tensor,     # [B, 2, H, W]  (dx, dy)
    tenMetric: torch.Tensor,   # [B, 1, H, W]
    eps: float = 1e-7,
    metric_clip_max: float = 20.0,
):
    """
    Pure PyTorch forward splat (soft mode):
      out = sum( in * exp(metric) * bilinear_w ) / ( sum( exp(metric) * bilinear_w ) + eps )

    Returns:
      tenOut: [B, C_in, H, W]
    """
    assert tenIn.ndim == 4 and tenFlow.ndim == 4 and tenMetric.ndim == 4
    B, C, H, W = tenIn.shape
    assert tenFlow.shape == (B, 2, H, W)
    assert tenMetric.shape == (B, 1, H, W)

    device = tenIn.device
    dtype = tenIn.dtype

    # 1) soft权重：w = exp(clip(metric))
    w = torch.exp(torch.clamp(tenMetric, max=metric_clip_max))  # [B,1,H,W]

    # 2) numerator / denominator 的“源像素值”
    src_num = tenIn * w            # [B,C,H,W]  利用广播机制,将置信度乘以原特征图的像素值
    src_den = w                    # [B,1,H,W]

    # 3) 生成每个源像素的目标连续坐标 (x+dx, y+dy)
    #    x: [H,W], y: [H,W]
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",  # matrix模式，第一个输出yy对应第一个输入arrange(H), 第二个输出xx对应第二个输入arrange(W)
    )
    xx = xx.unsqueeze(0).expand(B, -1, -1)  # [B,H,W]
    yy = yy.unsqueeze(0).expand(B, -1, -1)  # [B,H,W]

    dx = tenFlow[:, 0, :, :].to(dtype)      # [B,H,W]
    dy = tenFlow[:, 1, :, :].to(dtype)      # [B,H,W]

    x = xx + dx
    y = yy + dy

    # 4) 四邻居整数坐标
    x0 = torch.floor(x).to(torch.long)      # [B,H,W] 假设计算出的目标位置是 (x=10.4, y=5.7)：此时 (x0, y0) 就是 (10, 5)，x0 是左边界。数值含义是：原来住在源图像 (y, x) 位置的这个像素，经过光流搬运后，落在了目标图像的新位置。这个新位置的【整数左边界】是多少
    y0 = torch.floor(y).to(torch.long)      # 这是 y+dy后的上边界
    x1 = x0 + 1                             # 这是 x+dx后的右边界
    y1 = y0 + 1                             # 这是 y+dy后的下边界        

    # 5) 双线性权重（float）
    x0f = x0.to(dtype); x1f = x1.to(dtype)
    y0f = y0.to(dtype); y1f = y1.to(dtype)

    w_nw = (x1f - x) * (y1f - y)            # 表示和x0,y0的距离越近，权重越大
    w_ne = (x - x0f) * (y1f - y)            # 表示和x1,y0的距离越近，权重越大
    w_sw = (x1f - x) * (y - y0f)            # 表示和x0,y1的距离越近，权重越大
    w_se = (x - x0f) * (y - y0f)            # 表示和x1,y1的距离越近，权重越大

    # 6) 每个邻居的有效mask（越界的贡献应为0）,做越界检查，光流把像素“吹”到了一个新的位置，但这个新位置可能跑出了画布（图像）的范围。我们需要判断这个新位置周围的 4 个邻居，哪些还在画里，哪些已经跑出去了。
    def valid_mask(xi, yi):
        return (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) # 四个括号，每个括号输出的shape都是 [B,H,W]，表示每个像素点是否满足某个边界条件。最后一期在对应位置做与运算，得到最终的mask

    m_nw = valid_mask(x0, y0)       # 左上
    m_ne = valid_mask(x1, y0)       # 右上
    m_sw = valid_mask(x0, y1)       # 左下
    m_se = valid_mask(x1, y1)       # 右下

    # 7) flatten 到 HW 维度，用 scatter_add 做 “atomic add 累加”
    HW = H * W
    # 1.创建画布：申请全 0 的张量来存放结果
    out_num = tenIn.new_zeros((B, C, HW))
    out_den = tenIn.new_zeros((B, 1, HW))

    #2. 拉平输入：把源数据也从 (B, C, H, W) 变成 (B, C, HW)
    src_num_f = src_num.reshape(B, C, HW)       # 原特征图的像素值乘以置信度
    src_den_f = src_den.reshape(B, 1, HW)       # 置信度

    def splat_one(xi, yi, wij, mask):   # 这个函数负责把源图像的像素，搬运到目标图像的某一个角落（比如左上角）。它会被调用 4 次。
        # idx: [B,HW]
        idx = (yi * W + xi).reshape(B, HW)  # 将二维坐标 (y, x) 转换为一维索引 idx = y * Width + x

        # 越界 idx 会乱写，所以把越界 idx 改成 0，同时把权重置 0
        wij = (wij * mask.to(dtype)).reshape(B, 1, HW)  # [B,1,HW]      # 如果 mask 是 False（越界），就把权重 wij 变成 0
        idx_safe = idx.clone()
        idx_safe[~mask.reshape(B, HW)] = 0                          # # ~mask 表示“取反”，即选中那些越界（False）的点

        idx_num = idx_safe.unsqueeze(1).expand(B, C, HW)  # [B,C,HW]
        idx_den = idx_safe.unsqueeze(1)                   # [B,1,HW]

        out_num.scatter_add_(2, idx_num, src_num_f * wij)       # wij的shape([B,1,HW])
        out_den.scatter_add_(2, idx_den, src_den_f * wij)

    splat_one(x0, y0, w_nw, m_nw)   # x0是个矩阵，每个元素代表原来位置的像素，经过光流搬运后，落在目标像素最近的左上角位置。y0同理 ，w是距离权重，m是mask
    splat_one(x1, y0, w_ne, m_ne)   
    splat_one(x0, y1, w_sw, m_sw)
    splat_one(x1, y1, w_se, m_se)

    # 8) 归一化
    out_den = out_den + eps
    out = (out_num / out_den).reshape(B, C, H, W)
    return out



# 就是一个简单的循环
def scatter_add_cpu_3d(out, index, src):
    """
    针对 Softsplat 场景的手搓版 scatter_add
    假设 tensor 形状都是 [B, C, N]，且 dim=2
    """
    # 1. 获取维度信息
    B, C, N = src.shape
    
    # 2. 开始遍历每一个“源像素” (三层循环)
    for b in range(B):              # 遍历 Batch
        for c in range(C):          # 遍历 Channel
            for i in range(N):      # 遍历 Flatten 后的像素位置
                
                # A. 拿到“钱”：要存多少值？
                value = src[b, c, i]
                
                # B. 拿到“指令”：要去哪个柜子？
                # 注意：index 的形状和 src 是一样的，一一对应
                target_pos = index[b, c, i]
                
                # C. 存钱：找到 output 对应的位置累加
                # dim=2，说明我们在第3个维度上跳变，前两个维度 (b, c) 保持一致
                out[b, c, target_pos] += value
                
    return out
