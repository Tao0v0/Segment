import torch
from torch import Tensor
from typing import Tuple


class Metrics:
    def __init__(self, num_classes: int, ignore_label: int, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).to(device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        ious[ious.isnan()]=0.
        miou = ious.mean().item()
        # miou = ious[~ious.isnan()].mean().item()
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        f1[f1.isnan()]=0.
        mf1 = f1.mean().item()
        # mf1 = f1[~f1.isnan()].mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        acc = self.hist.diag() / self.hist.sum(1)
        acc[acc.isnan()]=0.
        macc = acc.mean().item()
        # macc = acc[~acc.isnan()].mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)


def compute_epe(pred_flow, gt_flow):
    """
    pred_flow: (B, 2, H, W) - 预测的光流
    gt_flow: (B, 2, H, W) - 真实的光流
    """
    # 计算每个像素点的 EPE
    epe = torch.sqrt((pred_flow[:, 0] - gt_flow[:, 0]) ** 2 + (pred_flow[:, 1] - gt_flow[:, 1]) ** 2)
    # 对所有像素点取平均
    mean_epe = epe.mean()
    return mean_epe

def compute_npe(pred_flow, gt_flow, n_values=[1, 2, 3]):
    """
    pred_flow: (B, 2, H, W) - 预测的光流
    gt_flow: (B, 2, H, W) - 真实的光流
    n_values: list of N values (e.g., [1, 2, 3])
    """
    # 计算 EPE
    epe = torch.sqrt((pred_flow[:, 0] - gt_flow[:, 0]) ** 2 + (pred_flow[:, 1] - gt_flow[:, 1]) ** 2)
    
    # 统计 EPE > N 的像素点数量
    n1pe = ((epe > 1).float().mean() * 100).item()  # 转换为百分比
    n2pe = ((epe > 2).float().mean() * 100).item()
    n3pe = ((epe > 3).float().mean() * 100).item()
    
    return n1pe, n2pe, n3pe