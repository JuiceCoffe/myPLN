import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PLNLoss"]


class PLNLoss(nn.Module):
    def __init__(
        self,
        branch_idx,
        num_classes=20,
        grid_size=14,
        coord_weight=2.0,
        link_weight=0.5,
        class_weight=0.5,
        noobj_weight=0.04,
        alpha=0.25, 
        gamma=2.0
    ):
        super().__init__()
        self.branch_idx = branch_idx
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.coord_weight = coord_weight
        self.link_weight = link_weight
        self.class_weight = class_weight
        self.noobj_weight = noobj_weight
        
        self.alpha = alpha
        self.gamma = gamma
        self.feature_size = 1 + 2 + (2 * grid_size) + num_classes
        self.last_stats = {}

    def forward(self, pred_tensor, target_tensor):
        if target_tensor.dim() != 5:
            raise ValueError(f"PLN target tensor must be [B,4,H,W,C], got {tuple(target_tensor.shape)}")

        target_branch = target_tensor[:, self.branch_idx, ...]
        pred = pred_tensor.permute(0, 2, 3, 1).contiguous().view(*target_branch.shape[:3], 4, self.feature_size)
        target = target_branch.view(*target_branch.shape[:3], 4, self.feature_size)

        # 1. 基础 Mask
        conf_target = target[..., 0]
        pos_mask = conf_target > 0
        pos_mask_f = pos_mask.float()
        neg_mask_f = (~pos_mask).float()

        # 【限制概率域，防止 log(0) 产生 NaN】
        conf_pred = torch.clamp(pred[..., 0], min=1e-6, max=1.0 - 1e-6)

        # ---------------------------------------------------------
        # 【修改 1: 点存在性损失 (Point Existence Loss) -> Focal Loss】
        # ---------------------------------------------------------
        # 正样本 (point) 的 Focal Loss: -alpha * (1-p)^gamma * log(p)
        point_loss_elem = -self.alpha * (1.0 - conf_pred) ** self.gamma * torch.log(conf_pred)
        point_loss = (point_loss_elem * pos_mask_f).sum()

        # 负样本 (noobj) 的 Focal Loss: -(1-alpha) * p^gamma * log(1-p)
        noobj_loss_elem = -(1.0 - self.alpha) * conf_pred ** self.gamma * torch.log(1.0 - conf_pred)
        noobj_loss = (noobj_loss_elem * neg_mask_f).sum()

        # ---------------------------------------------------------
        # 【修改 2: 坐标回归损失 (Coordinate Loss) -> L1 Loss】
        # ---------------------------------------------------------
        # L1 绝对值误差在 offset 预测中梯度的稳定性远优于 MSE
        coord_loss = (torch.abs(pred[..., 1:3] - target[..., 1:3]) * pos_mask_f.unsqueeze(-1)).sum()

        # ---------------------------------------------------------
        # 链接损失 (保持不变，因为是基于 Softmax 的分布约束)
        # ---------------------------------------------------------
        link_loss = ((pred[..., 3:3 + (2 * self.grid_size)] - target[..., 3:3 + (2 * self.grid_size)]) ** 2 * pos_mask_f.unsqueeze(-1)).sum()

        # ---------------------------------------------------------
        # 【修改 3: 分类损失 
        # ---------------------------------------------------------
        cls_pred = torch.clamp(pred[..., 3 + (2 * self.grid_size):], min=1e-6, max=1.0 - 1e-6)
        cls_target = target[..., 3 + (2 * self.grid_size):]

        bce_loss = F.binary_cross_entropy(cls_pred, cls_target, reduction='none')
        class_loss = (bce_loss * pos_mask_f.unsqueeze(-1)).sum()

        # ---------------------------------------------------------
        # 汇总
        # ---------------------------------------------------------
        total_loss = (
            point_loss
            + (self.coord_weight * coord_loss)
            + (self.link_weight * link_loss)
            + (self.class_weight * class_loss)
            + (self.noobj_weight * noobj_loss)
        )

        # 保持 logging 字段的兼容性
        self.last_stats = {
            "point": point_loss.detach(),
            "coord": coord_loss.detach(),
            "link": link_loss.detach(),
            "class": class_loss.detach(),
            "noobj": noobj_loss.detach(),
            "total": total_loss.detach(),
        }
        return total_loss