import torch.nn as nn

__all__ = ["PLNLoss"]


class PLNLoss(nn.Module):
    def __init__(
        self,
        branch_idx,
        num_classes=20,
        grid_size=14,
        point_weight=1.0,
        coord_weight=2.0,
        link_weight=0.5,
        class_weight=0.5,
        noobj_weight=0.04,
    ):
        super().__init__()
        self.branch_idx = branch_idx
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.point_weight = point_weight
        self.coord_weight = coord_weight
        self.link_weight = link_weight
        self.class_weight = class_weight
        self.noobj_weight = noobj_weight
        self.feature_size = 1 + 2 + (2 * grid_size) + num_classes
        self.last_stats = {}

    def forward(self, pred_tensor, target_tensor):
        if target_tensor.dim() != 5:
            raise ValueError(f"PLN target tensor must be [B,4,H,W,C], got {tuple(target_tensor.shape)}")

        target_branch = target_tensor[:, self.branch_idx, ...]
        pred = pred_tensor.permute(0, 2, 3, 1).contiguous().view(*target_branch.shape[:3], 4, self.feature_size)
        target = target_branch.view(*target_branch.shape[:3], 4, self.feature_size)

        conf_pred = pred[..., 0]
        conf_target = target[..., 0]
        pos_mask = conf_target > 0
        pos_mask_f = pos_mask.float()
        neg_mask_f = (~pos_mask).float()

        point_loss = ((conf_pred - conf_target) ** 2 * pos_mask_f).sum()
        coord_loss = ((pred[..., 1:3] - target[..., 1:3]) ** 2 * pos_mask_f.unsqueeze(-1)).sum()
        link_loss = ((pred[..., 3:3 + (2 * self.grid_size)] - target[..., 3:3 + (2 * self.grid_size)]) ** 2 * pos_mask_f.unsqueeze(-1)).sum()
        class_loss = ((pred[..., 3 + (2 * self.grid_size):] - target[..., 3 + (2 * self.grid_size):]) ** 2 * pos_mask_f.unsqueeze(-1)).sum()
        noobj_loss = ((conf_pred - conf_target) ** 2 * neg_mask_f).sum()

        total_loss = (
            (self.point_weight * point_loss)
            + (self.coord_weight * coord_loss)
            + (self.link_weight * link_loss)
            + (self.class_weight * class_loss)
            + (self.noobj_weight * noobj_loss)
        )

        self.last_stats = {
            "point": point_loss.detach(),
            "coord": coord_loss.detach(),
            "link": link_loss.detach(),
            "class": class_loss.detach(),
            "noobj": noobj_loss.detach(),
            "total": total_loss.detach(),
        }
        return total_loss
