import torch
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
        noobj_weight=0.05,
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
        print("point_weight:",point_weight)
        print("coord_weight:",coord_weight)
        print("link_weight:",link_weight)
        print("class_weight:",class_weight)
        print("noobj_weight:",noobj_weight)
        self.feature_size = 1 + 2 + (2 * grid_size) + num_classes
        self.last_stats = {}

    def forward(self, pred_tensor, target_tensor):
        if target_tensor.dim() != 5:
            raise ValueError(f"PLN target tensor must be [B,4,H,W,C], got {tuple(target_tensor.shape)}")

        target_branch = target_tensor[:, self.branch_idx, ...]
        pred = pred_tensor.permute(0, 2, 3, 1).contiguous().view(*target_branch.shape[:3], 4, self.feature_size)
        target = target_branch.view(*target_branch.shape[:3], 4, self.feature_size)
        
        # 获取当前 GPU 上的实际 batch_size (如果是 DataParallel，这里可能是 mini_batch_size / GPU数量)
        gpu_batch_size = pred.size(0)

        conf_pred = pred[..., 0]
        conf_target = target[..., 0]
        
        pos_mask = conf_target > 0
        neg_mask = ~pos_mask
        
        num_pos = pos_mask.sum().float().clamp(min=1.0)

        # ==========================================
        # 1. 负样本损失 (只取负样本计算，节省算力)
        # ==========================================
        noobj_pred = conf_pred[neg_mask]
        noobj_target = conf_target[neg_mask]
        # 注意：这里直接 sum，不除以 batch_size，交由外部 Engine 处理
        noobj_loss = ((noobj_pred - noobj_target) ** 2).sum()

        # ==========================================
        # 2. 正样本损失 (布尔索引提取)
        # ==========================================
        pos_pred_feats = pred[pos_mask]     # [N_pos, feature_size]
        pos_target_feats = target[pos_mask] # [N_pos, feature_size]

        if pos_pred_feats.numel() > 0:
            c_idx = 1
            coord_idx = c_idx + 2
            link_idx = coord_idx + (2 * self.grid_size)
            
            p_conf_pred, p_conf_tgt = pos_pred_feats[:, 0], pos_target_feats[:, 0]
            p_coord_pred, p_coord_tgt = pos_pred_feats[:, c_idx:coord_idx], pos_target_feats[:, c_idx:coord_idx]
            p_link_pred, p_link_tgt = pos_pred_feats[:, coord_idx:link_idx], pos_target_feats[:, coord_idx:link_idx]
            p_cls_pred, p_cls_tgt = pos_pred_feats[:, link_idx:], pos_target_feats[:, link_idx:]

            # 在通道维度用 mean，在样本维度用 sum。
            # 乘以 gpu_batch_size / num_pos 是为了：
            # 1. 消除一张图里目标数量波动带来的梯度方差 (除以 num_pos)
            # 2. 保证输出量级是 "当前 GPU 的 batch_size" 级别，以便外部引擎统一除以 batch_size。
            scale_factor = gpu_batch_size / num_pos

            point_loss = ((p_conf_pred - p_conf_tgt) ** 2).sum() * scale_factor
            coord_loss = ((p_coord_pred - p_coord_tgt) ** 2).mean(dim=1).sum() * scale_factor
            link_loss  = ((p_link_pred - p_link_tgt) ** 2).mean(dim=1).sum() * scale_factor
            class_loss = ((p_cls_pred - p_cls_tgt) ** 2).mean(dim=1).sum() * scale_factor
        else:
            # 兼容 DataParallel 下某一块 GPU 完全没有正样本的极端情况
            device = pred.device
            point_loss = torch.tensor(0.0, device=device)
            coord_loss = torch.tensor(0.0, device=device)
            link_loss  = torch.tensor(0.0, device=device)
            class_loss = torch.tensor(0.0, device=device)

        total_loss = (
            (self.point_weight * point_loss)
            + (self.coord_weight * coord_loss)
            + (self.link_weight * link_loss)
            + (self.class_weight * class_loss)
            + (self.noobj_weight * noobj_loss)
        )

        # 记录 detach 后的数据供 logging 使用
        self.last_stats = {
            "point": point_loss.detach(),
            "coord": coord_loss.detach(),
            "link": link_loss.detach(),
            "class": class_loss.detach(),
            "noobj": noobj_loss.detach(),
            "total": total_loss.detach(),
        }
        
        return total_loss
    



# import torch.nn as nn

# __all__ = ["PLNLoss"]


# class PLNLoss(nn.Module):
#     def __init__(
#         self,
#         branch_idx,
#         num_classes=20,
#         grid_size=14,
#         point_weight=1.0,
#         coord_weight=2.0,
#         link_weight=0.5,
#         class_weight=0.5,
#         noobj_weight=0.04,
#     ):
#         super().__init__()
#         self.branch_idx = branch_idx
#         self.num_classes = num_classes
#         self.grid_size = grid_size
#         self.point_weight = point_weight
#         self.coord_weight = coord_weight
#         self.link_weight = link_weight
#         self.class_weight = class_weight
#         self.noobj_weight = noobj_weight
#         self.feature_size = 1 + 2 + (2 * grid_size) + num_classes
#         self.last_stats = {}

#     def forward(self, pred_tensor, target_tensor):
#         if target_tensor.dim() != 5:
#             raise ValueError(f"PLN target tensor must be [B,4,H,W,C], got {tuple(target_tensor.shape)}")

#         target_branch = target_tensor[:, self.branch_idx, ...]
#         pred = pred_tensor.permute(0, 2, 3, 1).contiguous().view(*target_branch.shape[:3], 4, self.feature_size)
#         target = target_branch.view(*target_branch.shape[:3], 4, self.feature_size)

#         conf_pred = pred[..., 0]
#         conf_target = target[..., 0]
#         pos_mask = conf_target > 0
#         pos_mask_f = pos_mask.float()
#         neg_mask_f = (~pos_mask).float()

#         point_loss = ((conf_pred - conf_target) ** 2 * pos_mask_f).sum()
#         coord_loss = ((pred[..., 1:3] - target[..., 1:3]) ** 2 * pos_mask_f.unsqueeze(-1)).sum()
#         link_loss = ((pred[..., 3:3 + (2 * self.grid_size)] - target[..., 3:3 + (2 * self.grid_size)]) ** 2 * pos_mask_f.unsqueeze(-1)).sum()
#         class_loss = ((pred[..., 3 + (2 * self.grid_size):] - target[..., 3 + (2 * self.grid_size):]) ** 2 * pos_mask_f.unsqueeze(-1)).sum()
#         noobj_loss = ((conf_pred - conf_target) ** 2 * neg_mask_f).sum()

#         total_loss = (
#             (self.point_weight * point_loss)
#             + (self.coord_weight * coord_loss)
#             + (self.link_weight * link_loss)
#             + (self.class_weight * class_loss)
#             + (self.noobj_weight * noobj_loss)
#         )

#         self.last_stats = {
#             "point": point_loss.detach(),
#             "coord": coord_loss.detach(),
#             "link": link_loss.detach(),
#             "class": class_loss.detach(),
#             "noobj": noobj_loss.detach(),
#             "total": total_loss.detach(),
#         }
#         return total_loss
