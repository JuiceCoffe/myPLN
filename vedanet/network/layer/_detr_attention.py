#
#   DETR-style image self-attention layers
#

import math

import torch
from torch import nn


__all__ = ["PositionEmbeddingSine2D", "DETRImageSelfAttention"]


class PositionEmbeddingSine2D(nn.Module):
    """Fixed 2D sine/cosine positional encoding used by DETR.

    Args:
        num_pos_feats (int): Number of features per spatial axis. The returned
            encoding has ``2 * num_pos_feats`` channels.
        temperature (float): Temperature term used by the sine/cosine scaling.
        normalize (bool): Normalize the cumulative coordinates before encoding.
        scale (float, optional): Scaling factor applied when ``normalize=True``.
    """

    def __init__(self, num_pos_feats, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if x.dim() != 4:
            raise ValueError('Expected x with shape [B, C, H, W]')

        batch_size, _, height, width = x.shape
        if mask is None:
            mask = torch.zeros((batch_size, height, width), dtype=torch.bool, device=x.device)
        else:
            if mask.shape != (batch_size, height, width):
                raise ValueError('Expected mask with shape [B, H, W]')
            mask = mask.to(torch.bool)

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos.to(dtype=x.dtype)


class DETRImageSelfAttention(nn.Module):
    """DETR-style encoder layer for 2D feature maps.

    This module keeps the input and output tensor shape identical: ``[B, C, H, W]``.
    It injects fixed sine positional encoding into ``q`` and ``k`` only, and keeps
    the feed-forward network used by the original DETR encoder layer.

    Args:
        embed_dim (int): Channel dimension of the input feature map. Must be even.
        num_heads (int): Number of attention heads.
        dim_feedforward (int): Hidden width of the FFN sub-layer.
        dropout (float): Dropout used in attention and FFN residual branches.
        activation (str): Activation function used inside the FFN.
        normalize_before (bool): If True, use pre-norm; otherwise use post-norm.
    """

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError('embed_dim must be even so the 2D sine positional encoding matches the channel size')

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.normalize_before = normalize_before

        self.position_embedding = PositionEmbeddingSine2D(embed_dim // 2, normalize=True)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.ffn_dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

        self._reset_parameters()

    def _reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def _flatten(self, x, mask, pos):
        batch_size, channels, height, width = x.shape
        src = x.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)
        key_padding_mask = None if mask is None else mask.flatten(1).to(torch.bool)
        return src, pos, key_padding_mask, batch_size, channels, height, width

    def _forward_post(self, src, pos, key_padding_mask, need_weights):
        q = k = self.with_pos_embed(src, pos)
        src2, attn_weights = self.self_attn(
            q,
            k,
            value=src,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.ffn_dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

    def _forward_pre(self, src, pos, key_padding_mask, need_weights):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, attn_weights = self.self_attn(
            q,
            k,
            value=src2,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.ffn_dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, attn_weights

    def forward(self, x, mask=None, pos=None, need_weights=False):
        if x.dim() != 4:
            raise ValueError('Expected x with shape [B, C, H, W]')
        if x.shape[1] != self.embed_dim:
            raise ValueError(f'Expected input with {self.embed_dim} channels, got {x.shape[1]}')
        if mask is not None and mask.shape != (x.shape[0], x.shape[2], x.shape[3]):
            raise ValueError('Expected mask with shape [B, H, W]')

        if pos is None:
            pos = self.position_embedding(x, mask)

        src, pos, key_padding_mask, batch_size, channels, height, width = self._flatten(x, mask, pos)

        if self.normalize_before:
            src, attn_weights = self._forward_pre(src, pos, key_padding_mask, need_weights)
        else:
            src, attn_weights = self._forward_post(src, pos, key_padding_mask, need_weights)

        out = src.permute(1, 2, 0).reshape(batch_size, channels, height, width)

        if need_weights:
            return out, attn_weights
        return out


def _get_activation_fn(activation):
    if activation == "relu":
        return nn.functional.relu
    if activation == "gelu":
        return nn.functional.gelu
    if activation == "glu":
        return nn.functional.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
