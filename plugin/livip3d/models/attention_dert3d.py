import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule
from torch.nn.init import normal_


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@ATTENTION.register_module()
class Detr3DCrossAtten(BaseModule):
    """An attention module used in Detr3d. 
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(Detr3DCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams * num_levels * num_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)

        reference_points_3d, output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat


@ATTENTION.register_module()
class Detr3DCamRadarCrossAtten(BaseModule):
    """An attention module used in Detr3d. 
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 radar_dims=3,
                 radar_topk=8,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(Detr3DCamRadarCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams * num_levels * num_points)

        self.radar_dims = radar_dims

        self.attention_weights_radar = nn.Linear(embed_dims, radar_topk)
        self.radar_topk = radar_topk

        self.img_output_proj = nn.Linear(embed_dims, embed_dims)
        self.radar_output_proj = nn.Linear(self.radar_dims, self.radar_dims)

        self.img_radar_fusion = nn.Sequential(
            nn.Linear(embed_dims + radar_dims, embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
        )
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        constant_init(self.attention_weights_radar, val=0., bias=0.)
        xavier_init(self.img_output_proj, distribution='uniform', bias=0.)
        xavier_init(self.radar_output_proj, distribution='uniform', bias=0.)
        xavier_init(self.img_radar_fusion, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                ref_size=None,
                spatial_shapes=None,
                level_start_index=None,
                radar_feats=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 3),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
            ref_size (Tensor): the wlh(bbox size) associated with each query
                shape (bs, num_query, 3)
                value in log space. 
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)

        reference_points_3d, output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        # [bs, embed_dim, num_query]
        output = output.sum(-1).sum(-1).sum(-1)
        # chaneg to [num_query, bs, embed_dims]
        output = output.permute(2, 0, 1)

        output = self.img_output_proj(output)

        radar_feats, radar_mask = radar_feats[:, :, :-1], radar_feats[:, :, -1]
        
        # NOTE: baseline is limited to 100 radar points
        radar_xy = radar_feats[:, :, :2]
        # NOTE: we have 300 agents, so 300 XY reference points
        ref_xy = reference_points[:, :, :2]
        radar_feats = radar_feats[:, :, 2:]

        pad_xy = torch.ones_like(radar_xy) * 1000.0

        radar_xy = radar_xy + (1.0 - radar_mask.unsqueeze(dim=-1).type(torch.float)) * (pad_xy)

        # [B, num_query, M]
        ref_radar_dist = -1.0 * torch.cdist(ref_xy, radar_xy)

        # [B, num_query, topk]
        _value, indices = torch.topk(ref_radar_dist, self.radar_topk)

        # [B, num_query, M]
        radar_mask = radar_mask.unsqueeze(dim=1).repeat(1, num_query, 1)

        # [B, num_query, topk]
        top_mask = torch.gather(radar_mask, 2, indices)

        # [B, num_query, M, radar_dim]
        radar_feats = radar_feats.unsqueeze(dim=1).repeat(1, num_query, 1, 1)
        radar_dim = radar_feats.size(-1)
        # [B, num_query, topk, radar_dim]
        indices_pad = indices.unsqueeze(dim=-1).repeat(1, 1, 1, radar_dim)

        # [B, num_query, topk, radar_dim]
        radar_feats_topk = torch.gather(
            radar_feats, dim=2, index=indices_pad, sparse_grad=False)

        attention_weights_radar = self.attention_weights_radar(query).view(
            bs, num_query, self.radar_topk)

        # [B, num_query, topk]
        attention_weights_radar = attention_weights_radar.sigmoid() * top_mask
        # [B, num_query, topk, radar_dim]
        radar_out = radar_feats_topk * attention_weights_radar.unsqueeze(dim=-1)
        # [bs, num_query, radar_dim]
        radar_out = radar_out.sum(dim=2)

        # change to (num_query, bs, embed_dims)
        radar_out = radar_out.permute(1, 0, 2)

        radar_out = self.radar_output_proj(radar_out)

        output = torch.cat((output, radar_out), dim=-1)
        output = self.img_radar_fusion(output)

        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(
            inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat


def masked_softmax(vec: torch.Tensor, mask: torch.Tensor, dim: int = -1):
    # mask expected as 0/1 or bool, broadcastable to vec
    mask_bool = mask.bool()
    vec = vec.masked_fill(~mask_bool, float('-inf'))
    out = F.softmax(vec, dim=dim)
    # replace NaNs (from all -inf) with zeros
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


@ATTENTION.register_module()
class Detr3DCamLidarCrossAttenQGDF(BaseModule):
    """
    Drop-in replacement for Detr3DCamLidarCrossAtten with
    - Corrected shapes and variable names
    - Masked-softmax weights over cam/level/points (vs sigmoid)
    - Differentiable LiDAR sampling via grid_sample with learned offsets (no hard top-k)
    - Modality gating for fusion
    - Optional ablation to disable learnable LiDAR BEV offsets
    """
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=1,
                 num_cams=6,
                 lidar_dims=128,
                 lidar_points=5,
                 pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                 dropout=0.1,
                 batch_first=False,
                 gating_ablation=False,
                 lidar_offset_ablation=False,
                 attached_query=False):
        super().__init__()
        assert embed_dims % num_heads == 0
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_cams = num_cams
        self.lidar_dims = lidar_dims
        self.lidar_points = lidar_points
        self.pc_range = pc_range
        self.batch_first = batch_first
        self.gating_ablation = gating_ablation
        self.lidar_offset_ablation = lidar_offset_ablation
        self.attached_query = attached_query

        self.offset_ablation_print_flag = False
        self.step_counter = 0
        # Image attention (single set of weights for simplicity)
        self.img_attn_weights = nn.Linear(embed_dims, num_cams * num_levels * num_points)

        # LiDAR offsets and weights (per query)
        # Keep the layer defined for checkpoint/state_dict compatibility,
        # but optionally ignore it during forward when ablation is enabled.
        # Projections
        self.img_out_proj = nn.Linear(embed_dims, embed_dims)
        self.img_in_norm = nn.LayerNorm(embed_dims)

        self.lidar_in_proj = nn.Conv2d(lidar_dims, embed_dims, kernel_size=1, bias=False)
        self.lidar_in_norm2d = nn.GroupNorm(32, embed_dims)
        self.lidar_in_norm = nn.LayerNorm(embed_dims)
        assert embed_dims % num_heads == 0
        self.head_dim = embed_dims // num_heads
        self.lidar_offsets = nn.Linear(
            embed_dims, num_heads * (lidar_points - 1) * 2
        )
        self.lidar_attn_weights = nn.Sequential(
            nn.Linear(2 * self.head_dim + 2, self.head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.head_dim, 1)
        )
        # Explicit center-vs-context fusion
        self.lidar_center_ctx_gate = nn.Sequential(
            nn.Linear(2 * embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 2)
        )

        self.lidar_out_proj = nn.Linear(embed_dims, embed_dims)

        # Keep a small initial preference for the center branch
        self.center_logit_bias = nn.Parameter(torch.tensor(0.25))

        with torch.no_grad():
            nn.init.constant_(self.lidar_offsets.weight, 0.0)
            if self.lidar_offsets.bias is not None:
                Hh = self.num_heads
                Pm1 = max(self.lidar_points - 1, 0)

                if Pm1 > 0:
                    thetas = torch.arange(Hh, dtype=torch.float32) * (2.0 * math.pi / Hh)
                    dirs = torch.stack([thetas.cos(), thetas.sin()], dim=-1)   # (H,2)
                    dirs = dirs / dirs.abs().max(dim=-1, keepdim=True)[0]      

                    # Build a small radial pattern in BEV-cell units.
                    # radius 1,2,3,... across points
                    grid_init = dirs[:, None, :].repeat(1, Pm1, 1)             # (H,P-1,2)
                    for i in range(Pm1):
                        grid_init[:, i, :] *= (i + 1)

                    self.lidar_offsets.bias.copy_(grid_init.reshape(-1))

        self.position_encoder = nn.Sequential(
            nn.Linear(3, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )

        self.fuse_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dims * 2),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        if not self.gating_ablation:
            # Fusion gate with normalization for stability
            self.fuse_gate = nn.Sequential(
                nn.LayerNorm(embed_dims * 3),
                nn.Linear(embed_dims * 3, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, 2)  # logits for [img, lidar]
            )

            with torch.no_grad():
                if hasattr(self.fuse_gate[-1], 'bias') and self.fuse_gate[-1].bias is not None:
                    self.fuse_gate[-1].bias.data = torch.tensor(
                        [0.0, 0.2],
                        dtype=self.fuse_gate[-1].bias.dtype,
                        device=self.fuse_gate[-1].bias.device
                    )

        # Temperature for softmax attention (improves gradient flow)
        self.attn_temperature = 4.0
        self.register_buffer('offset_attn_temperature', torch.tensor(1.5))
        self.gate_min = 0.1
        # Scale factor for offset learning (start with small search radius)
        self.offset_scale = 1.5

        self.dropout = nn.Dropout(dropout)

        # Initialize offset weights small for stability
        nn.init.xavier_uniform_(self.lidar_offsets.weight, gain=0.01)
        if self.lidar_offsets.bias is not None:
            nn.init.zeros_(self.lidar_offsets.bias)

    def _xy_to_bev_grid(self, ref_xy: torch.Tensor, H: int, W: int):
        """
        Convert normalized [0,1] xy (in pc_range) to grid_sample coords [-1,1] with (x->W, y->H).
        ref_xy: (B, Q, 2) normalized in [0,1] w.r.t pc_range (as in DETR3D)
        """
        grid_x = ref_xy[..., 0] * 2 - 1  # width
        grid_y = ref_xy[..., 1] * 2 - 1  # height
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (B,Q,2)
        return grid

    def bev_sample(self, pts_feats: torch.Tensor, ref_xy: torch.Tensor, query: torch.Tensor):
        """
        Efficient multi-head BEV sampling with:
        - fixed center point per head
        - learned offset points per head
        - offsets predicted in BEV-cell units
        - separate center-vs-offset-context fusion

        Args:
            pts_feats: (B, C_lidar, H_bev, W_bev)
            ref_xy:    (B, Q, 2), normalized to [0,1]
            query:     (Q, B, E) or (B, Q, E)

        Returns:
            out:       (Q, B, E)
        """

        B, _, H_bev, W_bev = pts_feats.shape
        B2, Q, _ = ref_xy.shape
        assert B == B2, f"Batch mismatch: pts_feats B={B}, ref_xy B={B2}"

        # ------------------------------------------------------------
        # 1) Robust query layout -> (B, Q, E)
        # ------------------------------------------------------------
        if query.dim() != 3:
            raise ValueError(f"query must be 3D, got {tuple(query.shape)}")

        if query.shape[0] == Q and query.shape[1] == B:
            query = query.permute(1, 0, 2).contiguous()   # (B,Q,E)
        elif query.shape[0] == B and query.shape[1] == Q:
            query = query.contiguous()
        else:
            raise ValueError(
                f"Unexpected query shape {tuple(query.shape)}; "
                f"expected (Q,B,E)=({Q},{B},E) or (B,Q,E)=({B},{Q},E)"
            )

        E = self.embed_dims
        Hh = self.num_heads
        D = self.head_dim
        P = self.lidar_points

        assert E == Hh * D, f"embed_dims ({E}) must equal num_heads * head_dim ({Hh} * {D})"
        if P < 1:
            raise ValueError(f"self.lidar_points must be >= 1, got {P}")

        # ------------------------------------------------------------
        # 2) Project LiDAR BEV features and split into heads BEFORE sampling
        #    bev_embed_heads: (B*Hh, D, H_bev, W_bev)
        # ------------------------------------------------------------
        bev_embed = self.lidar_in_proj(pts_feats)                 # (B,E,H,W)
        bev_embed = self.lidar_in_norm2d(bev_embed)               # (B,E,H,W)
        bev_embed_heads = bev_embed.view(B, Hh, D, H_bev, W_bev)
        bev_embed_heads = bev_embed_heads.reshape(B * Hh, D, H_bev, W_bev)

        # query split into heads
        q_feat = query.view(B, Q, Hh, D)                         # (B,Q,H,D)

        # ------------------------------------------------------------
        # 3) Build per-head sampling grid
        # ------------------------------------------------------------
        center_grid = self._xy_to_bev_grid(ref_xy, H_bev, W_bev) # (B,Q,2)
        center_grid = center_grid.unsqueeze(2).expand(-1, -1, Hh, -1)   # (B,Q,H,2)

        center_point = center_grid.unsqueeze(3)                  # (B,Q,H,1,2)

        if P == 1:
            full_grid = center_point                             # (B,Q,H,1,2)
            offsets_grid = None

            if not self.offset_ablation_print_flag:
                print("[INFO] using center-only multi-head BEV sampling")
                self.offset_ablation_print_flag = True
        else:
            Pm1 = P - 1

            if self.lidar_offset_ablation:
                offsets_grid = torch.zeros(
                    B, Q, Hh, Pm1, 2,
                    device=pts_feats.device,
                    dtype=pts_feats.dtype
                )
                offset_points = center_grid.unsqueeze(3).expand(-1, -1, -1, Pm1, -1)

                if not self.offset_ablation_print_flag:
                    print("[INFO] using non-offset multi-head BEV sampling (ablation)")
                    self.offset_ablation_print_flag = True
            else:
                if not self.offset_ablation_print_flag:
                    print("[INFO] using multi-head offset BEV sampling")
                    self.offset_ablation_print_flag = True

                # Predict per-head offsets in CELL UNITS
                offsets = self.lidar_offsets(query)                          # (B,Q,H*(P-1)*2)
                offsets = offsets.view(B, Q, Hh, Pm1, 2)                    # (B,Q,H,P-1,2)
                offsets = offsets.tanh()

                # self.offset_scale is interpreted as max displacement in BEV cells
                offsets_cells = self.offset_scale * offsets                 # (B,Q,H,P-1,2)

                # Convert cell offsets to grid_sample coords
                # x: 1 cell -> 2/W, y: 1 cell -> 2/H
                cell_to_grid = offsets_cells.new_tensor(
                    [2.0 / W_bev, 2.0 / H_bev]
                ).view(1, 1, 1, 1, 2)

                offsets_grid = offsets_cells * cell_to_grid                 # (B,Q,H,P-1,2)
                offset_points = center_grid.unsqueeze(3) + offsets_grid     # (B,Q,H,P-1,2)

                eps = 1e-4
                offset_points = offset_points.clamp(-1.0 + eps, 1.0 - eps)

            full_grid = torch.cat([center_point, offset_points], dim=3)     # (B,Q,H,P,2)

        # ------------------------------------------------------------
        # 4) Sample per-head directly
        #
        # full_grid:       (B,Q,H,P,2)
        # -> reorder to    (B,H,Q,P,2)
        # -> flatten to    (B*H, Q*P, 1, 2)
        # ------------------------------------------------------------
        grid_heads = full_grid.permute(0, 2, 1, 3, 4).contiguous()         # (B,H,Q,P,2)
        grid_for_sample = grid_heads.view(B * Hh, Q * P, 1, 2)             # (B*H, QP, 1, 2)

        sampled = F.grid_sample(
            bev_embed_heads,
            grid_for_sample,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        # sampled: (B*H, D, QP, 1)
        sampled = sampled.squeeze(-1).transpose(1, 2).contiguous()         # (B*H, QP, D)
        sampled = sampled.view(B, Hh, Q, P, D)                             # (B,H,Q,P,D)
        sampled = sampled.permute(0, 2, 1, 3, 4).contiguous()              # (B,Q,H,P,D)

        # ------------------------------------------------------------
        # 5) Split center path and offset-context path
        # ------------------------------------------------------------
        center_feat = sampled[:, :, :, 0, :]                               # (B,Q,H,D)
        center_flat = center_feat.reshape(B, Q, E)                         # (B,Q,E)

        if P == 1:
            out = self.lidar_out_proj(center_flat)
            return out.permute(1, 0, 2).contiguous()

        offset_feat = sampled[:, :, :, 1:, :]                              # (B,Q,H,P-1,D)

        # ------------------------------------------------------------
        # 6) Score offset points per head using
        #    [offset_feature || query_head || relative_offset]
        # ------------------------------------------------------------
        q_expanded = q_feat.unsqueeze(3).expand(-1, -1, -1, P - 1, -1)     # (B,Q,H,P-1,D)
        rel_offsets = offsets_grid                                          # (B,Q,H,P-1,2)

        weight_input = torch.cat([offset_feat, q_expanded, rel_offsets], dim=-1)
        offset_logits = self.lidar_attn_weights(weight_input).squeeze(-1)   # (B,Q,H,P-1)

        offset_w = F.softmax(
            offset_logits / self.offset_attn_temperature,
            dim=-1
        ).unsqueeze(-1)                                                     # (B,Q,H,P-1,1)

        offset_ctx = (offset_feat * offset_w).sum(dim=3)                    # (B,Q,H,D)
        ctx_flat = offset_ctx.reshape(B, Q, E)                              # (B,Q,E)

        # ------------------------------------------------------------
        # 7) Fuse fixed center and offset-context explicitly
        # ------------------------------------------------------------
        gate_in = torch.cat([center_flat, ctx_flat], dim=-1)                # (B,Q,2E)
        center_ctx_logits = self.lidar_center_ctx_gate(gate_in)             # (B,Q,2)

        if self.center_logit_bias is not None:
            center_ctx_logits = torch.cat(
                [
                    center_ctx_logits[..., :1] + self.center_logit_bias,
                    center_ctx_logits[..., 1:]
                ],
                dim=-1
            )

        center_ctx_w = F.softmax(
            center_ctx_logits / self.offset_attn_temperature,
            dim=-1
        )                                                                   # (B,Q,2)

        center_w = center_ctx_w[..., 0:1]
        ctx_w = center_ctx_w[..., 1:2]

        fused = center_w * center_flat + ctx_w * ctx_flat                   # (B,Q,E)

        # ------------------------------------------------------------
        # 8) Output projection
        # ------------------------------------------------------------
        out = self.lidar_out_proj(fused)                                    # (B,Q,E)

        return out.permute(1, 0, 2).contiguous()

    def forward(self,
                query,  # (Q, B, E)
                key=None,
                value=None,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,  # (B,Q,3) normalized in [0,1]
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """MMCV attention forward-compatible signature.
        Expected external call passes: value=mlvl_img_feats, reference_points, and img_metas via kwargs.
        Optionally, LiDAR BEV features via kwargs['pts_feats'].
        """
        B = query.size(1)
        Q = query.size(0)
        E = query.size(2)

        # retrieve inputs following mmcv's calling convention
        mlvl_img_feats = value  # list[T]: each (B, N_cam, C, H, W)
        img_metas = kwargs.get('img_metas')
        pts_feats = kwargs.get('pts_feats', None)  # (B, C_lidar, H, W)

        if query_pos is not None:
            query = query + query_pos

        # Image branch: sample using provided function (outside)
        assert self.num_points == 1, "feature_sampling_v2 returns P=1; set num_points=1 or extend the sampler."

        reference_points_3d, sampled_feats, mask = feature_sampling_v2(
            mlvl_img_feats, reference_points, self.pc_range, img_metas
        )

        sampled_feats = torch.nan_to_num(sampled_feats)
        mask = torch.nan_to_num(mask)
        sampled_feats = sampled_feats * mask
        sampled_feats = torch.nan_to_num(sampled_feats)
        mask = torch.nan_to_num(mask)

        # weights over (cam, point, level)
        w_img = self.img_attn_weights(query.permute(1, 0, 2))           # (B,Q,N_cam*L)
        w_img = w_img.view(B, Q, self.num_cams, self.num_levels, 1)     # (B,Q,N_cam,L,1)
        w_img = w_img.permute(0, 1, 2, 4, 3).unsqueeze(1)               # (B,1,Q,N_cam,1,L)

        mask_full = mask.expand(-1, -1, -1, -1, 1, self.num_levels)     # (B,1,Q,N_cam,1,L)

        w_flat = w_img.reshape(B, 1, Q, -1)
        m_flat = mask_full.reshape(B, 1, Q, -1)
        w_flat = masked_softmax(w_flat, m_flat, dim=-1)
        w_img = w_flat.view_as(mask_full)

        img_agg = (sampled_feats * w_img).sum(dim=(3, 4, 5))            # (B,E,Q)
        img_agg = torch.nan_to_num(img_agg)
        img_agg = self.img_out_proj(img_agg.permute(2, 0, 1).contiguous())  # (Q,B,E)
        img_aligned = self.img_in_norm(img_agg)

        # LiDAR branch
        if pts_feats is not None:
            lidar_agg = self.bev_sample(pts_feats, reference_points[..., :2], query)  # (Q,B,E)
        else:
            lidar_agg = img_agg.new_zeros(img_agg.shape)

        lidar_agg = torch.nan_to_num(lidar_agg)
        lidar_aligned = self.lidar_in_norm(lidar_agg)

        if not self.attached_query:
            q_for_gate = query.detach()
        else:
            q_for_gate = query

        if not self.gating_ablation:
            gate_logits = self.fuse_gate(torch.cat([img_aligned, lidar_aligned, q_for_gate], dim=-1))  # (Q,B,2)
            gate_logits = gate_logits.clamp(min=-10.0, max=10.0)
            gate = F.softmax(gate_logits / self.attn_temperature, dim=-1)  # (Q,B,2)
            if self.gate_min > 0:
                gate = gate * (1 - 2 * self.gate_min) + self.gate_min

            f_img = gate[..., 0:1] * img_aligned
            f_lidar = gate[..., 1:2] * lidar_aligned
            fused = torch.cat([f_img, f_lidar], dim=-1)

            self.step_counter += 1
            if self.step_counter == 28000:
                print("Setting attn_temperature and gate_min to 2.0 and 0.0")
                self.attn_temperature = 1.0
                self.gate_min = 0.0
        else:
            fused = torch.cat([img_aligned, lidar_aligned], dim=-1)

        out = self.fuse_proj(fused)  # (Q,B,E)
        out = torch.nan_to_num(out)

        # positional code
        pos_feat = inverse_sigmoid(reference_points).permute(1, 0, 2)  # (Q,B,3)
        pos_feat = self.position_encoder(pos_feat)

        if torch.isnan(out).any():
            print("Found NaN in out")

        if torch.isnan(q_for_gate).any():
            print("Found NaN in q_for_gate")

        if torch.isnan(pos_feat).any():
            print("Found NaN in pos_feat")

        return self.dropout(out) + query + pos_feat


@ATTENTION.register_module()
class Detr3DCamLidarCrossAtten(BaseModule):
    """An attention module used in Detr3d. 
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 lidar_dims=32,
                 lidar_topk=400,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(Detr3DCamLidarCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams * num_levels * num_points)

        self.lidar_dims = lidar_dims

        self.attention_weights_lidar = nn.Linear(embed_dims, lidar_topk)
        self.lidar_topk = lidar_topk

        self.img_output_proj = nn.Linear(embed_dims, embed_dims)
        self.lidar_output_proj = nn.Linear(self.lidar_dims, self.lidar_dims)

        self.img_lidar_fusion = nn.Sequential(
            nn.Linear(embed_dims + lidar_dims, embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
        )
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first
        # Remove hardcoded coord_map - will be computed dynamically
        self.coord_map_cache = {}  # Cache coordinate maps by size


        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        constant_init(self.attention_weights_lidar, val=0., bias=0.)
        xavier_init(self.img_output_proj, distribution='uniform', bias=0.)
        xavier_init(self.lidar_output_proj, distribution='uniform', bias=0.)
        xavier_init(self.img_lidar_fusion, distribution='uniform', bias=0.)

    def get_lidar_coords(self, H, W, x_min, x_max, y_min, y_max):
        """
        Create a (H,W,2) tensor of the center coordinates for each grid cell.
        Then flatten to shape (H*W, 2).
        """
        # total range is 102.4 in each dimension
        cell_size_x = (x_max - x_min) / H  # = 0.2
        cell_size_y = (y_max - y_min) / W  # = 0.2

        # coordinate linspaces for center of each cell
        x_lin = torch.linspace(x_min, x_max, steps=H) + 0.5 * cell_size_x
        y_lin = torch.linspace(y_min, y_max, steps=W) + 0.5 * cell_size_y

        # create meshgrid => shape (H, W)
        yy, xx = torch.meshgrid(y_lin, x_lin, indexing='xy')
        # now xx, yy each is (W, H) if indexing='xy'
        # reorder or re-mesh as needed; just be consistent in flattening
        # We'll reorder to shape (H, W, 2):
        # be mindful about row/column vs x/y orientation
        coords = torch.stack((xx.t(), yy.t()), dim=-1)  # (H, W, 2)

        # flatten => (H*W, 2)
        coords_flat = coords.reshape(H*W, 2)
        return coords_flat  # (M, 2)


    def flatten_lidar_feat(self, lidar_feat):
        """
        lidar_feat: (B, H, W, C)
        Returns: 
        lidar_feats_flat: (B, M, C)
        """
        B, H, W, C = lidar_feat.shape
        M = H * W
        # Permute to (B, C, H, W) => (B, H, W, C) => flatten
        # If already (B, H, W, C), just reshape directly:
        lidar_feats_flat = lidar_feat.view(B, M, C)
        return lidar_feats_flat

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                ref_size=None,
                spatial_shapes=None,
                level_start_index=None,
                pts_feats=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 3),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
            ref_size (Tensor): the wlh(bbox size) associated with each query
                shape (bs, num_query, 3)
                value in log space. 
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)

        reference_points_3d, output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        # [bs, embed_dim, num_query]
        output = output.sum(-1).sum(-1).sum(-1)
        # chaneg to [num_query, bs, embed_dims]
        output = output.permute(2, 0, 1)

        output = self.img_output_proj(output)

        # Get actual LiDAR feature dimensions
        B_pts, C_pts, H_pts, W_pts = pts_feats.shape
        
        # Compute or retrieve cached coordinate map for this size
        cache_key = (H_pts, W_pts)
        if cache_key not in self.coord_map_cache:
            self.coord_map_cache[cache_key] = self.get_lidar_coords(
                H_pts, W_pts, -51.2, 51.2, -51.2, 51.2
            )
        
        coords = self.coord_map_cache[cache_key].to(pts_feats.device)
        coords = coords.unsqueeze(0).expand(bs, -1, -1)
        
        ref_xy = reference_points[:, :, :2]


        # [B, num_query, M]
        ref_lidar_dist = -1.0 * torch.cdist(ref_xy, coords)

        # Ensure k doesn't exceed available features
        M = coords.shape[1]  # Number of LiDAR grid cells
        k = min(self.lidar_topk, M)
        
        dist_vals, indices = torch.topk(ref_lidar_dist, k=k, dim=2)

        # TODO: investigate the viability of filtering out empty pillars to reduce computational load / improve relevancy
        pts_feats = pts_feats.permute(0, 2, 3, 1)
        # print(f"pts_feats {pts_feats.shape}")
        pts_feats = self.flatten_lidar_feat(pts_feats)
        # print(f"pts_feats {pts_feats.shape}")
#        c_lidar = pts_feats.size(-1)
#        print(pts_feats.shape)
#        lidar_feats = lidar_feats.unsqueeze(1).repeat(1, num_query, 1, 1)

        
        
        # [B, num_query, topk, c_lidar]
#        indices_pad = indices.unsqueeze(dim=-1).repeat(1, 1, 1, c_lidar)

        # [B, num_query, topk, c_lidar]
#        lidar_feats_topk = torch.gather(
#            lidar_feats, dim=2, index=indices_pad, sparse_grad=False)
 

#        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, c_lidar)
#        print(f"indices_expanded shape: {indices_expanded.shape}")
#        lidar_feats_topk = torch.gather(pts_feats, dim=1, index=indices_expanded)
#        print(f"topk shape: {lidar_feats_topk.shape}")

        B, M, C = pts_feats.shape
        _, Q, K = indices.shape

        # Make batch_idx so we pick from each batch. shape: [B, 1, 1] => [B, Q, K]
        batch_idx = torch.arange(B, device=pts_feats.device).view(B, 1, 1)
        batch_idx = batch_idx.expand(-1, Q, K)
        # print(f"batch_idx shape: {batch_idx.shape}")
        # print(f"indices shape: {indices.shape}")

        # Now we advanced-index:
        #   pts_feats[batch_idx, indices, :] => [B, Q, K, C]
        lidar_feats_topk = pts_feats[batch_idx, indices, :]
        # print(f"lidar_feats_topk shape: {lidar_feats_topk.shape}")

        attention_weights_lidar = self.attention_weights_lidar(query).view(
            bs, num_query, self.lidar_topk)
        
        # Slice to actual k if needed
        attention_weights_lidar = attention_weights_lidar[:, :, :k]

        # [B, num_query, topk]
        attention_weights_lidar = attention_weights_lidar.sigmoid()
        # [B, num_query, topk, radar_dim]
        lidar_out = lidar_feats_topk * attention_weights_lidar.unsqueeze(dim=-1)
        # print(lidar_out.shape)
        # [bs, num_query, radar_dim]
        lidar_out = lidar_out.sum(dim=2)

        # change to (num_query, bs, embed_dims)
        # print(lidar_out.shape)
        lidar_out = lidar_out.permute(1, 0, 2)

        lidar_out = self.lidar_output_proj(lidar_out)

        output = torch.cat((output, lidar_out), dim=-1)
        output = self.img_lidar_fusion(output)

        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(
            inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat


def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    # reference_points (B, num_queries, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
    # img_shape is [[(H, W, 3), ...]] (nested), so [0][0] gives first camera's (H, W, 3)
    # [0][0][1] is W (width), [0][0][0] is H (height)
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][0][1]  # divide x by width
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0][0]  # divide y by height
    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 1:2] > -1.0)
            & (reference_points_cam[..., 1:2] < 1.0))
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B * N, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B * N, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam, 1, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask


def feature_sampling_v2(mlvl_feats, reference_points, pc_range, img_metas):
    """
    Args:
        mlvl_feats: list of [B, N_cam, C, H, W] image features (L levels)
        reference_points: (B,Q,3) in [0,1] normalized to pc_range
        pc_range: [6]
        img_metas: list of dicts with key 'lidar2img' and 'img_shape'
    Returns:
        reference_points_3d: (B,Q,3) absolute xyz in lidar
        sampled_feats: (B, C, Q, N_cam, P=1, L)  -- we sample exactly the reference point per level/cam
        mask: (B, 1, Q, N_cam, 1, 1)
    """
    B, Q, _ = reference_points.shape
    num_cams = mlvl_feats[0].size(1)
    # to absolute 3D points
    ref = reference_points.clone()
    ref[..., 0] = ref[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    ref[..., 1] = ref[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    ref[..., 2] = ref[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    reference_points_3d = ref

    # append ones for homogeneous
    ref_homo = torch.cat([ref, torch.ones_like(ref[..., :1])], dim=-1)  # (B,Q,4)

    # collect lidar2img
    lidar2img = []
    img_hw = []
    for i in range(B):
        lidar2img.append(torch.as_tensor(np.asarray(img_metas[i]['lidar2img']), device=ref.device, dtype=ref.dtype))  # (N,4,4)
        # assume list of (H,W,3) or [[H, W]] per cam
        per_cam_shapes = img_metas[i]['img_shape'][0] if isinstance(img_metas[i]['img_shape'], list) else img_metas[i]['img_shape']
        # fallback to simple (H,W)
        per_cam = []
        for s in per_cam_shapes:
            if isinstance(s, (list, tuple)):
                H, W = s[0], s[1]
            else:
                H, W = s, s
            per_cam.append((H,W))
        img_hw.append(per_cam)
    lidar2img = torch.stack(lidar2img, dim=0)  # (B,N,4,4)

    # expand
    ref_h = ref_homo.view(B, 1, Q, 4, 1).repeat(1, num_cams, 1, 1, 1)    # (B,N,Q,4,1)
    l2i = lidar2img.view(B, num_cams, 1, 4, 4).repeat(1, 1, Q, 1, 1)    # (B,N,Q,4,4)
    cam_pts = torch.matmul(l2i, ref_h).squeeze(-1)                       # (B,N,Q,4)
    eps = 1e-5
    z = cam_pts[..., 2:3]
    mask = (z > eps)

    # project
    xy = cam_pts[..., :2] / torch.clamp(z, min=eps)  # (B,N,Q,2)
    # normalize to [-1,1] per-cam
    grids = []
    for b in range(B):
        per_cam = []
        for n in range(num_cams):
            H, W = img_hw[b][n]
            gx = (xy[b, n, :, 0] / W) * 2 - 1
            gy = (xy[b, n, :, 1] / H) * 2 - 1
            per_cam.append(torch.stack([gx, gy], dim=-1))  # (Q,2)
        grids.append(torch.stack(per_cam, dim=0))
    grids = torch.stack(grids, dim=0)  # (B,N,Q,2)

    # sample features on each level
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        Bf, N, C, H, W = feat.shape
        feat = feat.view(Bf*N, C, H, W)
        grid = grids.view(Bf*N, Q, 1, 2)
        sampled = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=False)  # (B*N, C, Q, 1)
        sampled = sampled.view(Bf, N, C, Q, 1).permute(0, 2, 3, 1, 4)  # (B, C, Q, N, 1)
        sampled_feats.append(sampled)
    sampled_feats = torch.stack(sampled_feats, dim=-1)  # (B, C, Q, N, 1, L)
    
    in_bounds = (grids[...,0] > -1) & (grids[...,0] < 1) & (grids[...,1] > -1) & (grids[...,1] < 1)
    mask = (mask.squeeze(-1).squeeze(-1) & in_bounds)  # (B,N_cam,Q)
    mask = mask.view(B, num_cams, Q, 1, 1).permute(0, 3, 2, 1, 4).unsqueeze(-1)
    mask = mask.to(dtype=sampled_feats.dtype)
    
    return reference_points_3d, sampled_feats, mask


@ATTENTION.register_module()
class Detr3DCrossAttenPetrFeature(BaseModule):
    """An attention module used in Detr3d.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=1,
                 num_points=5,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(Detr3DCrossAttenPetrFeature, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout)

        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        # constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        inp_residual = query

        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        if True:
            value = value[0]

            query = query.transpose(0, 1)
            value = value.transpose(0, 1)
            output = self.attn(query=query, key=value, value=value)[0]

            reference_points_3d = reference_points.clone()
            # petr_feature = value[0]
            # assert len(petr_feature.shape) == 2
            # petr_feature = petr_feature.view(-1, 6, 256) # num_query, num_cam, C
            # # petr_feature = petr_feature[:50, :, :]
            # petr_feature = petr_feature.permute(2, 0, 1).contiguous() # C, num_query, num_cam
            # s = petr_feature.shape
            # petr_feature = petr_feature.view(1, s[0], s[1], s[2], 1, 1)
            # output = petr_feature

        output = self.output_proj(output)  # (num_query, bs, embed_dims)

        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat
