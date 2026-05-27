
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, List, Optional, Union
from mmcv.runner import force_fp32
from mmcv.cnn import build_norm_layer

try:
    # mmdet3d hard voxelization
    from mmdet3d.ops import Voxelization as HardVoxelization
    from mmcv.utils import Registry
except Exception:
    HardVoxelization = None


__all__ = [
    "PillarFeatureNet",
    "PointPillarsScatter",
    "PPBackbone",
    "PPNeck",
    "LidarEncoder",
]

LIDAR_ENCODERS = Registry('lidar_encoder')
def build_lidar_encoder(cfg: dict) -> nn.Module:
    return LIDAR_ENCODERS.build(cfg)

def get_paddings_indicator(actual_num: torch.Tensor, max_num: int, axis: int = 0) -> torch.Tensor:
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape: list[int] = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num_tensor = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num_tensor
    return paddings_indicator


class PFNLayer(nn.Module):
    """
    PointPillars PFN layer (SECOND-style).
    If not last layer, halves out_channels and concatenates with max feature.
    """

    def __init__(self, in_channels: int, out_channels: int, norm_cfg: Optional[dict] = None, last_layer: bool = False) -> None:
        super().__init__()
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg

        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = build_norm_layer(self.norm_cfg, self.units)[1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (P, T, C_in)
        dtype = inputs.dtype
        
        # Linear in mixed precision is fine
        x = self.linear(inputs)
        
        # BN1d MUST be in FP32 for numerical stability with dynamic loss scaling
        # Convert to fp32, do BN, convert back
        x = x.float()  # Always FP32 for BN
        x_transposed = x.permute(0, 2, 1).contiguous()
        
        # Use torch.backends.cudnn.flags() context manager (thread-safe)
        with torch.backends.cudnn.flags(enabled=False):
            x_normed = self.norm(x_transposed)
        
        x = x_normed.permute(0, 2, 1).contiguous()
        
        # Convert back to original dtype before ReLU
        x = x.to(dtype)
        x = F.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]  # (P, 1, C)
        if self.last_vfe:
            return x_max  # (P, 1, C)
        x_repeat = x_max.repeat(1, inputs.shape[1], 1)
        x_concatenated = torch.cat([x, x_repeat], dim=2)
        return x_concatenated


@LIDAR_ENCODERS.register_module()
class PillarFeatureNet(nn.Module):
    """
    Implements the feature decorations of PointPillars:
      - raw point features (x,y,z[,i])
      - cluster offset (point - mean of pillar)
      - center offset (point - pillar center) in x,y
      - optional distance
    Then feeds through PFN layers.
    """

    def __init__(
        self,
        in_channels: int = 5,
        feat_channels: Tuple[int, ...] = (64,),
        with_distance: bool = False,
        voxel_size: Tuple[float, float, float] = (0.2, 0.2, 8.0),
        point_cloud_range: Tuple[float, float, float, float, float, float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        norm_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.with_distance = with_distance
        self.vx, self.vy, self.vz = voxel_size
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]

        # PFN input feature size: raw + f_cluster(3) + f_center(2) + optional distance(1)
        pfn_in_channels = in_channels + 3 + 2 + (1 if with_distance else 0)

        pfn_layers: list[nn.Module] = []
        in_filters = pfn_in_channels
        for i, out_filters in enumerate(feat_channels):
            last = (i == len(feat_channels) - 1)
            pfn_layers.append(PFNLayer(in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last))
            if not last:
                in_filters = out_filters
        self.pfn_layers = nn.ModuleList(pfn_layers)

    @staticmethod
    def _with_distance(pts: torch.Tensor) -> torch.Tensor:
        # pts: (..., 3 or 4) -> append distance
        return torch.norm(pts[..., :3], p=2, dim=-1, keepdim=True)

    def forward(self, features: torch.Tensor, num_voxels: torch.Tensor, coors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (P, T, C_in)
            num_voxels: (P,)
            coors: (P, 4)  [batch_idx, z, y, x]
        Returns:
            (P, C_out)
        """
        dtype = features.dtype

        # f_cluster: point - pillar mean
        # CRITICAL: Do mean computation in fp32 to avoid inf/nan with dynamic loss scaling
        features_fp32 = features.float()
        points_mean = features_fp32[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.float().view(-1, 1, 1).clamp(min=1e-6)
        f_cluster = features_fp32[:, :, :3] - points_mean
        
        # Convert back to original dtype
        f_cluster = f_cluster.to(dtype)
        points_mean = points_mean.to(dtype)

        # f_center: distance to pillar center in x,y
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)

        features_ls = [features, f_cluster, f_center]
        if self.with_distance:
            points_dist = torch.norm(features[:, :, :3], p=2, dim=2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # zero-out padded pillars
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features = features * mask

        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze(1)


@LIDAR_ENCODERS.register_module()
class PointPillarsScatter(nn.Module):
    """
    Convert sparse pillar features to a dense BEV pseudo-image.
    coors are expected to be [batch, z, y, x].
    output_shape: (ny, nx) == (H, W).
    """

    def __init__(self, in_channels: int, output_shape: Tuple[int, int]):
        super().__init__()
        self.in_channels = in_channels
        self.ny, self.nx = output_shape

    def forward(self, pillar_features: torch.Tensor, coors: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Args:
            pillar_features: (P, C)
            coors: (P, 4) [batch, z, y, x]
            batch_size: int
        Returns:
            canvas: (B, C, ny, nx)
        """
        device = pillar_features.device
        dtype = pillar_features.dtype
        B = batch_size
        C = self.in_channels
        canvas = pillar_features.new_zeros((B, C, self.ny * self.nx))
        coors = coors.long()

        # Transpose once for scatter
        pillars_T = pillar_features.t()  # (C, P)

        for b in range(B):
            mask_b = (coors[:, 0] == b)
            if not mask_b.any():
                continue
            this_coors = coors[mask_b]
            pillar_indices = torch.nonzero(mask_b, as_tuple=False).squeeze(1)

            # Guard against any out-of-bounds coordinates
            valid = (this_coors[:, 2] >= 0) & (this_coors[:, 2] < self.ny) & \
                    (this_coors[:, 3] >= 0) & (this_coors[:, 3] < self.nx)
            if not valid.any():
                continue

            this_coors = this_coors[valid]
            pillar_indices = pillar_indices[valid]

            # indices = y * nx + x   (ignore z)
            indices = (this_coors[:, 2] * self.nx + this_coors[:, 3]).long()
            canvas[b, :, indices] = pillars_T[:, pillar_indices]

        canvas = canvas.view(B, C, self.ny, self.nx)
        return canvas


@LIDAR_ENCODERS.register_module()
class PPBackbone(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: Tuple[int, int, int] = (64, 128, 256), layer_nums: Tuple[int, int, int] = (3, 5, 5), layer_strides: Tuple[int, int, int] = (2, 2, 2)) -> None:
        super().__init__()
        assert len(out_channels) == len(layer_nums) == len(layer_strides)
        self.blocks = nn.ModuleList()
        c_in = in_channels
        for i in range(len(layer_strides)):
            stride = layer_strides[i]
            c_out = out_channels[i]
            layers: list[nn.Module] = [
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
                build_norm_layer(dict(type="BN2d", eps=1e-3, momentum=0.01), c_out)[1],
                nn.ReLU(inplace=True),
            ]
            for _ in range(layer_nums[i]):
                layers += [
                    nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
                    build_norm_layer(dict(type="BN2d", eps=1e-3, momentum=0.01), c_out)[1],
                    nn.ReLU(inplace=True),
                ]
            self.blocks.append(nn.Sequential(*layers))
            c_in = c_out

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: list[torch.Tensor] = []
        dtype = x.dtype
        for blk in self.blocks:
            # Run in FP32 for numerical stability with dynamic loss scaling
            x_fp32 = x.float()
            x_fp32 = blk(x_fp32)
            x = x_fp32.to(dtype)
            feats.append(x)
        return feats


@LIDAR_ENCODERS.register_module()
class PPNeck(nn.Module):
    def __init__(self, in_channels: Tuple[int, int, int] = (64, 128, 256), upsample_strides: Tuple[int, int, int] = (1, 2, 4), out_channels: Tuple[int, int, int] = (64, 64, 64)) -> None:
        super().__init__()
        assert len(in_channels) == len(upsample_strides) == len(out_channels)
        deblocks: list[nn.Module] = []
        for c_in, stride, c_out in zip(in_channels, upsample_strides, out_channels):
            deblock = nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, kernel_size=stride, stride=stride, bias=False),
                build_norm_layer(dict(type="BN2d", eps=1e-3, momentum=0.01), c_out)[1],
                nn.ReLU(inplace=True),
            )
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)
        self.out_channels = sum(out_channels)
        self.proj: Optional[nn.Conv2d] = None

    def forward(self, xs: List[torch.Tensor], out_channels_proj: Optional[int] = None) -> torch.Tensor:
        assert len(xs) == len(self.deblocks)
        dtype = xs[0].dtype
        ups: list[torch.Tensor] = []
        for x, deblock in zip(xs, self.deblocks):
            # Run in FP32 for numerical stability with dynamic loss scaling
            x_fp32 = x.float()
            up_fp32 = deblock(x_fp32)
            up = up_fp32.to(dtype)
            ups.append(up)
        x = torch.cat(ups, dim=1)
        if out_channels_proj is not None:
            if self.proj is None or self.proj.out_channels != out_channels_proj:
                self.proj = nn.Conv2d(self.out_channels, out_channels_proj, kernel_size=1, bias=False).to(x.device)
            x = self.proj(x)
        return x


@LIDAR_ENCODERS.register_module()
class LidarEncoder(nn.Module):
    """
    A self-contained PointPillars encoder **without detection head**.
    It performs:
      points -> hard voxelize -> PFN -> scatter to BEV -> BEV backbone -> BEV features
    """

    def __init__(
        self,
        num_point_features: int = 5,
        voxel_size: Tuple[float, float, float] = (0.2, 0.2, 8.0),
        point_cloud_range: Tuple[float, float, float, float, float, float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        max_num_points_per_voxel: int = 32,
        max_voxels: int = 20000,
        with_distance: bool = False,
        bev_output_shape: Tuple[int, int] = (512, 512),
    ) -> None:
        super().__init__()
        self.pc_range = point_cloud_range
        # Compute effective voxel size so that BEV grid matches bev_output_shape
        x_min, y_min, _z_min, x_max, y_max, _z_max = point_cloud_range
        base_vx, base_vy, base_vz = voxel_size
        span_x = x_max - x_min
        span_y = y_max - y_min
        grid_x = int(round(span_x / base_vx))
        grid_y = int(round(span_y / base_vy))
        target_ny, target_nx = bev_output_shape
        scale_x = max(1, grid_x // int(target_nx))
        scale_y = max(1, grid_y // int(target_ny))
        eff_vx = base_vx * scale_x
        eff_vy = base_vy * scale_y
        eff_voxel_size = (eff_vx, eff_vy, base_vz)
        self.voxel_size = eff_voxel_size

        if HardVoxelization is None:
            raise ImportError("mmdet3d.ops.Voxelization is required for this encoder.")

        self.voxelize = HardVoxelization(
            voxel_size=eff_voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points_per_voxel,
            max_voxels=max_voxels,
        )

        # Original PointPillars module names for weight compatibility
        feat_channels = (64,)
        self.pts_voxel_encoder = PillarFeatureNet(
            in_channels=num_point_features,
            feat_channels=feat_channels,
            with_distance=with_distance,
            voxel_size=eff_voxel_size,
            point_cloud_range=point_cloud_range,
        )
        self.pts_middle_encoder = PointPillarsScatter(in_channels=feat_channels[-1], output_shape=bev_output_shape)
        self.pts_backbone = PPBackbone(in_channels=feat_channels[-1], out_channels=(64, 128, 256), layer_nums=(2, 4, 4), layer_strides=(2, 2, 2))
        self.pts_neck = PPNeck(in_channels=(64, 128, 256), upsample_strides=(1, 2, 4), out_channels=(64, 96, 96))

    def _hard_voxelize(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        points: (B, N, C) in LIDAR coords, where C >= num_point_features
        Returns voxels, coords, num_points_per_voxel
        Note: MMDet3D's HardVoxelization expects (N_points, C) per sample; we loop over batch.
        """
        assert points.ndim == 3, "Expected points of shape (B, N, C)"
        B = points.shape[0]
        original_dtype = points.dtype
        voxels_list, coords_list, num_points_list = [], [], []
        
        for b in range(B):
            points_b = points[b]
            if points_b.dtype != torch.float32:
                points_b = points_b.to(torch.float32)
            
            # MUST use torch.no_grad() - voxelization has no backward() implementation
            with torch.no_grad():
                voxels, coors, num_points = self.voxelize(points_b)
            
            # Convert back to original dtype if needed (maintains gradient connection)
            if original_dtype != torch.float32:
                voxels = voxels.to(original_dtype)

            # keep integer types from voxelizer
            if coors.dtype != torch.long:
                coors = coors.to(torch.long)
            if num_points.dtype != torch.long:
                num_points = num_points.to(torch.long)

            # prepend batch index without F.pad (works for ints)
            batch_col = torch.full((coors.size(0), 1), b, dtype=coors.dtype, device=coors.device)
            coors = torch.cat([batch_col, coors], dim=1)

            voxels_list.append(voxels)
            coords_list.append(coors)
            num_points_list.append(num_points)
        voxels = torch.cat(voxels_list, dim=0)
        coors = torch.cat(coords_list, dim=0)
        num_points = torch.cat(num_points_list, dim=0)
        return voxels, coors, num_points

    def forward(self, points: torch.Tensor, return_bev_hw: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            points: (B, N, C) float tensor.
        Returns:
            bev_feat: (B, C, H/2, W/2)  where (H, W) = bev_output_shape
        """
        B = points.size(0)
        voxels, coors, num_points = self._hard_voxelize(points)

        x = self.pts_voxel_encoder(voxels, num_points, coors)
        bev = self.pts_middle_encoder(x, coors, B)
        xs = self.pts_backbone(bev)
        bev_out = self.pts_neck(xs)

        if return_bev_hw:
            bev_hw = bev_out.permute(0, 2, 3, 1).contiguous()
            return bev_out, bev_hw
        return bev_out
