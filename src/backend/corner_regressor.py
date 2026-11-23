"""SOTA 4-Corner Heatmap Regressor (DocAligner-style) for perspective card rectification.

Operates on a rigidly rotated, similarity-padded (zero-perspective-distortion) ROI patch
extracted from any upstream detector proposal (OBB or Seg). Predicts 4 spatial corner
heatmaps (TL, TR, BR, BL) with differentiable 2D Soft-Argmax sub-pixel coordinate
extraction plus an auxiliary card-mask head.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.rectifier import TARGET_H, TARGET_W, order_quad, warp_perspective_antialiased

logger = logging.getLogger(__name__)

PATCH_H = 512
PATCH_W = 384
PAD_FRAC = 0.24

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class CornerPatchPrediction:
    """Output of the 4-corner heatmap regressor on one padded ROI patch."""

    patch_bgr: np.ndarray
    patch_corners: np.ndarray        # (4, 2) in patch [0..PATCH_W, 0..PATCH_H]
    scene_corners: np.ndarray        # (4, 2) in scene image coordinates
    patch_mask: np.ndarray           # (PATCH_H, PATCH_W) auxiliary foreground mask
    heatmaps: np.ndarray             # (4, H/2, W/2) corner probability maps
    rectified_rgb: np.ndarray        # (TARGET_H, TARGET_W, 3) anti-aliased rectified card
    M_sim_inv: np.ndarray            # (2, 3) inverse affine mapping patch -> scene


def extract_oriented_padded_patch(
    img_bgr: np.ndarray,
    quad: np.ndarray,
    patch_hw: tuple[int, int] = (PATCH_H, PATCH_W),
    pad_frac: float = PAD_FRAC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract an upright, rigidly rotated & padded ROI patch around ``quad``.

    Uses a pure 2D similarity transform (no projective distortion) so all tilted
    corners remain undistorted inside the padded field of view with fixed
    [TL, TR, BR, BL] quadrant ordering.
    """
    q = order_quad(quad)
    ph, pw = int(patch_hw[0]), int(patch_hw[1])

    center = q.mean(axis=0)
    top_vec = q[1] - q[0]
    bot_vec = q[2] - q[3]
    mean_horiz = 0.5 * (top_vec + bot_vec)
    angle_rad = float(np.arctan2(mean_horiz[1], mean_horiz[0]))

    w_est = max(float(np.linalg.norm(top_vec)), float(np.linalg.norm(bot_vec)), 20.0)
    h_est = max(float(np.linalg.norm(q[3] - q[0])), float(np.linalg.norm(q[2] - q[1])), 20.0)

    # Fit card + pad_frac margin on each side comfortably inside (pw, ph)
    scale_x = float(pw) / (w_est * (1.0 + 2.0 * pad_frac))
    scale_y = float(ph) / (h_est * (1.0 + 2.0 * pad_frac))
    scale = min(scale_x, scale_y)

    cos_a = float(np.cos(-angle_rad))
    sin_a = float(np.sin(-angle_rad))
    M = np.array(
        [
            [scale * cos_a, -scale * sin_a, 0.5 * pw - scale * (cos_a * center[0] - sin_a * center[1])],
            [scale * sin_a, scale * cos_a, 0.5 * ph - scale * (sin_a * center[0] + cos_a * center[1])],
        ],
        dtype=np.float64,
    )
    M_inv = cv2.invertAffineTransform(M)
    patch_bgr = cv2.warpAffine(
        img_bgr,
        M,
        (pw, ph),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return patch_bgr, M, M_inv


def soft_argmax_2d(
    logits: torch.Tensor,
    out_hw: tuple[int, int] = (PATCH_H, PATCH_W),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute differentiable 2D Soft-Argmax coordinates in ``out_hw`` space.

    Args:
        logits: Tensor of shape ``(B, K, H_hm, W_hm)``.
        out_hw: Target coordinate frame ``(H_out, W_out)``.

    Returns:
        coords: ``(B, K, 2)`` continuous ``(x, y)`` coordinates in ``[0, W_out-1] x [0, H_out-1]``.
        probs: ``(B, K, H_hm, W_hm)`` normalized spatial probability heatmaps.
    """
    b, k, h_hm, w_hm = logits.shape
    flat = logits.view(b, k, -1)
    probs = F.softmax(flat, dim=-1).view(b, k, h_hm, w_hm)

    xs = torch.linspace(0.0, float(out_hw[1] - 1), w_hm, device=logits.device, dtype=logits.dtype)
    ys = torch.linspace(0.0, float(out_hw[0] - 1), h_hm, device=logits.device, dtype=logits.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    exp_x = (probs * grid_x.view(1, 1, h_hm, w_hm)).sum(dim=(-2, -1))
    exp_y = (probs * grid_y.view(1, 1, h_hm, w_hm)).sum(dim=(-2, -1))
    coords = torch.stack([exp_x, exp_y], dim=-1)
    return coords, probs


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CornerHeatmapUNet(nn.Module):
    """MobileNetV3-Large U-Net predicting 4 corner heatmaps + 1 auxiliary card mask."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        feats = mobilenet_v3_large(weights=weights).features
        self.enc1 = feats[:2]    # stride 2, 16 ch
        self.enc2 = feats[2:4]   # stride 4, 24 ch
        self.enc3 = feats[4:7]   # stride 8, 40 ch
        self.enc4 = feats[7:13]  # stride 16, 112 ch
        self.enc5 = feats[13:]   # stride 32, 960 ch

        self.up4 = _ConvBlock(960 + 112, 256)
        self.up3 = _ConvBlock(256 + 40, 128)
        self.up2 = _ConvBlock(128 + 24, 64)
        self.up1 = _ConvBlock(64 + 16, 48)

        self.corner_head = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=1),
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s1 = self.enc1(x)   # stride 2 (H/2, W/2)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        s5 = self.enc5(s4)

        d4 = self.up4(torch.cat([F.interpolate(s5, size=s4.shape[-2:], mode="bilinear", align_corners=False), s4], dim=1))
        d3 = self.up3(torch.cat([F.interpolate(d4, size=s3.shape[-2:], mode="bilinear", align_corners=False), s3], dim=1))
        d2 = self.up2(torch.cat([F.interpolate(d3, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2], dim=1))
        d1 = self.up1(torch.cat([F.interpolate(d2, size=s1.shape[-2:], mode="bilinear", align_corners=False), s1], dim=1))

        corner_logits = self.corner_head(d1)
        mask_logits = self.mask_head(d1)
        return corner_logits, mask_logits


def patch_to_imagenet_tensor(patch_bgr: np.ndarray, device: str | torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0).to(device)


class CornerHeatmapRegressor:
    """Inference wrapper for 4-corner heatmap prediction and card rectification."""

    def __init__(
        self,
        weights_path: str | Path,
        device: Optional[str] = None,
        patch_hw: tuple[int, int] = (PATCH_H, PATCH_W),
        pad_frac: float = PAD_FRAC,
        target_size: tuple[int, int] = (TARGET_H, TARGET_W),
    ):
        self.patch_hw = patch_hw
        self.pad_frac = pad_frac
        self.target_size = target_size
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model = CornerHeatmapUNet(pretrained=False)
        ckpt = torch.load(str(weights_path), map_location=self.device, weights_only=True)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_from_proposal(
        self,
        img_bgr: np.ndarray,
        init_quad: np.ndarray,
    ) -> CornerPatchPrediction:
        patch_bgr, _, M_inv = extract_oriented_padded_patch(
            img_bgr,
            init_quad,
            patch_hw=self.patch_hw,
            pad_frac=self.pad_frac,
        )
        x = patch_to_imagenet_tensor(patch_bgr, self.device)
        corner_logits, mask_logits = self.model(x)
        coords, probs = soft_argmax_2d(corner_logits, out_hw=self.patch_hw)

        patch_corners = coords[0].cpu().numpy().astype(np.float64)
        mask_prob = torch.sigmoid(mask_logits[0, 0]).cpu().numpy().astype(np.float32)
        mask_full = cv2.resize(mask_prob, (self.patch_hw[1], self.patch_hw[0]), interpolation=cv2.INTER_LINEAR)

        # Map patch corners back to scene coordinates via inverse similarity affine M_inv
        ones = np.ones((4, 1), dtype=np.float64)
        scene_corners = (np.hstack([patch_corners, ones]) @ M_inv.T).astype(np.float64)
        scene_corners = order_quad(scene_corners)

        th, tw = self.target_size
        dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
        M_persp = cv2.getPerspectiveTransform(scene_corners.astype(np.float32), dst)
        rect_bgr = warp_perspective_antialiased(
            img_bgr,
            M_persp,
            (tw, th),
            quad_corners=scene_corners,
            flags=cv2.INTER_LANCZOS4,
            border_mode=cv2.BORDER_REPLICATE,
            antialiased=True,
        )
        rect_rgb = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2RGB)
        return CornerPatchPrediction(
            patch_bgr=patch_bgr,
            patch_corners=patch_corners,
            scene_corners=scene_corners,
            patch_mask=mask_full,
            heatmaps=probs[0].cpu().numpy(),
            rectified_rgb=rect_rgb,
            M_sim_inv=M_inv,
        )
