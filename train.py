
"""
train.py - Production Training Pipeline for Tiny-Object Drone Detection
=======================================================================
End-to-end pipeline:
  1. Auto-slicing (SAHI slice_coco)
  2. SlicedDroneDataset with tiny-object augmentations
  3. TinyDroneDetector: Backbone (C2-C5) + TinyFPN (P2-P5) + DecoupledHead
  4. ScaleAwareAssigner (<16px -> P2, 16-32px -> P3, 32-64px -> P4, >=64px -> P5)
  5. TinyDetectionLoss (Focal + Centerness + 0.5*GIoU + 0.5*NWD)
  6. AMP training loop with cosine warmup, checkpointing, and isolated mAP_tiny evaluation
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    COCO = None  # type: ignore[assignment, misc]
    COCOeval = None  # type: ignore[assignment, misc]


# ============================================================================
# 1. AUTOMATED DATASET SLICING
# ============================================================================

def check_and_slice_dataset(
    raw_img_dir: str,
    raw_ann: str,
    sliced_dir: str,
    slice_size: int = 640,
    overlap_ratio: float = 0.2,
    min_area_ratio: float = 0.1,
) -> Tuple[str, str]:
    """Slice high-res images and annotations into overlapping patches via SAHI."""
    ann_base = "sliced"
    sliced_ann_path = os.path.join(sliced_dir, f"{ann_base}_coco.json")
    sliced_img_dir = os.path.join(sliced_dir, "images")

    # Skip if pre-sliced dataset already exists
    if os.path.isfile(sliced_ann_path) and os.path.isdir(sliced_img_dir):
        imgs = [f for f in os.listdir(sliced_img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if len(imgs) > 0:
            print(f"[slice] Found existing sliced dataset at {sliced_dir} ({len(imgs)} tiles) -- skipping.")
            return sliced_img_dir, sliced_ann_path

    print(f"[slice] Slicing {raw_img_dir} -> {sliced_dir} (tile={slice_size}, overlap={overlap_ratio})")
    from sahi.slicing import slice_coco

    slice_coco(
        coco_annotation_file_path=raw_ann,
        image_dir=raw_img_dir,
        output_coco_annotation_file_name=ann_base,
        output_dir=sliced_dir,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        min_area_ratio=min_area_ratio,
        verbose=False,
    )

    # Standardize image location to sliced_dir/images
    if not os.path.isdir(sliced_img_dir):
        os.makedirs(sliced_img_dir, exist_ok=True)
        for f in os.listdir(sliced_dir):
            src = os.path.join(sliced_dir, f)
            if os.path.isfile(src) and f.lower().endswith((".jpg", ".png", ".jpeg")):
                shutil.move(src, os.path.join(sliced_img_dir, f))

    print(f"[slice] Slicing completed. Tiles saved in {sliced_img_dir}")
    return sliced_img_dir, sliced_ann_path


# ============================================================================
# 2. DATASET & TINY-OBJECT AUGMENTATION
# ============================================================================

class SlicedDroneDataset(Dataset):
    """Loads sliced COCO tiles with tiny-object safe augmentations."""

    def __init__(self, img_dir: str, ann_path: str, input_size: int = 640, augment: bool = True) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.input_size = input_size
        self.augment = augment

        with open(ann_path, "r") as f:
            coco_json = json.load(f)

        self.categories = coco_json.get("categories", [{"id": 1, "name": "person"}])
        self.cat_to_idx = {c["id"]: i for i, c in enumerate(self.categories)}
        self.num_classes = len(self.categories)
        self.images_meta = coco_json["images"]

        self.anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for ann in coco_json.get("annotations", []):
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int:
        return len(self.images_meta)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.images_meta[idx]
        img_id = meta["id"]

        fname = os.path.basename(meta["file_name"])
        img_path = os.path.join(self.img_dir, fname)
        if not os.path.isfile(img_path):
            img_path = os.path.join(self.img_dir, meta["file_name"])

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Parse ground-truth boxes [x1, y1, x2, y2]
        boxes_list, labels_list = [], []
        for ann in self.anns_by_img.get(img_id, []):
            x, y, bw, bh = ann["bbox"]
            if bw < 1 or bh < 1:
                continue
            boxes_list.append([x, y, x + bw, y + bh])
            labels_list.append(self.cat_to_idx.get(ann["category_id"], 0))

        boxes = np.array(boxes_list, dtype=np.float32).reshape(-1, 4)
        labels = np.array(labels_list, dtype=np.int64)

        # Tiny-object friendly augmentations
        if self.augment:
            # 1. Random horizontal flip (p=0.5)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if len(boxes) > 0:
                    x1 = boxes[:, 0].copy()
                    boxes[:, 0] = w - boxes[:, 2]
                    boxes[:, 2] = w - x1

            # 2. Scale jitter [0.85, 1.15] - keeps tiny objects within perceptible bounds
            scale = random.uniform(0.85, 1.15)
            nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
            img = img.resize((nw, nh), Image.BILINEAR)
            if len(boxes) > 0:
                boxes *= scale

            # 3. Color jitter for aerial illumination variance
            from torchvision.transforms import functional as TF
            img = TF.adjust_brightness(img, 1.0 + random.uniform(-0.15, 0.15))
            img = TF.adjust_contrast(img, 1.0 + random.uniform(-0.15, 0.15))
            img = TF.adjust_saturation(img, 1.0 + random.uniform(-0.15, 0.15))

        # Letterbox to square input_size (aspect-ratio preservation)
        w, h = img.size
        r = min(self.input_size / w, self.input_size / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        img = img.resize((nw, nh), Image.BILINEAR)

        pad_x = (self.input_size - nw) // 2
        pad_y = (self.input_size - nh) // 2
        canvas = Image.new("RGB", (self.input_size, self.input_size), (114, 114, 114))
        canvas.paste(img, (pad_x, pad_y))

        if len(boxes) > 0:
            boxes *= r
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] + pad_x, 0, self.input_size)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] + pad_y, 0, self.input_size)

        img_tensor = torch.from_numpy(np.array(canvas, dtype=np.float32) / 255.0).permute(2, 0, 1)  # (3, H, W)
        return {
            "image": img_tensor,
            "boxes": torch.from_numpy(boxes),      # (N, 4)
            "labels": torch.from_numpy(labels),    # (N,)
            "image_id": img_id,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stacks images into tensor; preserves variable-length boxes per image."""
    return {
        "images": torch.stack([s["image"] for s in batch], dim=0),  # (B, 3, H, W)
        "boxes": [s["boxes"] for s in batch],                      # list of (Ni, 4)
        "labels": [s["labels"] for s in batch],                    # list of (Ni,)
        "image_ids": [s["image_id"] for s in batch],
    }


# ============================================================================
# 3. P2-AUGMENTED MODEL ARCHITECTURE
# ============================================================================

class ConvBNSiLU(nn.Module):
    """Conv2d -> BatchNorm2d -> SiLU (smooth activation preserves weak responses)."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))  # (B, out_ch, H/s, W/s)


class ResidualBlock(nn.Module):
    """Bottleneck residual block maintaining channel dimension."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        mid = max(channels // 2, 32)
        self.conv1 = ConvBNSiLU(channels, mid, k=1, p=0)
        self.conv2 = ConvBNSiLU(mid, channels, k=3, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))  # (B, C, H, W)


class DroneBackbone(nn.Module):
    """Multi-scale residual backbone retaining C2 (stride 4) for tiny objects."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNSiLU(3, 32, k=3, s=2, p=1),
            ConvBNSiLU(32, 32, k=3, s=1, p=1),
        )  # (B, 32, H/2, W/2)

        self.stage1 = nn.Sequential(
            ConvBNSiLU(32, 64, k=3, s=2, p=1),
            ResidualBlock(64), ResidualBlock(64),
        )  # C2: (B, 64, H/4, W/4)

        self.stage2 = nn.Sequential(
            ConvBNSiLU(64, 128, k=3, s=2, p=1),
            ResidualBlock(128), ResidualBlock(128),
        )  # C3: (B, 128, H/8, W/8)

        self.stage3 = nn.Sequential(
            ConvBNSiLU(128, 256, k=3, s=2, p=1),
            ResidualBlock(256), ResidualBlock(256),
        )  # C4: (B, 256, H/16, W/16)

        self.stage4 = nn.Sequential(
            ConvBNSiLU(256, 512, k=3, s=2, p=1),
            ResidualBlock(512), ResidualBlock(512),
        )  # C5: (B, 512, H/32, W/32)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return c2, c3, c4, c5


class TinyFPN(nn.Module):
    """Feature Pyramid Network fusing semantic and spatial features down to P2."""

    def __init__(self, fpn_ch: int = 128) -> None:
        super().__init__()
        self.lat2 = nn.Conv2d(64, fpn_ch, 1)
        self.lat3 = nn.Conv2d(128, fpn_ch, 1)
        self.lat4 = nn.Conv2d(256, fpn_ch, 1)
        self.lat5 = nn.Conv2d(512, fpn_ch, 1)

        self.smooth2 = ConvBNSiLU(fpn_ch, fpn_ch, 3, 1, 1)
        self.smooth3 = ConvBNSiLU(fpn_ch, fpn_ch, 3, 1, 1)
        self.smooth4 = ConvBNSiLU(fpn_ch, fpn_ch, 3, 1, 1)
        self.smooth5 = ConvBNSiLU(fpn_ch, fpn_ch, 3, 1, 1)

    def forward(
        self, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        l5, l4, l3, l2 = self.lat5(c5), self.lat4(c4), self.lat3(c3), self.lat2(c2)

        p5 = self.smooth5(l5)                                                                 # (B, 128, H/32, W/32)
        p4 = self.smooth4(l4 + F.interpolate(l5, size=l4.shape[2:], mode="nearest"))          # (B, 128, H/16, W/16)
        p3 = self.smooth3(l3 + F.interpolate(p4, size=l3.shape[2:], mode="nearest"))          # (B, 128, H/8, W/8)
        p2 = self.smooth2(l2 + F.interpolate(p3, size=l2.shape[2:], mode="nearest"))          # (B, 128, H/4, W/4)
        return p2, p3, p4, p5


class DecoupledHead(nn.Module):
    """Decoupled anchor-free prediction head shared across FPN levels."""

    def __init__(self, fpn_ch: int = 128, num_classes: int = 1, num_convs: int = 4) -> None:
        super().__init__()
        cls_layers = [ConvBNSiLU(fpn_ch, fpn_ch, 3, 1, 1) for _ in range(num_convs)]
        reg_layers = [ConvBNSiLU(fpn_ch, fpn_ch, 3, 1, 1) for _ in range(num_convs)]

        self.cls_branch = nn.Sequential(*cls_layers)
        self.cls_out = nn.Conv2d(fpn_ch, num_classes, 3, 1, 1)

        self.reg_branch = nn.Sequential(*reg_layers)
        self.reg_out = nn.Conv2d(fpn_ch, 4, 3, 1, 1)
        self.ctr_out = nn.Conv2d(fpn_ch, 1, 3, 1, 1)

        # Prior bias initialization for focal loss stability
        nn.init.constant_(self.cls_out.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_feat = self.cls_branch(feat)
        reg_feat = self.reg_branch(feat)

        cls_logits = self.cls_out(cls_feat)        # (B, num_classes, H_l, W_l)
        reg_pred = F.relu(self.reg_out(reg_feat))  # (B, 4, H_l, W_l) -> (l, t, r, b) >= 0
        centerness = self.ctr_out(reg_feat)        # (B, 1, H_l, W_l)
        return cls_logits, reg_pred, centerness


class TinyDroneDetector(nn.Module):
    """End-to-end detector combining DroneBackbone, TinyFPN, and DecoupledHead."""

    STRIDES = [4, 8, 16, 32]

    def __init__(self, num_classes: int = 1, fpn_channels: int = 128) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fpn_channels = fpn_channels
        self.backbone = DroneBackbone()
        self.fpn = TinyFPN(fpn_ch=fpn_channels)
        self.head = DecoupledHead(fpn_ch=fpn_channels, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # x: (B, 3, H, W)
        c2, c3, c4, c5 = self.backbone(x)
        p2, p3, p4, p5 = self.fpn(c2, c3, c4, c5)
        return [self.head(p) for p in [p2, p3, p4, p5]]


# ============================================================================
# 4. SCALE-AWARE TARGET ASSIGNER
# ============================================================================

class ScaleAwareAssigner:
    """Routes GT boxes by sqrt(w*h) to pyramid levels and computes (l*, t*, r*, b*, ctr*)."""

    def __init__(
        self,
        strides: List[int] = [4, 8, 16, 32],
        bounds: List[float] = [16.0, 32.0, 64.0],
    ) -> None:
        self.strides = strides
        self.bounds = bounds

    def assign(
        self,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        feat_sizes: List[Tuple[int, int]],
        grid_centers: List[Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> List[Dict[str, torch.Tensor]]:
        """Assigns targets for a single image across all FPN levels."""
        num_levels = len(self.strides)
        N = gt_boxes.shape[0]

        # 1. Level routing: sqrt(w * h) bounds
        if N > 0:
            sizes = torch.sqrt((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]))
            level_idx = torch.zeros(N, dtype=torch.long, device=device)
            level_idx[sizes >= self.bounds[0]] = 1
            level_idx[sizes >= self.bounds[1]] = 2
            level_idx[sizes >= self.bounds[2]] = 3
        else:
            level_idx = torch.zeros(0, dtype=torch.long, device=device)

        targets = []
        for li in range(num_levels):
            H_l, W_l = feat_sizes[li]
            num_cells = H_l * W_l
            grid_x, grid_y = grid_centers[li]  # (num_cells,) each

            cls_tgt = torch.full((num_cells,), -1, dtype=torch.long, device=device)
            reg_tgt = torch.zeros((num_cells, 4), dtype=torch.float32, device=device)
            ctr_tgt = torch.zeros((num_cells,), dtype=torch.float32, device=device)

            if N > 0:
                mask = level_idx == li
                lvl_boxes = gt_boxes[mask]
                lvl_labels = gt_labels[mask]

                for gi in range(lvl_boxes.shape[0]):
                    gx1, gy1, gx2, gy2 = lvl_boxes[gi]
                    l = grid_x - gx1
                    t = grid_y - gy1
                    r = gx2 - grid_x
                    b = gy2 - grid_y

                    # Identify positive centers strictly inside box
                    inside = (l > 0) & (t > 0) & (r > 0) & (b > 0)
                    if not inside.any():
                        continue

                    # Safe centerness computation restricted to valid inside coordinates (prevents negative sqrt NaNs)
                    l_in, t_in, r_in, b_in = l[inside], t[inside], r[inside], b[inside]
                    lr_ratio = torch.min(l_in, r_in) / torch.max(l_in, r_in).clamp(min=1e-6)
                    tb_ratio = torch.min(t_in, b_in) / torch.max(t_in, b_in).clamp(min=1e-6)
                    ctr_in = torch.sqrt((lr_ratio * tb_ratio).clamp(min=0.0, max=1.0))

                    inside_indices = inside.nonzero(as_tuple=True)[0]
                    better = ctr_in > ctr_tgt[inside_indices]
                    if not better.any():
                        continue

                    chosen_idx = inside_indices[better]
                    cls_tgt[chosen_idx] = lvl_labels[gi]
                    reg_tgt[chosen_idx] = torch.stack([l_in[better], t_in[better], r_in[better], b_in[better]], dim=1)
                    ctr_tgt[chosen_idx] = ctr_in[better]

            targets.append({
                "cls": cls_tgt,
                "reg": reg_tgt,
                "ctr": ctr_tgt,
                "pos": cls_tgt >= 0,
            })
        return targets


# ============================================================================
# 5. LOSS OBJECTIVES (FOCAL + NWD + GIOU + CENTERNESS)
# ============================================================================

def sigmoid_focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss for dense background classification imbalance."""
    p = torch.sigmoid(pred)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * ((1 - p_t) ** gamma) * bce).sum()


def giou_loss(p_boxes: torch.Tensor, g_boxes: torch.Tensor) -> torch.Tensor:
    """Generalized IoU loss for bounding box regression."""
    ix1, iy1 = torch.max(p_boxes[:, 0], g_boxes[:, 0]), torch.max(p_boxes[:, 1], g_boxes[:, 1])
    ix2, iy2 = torch.min(p_boxes[:, 2], g_boxes[:, 2]), torch.min(p_boxes[:, 3], g_boxes[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    area_p = (p_boxes[:, 2] - p_boxes[:, 0]).clamp(min=0) * (p_boxes[:, 3] - p_boxes[:, 1]).clamp(min=0)
    area_g = (g_boxes[:, 2] - g_boxes[:, 0]).clamp(min=0) * (g_boxes[:, 3] - g_boxes[:, 1]).clamp(min=0)
    union = area_p + area_g - inter
    iou = inter / union.clamp(min=1e-6)

    ex1, ey1 = torch.min(p_boxes[:, 0], g_boxes[:, 0]), torch.min(p_boxes[:, 1], g_boxes[:, 1])
    ex2, ey2 = torch.max(p_boxes[:, 2], g_boxes[:, 2]), torch.max(p_boxes[:, 3], g_boxes[:, 3])
    enclosing = (ex2 - ex1).clamp(min=0) * (ey2 - ey1).clamp(min=0)

    giou = iou - (enclosing - union) / enclosing.clamp(min=1e-6)
    return (1.0 - giou).mean()


def nwd_loss(p_boxes: torch.Tensor, g_boxes: torch.Tensor, C: float = 12.8) -> torch.Tensor:
    """Normalized Wasserstein Distance (NWD) loss: models boxes as 2D Gaussians."""
    px_c, py_c = (p_boxes[:, 0] + p_boxes[:, 2]) / 2.0, (p_boxes[:, 1] + p_boxes[:, 3]) / 2.0
    pw, ph = p_boxes[:, 2] - p_boxes[:, 0], p_boxes[:, 3] - p_boxes[:, 1]

    gx_c, gy_c = (g_boxes[:, 0] + g_boxes[:, 2]) / 2.0, (g_boxes[:, 1] + g_boxes[:, 3]) / 2.0
    gw, gh = g_boxes[:, 2] - g_boxes[:, 0], g_boxes[:, 3] - g_boxes[:, 1]

    w2_sq = (px_c - gx_c)**2 + (py_c - gy_c)**2 + ((pw - gw)**2 + (ph - gh)**2) / 4.0
    nwd_sim = torch.exp(-torch.sqrt(w2_sq.clamp(min=1e-7)) / C)
    return (1.0 - nwd_sim).mean()


class TinyDetectionLoss(nn.Module):
    """Total Loss = Focal + Centerness + 0.5 * GIoU + 0.5 * NWD."""

    def __init__(self, num_classes: int = 1, strides: List[int] = [4, 8, 16, 32], nwd_c: float = 12.8) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.nwd_c = nwd_c
        self.assigner = ScaleAwareAssigner(strides=strides)

    def forward(
        self,
        outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = outputs[0][0].device
        B = outputs[0][0].shape[0]
        L = len(outputs)

        # Precompute spatial dimensions & grid coordinates once per forward pass
        feat_sizes = [(outputs[li][0].shape[2], outputs[li][0].shape[3]) for li in range(L)]
        grid_centers = []
        for li in range(L):
            H_l, W_l = feat_sizes[li]
            s = self.strides[li]
            sy = (torch.arange(0, H_l, device=device, dtype=torch.float32) + 0.5) * s
            sx = (torch.arange(0, W_l, device=device, dtype=torch.float32) + 0.5) * s
            gy, gx = torch.meshgrid(sy, sx, indexing="ij")
            grid_centers.append((gx.reshape(-1), gy.reshape(-1)))

        tot_cls, tot_reg, tot_ctr = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        total_pos = 0

        for b in range(B):
            b_boxes = gt_boxes[b].to(device)
            b_labels = gt_labels[b].to(device)
            targets = self.assigner.assign(b_boxes, b_labels, feat_sizes, grid_centers, device)

            for li in range(L):
                cls_logits = outputs[li][0][b].permute(1, 2, 0).reshape(-1, self.num_classes)
                reg_pred = outputs[li][1][b].permute(1, 2, 0).reshape(-1, 4)
                ctr_pred = outputs[li][2][b].permute(1, 2, 0).reshape(-1)

                tgt = targets[li]
                pos_mask = tgt["pos"]
                n_pos = pos_mask.sum().item()

                # Classification focal loss
                cls_oh = torch.zeros_like(cls_logits)
                if n_pos > 0:
                    cls_oh[pos_mask.nonzero(as_tuple=True)[0], tgt["cls"][pos_mask]] = 1.0
                tot_cls = tot_cls + sigmoid_focal_loss(cls_logits, cls_oh)

                if n_pos == 0:
                    continue
                total_pos += n_pos

                # Centerness BCE loss
                tot_ctr = tot_ctr + F.binary_cross_entropy_with_logits(ctr_pred[pos_mask], tgt["ctr"][pos_mask], reduction="sum")

                # Decode boxes at positive cells
                gx, gy = grid_centers[li]
                cx, cy = gx[pos_mask], gy[pos_mask]

                p_ltrb = reg_pred[pos_mask]
                p_xyxy = torch.stack([cx - p_ltrb[:, 0], cy - p_ltrb[:, 1], cx + p_ltrb[:, 2], cy + p_ltrb[:, 3]], dim=1)

                t_ltrb = tgt["reg"][pos_mask]
                t_xyxy = torch.stack([cx - t_ltrb[:, 0], cy - t_ltrb[:, 1], cx + t_ltrb[:, 2], cy + t_ltrb[:, 3]], dim=1)

                # Combined regression: 0.5 * GIoU + 0.5 * NWD
                loss_reg = 0.5 * giou_loss(p_xyxy, t_xyxy) + 0.5 * nwd_loss(p_xyxy, t_xyxy, C=self.nwd_c)
                tot_reg = tot_reg + loss_reg * n_pos

        norm = max(total_pos, 1)
        loss_cls = tot_cls / norm
        loss_reg = tot_reg / norm
        loss_ctr = tot_ctr / norm
        total = loss_cls + loss_reg + loss_ctr

        return {"total": total, "cls": loss_cls.detach(), "reg": loss_reg.detach(), "ctr": loss_ctr.detach()}


# ============================================================================
# 6. DECODER & VALIDATION (mAP_tiny)
# ============================================================================

def decode_predictions(
    outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    strides: List[int],
    score_thr: float = 0.05,
    nms_thr: float = 0.5,
    max_per_img: int = 300,
) -> List[Dict[str, torch.Tensor]]:
    """Decodes raw head outputs into [x1, y1, x2, y2] detections with per-class NMS."""
    device = outputs[0][0].device
    B = outputs[0][0].shape[0]
    from torchvision.ops import batched_nms

    results = []
    for b in range(B):
        all_boxes, all_scores, all_labels = [], [], []
        for li, s in enumerate(strides):
            cls_logits, reg_pred, ctr_pred = outputs[li][0][b], outputs[li][1][b], outputs[li][2][b]
            C, H_l, W_l = cls_logits.shape

            sy = (torch.arange(0, H_l, device=device, dtype=torch.float32) + 0.5) * s
            sx = (torch.arange(0, W_l, device=device, dtype=torch.float32) + 0.5) * s
            gy, gx = torch.meshgrid(sy, sx, indexing="ij")
            gx, gy = gx.reshape(-1), gy.reshape(-1)

            scores = torch.sigmoid(cls_logits.permute(1, 2, 0).reshape(-1, C)) * torch.sigmoid(ctr_pred.reshape(-1)).unsqueeze(1)
            max_s, max_c = scores.max(dim=1)
            keep = max_s > score_thr
            if not keep.any():
                continue

            k_reg = reg_pred.permute(1, 2, 0).reshape(-1, 4)[keep]
            cx, cy = gx[keep], gy[keep]
            boxes = torch.stack([cx - k_reg[:, 0], cy - k_reg[:, 1], cx + k_reg[:, 2], cy + k_reg[:, 3]], dim=1)

            all_boxes.append(boxes)
            all_scores.append(max_s[keep])
            all_labels.append(max_c[keep])

        if len(all_boxes) == 0:
            results.append({"boxes": torch.zeros((0, 4), device=device), "scores": torch.zeros((0,), device=device), "labels": torch.zeros((0,), dtype=torch.long, device=device)})
            continue

        c_boxes = torch.cat(all_boxes, dim=0)
        c_scores = torch.cat(all_scores, dim=0)
        c_labels = torch.cat(all_labels, dim=0)
        keep_idx = batched_nms(c_boxes, c_scores, c_labels, nms_thr)[:max_per_img]
        results.append({"boxes": c_boxes[keep_idx], "scores": c_scores[keep_idx], "labels": c_labels[keep_idx]})
    return results


@torch.no_grad()
def evaluate_map_tiny(model: nn.Module, val_loader: DataLoader, device: torch.device, ann_path: str, max_area: int = 256) -> Tuple[float, float]:
    """Computes mAP@0.5 and mAP@0.5:0.95 for tiny objects (< 16x16 px) via pycocotools."""
    if COCO is None or COCOeval is None:
        return 0.0, 0.0

    model.eval()
    coco_gt = COCO(ann_path)
    cat_ids = sorted(coco_gt.getCatIds())
    coco_results = []
    val_image_ids: Set[int] = set()

    for batch in val_loader:
        images = batch["images"].to(device)
        image_ids = batch["image_ids"]
        for iid in image_ids:
            val_image_ids.add(int(iid))

        dets = decode_predictions(model(images), TinyDroneDetector.STRIDES, score_thr=0.01, nms_thr=0.6)

        for bi, det in enumerate(dets):
            boxes, scores, labels = det["boxes"].cpu().numpy(), det["scores"].cpu().numpy(), det["labels"].cpu().numpy()
            for k in range(len(boxes)):
                x1, y1, x2, y2 = boxes[k]
                coco_results.append({
                    "image_id": int(image_ids[bi]),
                    "category_id": int(cat_ids[int(labels[k])] if int(labels[k]) < len(cat_ids) else cat_ids[0]),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(scores[k]),
                })

    if len(coco_results) == 0:
        return 0.0, 0.0

    coco_dt = coco_gt.loadRes(coco_results)

    # Restrict evaluation strictly to validation images (prevents artificially collapsed mAP during subset splits)
    eval_ids = sorted(list(val_image_ids))

    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = eval_ids
    evaluator.params.areaRng = [[0, max_area]]
    evaluator.params.areaRngLbl = ["tiny"]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    map_50_95 = float(evaluator.stats[0])

    evaluator2 = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator2.params.imgIds = eval_ids
    evaluator2.params.areaRng = [[0, max_area]]
    evaluator2.params.areaRngLbl = ["tiny"]
    evaluator2.params.iouThrs = [0.5]
    evaluator2.evaluate()
    evaluator2.accumulate()
    map_50 = float(evaluator2.stats[0])

    return map_50, map_50_95


# ============================================================================
# 7. SYNTHETIC BENCHMARK GENERATOR
# ============================================================================

def create_synthetic_dataset(output_dir: str, num_images: int = 4, img_size: int = 640) -> Tuple[str, str]:
    """Generates synthetic aerial tiles with tiny colored rectangles for smoke tests."""
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    images_json, annotations_json = [], []
    ann_id = 1

    for i in range(num_images):
        arr = np.random.randint(90, 160, (img_size, img_size, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

        for _ in range(random.randint(1, 4)):
            w, h = random.randint(4, 15), random.randint(4, 15)
            x, y = random.randint(0, img_size - w - 1), random.randint(0, img_size - h - 1)
            draw.rectangle([x, y, x + w, y + h], fill=(240, 50, 50))
            annotations_json.append({"id": ann_id, "image_id": i + 1, "category_id": 1, "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            ann_id += 1

        fname = f"synth_{i:04d}.jpg"
        img.save(os.path.join(img_dir, fname))
        images_json.append({"id": i + 1, "file_name": fname, "width": img_size, "height": img_size})

    ann_path = os.path.join(output_dir, "annotations.json")
    with open(ann_path, "w") as f:
        json.dump({"images": images_json, "annotations": annotations_json, "categories": [{"id": 1, "name": "person"}]}, f)

    return img_dir, ann_path


# ============================================================================
# 8. TRAINING ENGINE
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tiny Drone Detector (P2-FPN + NWD)")
    parser.add_argument("--raw_img_dir", type=str, default="VisDrone2019-DET-train/images")
    parser.add_argument("--raw_ann", type=str, default="VisDrone2019-DET-train/train_coco.json")
    parser.add_argument("--val_img_dir", type=str, default="")
    parser.add_argument("--val_ann", type=str, default="")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--sliced_dir", type=str, default="data/sliced_train")
    parser.add_argument("--sliced_val_dir", type=str, default="data/sliced_val")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--input_size", type=int, default=640)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[train] Running on: {device}")

    # Handle smoke test / synthetic fallback
    if args.smoke_test or (not args.raw_img_dir and not args.raw_ann):
        print("[train] Smoke test mode: generating synthetic benchmark dataset.")
        args.raw_img_dir, args.raw_ann = create_synthetic_dataset("data/smoke_test", 4, args.input_size)
        args.epochs, args.batch_size, args.warmup_epochs = 2, 2, 0

    # Auto-slice raw dataset
    sliced_img_dir, sliced_ann = check_and_slice_dataset(
        args.raw_img_dir, args.raw_ann, args.sliced_dir,
        slice_size=args.input_size, overlap_ratio=0.2, min_area_ratio=0.1
    )

    full_ds = SlicedDroneDataset(sliced_img_dir, sliced_ann, input_size=args.input_size, augment=True)
    num_classes = full_ds.num_classes

    # Validation split setup
    if args.val_img_dir and args.val_ann:
        val_img_dir, val_ann = check_and_slice_dataset(
            args.val_img_dir, args.val_ann, args.sliced_val_dir,
            slice_size=args.input_size, overlap_ratio=0.2, min_area_ratio=0.1
        )
        val_ds = SlicedDroneDataset(val_img_dir, val_ann, input_size=args.input_size, augment=False)
        train_ds = full_ds
    else:
        n_val = max(int(len(full_ds) * args.val_split), 1)
        indices = list(range(len(full_ds)))
        random.seed(42)
        random.shuffle(indices)
        train_ds = Subset(full_ds, indices[n_val:])
        val_ds = Subset(SlicedDroneDataset(sliced_img_dir, sliced_ann, input_size=args.input_size, augment=False), indices[:n_val])
        val_ann = sliced_ann

    print(f"[train] Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)} | Classes: {num_classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model = TinyDroneDetector(num_classes=num_classes).to(device)
    criterion = TinyDetectionLoss(num_classes=num_classes, strides=TinyDroneDetector.STRIDES)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    warmup_steps = args.warmup_epochs * len(train_loader)
    global_step = 0
    best_map = -1.0
    os.makedirs(args.weights_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_cls, ep_reg, ep_ctr = 0.0, 0.0, 0.0, 0.0
        t0 = time.time()

        for batch in train_loader:
            global_step += 1
            if global_step <= warmup_steps:
                warmup_lr = args.lr * max(global_step / max(warmup_steps, 1), 1e-4)
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr

            imgs = batch["images"].to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                losses = criterion(model(imgs), batch["boxes"], batch["labels"])
                loss = losses["total"]

            if not torch.isfinite(loss):
                print(f"[train] Warning: non-finite loss at step {global_step}, skipping.")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            ep_loss += loss.item()
            ep_cls += losses["cls"].item()
            ep_reg += losses["reg"].item()
            ep_ctr += losses["ctr"].item()
            # to output progress of epoch
            if global_step % 20 == 0:
                cur_batch = (global_step - 1) % len(train_loader) + 1
                print(f"[Epoch {epoch:2d}] Batch {cur_batch:4d}/{len(train_loader)} | Total: {loss.item():.4f} (cls: {losses['cls'].item():.3f}, reg: {losses['reg'].item():.3f}, ctr: {losses['ctr'].item():.3f})", end="\r")

        if epoch > args.warmup_epochs:
            scheduler.step()

        N_b = max(len(train_loader), 1)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch:2d}/{args.epochs}] loss={ep_loss/N_b:.4f} cls={ep_cls/N_b:.4f} reg={ep_reg/N_b:.4f} ctr={ep_ctr/N_b:.4f} lr={lr_now:.6f} ({elapsed:.1f}s)")

        # Validation mAP_tiny evaluation
        map50, map50_95 = evaluate_map_tiny(model, val_loader, device, val_ann)
        print(f"         mAP_tiny@0.5: {map50:.4f} | mAP_tiny@0.5:0.95: {map50_95:.4f}")

        # Checkpoint saving
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "fpn_channels": model.fpn_channels,
            "map50": map50,
            "map50_95": map50_95,
        }
        torch.save(ckpt, os.path.join(args.weights_dir, "last_model.pth"))
        if map50 > best_map:
            best_map = map50
            torch.save(ckpt, os.path.join(args.weights_dir, "best_model.pth"))
            print(f"         * Saved new best checkpoint (mAP@0.5: {best_map:.4f})")

    print(f"\n[train] Complete. Best mAP_tiny@0.5: {best_map:.4f} saved in {args.weights_dir}/")


if __name__ == "__main__":
    main()