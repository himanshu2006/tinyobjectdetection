"""
train_2.py - Production Linux-Optimized Training Pipeline for Aerial Detection
=============================================================================
Critical Fixes & Architectural Enhancements:
  - Real Scale Jittering: Jitter + crop/pad to fixed canvas (no letterbox cancellation)
  - Zero Train/Val Leakage: Splits tiles strictly by underlying source image ID
  - Decoupled Head: GroupNorm(32, 128) + Learnable scale-indexed exp() box regression
  - Robust Vectorized Assigner: Soft-overlapping bands + Nearest-Center GT fallback
  - Scale-Normalized NWD: Dynamic Gaussian similarity matching target diagonals
  - Mixed Precision Safety: FP32 loss computation under AMP + Atomic checkpointing
  - Pre-NMS Top-K filtering (prevents CPU memory stalls on dense grids)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    raise ImportError("pycocotools is required for evaluation. Run: pip install pycocotools")


# ============================================================================
# 1. EARLY STOPPING HANDLER
# ============================================================================

class EarlyStopping:
    """Monitors validation metrics with warmup grace period and atomic state."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, min_epoch: int = 15) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.min_epoch = min_epoch
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False

    def step(self, score: float, current_epoch: int) -> bool:
        if current_epoch < self.min_epoch:
            if self.best_score is None or score > self.best_score:
                self.best_score = score
            return False

        if self.best_score is None:
            self.best_score = score
            return False

        if score > (self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


# ============================================================================
# 2. AUTOMATED DATASET SLICING
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
    sliced_path = Path(sliced_dir)
    sliced_ann_path = sliced_path / "sliced_coco.json"
    sliced_img_dir = sliced_path / "images"

    # Integrity verification: json exists, dir exists, and images match JSON count
    if sliced_ann_path.is_file() and sliced_img_dir.is_dir():
        try:
            with open(sliced_ann_path, "r") as f:
                meta = json.load(f)
            imgs = [f for f in os.listdir(sliced_img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            if len(imgs) > 0 and len(imgs) == len(meta.get("images", [])):
                print(f"[slice] Found validated sliced dataset at {sliced_dir} ({len(imgs)} tiles) -- skipping.")
                return str(sliced_img_dir), str(sliced_ann_path)
        except Exception:
            print("[slice] Existing sliced dataset corrupt or incomplete. Re-slicing...")

    print(f"[slice] Slicing {raw_img_dir} -> {sliced_dir} (tile={slice_size}, overlap={overlap_ratio})")
    from sahi.slicing import slice_coco

    slice_coco(
        coco_annotation_file_path=raw_ann,
        image_dir=raw_img_dir,
        output_coco_annotation_file_name="sliced",
        output_dir=sliced_dir,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        min_area_ratio=min_area_ratio,
        verbose=False,
    )

    sliced_img_dir.mkdir(parents=True, exist_ok=True)
    for f in os.listdir(sliced_dir):
        src = sliced_path / f
        if src.is_file() and f.lower().endswith((".jpg", ".png", ".jpeg")):
            shutil.move(str(src), str(sliced_img_dir / f))

    print(f"[slice] Slicing completed. Tiles saved in {sliced_img_dir}")
    return str(sliced_img_dir), str(sliced_ann_path)


# ============================================================================
# 3. DATASET & FIXED-CANVAS TINY-OBJECT AUGMENTATION
# ============================================================================

class SlicedDroneDataset(Dataset):
    """Loads sliced COCO tiles with canvas placement and true scale jittering."""

    def __init__(self, img_dir: str, ann_path: str, input_size: int = 640, augment: bool = True) -> None:
        super().__init__()
        self.img_dir = Path(img_dir)
        self.input_size = input_size
        self.augment = augment

        with open(ann_path, "r") as f:
            coco_json = json.load(f)

        self.categories = coco_json.get("categories", [{"id": 1, "name": "person"}])
        self.cat_to_idx = {c["id"]: i for i, c in enumerate(self.categories)}
        self.num_classes = len(self.categories)
        self.images_meta = coco_json["images"]

        # Parse source image identifiers to prevent data leakage across slices
        self.source_image_ids: List[Any] = []
        for m in self.images_meta:
            fname = Path(m["file_name"]).stem
            # SAHI standard naming: <orig_id>_<x>_<y>_<w>_<h>
            parts = fname.split("_")
            self.source_image_ids.append(parts[0] if len(parts) > 1 else m.get("original_image_id", m["id"]))

        # Pre-pack annotations into contiguous float arrays to avoid refcount memory bloat
        self.anns_by_img: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for ann in coco_json.get("annotations", []):
            if ann.get("iscrowd", 0) == 1:
                continue
            x, y, bw, bh = ann["bbox"]
            if bw < 2 or bh < 2:
                continue
            iid = ann["image_id"]
            box = [x, y, x + bw, y + bh]
            cat = self.cat_to_idx.get(ann["category_id"], 0)
            if iid not in self.anns_by_img:
                self.anns_by_img[iid] = ([box], [cat])
            else:
                self.anns_by_img[iid][0].append(box)
                self.anns_by_img[iid][1].append(cat)

        self.packed_anns: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for iid, (b_list, c_list) in self.anns_by_img.items():
            self.packed_anns[iid] = (
                np.array(b_list, dtype=np.float32).reshape(-1, 4),
                np.array(c_list, dtype=np.int64),
            )

    def __len__(self) -> int:
        return len(self.images_meta)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.images_meta[idx]
        img_id = meta["id"]

        fname = Path(meta["file_name"]).name
        img_path = self.img_dir / fname
        if not img_path.is_file():
            img_path = self.img_dir / meta["file_name"]

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        if img_id in self.packed_anns:
            boxes = self.packed_anns[img_id][0].copy()
            labels = self.packed_anns[img_id][1].copy()
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        canvas = Image.new("RGB", (self.input_size, self.input_size), (114, 114, 114))

        if self.augment:
            # 1. Scale Jittering [0.5, 1.5]
            scale = random.uniform(0.5, 1.5)
            nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
            img = img.resize((nw, nh), Image.BILINEAR)
            if len(boxes) > 0:
                boxes *= scale

            # 2. Random Crop or Pad placement directly on canvas (No Letterbox Inversion)
            if nw > self.input_size:
                x_offset = random.randint(0, nw - self.input_size)
                paste_x = 0
                crop_x1 = x_offset
                crop_x2 = x_offset + self.input_size
            else:
                x_offset = 0
                paste_x = random.randint(0, self.input_size - nw)
                crop_x1 = 0
                crop_x2 = nw

            if nh > self.input_size:
                y_offset = random.randint(0, nh - self.input_size)
                paste_y = 0
                crop_y1 = y_offset
                crop_y2 = y_offset + self.input_size
            else:
                y_offset = 0
                paste_y = random.randint(0, self.input_size - nh)
                crop_y1 = 0
                crop_y2 = nh

            img_cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            canvas.paste(img_cropped, (paste_x, paste_y))

            if len(boxes) > 0:
                # Coordinate translation
                boxes[:, [0, 2]] = boxes[:, [0, 2]] - crop_x1 + paste_x
                boxes[:, [1, 3]] = boxes[:, [1, 3]] - crop_y1 + paste_y
                # Box integrity clipping
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.input_size)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.input_size)

                # Discard cropped degenerate boxes
                bw = boxes[:, 2] - boxes[:, 0]
                bh = boxes[:, 3] - boxes[:, 1]
                valid = (bw >= 2.0) & (bh >= 2.0)
                boxes = boxes[valid]
                labels = labels[valid]

            # 3. Flips and Photometric Distortions
            if random.random() < 0.5:
                canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)
                if len(boxes) > 0:
                    x1 = boxes[:, 0].copy()
                    boxes[:, 0] = self.input_size - boxes[:, 2]
                    boxes[:, 2] = self.input_size - x1

            if random.random() < 0.5:
                canvas = canvas.transpose(Image.FLIP_TOP_BOTTOM)
                if len(boxes) > 0:
                    y1 = boxes[:, 1].copy()
                    boxes[:, 1] = self.input_size - boxes[:, 3]
                    boxes[:, 3] = self.input_size - y1

            # Mild photometric jitter
            arr = np.array(canvas, dtype=np.float32)
            contrast = random.uniform(0.85, 1.15)
            arr = np.clip((arr - 128.0) * contrast + 128.0, 0, 255)
            brightness = random.uniform(-20, 20)
            arr = np.clip(arr + brightness, 0, 255)
            canvas = Image.fromarray(arr.astype(np.uint8))

        else:
            # Deterministic evaluation letterbox
            r = min(self.input_size / w, self.input_size / h)
            nw, nh = int(round(w * r)), int(round(h * r))
            img = img.resize((nw, nh), Image.BILINEAR)
            pad_x = (self.input_size - nw) // 2
            pad_y = (self.input_size - nh) // 2
            canvas.paste(img, (pad_x, pad_y))

            if len(boxes) > 0:
                boxes *= r
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] + pad_x, 0, self.input_size)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] + pad_y, 0, self.input_size)

        img_tensor = torch.from_numpy(np.array(canvas, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return {
            "image": img_tensor,
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "image_id": img_id,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": torch.stack([s["image"] for s in batch], dim=0),
        "boxes": [s["boxes"] for s in batch],
        "labels": [s["labels"] for s in batch],
        "image_ids": [s["image_id"] for s in batch],
    }


# ============================================================================
# 4. NEURAL ARCHITECTURE (GROUPNORM & SCALE-INDEXED HEAD)
# ============================================================================

class ConvGNAct(nn.Module):
    """Conv2d -> GroupNorm(32) -> SiLU (Independent of batch size and tensor shape)."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.gn = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class ConvBNSiLU(nn.Module):
    """Conv2d -> BatchNorm2d -> SiLU (Used strictly in backbone feature extraction)."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CSPBottleneck(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBNSiLU(channels, channels, k=3, s=1, p=1)
        self.conv2 = ConvBNSiLU(channels, channels, k=3, s=1, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class CSPBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 1) -> None:
        super().__init__()
        mid_ch = out_ch // 2
        self.conv_main = ConvBNSiLU(in_ch, mid_ch, k=1, s=1, p=0)
        self.conv_bypass = ConvBNSiLU(in_ch, mid_ch, k=1, s=1, p=0)
        self.bottlenecks = nn.Sequential(*[CSPBottleneck(mid_ch) for _ in range(num_blocks)])
        self.conv_out = ConvBNSiLU(mid_ch * 2, out_ch, k=1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_feat = self.bottlenecks(self.conv_main(x))
        bypass_feat = self.conv_bypass(x)
        return self.conv_out(torch.cat([main_feat, bypass_feat], dim=1))


class CSPDarknetBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Stem: Two stride-2 passes prevent blind sampling
        self.stem1 = ConvBNSiLU(3, 32, k=3, s=2, p=1)   # 640 -> 320
        self.stem2 = ConvBNSiLU(32, 64, k=3, s=2, p=1)  # 320 -> 160

        self.c2_csp = CSPBlock(in_ch=64, out_ch=64, num_blocks=1)
        self.c3_down = ConvBNSiLU(64, 128, k=3, s=2, p=1)
        self.c3_csp = CSPBlock(in_ch=128, out_ch=128, num_blocks=2)
        self.c4_down = ConvBNSiLU(128, 256, k=3, s=2, p=1)
        self.c4_csp = CSPBlock(in_ch=256, out_ch=256, num_blocks=2)
        self.c5_down = ConvBNSiLU(256, 512, k=3, s=2, p=1)
        self.c5_csp = CSPBlock(in_ch=512, out_ch=512, num_blocks=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem2(self.stem1(x))
        c2 = self.c2_csp(x)
        c3 = self.c3_csp(self.c3_down(c2))
        c4 = self.c4_csp(self.c4_down(c3))
        c5 = self.c5_csp(self.c5_down(c4))
        return c2, c3, c4, c5


class TinyFPN(nn.Module):
    def __init__(self, fpn_ch: int = 128) -> None:
        super().__init__()
        self.lat2 = nn.Conv2d(64, fpn_ch, 1)
        self.lat3 = nn.Conv2d(128, fpn_ch, 1)
        self.lat4 = nn.Conv2d(256, fpn_ch, 1)
        self.lat5 = nn.Conv2d(512, fpn_ch, 1)

        self.smooth2 = ConvGNAct(fpn_ch, fpn_ch, 3, 1, 1)
        self.smooth3 = ConvGNAct(fpn_ch, fpn_ch, 3, 1, 1)
        self.smooth4 = ConvGNAct(fpn_ch, fpn_ch, 3, 1, 1)
        self.smooth5 = ConvGNAct(fpn_ch, fpn_ch, 3, 1, 1)

    def forward(
        self, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        l5, l4, l3, l2 = self.lat5(c5), self.lat4(c4), self.lat3(c3), self.lat2(c2)

        p5 = self.smooth5(l5)
        # Consistent top-down pipeline
        p4 = self.smooth4(l4 + F.interpolate(p5, size=l4.shape[2:], mode="nearest"))
        p3 = self.smooth3(l3 + F.interpolate(p4, size=l3.shape[2:], mode="nearest"))
        p2 = self.smooth2(l2 + F.interpolate(p3, size=l2.shape[2:], mode="nearest"))
        return p2, p3, p4, p5


class ScaleExp(nn.Module):
    """Learnable per-level exponential scaling: exp(scale * x) * stride."""

    def __init__(self, init_scale: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor, stride: int) -> torch.Tensor:
        # Clamped exponent prevents FP16 gradient explosion
        return torch.exp((x * self.scale).clamp(-8.0, 8.0)) * stride


class DecoupledHead(nn.Module):
    """GroupNorm-regularized decoupled prediction head with scale-indexed regression."""

    def __init__(self, fpn_ch: int = 128, num_classes: int = 1, num_convs: int = 4) -> None:
        super().__init__()
        self.cls_branch = nn.Sequential(*[ConvGNAct(fpn_ch, fpn_ch, 3, 1, 1) for _ in range(num_convs)])
        self.cls_out = nn.Conv2d(fpn_ch, num_classes, 3, 1, 1)

        self.reg_branch = nn.Sequential(*[ConvGNAct(fpn_ch, fpn_ch, 3, 1, 1) for _ in range(num_convs)])
        self.reg_out = nn.Conv2d(fpn_ch, 4, 3, 1, 1)

        # 4 learnable scalars for levels P2, P3, P4, P5
        self.scale_reg = nn.ModuleList([ScaleExp(init_scale=1.0) for _ in range(4)])

        # Prior bias initialization: p = 0.01 on epoch 0
        nn.init.constant_(self.cls_out.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, feats: List[torch.Tensor], strides: List[int]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        outputs = []
        for li, feat in enumerate(feats):
            cls_feat = self.cls_branch(feat)
            reg_feat = self.reg_branch(feat)

            cls_logits = self.cls_out(cls_feat)
            raw_reg = self.reg_out(reg_feat)
            scaled_reg = self.scale_reg[li](raw_reg, strides[li])
            outputs.append((cls_logits, scaled_reg))
        return outputs


class TinyDroneDetector(nn.Module):
    STRIDES = [4, 8, 16, 32]

    def __init__(self, num_classes: int = 1, fpn_channels: int = 128) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fpn_channels = fpn_channels
        self.backbone = CSPDarknetBackbone()
        self.fpn = TinyFPN(fpn_ch=fpn_channels)
        self.head = DecoupledHead(fpn_ch=fpn_channels, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        c2, c3, c4, c5 = self.backbone(x)
        p2, p3, p4, p5 = self.fpn(c2, c3, c4, c5)
        return self.head([p2, p3, p4, p5], self.STRIDES)


# ============================================================================
# 5. VECTORIZED ASSIGNER (SOFT-BANDS & NEAREST-CENTER FALLBACK)
# ============================================================================

class ScaleAwareAssigner:
    """Fully vectorized anchor-free assigner with guaranteed sub-6px target fallback."""

    def __init__(
        self,
        strides: List[int] = [4, 8, 16, 32],
        scale_ranges: List[Tuple[float, float]] = [
            (0.0, 24.0),   # P2
            (12.0, 48.0),  # P3
            (24.0, 96.0),  # P4
            (48.0, 1e5),   # P5
        ],
    ) -> None:
        self.strides = strides
        self.scale_ranges = scale_ranges

    @torch.no_grad()
    def assign(
        self,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        feat_sizes: List[Tuple[int, int]],
        grid_centers: List[Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> List[Dict[str, torch.Tensor]]:
        num_levels = len(self.strides)
        N = gt_boxes.shape[0]

        targets = []
        if N == 0:
            for li in range(num_levels):
                H_l, W_l = feat_sizes[li]
                num_cells = H_l * W_l
                targets.append({
                    "cls": torch.full((num_cells,), -1, dtype=torch.long, device=device),
                    "reg": torch.zeros((num_cells, 4), dtype=torch.float32, device=device),
                    "pos": torch.zeros((num_cells,), dtype=torch.bool, device=device),
                    "gt_idx": torch.full((num_cells,), -1, dtype=torch.long, device=device),
                })
            return targets

        gt_w = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1e-4)
        gt_h = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1e-4)
        sizes = torch.sqrt(gt_w * gt_h)

        # Primary level candidate assignment
        assigned_level = torch.zeros((N,), dtype=torch.long, device=device)
        for li in range(num_levels):
            min_s, max_s = self.scale_ranges[li]
            assigned_level[(sizes >= min_s) & (sizes <= max_s)] = li

        for li in range(num_levels):
            H_l, W_l = feat_sizes[li]
            num_cells = H_l * W_l
            grid_x, grid_y = grid_centers[li]  # (num_cells,)

            cls_tgt = torch.full((num_cells,), -1, dtype=torch.long, device=device)
            reg_tgt = torch.zeros((num_cells, 4), dtype=torch.float32, device=device)
            assigned_gt_idx = torch.full((num_cells,), -1, dtype=torch.long, device=device)
            best_proximity = torch.zeros((num_cells,), dtype=torch.float32, device=device)

            # Mask GTs belonging to this level
            min_s, max_s = self.scale_ranges[li]
            lvl_mask = (sizes >= min_s) & (sizes <= max_s)
            lvl_indices = lvl_mask.nonzero(as_tuple=True)[0]

            if len(lvl_indices) > 0:
                sub_boxes = gt_boxes[lvl_indices]    # (M, 4)
                sub_labels = gt_labels[lvl_indices]  # (M,)

                # Vectorized point-in-box tensor [M, num_cells]
                gx = grid_x.unsqueeze(0)  # (1, num_cells)
                gy = grid_y.unsqueeze(0)

                l = gx - sub_boxes[:, 0].unsqueeze(1)
                t = gy - sub_boxes[:, 1].unsqueeze(1)
                r = sub_boxes[:, 2].unsqueeze(1) - gx
                b = sub_boxes[:, 3].unsqueeze(1) - gy

                inside = (l > 0) & (t > 0) & (r > 0) & (b > 0)  # (M, num_cells)

                # Centerness proximity metric for conflict resolution
                lr = torch.min(l, r) / torch.max(l, r).clamp(min=1e-6)
                tb = torch.min(t, b) / torch.max(t, b).clamp(min=1e-6)
                prox = torch.sqrt((lr * tb).clamp(min=0.0, max=1.0)) * inside.float()

                # CRITICAL FALLBACK: Ensure sub-6px micro-targets get at least their nearest cell
                has_pos = inside.any(dim=1)
                for mi in range(len(lvl_indices)):
                    if not has_pos[mi]:
                        # Identify nearest grid center
                        gc_x = (sub_boxes[mi, 0] + sub_boxes[mi, 2]) * 0.5
                        gc_y = (sub_boxes[mi, 1] + sub_boxes[mi, 3]) * 0.5
                        dists = (grid_x - gc_x) ** 2 + (grid_y - gc_y) ** 2
                        nearest_cell = torch.argmin(dists)
                        inside[mi, nearest_cell] = True
                        prox[mi, nearest_cell] = 1.0

                # Resolve multi-GT spatial competition across grid cells
                max_prox, max_gt_sub_idx = torch.max(prox, dim=0)  # (num_cells,)
                pos_cells = max_prox > 0

                if pos_cells.any():
                    chosen_gts = lvl_indices[max_gt_sub_idx[pos_cells]]
                    cls_tgt[pos_cells] = gt_labels[chosen_gts]
                    assigned_gt_idx[pos_cells] = chosen_gts

                    chosen_boxes = gt_boxes[chosen_gts]
                    p_gx = grid_x[pos_cells]
                    p_gy = grid_y[pos_cells]

                    reg_tgt[pos_cells] = torch.stack([
                        (p_gx - chosen_boxes[:, 0]).clamp(min=0.0),
                        (p_gy - chosen_boxes[:, 1]).clamp(min=0.0),
                        (chosen_boxes[:, 2] - p_gx).clamp(min=0.0),
                        (chosen_boxes[:, 3] - p_gy).clamp(min=0.0),
                    ], dim=1)

            targets.append({
                "cls": cls_tgt,
                "reg": reg_tgt,
                "pos": cls_tgt >= 0,
                "gt_idx": assigned_gt_idx,
            })
        return targets


# ============================================================================
# 6. SCALE-NORMALIZED LOSS OBJECTIVES (QFL + ADAPTIVE NWD)
# ============================================================================

def quality_focal_loss(pred_logits: torch.Tensor, target_score: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    pred_prob = torch.sigmoid(pred_logits)
    scale = (pred_prob - target_score).abs().pow(beta)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_score, reduction="none")
    return (scale * bce).sum()


def giou_loss(p_boxes: torch.Tensor, g_boxes: torch.Tensor) -> torch.Tensor:
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


def scale_normalized_nwd(p_boxes: torch.Tensor, g_boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes Normalized Wasserstein Distance with scale-adaptive diagonal calibration."""
    px_c, py_c = (p_boxes[:, 0] + p_boxes[:, 2]) * 0.5, (p_boxes[:, 1] + p_boxes[:, 3]) * 0.5
    pw, ph = p_boxes[:, 2] - p_boxes[:, 0], p_boxes[:, 3] - p_boxes[:, 1]

    gx_c, gy_c = (g_boxes[:, 0] + g_boxes[:, 2]) * 0.5, (g_boxes[:, 1] + g_boxes[:, 3]) * 0.5
    gw, gh = (g_boxes[:, 2] - g_boxes[:, 0]).clamp(min=1e-4), (g_boxes[:, 3] - g_boxes[:, 1]).clamp(min=1e-4)

    # Scale-adaptive C parameter normalized to base 16px diagonal
    target_diag = torch.sqrt(gw ** 2 + gh ** 2)
    C_adaptive = (12.8 * (target_diag / 16.0)).clamp(min=6.0, max=128.0)

    w2_sq = (px_c - gx_c) ** 2 + (py_c - gy_c) ** 2 + ((pw - gw) ** 2 + (ph - gh) ** 2) / 4.0
    nwd_sim = torch.exp(-torch.sqrt(w2_sq.clamp(min=1e-7)) / C_adaptive)
    loss = (1.0 - nwd_sim).mean()
    return loss, nwd_sim


class TinyDetectionLoss(nn.Module):
    """Multi-task loss: FP32-stabilized QFL with Scale-Normalized Wasserstein Regression."""

    def __init__(self, num_classes: int = 1, strides: List[int] = [4, 8, 16, 32]) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.assigner = ScaleAwareAssigner(strides=strides)

    def forward(
        self,
        outputs: List[Tuple[torch.Tensor, torch.Tensor]],
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = outputs[0][0].device
        B = outputs[0][0].shape[0]
        L = len(outputs)

        feat_sizes = [(outputs[li][0].shape[2], outputs[li][0].shape[3]) for li in range(L)]
        grid_centers = []
        for li in range(L):
            H_l, W_l = feat_sizes[li]
            s = self.strides[li]
            sy = (torch.arange(0, H_l, device=device, dtype=torch.float32) + 0.5) * s
            sx = (torch.arange(0, W_l, device=device, dtype=torch.float32) + 0.5) * s
            gy, gx = torch.meshgrid(sy, sx, indexing="ij")
            grid_centers.append((gx.reshape(-1), gy.reshape(-1)))

        tot_cls = torch.tensor(0.0, device=device, dtype=torch.float32)
        tot_reg = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_pos = 0

        for b in range(B):
            b_boxes = gt_boxes[b].to(device)
            b_labels = gt_labels[b].to(device)
            targets = self.assigner.assign(b_boxes, b_labels, feat_sizes, grid_centers, device)

            for li in range(L):
                # Always compute loss in FP32
                cls_logits = outputs[li][0][b].float().permute(1, 2, 0).reshape(-1, self.num_classes)
                reg_pred = outputs[li][1][b].float().permute(1, 2, 0).reshape(-1, 4)

                tgt = targets[li]
                pos_mask = tgt["pos"]
                n_pos = pos_mask.sum().item()

                target_score = torch.zeros_like(cls_logits)

                if n_pos > 0:
                    total_pos += n_pos
                    gx, gy = grid_centers[li]
                    cx, cy = gx[pos_mask], gy[pos_mask]

                    p_ltrb = reg_pred[pos_mask]
                    p_xyxy = torch.stack([cx - p_ltrb[:, 0], cy - p_ltrb[:, 1], cx + p_ltrb[:, 2], cy + p_ltrb[:, 3]], dim=1)

                    t_ltrb = tgt["reg"][pos_mask]
                    t_xyxy = torch.stack([cx - t_ltrb[:, 0], cy - t_ltrb[:, 1], cx + t_ltrb[:, 2], cy + t_ltrb[:, 3]], dim=1)

                    loss_giou = giou_loss(p_xyxy, t_xyxy)
                    loss_nwd_val, nwd_sim = scale_normalized_nwd(p_xyxy, t_xyxy)
                    tot_reg = tot_reg + (0.5 * loss_giou + 0.5 * loss_nwd_val) * n_pos

                    # Safe soft-label assignment
                    pos_idx = pos_mask.nonzero(as_tuple=True)[0]
                    target_score[pos_idx, tgt["cls"][pos_mask]] = nwd_sim.detach().clamp(0.0, 1.0).to(target_score.dtype)

                tot_cls = tot_cls + quality_focal_loss(cls_logits, target_score)

        # Smooth normalization protects against zero-foreground batches
        norm = max(total_pos, B)
        loss_cls = tot_cls / norm
        loss_reg = tot_reg / norm
        total = loss_cls + loss_reg

        return {"total": total, "cls": loss_cls.detach(), "reg": loss_reg.detach()}


# ============================================================================
# 7. DECODER & VALIDATION ENGINE
# ============================================================================

def decode_predictions(
    outputs: List[Tuple[torch.Tensor, torch.Tensor]],
    strides: List[int],
    score_thr: float = 0.05,
    nms_thr: float = 0.5,
    pre_nms_topk: int = 1000,
    max_per_img: int = 300,
) -> List[Dict[str, torch.Tensor]]:
    """Decodes predictions with Pre-NMS Top-K filtering to eliminate CPU bottlenecks."""
    device = outputs[0][0].device
    B = outputs[0][0].shape[0]
    from torchvision.ops import batched_nms

    results = []
    for b in range(B):
        all_boxes, all_scores, all_labels = [], [], []
        for li, s in enumerate(strides):
            cls_logits, reg_pred = outputs[li][0][b], outputs[li][1][b]
            C, H_l, W_l = cls_logits.shape

            sy = (torch.arange(0, H_l, device=device, dtype=torch.float32) + 0.5) * s
            sx = (torch.arange(0, W_l, device=device, dtype=torch.float32) + 0.5) * s
            gy, gx = torch.meshgrid(sy, sx, indexing="ij")
            gx, gy = gx.reshape(-1), gy.reshape(-1)

            scores = torch.sigmoid(cls_logits.permute(1, 2, 0).reshape(-1, C))
            max_s, max_c = scores.max(dim=1)
            keep = max_s > score_thr
            if not keep.any():
                continue

            # Pre-NMS candidate filtering per level
            if keep.sum() > pre_nms_topk:
                topk_idx = torch.topk(max_s[keep], pre_nms_topk)[1]
                keep_indices = keep.nonzero(as_tuple=True)[0][topk_idx]
            else:
                keep_indices = keep.nonzero(as_tuple=True)[0]

            k_reg = reg_pred.permute(1, 2, 0).reshape(-1, 4)[keep_indices]
            cx, cy = gx[keep_indices], gy[keep_indices]

            boxes = torch.stack([
                (cx - k_reg[:, 0]).clamp(min=0.0, max=640.0),
                (cy - k_reg[:, 1]).clamp(min=0.0, max=640.0),
                (cx + k_reg[:, 2]).clamp(min=0.0, max=640.0),
                (cy + k_reg[:, 3]).clamp(min=0.0, max=640.0),
            ], dim=1)

            all_boxes.append(boxes)
            all_scores.append(max_s[keep_indices])
            all_labels.append(max_c[keep_indices])

        if len(all_boxes) == 0:
            results.append({
                "boxes": torch.zeros((0, 4), device=device),
                "scores": torch.zeros((0,), device=device),
                "labels": torch.zeros((0,), dtype=torch.long, device=device),
            })
            continue

        c_boxes = torch.cat(all_boxes, dim=0)
        c_scores = torch.cat(all_scores, dim=0)
        c_labels = torch.cat(all_labels, dim=0)
        keep_idx = batched_nms(c_boxes, c_scores, c_labels, nms_thr)[:max_per_img]
        results.append({"boxes": c_boxes[keep_idx], "scores": c_scores[keep_idx], "labels": c_labels[keep_idx]})
    return results


@torch.no_grad()
def evaluate_map_tiny(model: nn.Module, val_loader: DataLoader, device: torch.device, coco_gt: COCO) -> Tuple[float, float]:
    model.eval()
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
            boxes = det["boxes"].cpu().numpy()
            scores = det["scores"].cpu().numpy()
            labels = det["labels"].cpu().numpy()
            for k in range(len(boxes)):
                x1, y1, x2, y2 = boxes[k]
                cls_idx = int(labels[k])
                target_cat = cat_ids[cls_idx] if cls_idx < len(cat_ids) else cat_ids[0]
                coco_results.append({
                    "image_id": int(image_ids[bi]),
                    "category_id": int(target_cat),
                    "bbox": [float(x1), float(y1), float(max(x2 - x1, 0.1)), float(max(y2 - y1, 0.1))],
                    "score": float(scores[k]),
                })

    if len(coco_results) == 0:
        return 0.0, 0.0

    coco_dt = coco_gt.loadRes(coco_results)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = sorted(list(val_image_ids))
    evaluator.params.areaRng = [
        [0 ** 2, 1e5 ** 2],
        [0 ** 2, 256],  # Tiny objects (<16x16)
        [32 ** 2, 96 ** 2],
        [96 ** 2, 1e5 ** 2],
    ]
    evaluator.params.areaRngLbl = ["all", "small", "medium", "large"]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    try:
        p_tiny_50 = evaluator.eval["precision"][0, :, :, 1, -1]
        map_50 = float(np.mean(p_tiny_50[p_tiny_50 > -1])) if (p_tiny_50 > -1).any() else 0.0

        p_tiny_all = evaluator.eval["precision"][:, :, :, 1, -1]
        map_50_95 = float(np.mean(p_tiny_all[p_tiny_all > -1])) if (p_tiny_all > -1).any() else 0.0
    except Exception:
        map_50 = float(evaluator.stats[1]) if len(evaluator.stats) > 1 and evaluator.stats[1] >= 0 else 0.0
        map_50_95 = float(evaluator.stats[0]) if len(evaluator.stats) > 0 and evaluator.stats[0] >= 0 else 0.0

    return map_50, map_50_95


# ============================================================================
# 8. TRAINING ENGINE
# ============================================================================

def main() -> None:
    # Reproducible seeding across all subsystems
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # Fixed 640x640 shape optimization

    default_workers = min(8, os.cpu_count() or 4)

    parser = argparse.ArgumentParser(description="Production Tiny Drone Detector (Linux / Multi-Process Optimized)")
    parser.add_argument("--raw_img_dir", type=str, default="VisDrone2019-DET-train/images")
    parser.add_argument("--raw_ann", type=str, default="VisDrone2019-DET-train/train_coco.json")
    parser.add_argument("--val_img_dir", type=str, default="")
    parser.add_argument("--val_ann", type=str, default="")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--sliced_dir", type=str, default="data/sliced_train")
    parser.add_argument("--sliced_val_dir", type=str, default="data/sliced_val")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--input_size", type=int, default=640)
    parser.add_argument("--num_workers", type=int, default=default_workers)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=6)
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[train] Running on: {device} | CUDA Benchmark: Active | Workers: {args.num_workers}")

    sliced_img_dir, sliced_ann = check_and_slice_dataset(
        args.raw_img_dir, args.raw_ann, args.sliced_dir,
        slice_size=args.input_size, overlap_ratio=0.2, min_area_ratio=0.1
    )

    full_ds = SlicedDroneDataset(sliced_img_dir, sliced_ann, input_size=args.input_size, augment=True)
    num_classes = full_ds.num_classes

    # PREVENT TRAIN/VAL LEAKAGE: Group tiles by root image ID before partitioning
    if args.val_img_dir and args.val_ann:
        val_img_dir, val_ann = check_and_slice_dataset(
            args.val_img_dir, args.val_ann, args.sliced_val_dir,
            slice_size=args.input_size, overlap_ratio=0.2, min_area_ratio=0.1
        )
        val_ds = SlicedDroneDataset(val_img_dir, val_ann, input_size=args.input_size, augment=False)
        train_ds = full_ds
    else:
        # Group indices by root image ID
        groups: Dict[Any, List[int]] = {}
        for idx, src_id in enumerate(full_ds.source_image_ids):
            groups.setdefault(src_id, []).append(idx)

        unique_src_ids = list(groups.keys())
        random.shuffle(unique_src_ids)

        n_val_src = max(int(len(unique_src_ids) * args.val_split), 1)
        val_src_set = set(unique_src_ids[:n_val_src])

        train_indices, val_indices = [], []
        for src_id, tile_idxs in groups.items():
            if src_id in val_src_set:
                val_indices.extend(tile_idxs)
            else:
                train_indices.extend(tile_idxs)

        train_ds = Subset(full_ds, train_indices)
        val_ds = Subset(
            SlicedDroneDataset(sliced_img_dir, sliced_ann, input_size=args.input_size, augment=False),
            val_indices
        )
        val_ann = sliced_ann

    print(f"[train] Leakage-Free Partition: {len(train_ds)} train tiles | {len(val_ds)} val tiles")

    # Shared GT object loaded once
    coco_gt = COCO(val_ann)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": (device.type == "cuda"),
        "persistent_workers": (args.num_workers > 0),
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
    }

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, **loader_kwargs
    )

    model = TinyDroneDetector(num_classes=num_classes).to(device)
    criterion = TinyDetectionLoss(num_classes=num_classes, strides=TinyDroneDetector.STRIDES)

    # Exclude 1D parameters (biases, Norm weights) from weight decay
    decay_params, no_decay_params = [], []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            decay_params.append(m.weight)
            if m.bias is not None:
                no_decay_params.append(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                no_decay_params.append(m.weight)
            if m.bias is not None:
                no_decay_params.append(m.bias)
    for p in model.head.scale_reg.parameters():
        no_decay_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr)

    # Corrected Cosine Schedule: Anneals across actual post-warmup epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=args.lr * 0.01
    )

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    warmup_steps = args.warmup_epochs * len(train_loader)
    global_step = 0
    start_epoch = 1
    best_map = -1.0
    os.makedirs(args.weights_dir, exist_ok=True)

    early_stopper = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=1e-4,
        min_epoch=max(args.warmup_epochs + 5, 15)  # Grace period prevents early schedule cutoffs
    )

    # Resume capability
    if args.resume and os.path.isfile(args.resume):
        print(f"[train] Resuming state from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", start_epoch * len(train_loader))
        best_map = ckpt.get("best_map", -1.0)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        ep_loss, ep_cls, ep_reg = 0.0, 0.0, 0.0
        steps_trained = 0
        t0 = time.time()

        for batch in train_loader:
            global_step += 1
            if global_step <= warmup_steps:
                warmup_lr = args.lr * max(global_step / max(warmup_steps, 1), 1e-4)
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr

            imgs = batch["images"].to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(imgs)
                losses = criterion(outputs, batch["boxes"], batch["labels"])
                loss = losses["total"]

            if not torch.isfinite(loss):
                print(f"[train] Warning: Non-finite loss at step {global_step}, skipping.")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            ep_loss += loss.item()
            ep_cls += losses["cls"].item()
            ep_reg += losses["reg"].item()
            steps_trained += 1

            if global_step % 20 == 0:
                cur_b = (global_step - 1) % len(train_loader) + 1
                print(f"[Epoch {epoch:2d}] Batch {cur_b:4d}/{len(train_loader)} | Total: {loss.item():.4f} (cls: {losses['cls'].item():.3f}, reg: {losses['reg'].item():.3f})", end="\r")

        if epoch > args.warmup_epochs:
            scheduler.step()

        N_b = max(steps_trained, 1)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        avg_loss = ep_loss / N_b
        print(f"\r[Epoch {epoch:2d}/{args.epochs}] Loss={avg_loss:.4f} Cls={ep_cls/N_b:.4f} Reg={ep_reg/N_b:.4f} LR={lr_now:.6f} ({elapsed:.1f}s)")

        map50, map50_95 = evaluate_map_tiny(model, val_loader, device, coco_gt)
        print(f"         mAP_tiny@0.5: {map50:.4f} | mAP_tiny@0.5:0.95: {map50_95:.4f}")

        # Atomic checkpoint dictionary
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "num_classes": num_classes,
            "best_map": best_map,
            "map50": map50,
            "map50_95": map50_95,
            "loss": avg_loss,
        }

        # Atomic saving prevents checkpoint corruption during unexpected interrupts
        def save_atomic(state: Dict[str, Any], path: Path) -> None:
            tmp_path = path.with_suffix(".tmp")
            torch.save(state, tmp_path)
            tmp_path.replace(path)

        weights_p = Path(args.weights_dir)
        save_atomic(ckpt, weights_p / "last_model.pth")

        if map50 > best_map:
            best_map = map50
            ckpt["best_map"] = best_map
            save_atomic(ckpt, weights_p / "best_model.pth")
            print(f"         * New best checkpoint saved (mAP@0.5: {best_map:.4f})")

        if epoch % args.save_interval == 0:
            save_atomic(ckpt, weights_p / f"epoch_{epoch}.pth")
            print(f"         * Periodic checkpoint saved: epoch_{epoch}.pth")

        if early_stopper.step(map50, epoch):
            print(f"\n[train] Early stopping triggered! Validation progress saturated across {early_stopper.patience} epochs.")
            save_atomic(ckpt, weights_p / "early_stop_model.pth")
            break

    print(f"\n[train] Training completed. Best mAP_tiny@0.5: {best_map:.4f} saved in {args.weights_dir}/")


if __name__ == "__main__":
    main()