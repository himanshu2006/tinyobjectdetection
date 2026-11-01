"""
test_infer.py - Custom SAHI Sliced Inference, Evaluation, & Visualization
========================================================================
Standalone inference pipeline:
  1. TinyDroneDetector: Backbone + TinyFPN + DecoupledHead (zero external dependencies)
  2. CustomSAHIWrapper: Subclasses sahi.models.base.DetectionModel with vectorized decoding
  3. Sliced inference via get_sliced_prediction with GreedyNMM / IOS merging
  4. Single-image visualization and batch directory benchmarking (FPS, tiny detection counts)
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.predict import get_sliced_prediction


# ============================================================================
# 1. STANDALONE MODEL ARCHITECTURE
# ============================================================================

class ConvBNSiLU(nn.Module):
    """Conv2d -> BatchNorm2d -> SiLU activation."""

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
# 2. CUSTOM SAHI DETECTION MODEL WRAPPER
# ============================================================================

class CustomSAHIWrapper(DetectionModel):
    """Wraps TinyDroneDetector for seamless integration with SAHI tiled inference."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[TinyDroneDetector] = None,
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        category_mapping: Optional[Dict[str, str]] = None,
        category_remapping: Optional[Dict[str, int]] = None,
        load_at_init: bool = True,
        image_size: int = 640,
        nms_iou_threshold: float = 0.45,
        **kwargs: Any,
    ) -> None:
        self.nms_iou_threshold = nms_iou_threshold
        if category_mapping is None:
            category_mapping = {"0": "person"}

        super().__init__(
            model_path=model_path,
            model=model,
            confidence_threshold=confidence_threshold,
            device=device,
            category_mapping=category_mapping,
            category_remapping=category_remapping,
            load_at_init=load_at_init,
            image_size=image_size,
            **kwargs,
        )

    def load_model(self) -> None:
        """Loads weights from checkpoint into TinyDroneDetector."""
        if self.model is not None:
            self.model.eval().to(self.device)
            return

        if not self.model_path or not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model checkpoint path not found: {self.model_path}")

        try:
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(self.model_path, map_location=self.device)

        num_classes = ckpt.get("num_classes", len(self.category_mapping))
        fpn_channels = ckpt.get("fpn_channels", 128)

        if len(self.category_mapping) < num_classes:
            self.category_mapping = {str(i): f"class_{i}" for i in range(num_classes)}
            if num_classes == 1:
                self.category_mapping = {"0": "person"}

        detector = TinyDroneDetector(num_classes=num_classes, fpn_channels=fpn_channels)
        detector.load_state_dict(ckpt["model_state_dict"])
        detector.to(self.device).eval()
        self.set_model(detector)

    def set_model(self, model: Any, **kwargs: Any) -> None:
        self.model = model
        self.model.eval()

    def _letterbox(self, image: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, float, int, int]:
        """Aspect-preserving letterbox resize with grey (114) border padding."""
        h, w = image.shape[:2]
        scale = min(target_size / w, target_size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))

        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_x = (target_size - nw) // 2
        pad_y = (target_size - nh) // 2

        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
        return canvas, scale, pad_x, pad_y

    def perform_inference(self, image: np.ndarray) -> None:
        """Executes forward pass and decodes (l, t, r, b) boxes for a single slice."""
        orig_h, orig_w = image.shape[:2]
        input_size = self.image_size or 640

        padded_img, scale, pad_x, pad_y = self._letterbox(image, input_size)
        tensor = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)

        all_boxes, all_scores, all_labels = [], [], []

        for li, stride in enumerate(TinyDroneDetector.STRIDES):
            cls_logits = outputs[li][0][0]  # (C, H_l, W_l)
            reg_pred = outputs[li][1][0]    # (4, H_l, W_l)
            ctr_pred = outputs[li][2][0]    # (1, H_l, W_l)
            C, H_l, W_l = cls_logits.shape

            # Generate grid cell coordinates in padded image space
            sy = (torch.arange(0, H_l, device=self.device, dtype=torch.float32) + 0.5) * stride
            sx = (torch.arange(0, W_l, device=self.device, dtype=torch.float32) + 0.5) * stride
            gy, gx = torch.meshgrid(sy, sx, indexing="ij")
            gx, gy = gx.reshape(-1), gy.reshape(-1)

            # Combined score: Sigmoid(class) * Sigmoid(centerness)
            scores = torch.sigmoid(cls_logits.permute(1, 2, 0).reshape(-1, C)) * torch.sigmoid(ctr_pred.reshape(-1)).unsqueeze(1)
            max_scores, max_labels = scores.max(dim=1)
            keep = max_scores >= self.confidence_threshold

            if not keep.any():
                continue

            k_reg = reg_pred.permute(1, 2, 0).reshape(-1, 4)[keep]
            cx, cy = gx[keep], gy[keep]

            # Decode (l, t, r, b) to [x1, y1, x2, y2]
            x1 = (cx - k_reg[:, 0] - pad_x) / scale
            y1 = (cy - k_reg[:, 1] - pad_y) / scale
            x2 = (cx + k_reg[:, 2] - pad_x) / scale
            y2 = (cy + k_reg[:, 3] - pad_y) / scale

            boxes = torch.stack([x1.clamp(0, orig_w), y1.clamp(0, orig_h), x2.clamp(0, orig_w), y2.clamp(0, orig_h)], dim=1)
            all_boxes.append(boxes)
            all_scores.append(max_scores[keep])
            all_labels.append(max_labels[keep])

        if len(all_boxes) == 0:
            self._original_predictions = {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
            }
            return

        cat_boxes = torch.cat(all_boxes, dim=0)
        cat_scores = torch.cat(all_scores, dim=0)
        cat_labels = torch.cat(all_labels, dim=0)

        # Slice-level NMS via torchvision
        from torchvision.ops import batched_nms
        keep_idx = batched_nms(cat_boxes, cat_scores, cat_labels, self.nms_iou_threshold)

        self._original_predictions = {
            "boxes": cat_boxes[keep_idx].cpu().numpy(),
            "scores": cat_scores[keep_idx].cpu().numpy(),
            "labels": cat_labels[keep_idx].cpu().numpy(),
        }

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[Union[int, float]]]] = None,
        full_shape_list: Optional[List[List[Union[int, float]]]] = None,
    ) -> None:
        """Converts raw predictions into SAHI ObjectPrediction items with slice offset."""
        shift_x, shift_y = 0, 0
        if shift_amount_list and len(shift_amount_list) > 0:
            shift_x = int(shift_amount_list[0][0])
            shift_y = int(shift_amount_list[0][1])

        full_shape = None
        if full_shape_list and len(full_shape_list) > 0:
            full_shape = [int(full_shape_list[0][0]), int(full_shape_list[0][1])]

        raw = self._original_predictions
        if raw is None or len(raw["boxes"]) == 0:
            self._object_prediction_list = []
            self._object_prediction_list_per_image = [[]]
            return

        boxes, scores, labels = raw["boxes"], raw["scores"], raw["labels"]
        predictions: List[ObjectPrediction] = []

        for i in range(len(boxes)):
            x1, y1, x2, y2 = float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue

            lbl = int(labels[i])
            name = self.category_mapping.get(str(lbl), f"class_{lbl}")
            predictions.append(
                ObjectPrediction(
                    bbox=[x1, y1, x2, y2],
                    category_id=lbl,
                    category_name=name,
                    score=float(scores[i]),
                    shift_amount=[shift_x, shift_y],
                    full_shape=full_shape,
                )
            )

        # Populate both attributes for complete SAHI version interoperability
        self._object_prediction_list = predictions
        self._object_prediction_list_per_image = [predictions]


# ============================================================================
# 3. HIGH-RESOLUTION SLICED INFERENCE & VISUALIZATION
# ============================================================================

def visualize_detections(
    image_path: str,
    predictions: List[ObjectPrediction],
    output_path: str,
    box_color: Tuple[int, int, int] = (0, 255, 127),
    line_thickness: int = 2,
) -> None:
    """Draws bounding boxes and labels on the full-resolution image."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for pred in predictions:
        box = pred.bbox.to_xyxy()
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_thickness)

        label = f"{pred.category.name} {pred.score.value:.2f} ({int(x2-x1)}x{int(y2-y1)}px)"
        draw.text((x1 + 2, max(0, y1 - 14)), label, fill=box_color)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path)
    print(f"[vis] Saved {len(predictions)} detections to {output_path}")


def run_sliced_inference_on_image(
    image_path: str,
    sahi_model: CustomSAHIWrapper,
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_ratio: float = 0.25,
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    visualize: bool = True,
    output_path: Optional[str] = None,
) -> Tuple[List[ObjectPrediction], float]:
    """Runs SAHI get_sliced_prediction with GreedyNMM / IOS merging."""
    t0 = time.perf_counter()

    result = get_sliced_prediction(
        image_path,
        sahi_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        perform_standard_pred=False,
        postprocess_type=postprocess_type,
        postprocess_match_metric=postprocess_match_metric,
        postprocess_match_threshold=postprocess_match_threshold,
        verbose=0,
    )

    latency = time.perf_counter() - t0
    preds = result.object_prediction_list

    if visualize and output_path:
        visualize_detections(image_path, preds, output_path)

    return preds, latency


def evaluate_directory(
    image_dir: str,
    sahi_model: CustomSAHIWrapper,
    output_dir: str = "outputs",
    slice_size: int = 640,
    overlap_ratio: float = 0.25,
    postprocess_type: str = "GREEDYNMM",
    tiny_thresh_px: int = 16,
) -> Dict[str, Any]:
    """Benchmarks sliced inference across a directory of high-resolution images."""
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    files = [f for f in sorted(os.listdir(image_dir)) if f.lower().endswith(valid_exts)]
    if not files:
        print(f"[eval] No valid images found in {image_dir}")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    total_time, total_dets, tiny_dets = 0.0, 0, 0

    print(f"\n[eval] Evaluating {len(files)} images from: {image_dir}")
    print("-" * 75)

    for i, fname in enumerate(files, 1):
        fpath = os.path.join(image_dir, fname)
        out_path = os.path.join(output_dir, f"det_{fname}")

        preds, latency = run_sliced_inference_on_image(
            fpath, sahi_model, slice_height=slice_size, slice_width=slice_size,
            overlap_ratio=overlap_ratio, postprocess_type=postprocess_type,
            visualize=True, output_path=out_path,
        )

        n_dets = len(preds)
        n_tiny = sum(1 for p in preds if max(p.bbox.to_xyxy()[2] - p.bbox.to_xyxy()[0], p.bbox.to_xyxy()[3] - p.bbox.to_xyxy()[1]) <= tiny_thresh_px)

        total_time += latency
        total_dets += n_dets
        tiny_dets += n_tiny

        fps = 1.0 / max(latency, 1e-6)
        print(f"[{i:3d}/{len(files)}] {fname:25s} | Dets: {n_dets:3d} (Tiny: {n_tiny:2d}) | {latency*1000:6.1f}ms ({fps:4.1f} FPS)")

    avg_fps = len(files) / max(total_time, 1e-6)
    print("-" * 75)
    print(f"[Summary] Images: {len(files)} | Detections: {total_dets} (Tiny: {tiny_dets}) | Speed: {avg_fps:.2f} FPS")
    return {"total_images": len(files), "total_dets": total_dets, "tiny_dets": tiny_dets, "avg_fps": avg_fps}


# ============================================================================
# 4. CLI ENTRY POINT
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny-Object Drone Detector Sliced Inference with SAHI")
    parser.add_argument("--image", type=str, default="", help="Single high-resolution input image")
    parser.add_argument("--image_dir", type=str, default="", help="Directory of input images")
    parser.add_argument("--weights", type=str, default="weights/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for visual results")
    parser.add_argument("--slice_size", type=int, default=640, help="Tile slice size in pixels")
    parser.add_argument("--overlap_ratio", type=float, default=0.25, help="Slice overlap ratio")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="", help="Compute device: cuda, cpu, or auto")
    parser.add_argument("--postprocess", type=str, default="GREEDYNMM", choices=["GREEDYNMM", "NMS"])
    args = parser.parse_args()

    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_path = args.weights
    if not os.path.isfile(weights_path):
        alt_weights = os.path.join(os.path.dirname(weights_path), "last_model.pth")
        if os.path.isfile(alt_weights):
            weights_path = alt_weights
        else:
            raise FileNotFoundError(f"No checkpoint found at {weights_path} or {alt_weights}")

    sahi_model = CustomSAHIWrapper(
        model_path=weights_path,
        confidence_threshold=args.conf_thresh,
        device=device,
        image_size=args.slice_size,
        load_at_init=True,
    )

    if args.image:
        preds, latency = run_sliced_inference_on_image(
            args.image, sahi_model, slice_height=args.slice_size, slice_width=args.slice_size,
            overlap_ratio=args.overlap_ratio, postprocess_type=args.postprocess,
            visualize=True, output_path=os.path.join(args.output_dir, f"det_{os.path.basename(args.image)}"),
        )
        print(f"[Done] Detected {len(preds)} objects in {latency*1000:.1f}ms ({1.0/latency:.1f} FPS)")
    elif args.image_dir:
        evaluate_directory(
            args.image_dir, sahi_model, output_dir=args.output_dir,
            slice_size=args.slice_size, overlap_ratio=args.overlap_ratio,
            postprocess_type=args.postprocess,
        )
    else:
        sample = "data/smoke_test/images/synth_0000.jpg"
        if os.path.isfile(sample):
            preds, latency = run_sliced_inference_on_image(
                sample, sahi_model, slice_height=args.slice_size, slice_width=args.slice_size,
                overlap_ratio=args.overlap_ratio, postprocess_type=args.postprocess,
                visualize=True, output_path=os.path.join(args.output_dir, "sample_detection.jpg"),
            )
            print(f"[Done] Sample test: {len(preds)} detections in {latency*1000:.1f}ms ({1.0/latency:.1f} FPS)")
        else:
            print("[test_infer] Usage error: specify --image or --image_dir.")


if __name__ == "__main__":
    main()