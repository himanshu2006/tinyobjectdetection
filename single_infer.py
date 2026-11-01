import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def preprocess_image(image_path: str, input_size: int = 640):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    orig_h, orig_w = img.shape[:2]

    # Deterministic letterbox keeping aspect ratio
    r = min(input_size / orig_w, input_size / orig_h)
    nw, nh = int(round(orig_w * r)), int(round(orig_h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_x = (input_size - nw) // 2
    pad_y = (input_size - nh) // 2

    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

    # Normalization (0-1), BGR to RGB, and HWC to CHW
    blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None, ...])

    meta = {"orig_w": orig_w, "orig_h": orig_h, "scale": r, "pad_x": pad_x, "pad_y": pad_y}
    return blob, img, meta


def nms_cpu(boxes, scores, score_thr=0.25, iou_thr=0.45):
    keep_mask = scores > score_thr
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty((0,))

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=scores.tolist(),
        score_threshold=score_thr,
        nms_threshold=iou_thr,
    )
    if len(indices) == 0:
        return np.empty((0, 4)), np.empty((0,))
    indices = np.array(indices).flatten()
    return boxes[indices], scores[indices]


def run_inference(
    model_path="weights/drone_detector.onnx",
    image_path="test.jpg",
    score_thr=0.25,
    iou_thr=0.45,
    save_img=True,
):
    print("=" * 60)
    print(f"[INFERENCE ENGINE] Model: {model_path}")
    print(f"[INFERENCE ENGINE] Source Image: {image_path}")

    # Load ONNX Runtime Session
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    t0_load = time.time()
    session = ort.InferenceSession(model_path, providers=providers)
    active_provider = session.get_providers()[0]
    print(f"[INFERENCE ENGINE] Active Provider: {active_provider}")
    print(f"[INFERENCE ENGINE] Session Init Time: {(time.time() - t0_load)*1000:.2f} ms")

    # 1. Preprocess
    t0_pre = time.time()
    blob, raw_img, meta = preprocess_image(image_path, input_size=640)
    pre_time = (time.time() - t0_pre) * 1000

    # 2. Forward Pass
    t0_infer = time.time()
    boxes_pred, scores_pred = session.run(["boxes", "scores"], {"images": blob})
    infer_time = (time.time() - t0_infer) * 1000

    # 3. Post-process (Coordinate recovery + NMS)
    t0_post = time.time()
    boxes_640 = boxes_pred[0]  # Shape: (34000, 4)
    scores_640 = scores_pred[0].squeeze(-1)  # Shape: (34000,)

    # Convert [x1, y1, x2, y2] to [x, y, w, h] for cv2 NMS
    boxes_xywh = np.zeros_like(boxes_640)
    boxes_xywh[:, 0] = boxes_640[:, 0]
    boxes_xywh[:, 1] = boxes_640[:, 1]
    boxes_xywh[:, 2] = boxes_640[:, 2] - boxes_640[:, 0]
    boxes_xywh[:, 3] = boxes_640[:, 3] - boxes_640[:, 1]

    final_boxes, final_scores = nms_cpu(boxes_xywh, scores_640, score_thr=score_thr, iou_thr=iou_thr)

    # Scale back to original resolution
    r, pad_x, pad_y = meta["scale"], meta["pad_x"], meta["pad_y"]
    scaled_detections = []
    for (x, y, w, h), score in zip(final_boxes, final_scores):
        orig_x1 = max(0, int((x - pad_x) / r))
        orig_y1 = max(0, int((y - pad_y) / r))
        orig_w = int(w / r)
        orig_h = int(h / r)
        orig_x2 = min(meta["orig_w"], orig_x1 + orig_w)
        orig_y2 = min(meta["orig_h"], orig_y1 + orig_h)
        scaled_detections.append((orig_x1, orig_y1, orig_x2, orig_y2, score))

    post_time = (time.time() - t0_post) * 1000

    # Print Full Standard Output Diagnostics
    print("-" * 60)
    print(f"Latency Profile:")
    print(f"  • Preprocessing : {pre_time:.2f} ms")
    print(f"  • Model Inference: {infer_time:.2f} ms")
    print(f"  • Post-proc/NMS : {post_time:.2f} ms")
    print(f"  • Total Runtime : {pre_time + infer_time + post_time:.2f} ms ({(1000/(pre_time + infer_time + post_time)):.1f} FPS)")
    print("-" * 60)
    print(f"Detection Results (Threshold >= {score_thr}):")
    print(f"  Total Human Targets Identified: {len(scaled_detections)}")

    for idx, (x1, y1, x2, y2, score) in enumerate(scaled_detections, 1):
        target_w = x2 - x1
        target_h = y2 - y1
        print(f"  [{idx:02d}] Class: Person | Conf: {score*100:5.2f}% | Box: [{x1}, {y1}, {x2}, {y2}] | Size: {target_w}x{target_h}px")

    if save_img:
        for x1, y1, x2, y2, score in scaled_detections:
            cv2.rectangle(raw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(raw_img, f"Person {score:.2f}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out_name = f"pred_{Path(image_path).name}"
        cv2.imwrite(out_name, raw_img)
        print(f"[OUTPUT] Annotated detection rendered to: {out_name}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ONNX Detection Inference")
    parser.add_argument("--model", type=str, default="weights/drone_detector.onnx")
    parser.add_argument("--img", type=str, required=True, help="Path to input aerial image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS threshold")
    args = parser.parse_args()

    run_inference(model_path=args.model, image_path=args.img, score_thr=args.conf, iou_thr=args.iou)