import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def preprocess(img_path: str, input_size: int = 640):
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None
    h, w = img.shape[:2]
    r = min(input_size / w, input_size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_x = (input_size - nw) // 2
    pad_y = (input_size - nh) // 2
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

    blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None, ...])
    return blob, img, (w, h, r, pad_x, pad_y)


def run_batch(folder="test", model_path="weights/drone_detector.onnx", conf_thr=0.25, out_dir="outputs"):
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)

    session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print(f"Provider: {session.get_providers()[0]}")

    image_paths = sorted([p for p in Path(folder).iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    print(f"Processing {len(image_paths)} images from '{folder}' -> saving to '{out_dir}/'\n")
    print(f"{'Image Name':<35} | {'Detections':<10} | {'Latency':<10} | {'Max Conf'}")
    print("-" * 72)

    total_time = 0.0
    total_detections = 0

    for p in image_paths:
        blob, raw_img, meta = preprocess(str(p))
        if blob is None:
            continue

        t0 = time.time()
        boxes_pred, scores_pred = session.run(["boxes", "scores"], {"images": blob})
        dt = (time.time() - t0) * 1000
        total_time += dt

        boxes_640 = boxes_pred[0]
        scores_640 = scores_pred[0].squeeze(-1)

        boxes_xywh = np.zeros_like(boxes_640)
        boxes_xywh[:, 0] = boxes_640[:, 0]
        boxes_xywh[:, 1] = boxes_640[:, 1]
        boxes_xywh[:, 2] = boxes_640[:, 2] - boxes_640[:, 0]
        boxes_xywh[:, 3] = boxes_640[:, 3] - boxes_640[:, 1]

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_xywh.tolist(),
            scores=scores_640.tolist(),
            score_threshold=conf_thr,
            nms_threshold=0.45,
        )

        orig_w, orig_h, r, pad_x, pad_y = meta
        dets_count = len(indices)
        total_detections += dets_count
        max_conf = 0.0

        if dets_count > 0:
            for idx in np.array(indices).flatten():
                x, y, bw, bh = boxes_xywh[idx]
                score = float(scores_640[idx])
                max_conf = max(max_conf, score)

                x1 = max(0, int((x - pad_x) / r))
                y1 = max(0, int((y - pad_y) / r))
                x2 = min(orig_w, int((x + bw - pad_x) / r))
                y2 = min(orig_h, int((y + bh - pad_y) / r))

                cv2.rectangle(raw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(raw_img, f"{score:.2f}", (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imwrite(str(out_path / f"pred_{p.name}"), raw_img)
        print(f"{p.name:<35} | {dets_count:<10} | {dt:6.2f} ms  | {max_conf * 100:5.1f}%")

    avg_time = total_time / max(len(image_paths), 1)
    print("-" * 72)
    print(f"Total Targets: {total_detections} | Avg Latency: {avg_time:.2f} ms ({(1000/avg_time):.1f} FPS)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="test")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    run_batch(folder=args.folder, conf_thr=args.conf)