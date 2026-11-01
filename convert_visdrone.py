import os
import json
from PIL import Image
from tqdm import tqdm

def convert_split(visdrone_dir, output_json):
    img_dir = os.path.join(visdrone_dir, "images")
    ann_dir = os.path.join(visdrone_dir, "annotations")

    valid_exts = (".jpg", ".jpeg", ".png")
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)])

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "person", "supercategory": "human"}]
    }

    ann_id = 1
    print(f"Converting {len(img_files)} images from: {visdrone_dir}")

    for img_id, fname in enumerate(tqdm(img_files), start=1):
        img_path = os.path.join(img_dir, fname)
        with Image.open(img_path) as img:
            w_img, h_img = img.size

        coco_data["images"].append({
            "id": img_id,
            "file_name": fname,
            "width": w_img,
            "height": h_img
        })

        txt_file = os.path.join(ann_dir, os.path.splitext(fname)[0] + ".txt")
        if not os.path.isfile(txt_file):
            continue

        with open(txt_file, "r") as f:
            for line in f:
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) < 8:
                    continue

                x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                score = int(parts[4])
                category = int(parts[5])

                # Keep only score 1 (considered) and category 1 (pedestrian) or 2 (people)
                if score == 0 or category not in (1, 2):
                    continue

                if w <= 1 or h <= 1:
                    continue

                # Clamp bounding boxes to image dimensions
                x = max(0.0, min(x, w_img - 1))
                y = max(0.0, min(y, h_img - 1))
                w = max(1.0, min(w, w_img - x))
                h = max(1.0, min(h, h_img - y))

                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "iscrowd": 0
                })
                ann_id += 1

    with open(output_json, "w") as out:
        json.dump(coco_data, out)

    print(f"Created: {output_json} with {len(coco_data['annotations'])} human boxes.")

if __name__ == "__main__":
    convert_split(
        visdrone_dir="VisDrone2019-DET-train",
        output_json="VisDrone2019-DET-train/train_coco.json"
    )