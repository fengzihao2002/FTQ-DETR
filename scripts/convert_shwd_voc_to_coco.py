#!/usr/bin/env python3
"""Generic VOC-to-COCO converter for SHWD supplementary validation.

The manuscript uses SHWD as a two-class supplementary validation dataset with a
6065/1516 train/val split. Because different SHWD releases may use slightly
different class names, this script lets users define the class mapping from the
command line.

Example:
    python scripts/convert_shwd_voc_to_coco.py \
      --src /path/to/SHWD \
      --out data/SHWD_COCO \
      --categories helmet,head \
      --seed 42 \
      --overwrite

The output category IDs are one-based. If two categories are used, run FTQ-DETR
with `--num_classes 3`.
"""

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def norm_name(name):
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def find_image(images_dir, name):
    p = Path(name)
    candidates = [images_dir / p.name] if p.suffix else []
    if not p.suffix:
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]:
            candidates.append(images_dir / f"{name}{ext}")
    for item in candidates:
        if item.exists():
            return item
    return None


def parse_xml(xml_path, images_dir, name_to_id):
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename") or xml_path.stem
    image_path = find_image(images_dir, filename) or find_image(images_dir, xml_path.stem)
    if image_path is None:
        raise FileNotFoundError(f"Image not found for {xml_path}")
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")
    width = int(float(size.findtext("width")))
    height = int(float(size.findtext("height")))
    objects = []
    unknown = []
    for obj in root.findall("object"):
        raw = obj.findtext("name")
        if not raw:
            continue
        label = norm_name(raw)
        if label not in name_to_id:
            unknown.append(raw)
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = max(0.0, min(float(bnd.findtext("xmin")), width - 1))
        ymin = max(0.0, min(float(bnd.findtext("ymin")), height - 1))
        xmax = max(0.0, min(float(bnd.findtext("xmax")), width))
        ymax = max(0.0, min(float(bnd.findtext("ymax")), height))
        bw, bh = xmax - xmin, ymax - ymin
        if bw <= 0 or bh <= 0:
            continue
        objects.append({
            "category_id": name_to_id[label],
            "bbox": [xmin, ymin, bw, bh],
            "area": bw * bh,
            "iscrowd": 0,
        })
    return {"image_path": image_path, "file_name": image_path.name, "width": width, "height": height, "objects": objects, "unknown": unknown}


def write_coco(records, split, out_dir, categories):
    image_dir = out_dir / f"{split}2017"
    image_dir.mkdir(parents=True, exist_ok=True)
    coco = {"info": {"description": "SHWD COCO for FTQ-DETR"}, "licenses": [], "images": [], "annotations": [], "categories": categories}
    ann_id = 1
    counter = Counter()
    for img_id, rec in enumerate(records, 1):
        dst = image_dir / rec["file_name"]
        if not dst.exists():
            shutil.copy2(rec["image_path"], dst)
        coco["images"].append({"id": img_id, "file_name": rec["file_name"], "width": rec["width"], "height": rec["height"]})
        for obj in rec["objects"]:
            coco["annotations"].append({"id": ann_id, "image_id": img_id, **obj})
            counter[obj["category_id"]] += 1
            ann_id += 1
    ann_dir = out_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    path = ann_dir / f"instances_{split}2017.json"
    path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Original SHWD root containing Annotations and Images")
    parser.add_argument("--out", required=True, help="Output COCO dataset root")
    parser.add_argument("--categories", required=True, help="Comma-separated class names in the desired order, e.g. helmet,head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = Path(args.src)
    ann_dir = src / "Annotations"
    images_dir = src / "Images"
    if not ann_dir.exists() or not images_dir.exists():
        raise FileNotFoundError("The source directory must contain Annotations/ and Images/.")
    out_dir = Path(args.out)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = [norm_name(x) for x in args.categories.split(",") if x.strip()]
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    name_to_id = {c["name"]: c["id"] for c in categories}

    records = []
    unknown_counter = Counter()
    for xml in sorted(ann_dir.glob("*.xml")):
        rec = parse_xml(xml, images_dir, name_to_id)
        records.append(rec)
        unknown_counter.update(rec["unknown"])

    if len(records) < 7581:
        raise RuntimeError(f"Expected at least 7581 XML files for the 6065/1516 split, found {len(records)}")
    random.seed(args.seed)
    random.shuffle(records)
    records = records[:7581]
    split_map = {"train": records[:6065], "val": records[6065:7581]}

    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, items in split_map.items():
        (split_dir / f"{split}.txt").write_text("\n".join(Path(r["file_name"]).stem for r in items) + "\n", encoding="utf-8")

    for split, items in split_map.items():
        path, counter = write_coco(items, split, out_dir, categories)
        print(f"{split}: {len(items)} images -> {path}")
        for c in categories:
            print(f"  {c['id']} {c['name']}: {counter[c['id']]}")
    if unknown_counter:
        print("Unknown labels skipped:")
        for k, v in unknown_counter.items():
            print(f"  {k}: {v}")
    print(f"Use --num_classes {len(categories) + 1} if category IDs are one-based.")


if __name__ == "__main__":
    main()
