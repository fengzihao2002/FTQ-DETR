#!/usr/bin/env python3
"""Convert original SHEL5K VOC-style annotations to the COCO format used by FTQ-DETR.

Expected original structure:
    Safety Helmet Wearing Dataset/
    ├── Annotations/*.xml
    └── Images/*

The script removes the rare `person` category and keeps six one-based category IDs:
    1 helmet
    2 head_with_helmet
    3 head
    4 person_with_helmet
    5 face
    6 person_no_helmet

The generated split is fixed to 3500/1000/500 using the specified random seed.
"""

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

CATEGORIES = [
    {"id": 1, "name": "helmet"},
    {"id": 2, "name": "head_with_helmet"},
    {"id": 3, "name": "head"},
    {"id": 4, "name": "person_with_helmet"},
    {"id": 5, "name": "face"},
    {"id": 6, "name": "person_no_helmet"},
]
NAME_TO_ID = {c["name"]: c["id"] for c in CATEGORIES}

ALIASES = {
    "helmet": "helmet",
    "head_with_helmet": "head_with_helmet",
    "head-with-helmet": "head_with_helmet",
    "head with helmet": "head_with_helmet",
    "head": "head",
    "person_with_helmet": "person_with_helmet",
    "person-with-helmet": "person_with_helmet",
    "person with helmet": "person_with_helmet",
    "face": "face",
    "person_no_helmet": "person_no_helmet",
    "person-no-helmet": "person_no_helmet",
    "person no helmet": "person_no_helmet",
    "person_without_helmet": "person_no_helmet",
    "person-without-helmet": "person_no_helmet",
    "person without helmet": "person_no_helmet",
    "person": "person",
}


def norm_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def find_image(images_dir: Path, name: str):
    p = Path(name)
    candidates = [images_dir / p.name] if p.suffix else []
    if not p.suffix:
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]:
            candidates.append(images_dir / f"{name}{ext}")
    for item in candidates:
        if item.exists():
            return item
    return None


def parse_xml(xml_path: Path, images_dir: Path) -> dict:
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
    removed_person = 0
    unknown = []
    for obj in root.findall("object"):
        raw = obj.findtext("name")
        if not raw:
            continue
        label = ALIASES.get(norm_name(raw), norm_name(raw))
        if label == "person":
            removed_person += 1
            continue
        if label not in NAME_TO_ID:
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
            "category_id": NAME_TO_ID[label],
            "bbox": [xmin, ymin, bw, bh],
            "area": bw * bh,
            "iscrowd": 0,
        })
    return {
        "image_path": image_path,
        "file_name": image_path.name,
        "width": width,
        "height": height,
        "objects": objects,
        "removed_person": removed_person,
        "unknown": unknown,
    }


def write_coco(records: list[dict], split: str, out_dir: Path) -> tuple[Path, Counter]:
    image_dir = out_dir / f"{split}2017"
    image_dir.mkdir(parents=True, exist_ok=True)
    coco = {"info": {"description": "SHEL5K COCO6 for FTQ-DETR"}, "licenses": [], "images": [], "annotations": [], "categories": CATEGORIES}
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
    json_path = ann_dir / f"instances_{split}2017.json"
    json_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path, counter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Original SHEL5K root containing Annotations and Images")
    parser.add_argument("--out", required=True, help="Output COCO dataset root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out)
    ann_dir = src / "Annotations"
    images_dir = src / "Images"
    if not ann_dir.exists() or not images_dir.exists():
        raise FileNotFoundError("The source directory must contain Annotations/ and Images/.")
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(ann_dir.glob("*.xml"))
    if len(xml_files) < 5000:
        raise RuntimeError(f"Expected at least 5000 XML files, found {len(xml_files)}")
    if len(xml_files) != 5000:
        print(f"Warning: found {len(xml_files)} XML files; the first 5000 after shuffling will be split as 3500/1000/500.")

    records = []
    removed_person = 0
    unknown_counter = Counter()
    for xml in xml_files:
        rec = parse_xml(xml, images_dir)
        records.append(rec)
        removed_person += rec["removed_person"]
        unknown_counter.update(rec["unknown"])

    random.seed(args.seed)
    random.shuffle(records)
    records = records[:5000]
    split_map = {"train": records[:3500], "val": records[3500:4500], "test": records[4500:5000]}

    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, items in split_map.items():
        (split_dir / f"{split}.txt").write_text("\n".join(Path(r["file_name"]).stem for r in items) + "\n", encoding="utf-8")

    print("Writing COCO files...")
    for split, items in split_map.items():
        path, counter = write_coco(items, split, out_dir)
        print(f"{split}: {len(items)} images -> {path}")
        for c in CATEGORIES:
            print(f"  {c['id']} {c['name']}: {counter[c['id']]}")
    print(f"Removed person annotations: {removed_person}")
    if unknown_counter:
        print("Unknown labels skipped:")
        for k, v in unknown_counter.items():
            print(f"  {k}: {v}")
    print("Use --num_classes 7 for this processed SHEL5K dataset.")


if __name__ == "__main__":
    main()
