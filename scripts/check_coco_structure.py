#!/usr/bin/env python3
"""Check COCO-format dataset structure for FTQ-DETR experiments.

Example:
    python scripts/check_coco_structure.py --coco_path data/SHEL5K_COCO6 --splits train val test
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def split_to_paths(root: Path, split: str):
    image_dir = root / f"{split}2017"
    ann_path = root / "annotations" / f"instances_{split}2017.json"
    return image_dir, ann_path


def check_split(root: Path, split: str, max_missing_report: int = 10):
    image_dir, ann_path = split_to_paths(root, split)
    print(f"\n[{split}]")
    print(f"image_dir: {image_dir}")
    print(f"ann_path : {ann_path}")

    if not image_dir.exists():
        print("ERROR: image directory does not exist")
        return False
    if not ann_path.exists():
        print("ERROR: annotation file does not exist")
        return False

    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    print(f"images      : {len(images)}")
    print(f"annotations : {len(anns)}")
    print(f"categories  : {len(cats)}")

    cat_ids = sorted({int(c["id"]) for c in cats})
    ann_cat_counter = Counter(int(a["category_id"]) for a in anns)
    print(f"category ids: {cat_ids}")
    print("category names:")
    id_to_name = {int(c["id"]): c.get("name", "") for c in cats}
    for cid in cat_ids:
        print(f"  {cid}: {id_to_name.get(cid, '')} | anns={ann_cat_counter.get(cid, 0)}")

    image_files = {p.name for p in image_dir.iterdir() if p.is_file()}
    missing = []
    for img in images:
        fn = img.get("file_name")
        if fn and Path(fn).name not in image_files and fn not in image_files:
            missing.append(fn)
    if missing:
        print(f"WARNING: {len(missing)} images listed in JSON were not found in {image_dir}")
        for x in missing[:max_missing_report]:
            print(f"  missing: {x}")
    else:
        print("image-file check: OK")

    max_cat = max(cat_ids) if cat_ids else -1
    print(f"suggested --num_classes for one-based labels: {max_cat + 1 if max_cat >= 0 else 'N/A'}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_path", required=True, help="Path to COCO-format dataset root")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    args = parser.parse_args()

    root = Path(args.coco_path)
    if not root.exists():
        raise FileNotFoundError(root)

    ok = True
    for split in args.splits:
        ok = check_split(root, split) and ok

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
