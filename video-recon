"""
Per-frame, per-category GT video reconstruction for a single CRCD case,
using a single COCO file: annotations.json.

Assumptions:
  - annotations.json in CASE_DIR contains:
      * all 'images' entries for this case
      * all 'annotations' for those images
      * 'categories' with ids 0..5 (Meat, Skin, Liver, Gallbladder, FBF, PCH)
  - Each image 'file_name' is something like "split_0/00037.png"
    that matches frames on disk under:
      CASE_DIR/split_imgs/split_imgs/split_0/00037.png

This script:
  - Loads annotations.json.
  - Builds an index from (split_name, basename) -> image_id.
  - For each frame on disk, looks up its image_id.
  - Builds per-category masks for that frame.
  - Overlays masks with distinct colors.
  - Writes one GT video per split: split_X_gt.mp4.
"""

import os
import json
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np


# ======================= CONFIG =======================

# Inner case directory that holds split_imgs + JSON
# Example: r"C:\Users\BenWright\CRCD\annotated\C_1\C_1"
CASE_DIR = r"C:\Users\BenWright\CRCD\annotated\C_1\C_1"

# Single combined COCO file
JSON_NAME = "annotations.json"

# Splits root
SPLIT_ROOT = os.path.join(CASE_DIR, "split_imgs", "split_imgs")

# Output video naming
GT_VIDEO_SUFFIX = "_gt.mp4"
OUTPUT_FPS = 10

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Per-category colors (BGR) for ids 0..5
CATEGORY_COLORS = {
    0: (255, 0, 0),      # Meat - blue
    1: (0, 255, 0),      # Skin - green
    2: (0, 0, 255),      # Liver - red
    3: (0, 255, 255),    # Gallbladder - yellow
    4: (255, 0, 255),    # FBF - magenta
    5: (255, 255, 0),    # PCH - cyan
}
MASK_ALPHA = 0.4

# ======================================================


def polygons_to_mask(polygons: List[List[float]], height: int, width: int) -> np.ndarray:
    """
    Convert COCO polygon segmentation to a binary mask.
    polygons: list of [x0, y0, x1, y1, ...] lists.
    Returns: bool array [H, W]
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        if len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def overlay_multi_category(
    img_bgr: np.ndarray,
    cat_masks: Dict[int, np.ndarray],
    colors: Dict[int, tuple],
) -> np.ndarray:
    """
    Overlay multiple category masks on an image with different colors.
    cat_masks: category_id -> bool mask [H,W]
    colors:    category_id -> BGR tuple
    """
    out = img_bgr.copy()
    h, w, _ = out.shape
    alpha = MASK_ALPHA

    for cat_id, mask in cat_masks.items():
        if cat_id not in colors:
            continue
        if mask is None or mask.shape != (h, w) or not mask.any():
            continue

        color_layer = np.zeros_like(out, dtype=np.uint8)
        color_layer[mask] = colors[cat_id]

        blended = (
            out.astype(np.float32) * (1 - alpha)
            + color_layer.astype(np.float32) * alpha
        ).astype(np.uint8)

        out[mask] = blended[mask]

    return out


def load_coco_annotations(case_dir: str, json_name: str):
    """
    Load a single COCO JSON from CASE_DIR/json_name and build:
      - images_by_id: image_id -> image dict
      - anns_by_image: image_id -> list[annotation dict]
      - categories_by_id: category_id -> category dict
      - frame_index: (split_name, basename) -> image_id
    """
    json_path = os.path.join(case_dir, json_name)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")

    print(f"Loading COCO annotations from: {json_path}")
    with open(json_path, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    print(f"  images      : {len(images)}")
    print(f"  annotations : {len(annotations)}")
    print(f"  categories  : {len(categories)}")

    images_by_id: Dict[int, dict] = {img["id"]: img for img in images}
    anns_by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    categories_by_id: Dict[int, dict] = {cat["id"]: cat for cat in categories}

    # Build (split_name, basename) -> image_id index from file_name
    frame_index: Dict[Tuple[str, str], int] = {}

    for img in images:
        img_id = img["id"]
        file_name = img.get("file_name", "")

        # Normalise path: strip leading "./", unify slashes
        cleaned = file_name.strip().lstrip("./\\")
        cleaned = cleaned.replace("\\", "/")
        parts = cleaned.split("/")

        if len(parts) == 1:
            # No split folder in file_name; treat as basename only
            split_name = ""
            basename = parts[0]
        else:
            split_name = parts[-2]  # e.g. "split_0"
            basename = parts[-1]    # e.g. "00037.png"

        key = (split_name, basename)
        if key in frame_index:
            print(f"[WARN] Duplicate (split,basename) in JSON: {key}, overwriting previous image_id.")
        frame_index[key] = img_id

    print(f"  distinct (split,basename) keys in JSON: {len(frame_index)}")

    return images_by_id, anns_by_image, categories_by_id, frame_index


def collect_frames_by_split(split_root: str):
    """
    Collect frames on disk, grouped by split.
    Returns:
      split_to_frames: split_name -> list[(fname, full_path)]
    """
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"SPLIT_ROOT not found: {split_root}")

    split_names = sorted(
        d for d in os.listdir(split_root)
        if os.path.isdir(os.path.join(split_root, d))
    )
    print("Found splits on disk:", split_names)

    split_to_frames: Dict[str, List[Tuple[str, str]]] = {}

    for split_name in split_names:
        split_dir = os.path.join(split_root, split_name)
        frames = sorted(
            f for f in os.listdir(split_dir)
            if os.path.isfile(os.path.join(split_dir, f))
            and f.lower().endswith(IMAGE_EXTS)
        )
        full_paths = [(f, os.path.join(split_dir, f)) for f in frames]
        split_to_frames[split_name] = full_paths

        print(f"  split {split_name}: {len(frames)} frames")

    return split_to_frames


def build_cat_masks_for_frame(
    img_id: int,
    anns_by_image: Dict[int, List[dict]],
    height: int,
    width: int,
) -> Dict[int, np.ndarray]:
    """
    Build one mask per category for a given image_id.
    Returns: category_id -> bool mask [H,W]
    """
    anns = anns_by_image.get(img_id, [])
    cat_masks: Dict[int, np.ndarray] = {}

    if not anns:
        return cat_masks

    for ann in anns:
        cat_id = ann.get("category_id", None)
        seg = ann.get("segmentation", None)

        if cat_id is None:
            continue
        if not isinstance(seg, list) or len(seg) == 0:
            # RLE or empty; ignore here
            continue

        mask = polygons_to_mask(seg, height, width)
        if not mask.any():
            continue

        if cat_id not in cat_masks:
            cat_masks[cat_id] = mask
        else:
            cat_masks[cat_id] |= mask

    return cat_masks


def reconstruct_videos_by_filename():
    # 1) Load annotations.json and build frame index
    images_by_id, anns_by_image, categories_by_id, frame_index = load_coco_annotations(CASE_DIR, JSON_NAME)

    # 2) Collect frames on disk, grouped by split
    split_to_frames = collect_frames_by_split(SPLIT_ROOT)

    # 3) For each split, open frames and overlay masks by (split_name, basename) lookup
    for split_name, frame_infos in split_to_frames.items():
        if not frame_infos:
            print(f"[INFO] Split {split_name} has no frames; skipping.")
            continue

        first_frame_path = frame_infos[0][1]
        first_img = cv2.imread(first_frame_path)
        if first_img is None:
            print(f"[WARN] Cannot read first frame in {split_name}, skipping video.")
            continue

        H, W, _ = first_img.shape
        out_path = os.path.join(os.path.dirname(first_frame_path), f"{split_name}{GT_VIDEO_SUFFIX}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, OUTPUT_FPS, (W, H))

        print(f"[VIDEO] Writing GT video for split {split_name} -> {out_path}")

        n_written = 0
        n_with_masks = 0
        n_missing_in_json = 0

        for fname, fpath in frame_infos:
            img = cv2.imread(fpath)
            if img is None:
                print(f"[WARN] Could not read frame {fpath}; skipping.")
                continue

            basename = os.path.basename(fname)
            key = (split_name, basename)
            img_id = frame_index.get(key, None)

            if img_id is None:
                # Frame exists on disk but not in JSON
                n_missing_in_json += 1
                cat_masks = {}
            else:
                cat_masks = build_cat_masks_for_frame(img_id, anns_by_image, H, W)
                if cat_masks:
                    n_with_masks += 1

            overlaid = overlay_multi_category(img, cat_masks, CATEGORY_COLORS)
            writer.write(overlaid)
            n_written += 1

        writer.release()
        print(
            f"[VIDEO] Finished split {split_name}: {n_written} frames written, "
            f"{n_with_masks} had at least one mask, "
            f"{n_missing_in_json} frames had no JSON entry."
        )


def main():
    if not os.path.isdir(CASE_DIR):
        raise FileNotFoundError(f"CASE_DIR does not exist: {CASE_DIR}")
    if not os.path.isdir(SPLIT_ROOT):
        raise FileNotFoundError(f"SPLIT_ROOT does not exist: {SPLIT_ROOT}")

    print("CASE_DIR  :", CASE_DIR)
    print("SPLIT_ROOT:", SPLIT_ROOT)
    print("JSON_NAME :", JSON_NAME)

    reconstruct_videos_by_filename()


if __name__ == "__main__":
    main()
