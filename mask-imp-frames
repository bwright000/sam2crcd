"""
Quick sanity check for CRCD annotated masks.

What it does:
  - Loads global COCO JSON (train/test/annotations).
  - For a sample of images:
      * loads the image from disk
      * converts segmentation polygons to binary masks
      * overlays masks on the image
      * prints basic stats
      * saves overlay PNGs to an output folder

Run:
    conda activate sam2
    python inspect_crcd_masks.py
"""

import os
import json
import random
from collections import defaultdict
from typing import List, Dict

import cv2
import numpy as np


# ======================= CONFIG =======================

# Root folder containing annotated images and JSON
DATA_ROOT_ANNOT = r"C:\Users\BenWright\CRCD\annotated\C_1\C_1"

# Which JSON file to use at the root (global)
ANNOT_JSON_NAME = "train.json"   # or "test.json", "annotations.json"

# How many random images to inspect
NUM_SAMPLES = 10

# Where to write overlay images
OVERLAY_OUT_DIR = os.path.join(DATA_ROOT_ANNOT, "mask_overlays")

# Image root
IMAGE_ROOT = r"C:\Users\BenWright\CRCD\annotated\C_1\C_1\split_imgs\split_imgs"

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


def overlay_mask_on_image(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Overlay a single binary mask (H,W) on a BGR image (H,W,3) in semi-transparent red.
    """
    h, w, _ = img_bgr.shape
    if mask.shape != (h, w):
        # Shape mismatch: just return original
        return img_bgr

    color = np.array([0, 0, 255], dtype=np.uint8)  # red in BGR
    alpha = 0.4

    color_layer = np.zeros_like(img_bgr, dtype=np.uint8)
    color_layer[mask] = color

    blended = (
        img_bgr.astype(np.float32) * (1 - alpha)
        + color_layer.astype(np.float32) * alpha
    ).astype(np.uint8)

    out = img_bgr.copy()
    out[mask] = blended[mask]
    return out


def main():
    json_path = os.path.join(DATA_ROOT_ANNOT, ANNOT_JSON_NAME)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find annotation JSON: {json_path}")

    print(f"Loading annotations from: {json_path}")
    with open(json_path, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    print(f"Found {len(images)} images, {len(annotations)} annotations in JSON.")

    # Build lookup: image_id -> image_dict
    images_by_id: Dict[int, dict] = {img["id"]: img for img in images}

    # Build lookup: image_id -> list[annotation_dict]
    anns_by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in annotations:
        img_id = ann["image_id"]
        anns_by_image[img_id].append(ann)

    # Create output directory
    os.makedirs(OVERLAY_OUT_DIR, exist_ok=True)

    # Choose random image_ids to inspect
    if len(images) <= NUM_SAMPLES:
        sample_images = images
    else:
        sample_images = random.sample(images, NUM_SAMPLES)

    print(f"Sampling {len(sample_images)} images for inspection...")

    for img in sample_images:
        img_id = img["id"]
        file_name = img["file_name"]
        # CRCD gives folder names, not actual image paths.
        # Convert "./split_42/" → "C:\...\split_imgs\split_imgs\split_42"
        folder_name = os.path.basename(file_name.strip("./"))  # "split_42"

        folder_path = os.path.join(IMAGE_ROOT, folder_name)

        # Now pick the first frame (00000.png)
        frames = sorted(
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        if not frames:
            print("  ! No frames found in folder, skipping.")
            continue

        img_path = os.path.join(folder_path, frames[0])


        print(f"\n=== image_id={img_id} ===")
        print(f"file_name: {file_name}")
        print(f"full path: {img_path}")

        if not os.path.exists(img_path):
            print("  ! Image file missing on disk, skipping.")
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print("  ! Failed to read image, skipping.")
            continue

        H, W, _ = img_bgr.shape
        anns = anns_by_image.get(img_id, [])
        print(f"  #annotations for this image: {len(anns)}")

        if not anns:
            print("  ! No annotations for this image in JSON.")
            # still save raw image for reference
            out_no_mask = os.path.join(OVERLAY_OUT_DIR, f"img_{img_id}_nomask.png")
            cv2.imwrite(out_no_mask, img_bgr)
            continue

        # Build combined mask and per-ann stats
        combined_mask = np.zeros((H, W), dtype=bool)
        ann_stats = []

        for ann in anns:
            seg = ann.get("segmentation", None)
            if not isinstance(seg, list) or len(seg) == 0:
                # RLE or empty: skip in this script
                continue

            mask = polygons_to_mask(seg, H, W)
            area_pixels = mask.sum()
            area_frac = area_pixels / float(H * W)

            ann_stats.append({
                "id": ann.get("id", None),
                "category_id": ann.get("category_id", None),
                "area_pixels": int(area_pixels),
                "area_frac": float(area_frac),
            })

            combined_mask |= mask

        if not combined_mask.any():
            print("  ! No usable polygon masks for this image (maybe all RLE or empty).")
            out_no_mask = os.path.join(OVERLAY_OUT_DIR, f"img_{img_id}_nomask.png")
            cv2.imwrite(out_no_mask, img_bgr)
            continue

        # Print simple stats
        print(f"  usable polygon annotations: {len(ann_stats)}")
        for s in ann_stats:
            print(f"    ann_id={s['id']}, cat={s['category_id']}, "
                  f"area_pixels={s['area_pixels']}, area_frac={s['area_frac']:.4f}")

        # Overlay combined mask
        overlaid = overlay_mask_on_image(img_bgr, combined_mask)
        out_path = os.path.join(OVERLAY_OUT_DIR, f"img_{img_id}_overlay.png")
        cv2.imwrite(out_path, overlaid)
        print(f"  → Saved overlay to: {out_path}")

    print("\nDone. Check", OVERLAY_OUT_DIR, "for overlay PNGs.")


if __name__ == "__main__":
    main()
