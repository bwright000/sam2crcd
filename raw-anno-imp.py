"""
 SAM2 + CRCD pipeline (RAW + ANNOTATED + evaluation)

Environment setup (CPU only):

    conda activate sam2

    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install opencv-python pillow
    pip install huggingface_hub einops decord ffmpeg-python

SAM2 install from local clone:

    cd C:\path\to\sam2
    pip install -e .

This script supports two modes:

    RUN_MODE = "raw"
    RUN_MODE = "annotated"

RAW mode:
    - expects CRCD raw videos laid out as:
        DATA_ROOT_RAW/
            A_1/
              left/left.mp4
              right/right.mp4
            ...

ANNOTATED mode:
    - expects CRCD annotated cases laid out as:
        DATA_ROOT_ANNOT/
            C_1/
              C_1/
                split_imgs/split_imgs/split_0/00000.png ...
                annotations.json  (where 'annotations.json' = train.json / test.json)
            A_1/
              A_1/
                ...

    - computes IoU and Dice between SAM2 masks and COCO masks.

NOTE:
    COCO masks are assumed to be polygon segmentations.
    RLE segmentations are not currently decoded in this script.
"""

import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import cv2
import torch
import numpy as np

from sam2.sam2_video_predictor import SAM2VideoPredictor


# =============================== GLOBAL CONFIG ===============================

RUN_MODE = "annotated"   # "raw" or "annotated"

MODEL_NAME = "facebook/sam2-hiera-small"   # small = faster

# -------- RAW dataset config --------
DATA_ROOT_RAW = r"C:\Users\BenWright\CRCD\raw\original_dataset\original_dataset"

# How much of each RAW video to keep for SAM2
MAX_FRAMES_FOR_SAM2 = 60   # after subsampling
FRAME_STRIDE = 4           # use every 4th frame
TRIM_SUFFIX = "_sam2clip.mp4"
OUTPUT_SUFFIX_RAW = "_sam2.mp4"

# -------- ANNOTATED dataset config --------
DATA_ROOT_ANNOT = r"C:\Users\BenWright\CRCD\annotated"

# Which annotation JSON to use per case: "train", "test", or "annotations"
ANNOTATION_JSON_BASENAME = "train"   # e.g. C_1/C_1/train

OUTPUT_SUFFIX_ANNOT = "_sam2.mp4"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
MAX_FRAMES_PER_SPLIT = 0   # 0 => no limit, otherwise cap per split

# -------- Overlay config --------
MASK_COLOR = (0, 255, 0)    # BGR green
MASK_ALPHA = 0.2            # transparency

# ============================================================================


def overlay_masks(frame_bgr: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Overlay masks on a BGR frame and return:
      - the overlayed frame
      - the combined boolean mask [H,W] used (or None if invalid).

    masks can be:
      - [H, W]
      - [N, H, W]
      - [N, 1, H, W]
      - or anything where the last two dims are (H, W)
    """
    H, W, _ = frame_bgr.shape

    masks_bool = masks.astype(bool)

    if masks_bool.ndim == 2:
        combined = masks_bool
    elif masks_bool.ndim >= 3:
        leading_axes = tuple(range(masks_bool.ndim - 2))
        combined = np.any(masks_bool, axis=leading_axes)
    else:
        return frame_bgr, None

    if combined.shape != (H, W):
        return frame_bgr, None

    if not combined.any():
        return frame_bgr, combined

    overlay = frame_bgr.copy()
    color_layer = np.zeros_like(frame_bgr, dtype=np.uint8)
    color_layer[combined] = MASK_COLOR

    blended = (
        frame_bgr.astype(np.float32) * (1.0 - MASK_ALPHA)
        + color_layer.astype(np.float32) * MASK_ALPHA
    ).astype(np.uint8)

    out = frame_bgr.copy()
    out[combined] = blended[combined]
    return out, combined


# =========================== RAW VIDEO PIPELINE =============================


def build_short_clip(src_path: str) -> str:
    """
    Create a shorter, subsampled version of the video to keep RAM sane.
    - Uses only every FRAME_STRIDE-th frame.
    - Stops after MAX_FRAMES_FOR_SAM2 written frames.
    - Saves next to the original as <name>_sam2clip.mp4.

    If the trimmed file already exists, it is reused.
    """
    dirpath, filename = os.path.split(src_path)
    basename, ext = os.path.splitext(filename)
    dst_path = os.path.join(dirpath, f"{basename}{TRIM_SUFFIX}")

    if os.path.exists(dst_path):
        return dst_path  # reuse existing trimmed clip

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dst_path, fourcc, max(fps / FRAME_STRIDE, 1), (width, height))

    read_idx = 0
    written = 0

    while written < MAX_FRAMES_FOR_SAM2:
        ok, frame = cap.read()
        if not ok:
            break

        if read_idx % FRAME_STRIDE == 0:
            out.write(frame)
            written += 1

        read_idx += 1

    cap.release()
    out.release()

    print(f"    → Built trimmed clip: {dst_path} "
          f"(used {written} frames, read {read_idx} frames from source)")
    return dst_path


def process_video_raw(predictor: SAM2VideoPredictor, video_path: str, output_path: str) -> bool:
    """Run SAM2 on a single RAW video file (left/right video)."""
    print(f"  → [RAW VIDEO] Processing: {video_path}")

    trimmed_path = build_short_clip(video_path)

    cap = cv2.VideoCapture(trimmed_path)
    ok, first_frame = cap.read()
    if not ok:
        print("    ! ERROR: Could not read trimmed video")
        cap.release()
        return False

    H, W, _ = first_frame.shape

    box = np.array(
        [[int(0.2 * W), int(0.2 * H), int(0.8 * W), int(0.8 * H)]],
        dtype=np.float32,
    )

    with torch.inference_mode():
        state = predictor.init_state(trimmed_path)

        obj_id = 1
        _, out_obj_ids, _ = predictor.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=obj_id,
            box=box,
        )
        print("    → Initialized object ids:", out_obj_ids)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )

    frame_i = 0
    cap.release()
    cap = cv2.VideoCapture(trimmed_path)

    print("    → Propagating masks...")
    with torch.inference_mode():
        for frame_idx_prop, obj_ids_prop, mask_logits in predictor.propagate_in_video(state):
            ok, frame = cap.read()
            if not ok:
                break

            if mask_logits is not None:
                masks_np = (mask_logits > 0.0).cpu().numpy()
                frame, _ = overlay_masks(frame, masks_np)

            out.write(frame)
            frame_i += 1

            if frame_i % 10 == 0:
                print(f"    frame {frame_i}")

            if frame_i >= MAX_FRAMES_FOR_SAM2:
                break

    out.release()
    cap.release()

    print(f"    ✓ Saved (raw video) to: {output_path}  ({frame_i} frames)")
    return True


def run_raw_pipeline(predictor: SAM2VideoPredictor):
    """Process all RAW videos under DATA_ROOT_RAW."""
    cases = sorted(
        d for d in os.listdir(DATA_ROOT_RAW)
        if os.path.isdir(os.path.join(DATA_ROOT_RAW, d))
    )
    print("RAW cases:", cases)

    for case in cases:
        case_dir = os.path.join(DATA_ROOT_RAW, case)

        for side in ["left", "right"]:
            side_dir = os.path.join(case_dir, side)
            video_path = os.path.join(side_dir, f"{side}.mp4")
            if not os.path.exists(video_path):
                print(f"[SKIP] Missing video: {video_path}")
                continue

            output_path = os.path.join(side_dir, f"{side}{OUTPUT_SUFFIX_RAW}")
            os.makedirs(side_dir, exist_ok=True)
            process_video_raw(predictor, video_path, output_path)


# ======================= ANNOTATED + EVALUATION ======================

def collect_frame_paths(frames_dir: str) -> List[str]:
    files = [
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if os.path.isfile(os.path.join(frames_dir, f))
        and f.lower().endswith(IMAGE_EXTS)
    ]
    files.sort()
    return files


def load_coco_annotations(json_path: str):
    """
    Load COCO-style annotations and build fast lookup tables.

    Returns:
        images_by_id: dict[image_id] -> image_dict
        anns_by_image: dict[image_id] -> list[ann_dict]
        basenames_to_image_ids: dict[basename] -> list[image_id]
    """
    print("    → Loading COCO annotations from:", json_path)
    with open(json_path, "r") as f:
        coco = json.load(f)

    images_by_id: Dict[int, dict] = {}
    basenames_to_image_ids: Dict[str, List[int]] = defaultdict(list)

    for img in coco.get("images", []):
        img_id = img["id"]
        images_by_id[img_id] = img
        basename = os.path.basename(img["file_name"])
        basenames_to_image_ids[basename].append(img_id)

    anns_by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        anns_by_image[img_id].append(ann)

    return images_by_id, anns_by_image, basenames_to_image_ids


def polygons_to_mask(polygons: List[List[float]], height: int, width: int) -> np.ndarray:
    """
    Convert COCO polygon segmentation to a binary mask.
    polygons: list of [x0, y0, x1, y1, ...] lists.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        if len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def find_image_id_for_frame(
    basename: str,
    split_name: str,
    basenames_to_image_ids: Dict[str, List[int]],
    images_by_id: Dict[int, dict],
) -> Optional[int]:
    """
    Try to find the COCO image_id corresponding to a given frame basename and split name.
    Strategy:
      - Use basename to get candidate image_ids.
      - If multiple candidates, prefer ones whose file_name contains the split_name.
    """
    candidates = basenames_to_image_ids.get(basename, [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # More than one; disambiguate using split_name if possible
    for img_id in candidates:
        file_name = images_by_id[img_id]["file_name"].replace("\\", "/")
        if split_name in file_name:
            return img_id
    # Fallback: first candidate
    return candidates[0]


def compute_iou_and_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    eps = 1e-6

    iou = inter / (union + eps)
    dice = 2 * inter / (pred_sum + gt_sum + eps)
    return iou, dice


def process_split_annotated(
    predictor: SAM2VideoPredictor,
    split_dir: str,
    split_name: str,
    output_path: str,
    images_by_id: Dict[int, dict],
    anns_by_image: Dict[int, List[dict]],
    basenames_to_image_ids: Dict[str, List[int]],
) -> Tuple[float, float, int]:
    """
    Run SAM2 on a folder of frames for one split and evaluate against COCO masks.

    Returns:
        mean_iou, mean_dice, n_eval_frames
    """
    print(f"  → [ANNOT SPLIT] {split_name} in {os.path.dirname(split_dir)}")

    frame_paths = collect_frame_paths(split_dir)
    if not frame_paths:
        print("    ! ERROR: No images found in", split_dir)
        return 0.0, 0.0, 0

    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        print("    ! ERROR: Could not read first image in", split_dir)
        return 0.0, 0.0, 0

    H, W, _ = first_frame.shape

    box = np.array(
        [[int(0.2 * W), int(0.2 * H), int(0.8 * W), int(0.8 * H)]],
        dtype=np.float32,
    )

    # Build mapping from frame index -> (image_id, gt_mask)
    gt_masks_by_idx: Dict[int, np.ndarray] = {}

    for idx, frame_path in enumerate(frame_paths):
        basename = os.path.basename(frame_path)
        img_id = find_image_id_for_frame(basename, split_name, basenames_to_image_ids, images_by_id)
        if img_id is None:
            continue
        anns = anns_by_image.get(img_id, [])
        if not anns:
            continue

        # Combine all polygons for this image
        combined_gt = np.zeros((H, W), dtype=bool)
        for ann in anns:
            seg = ann.get("segmentation", None)
            if seg is None:
                continue
            # Only support polygon segmentations (list of lists)
            if isinstance(seg, list):
                mask = polygons_to_mask(seg, H, W)
                combined_gt |= mask
            else:
                # RLE or other – not handled in this script
                continue

        if combined_gt.any():
            gt_masks_by_idx[idx] = combined_gt

    if not gt_masks_by_idx:
        print("    ! WARNING: No usable GT masks found for this split.")
    else:
        print(f"    → GT masks available for {len(gt_masks_by_idx)} frames in this split.")

    # Initialize SAM2 state using the frames directory
    with torch.inference_mode():
        state = predictor.init_state(split_dir)

        obj_id = 1
        _, out_obj_ids, _ = predictor.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=obj_id,
            box=box,
        )
        print("    → Initialized object ids:", out_obj_ids)

    fps = 10
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )

    ious: List[float] = []
    dices: List[float] = []

    frame_i = 0
    print("    → Propagating masks over frame sequence...")
    with torch.inference_mode():
        for frame_idx_prop, obj_ids_prop, mask_logits in predictor.propagate_in_video(state):
            if frame_idx_prop >= len(frame_paths):
                break

            frame = cv2.imread(frame_paths[frame_idx_prop])
            if frame is None:
                break

            if mask_logits is not None:
                masks_np = (mask_logits > 0.0).cpu().numpy()
                frame, pred_combined = overlay_masks(frame, masks_np)
            else:
                pred_combined = None

            # If we have GT for this frame and a predicted mask, compute metrics
            if pred_combined is not None and frame_idx_prop in gt_masks_by_idx:
                gt_mask = gt_masks_by_idx[frame_idx_prop]
                iou, dice = compute_iou_and_dice(pred_combined, gt_mask)
                ious.append(iou)
                dices.append(dice)

            out.write(frame)
            frame_i += 1

            if frame_i % 50 == 0:
                print(f"    frame {frame_i}")

            if MAX_FRAMES_PER_SPLIT > 0 and frame_i >= MAX_FRAMES_PER_SPLIT:
                break

    out.release()
    print(f"    ✓ Saved (annot split) to: {output_path}  ({frame_i} frames)")

    if ious:
        mean_iou = float(np.mean(ious))
        mean_dice = float(np.mean(dices))
        print(f"    → Split metrics: IoU={mean_iou:.4f}, Dice={mean_dice:.4f} over {len(ious)} frames")
        return mean_iou, mean_dice, len(ious)
    else:
        print("    ! No frames with both prediction and GT – metrics undefined.")
        return 0.0, 0.0, 0


def run_annotated_pipeline(predictor: SAM2VideoPredictor):
    """Process all annotated cases under DATA_ROOT_ANNOT and evaluate."""
    cases = sorted(
        d for d in os.listdir(DATA_ROOT_ANNOT)
        if os.path.isdir(os.path.join(DATA_ROOT_ANNOT, d))
    )
    print("ANNOTATED cases:", cases)

    global_ious: List[float] = []
    global_dices: List[float] = []

    for case in cases:
        outer_case_dir = os.path.join(DATA_ROOT_ANNOT, case)
        # Some cases have nested folder with same name (e.g., C_1/C_1/)
        inner_case_dir = os.path.join(outer_case_dir, case)
        if os.path.isdir(inner_case_dir):
            case_dir = inner_case_dir
        else:
            case_dir = outer_case_dir

        # Load annotations JSON for this case
        json_candidate = os.path.join(case_dir, f"{ANNOTATION_JSON_BASENAME}.json")
        if not os.path.exists(json_candidate):
            # Fallback to "annotations.json"
            json_candidate = os.path.join(case_dir, "annotations")
            if not json_candidate.endswith(".json"):
                json_candidate = json_candidate + ".json"
        if not os.path.exists(json_candidate):
            print(f"[SKIP] No annotation JSON found for case {case}")
            continue

        images_by_id, anns_by_image, basenames_to_image_ids = load_coco_annotations(json_candidate)

        # split_imgs/split_imgs/split_x
        splits_root = os.path.join(case_dir, "split_imgs", "split_imgs")
        if not os.path.isdir(splits_root):
            print(f"[SKIP] No split_imgs/split_imgs found for case {case}")
            continue

        split_names = sorted(
            d for d in os.listdir(splits_root)
            if os.path.isdir(os.path.join(splits_root, d))
        )
        print(f"\n=== Case {case} | splits: {split_names} ===")

        case_ious: List[float] = []
        case_dices: List[float] = []

        for split_name in split_names:
            split_dir = os.path.join(splits_root, split_name)
            output_path = os.path.join(split_dir, f"{split_name}{OUTPUT_SUFFIX_ANNOT}")
            os.makedirs(split_dir, exist_ok=True)

            mean_iou, mean_dice, n_eval = process_split_annotated(
                predictor,
                split_dir,
                split_name,
                output_path,
                images_by_id,
                anns_by_image,
                basenames_to_image_ids,
            )

            if n_eval > 0:
                case_ious.append(mean_iou)
                case_dices.append(mean_dice)
                global_ious.append(mean_iou)
                global_dices.append(mean_dice)

        if case_ious:
            print(f"\n>>> Case {case} mean IoU={np.mean(case_ious):.4f}, "
                  f"mean Dice={np.mean(case_dices):.4f} over {len(case_ious)} splits")

    if global_ious:
        print("\n=== GLOBAL METRICS OVER ALL CASES ===")
        print(f"Mean IoU = {np.mean(global_ious):.4f}")
        print(f"Mean Dice = {np.mean(global_dices):.4f}")
    else:
        print("\nNo global metrics – no evaluated frames with GT + predictions.")


# ================================ MAIN ======================================

def main():
    device = "cpu"
    print("Using device:", device)
    print("Model:", MODEL_NAME)
    print("RUN_MODE:", RUN_MODE)

    predictor = SAM2VideoPredictor.from_pretrained(
        MODEL_NAME,
        device=device,
    )

    if RUN_MODE == "raw":
        print("Running RAW video pipeline...")
        run_raw_pipeline(predictor)
    elif RUN_MODE == "annotated":
        print("Running ANNOTATED pipeline with GT evaluation...")
        run_annotated_pipeline(predictor)
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()
