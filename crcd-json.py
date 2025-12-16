import os
import json
from collections import defaultdict, Counter


# ======================= CONFIG =======================
# Root directory for a single CRCD case (e.g. C_1).
# Expected to contain COCO JSONs and extracted video frames.
CASE_DIR = r"YOUR DIRECTORY"

# COCO-format annotation files.
# These typically correspond to frame-level annotations
# produced for SAM2 mask initialization.
TRAIN_JSON_PATH = os.path.join(CASE_DIR, "train.json")
TEST_JSON_PATH  = os.path.join(CASE_DIR, "test.json")  # set to None if absent

# Directory containing per-video or per-clip frame folders.
# Used to validate that JSON image counts align with frames on disk.
SPLIT_ROOT = os.path.join(CASE_DIR, "split_imgs", "split_imgs")

# Supported image extensions for frame counting.
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
# =====================================================


def load_coco_json(json_path: str):
    """
    Load a COCO-format JSON annotation file.

    Parameters
    ----------
    json_path : str
        Path to a COCO-style JSON file containing `images`,
        `annotations`, and `categories`.

    Returns
    -------
    images : list of dict
        COCO `images` entries. Each item typically includes
        `id`, `file_name`, `width`, and `height`.
    annotations : list of dict
        COCO `annotations` entries. Each annotation corresponds
        to one mask instance on a single frame.
    categories : list of dict
        COCO `categories` entries defining semantic classes.

    Notes
    -----
    - If the JSON path is missing or disabled, empty lists are returned.
    - This function performs *no schema validation* beyond key existence.
    """
    if not json_path or not os.path.exists(json_path):
        print(f"[INFO] JSON not found or disabled: {json_path}")
        return [], [], []

    with open(json_path, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    return images, annotations, categories


def category_summary(name, categories, annotations):
    """
    Print a per-category annotation count summary.

    Parameters
    ----------
    name : str
        Label for the dataset split (e.g. 'TRAIN', 'TEST').
    categories : list of dict
        COCO categories, each with `id` and `name`.
    annotations : list of dict
        COCO annotations containing `category_id`.

    Notes
    -----
    Useful for quickly verifying:
    - Class imbalance
    - Missing or unused categories
    - Train/test category mismatches
    """
    print(f"\n=== CATEGORY SUMMARY ({name}) ===")

    if not categories:
        print("No categories found.")
        return

    # Map category_id -> human-readable name
    cat_id_to_name = {c["id"]: c["name"] for c in categories}

    # Count number of masks per category
    counts = Counter(ann["category_id"] for ann in annotations)

    for cat_id, cat_name in cat_id_to_name.items():
        print(
            f"Category {cat_id:>2} | {cat_name:<15} | "
            f"Masks: {counts.get(cat_id, 0)}"
        )


def image_annotation_stats(name, images, annotations):
    """
    Compute and print image-level annotation statistics.

    Parameters
    ----------
    name : str
        Dataset label (TRAIN / TEST / COMBINED).
    images : list of dict
        COCO image entries.
    annotations : list of dict
        COCO annotation entries.

    Notes
    -----
    This function highlights assumptions critical for SAM2:
    - Whether masks exist on *every* frame
    - How many instances are typically initialized per frame
    - Whether some frames are completely unannotated
    """
    print(f"\n=== IMAGE / ANNOTATION STATS ({name}) ===")

    # Group annotations by image_id
    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    total_images = len(images)
    total_annotations = len(annotations)

    images_with_masks = sum(1 for img in images if img["id"] in anns_by_image)
    images_without_masks = total_images - images_with_masks

    # Masks per annotated image
    masks_per_image = [
        len(anns_by_image[img["id"]])
        for img in images
        if img["id"] in anns_by_image
    ]

    print(f"Total images           : {total_images}")
    print(f"Total annotations      : {total_annotations}")
    print(f"Images WITH masks      : {images_with_masks}")
    print(f"Images WITHOUT masks   : {images_without_masks}")

    if masks_per_image:
        print(f"Min masks per image    : {min(masks_per_image)}")
        print(f"Max masks per image    : {max(masks_per_image)}")
        print(f"Mean masks per image   : {sum(masks_per_image)/len(masks_per_image):.2f}")
    else:
        print("No masks found on any image.")


def file_name_sample(name, images, n=5):
    """
    Print a small sample of image IDs and filenames.

    Parameters
    ----------
    name : str
        Dataset split label.
    images : list of dict
        COCO image entries.
    n : int, optional
        Number of samples to print (default: 5).

    Notes
    -----
    Useful for sanity-checking:
    - Frame ordering
    - Naming conventions
    - Alignment with frame folders on disk
    """
    print(f"\n=== FILE NAME SAMPLING ({name}, first {n}) ===")
    for img in images[:n]:
        print(f"id={img['id']:<5} file_name={img.get('file_name')}")


def count_frames_on_disk(split_root: str):
    """
    Count image frames present on disk across all split folders.

    Parameters
    ----------
    split_root : str
        Root directory containing per-video or per-clip subfolders.

    Returns
    -------
    total_frames : int
        Total number of image files found.
    split_frame_counts : dict
        Mapping {split_name: frame_count}.

    Notes
    -----
    This is the *ground truth* for how many frames SAM2 can
    actually process, independent of JSON metadata.
    """
    print("\n=== FRAME COUNTS ON DISK (C_1) ===")

    if not os.path.isdir(split_root):
        print(f"SPLIT_ROOT does not exist: {split_root}")
        return 0, {}

    split_names = sorted(
        d for d in os.listdir(split_root)
        if os.path.isdir(os.path.join(split_root, d))
    )
    if not split_names:
        print(f"No split folders found under {split_root}")
        return 0, {}

    split_frame_counts = {}
    total_frames = 0

    for split_name in split_names:
        split_dir = os.path.join(split_root, split_name)
        frames = [
            f for f in os.listdir(split_dir)
            if os.path.isfile(os.path.join(split_dir, f))
            and f.lower().endswith(IMAGE_EXTS)
        ]
        n = len(frames)
        split_frame_counts[split_name] = n
        total_frames += n

    for split_name, n in split_frame_counts.items():
        print(f"Split {split_name:<10} : {n} frames")

    print(f"\nTotal frames on disk   : {total_frames}")

    return total_frames, split_frame_counts


def compare_json_vs_frames(label, num_images: int, num_frames: int):
    """
    Compare COCO image counts against frames present on disk.

    Parameters
    ----------
    label : str
        Dataset label (TRAIN / TEST / COMBINED).
    num_images : int
        Number of images listed in the COCO JSON.
    num_frames : int
        Number of frames found on disk.

    Notes
    -----
    Any mismatch here is a red flag for:
    - Dropped frames
    - Incorrect JSON generation
    - Broken assumptions in SAM2 propagation
    """
    print(f"\n=== JSON vs DISK FRAME COUNT ({label}) ===")
    print(f"JSON images : {num_images}")
    print(f"Frames on disk: {num_frames}")

    diff = num_images - num_frames
    if diff == 0:
        print("✔ JSON image count matches frame count on disk.")
    elif diff > 0:
        print(f"⚠ JSON has {diff} MORE images than frames on disk.")
    else:
        print(f"⚠ JSON has {-diff} FEWER images than frames on disk.")


def main():
    """
    Run full diagnostic analysis for CRCD COCO annotations.

    Workflow
    --------
    1. Load train and test COCO JSONs
    2. Report category and image-level statistics
    3. Merge splits for a combined view
    4. Count actual frames on disk
    5. Compare JSON metadata vs physical frames

    Notes
    -----
    This script is intended as a *pre-flight check* before:
    - SAM2 video propagation
    - Temporal evaluation
    - Dataset release or training
    """
    print("\n==============================")
    print("CRCD JSON DIAGNOSTICS (TRAIN + TEST)")
    print("==============================")
    print("CASE_DIR  :", CASE_DIR)
    print("TRAIN_JSON:", TRAIN_JSON_PATH)
    print("TEST_JSON :", TEST_JSON_PATH)
    print("SPLIT_ROOT:", SPLIT_ROOT)

    # ---- Load train + test ----
    train_images, train_anns, train_cats = load_coco_json(TRAIN_JSON_PATH)
    test_images,  test_anns,  test_cats  = load_coco_json(TEST_JSON_PATH)

    print("\n=== BASIC COUNTS ===")
    print("Train images      :", len(train_images))
    print("Train annotations :", len(train_anns))
    print("Test images       :", len(test_images))
    print("Test annotations  :", len(test_anns))

    # ---- Per-split stats ----
    if train_images:
        category_summary("TRAIN", train_cats, train_anns)
        image_annotation_stats("TRAIN", train_images, train_anns)
        file_name_sample("TRAIN", train_images, n=5)

    if test_images:
        category_summary("TEST", test_cats, test_anns)
        image_annotation_stats("TEST", test_images, test_anns)
        file_name_sample("TEST", test_images, n=5)

    # ---- Combined view ----
    print("\n=== COMBINED (TRAIN + TEST) ===")

    # Merge images by unique ID
    combined_images_by_id = {img["id"]: img for img in train_images}
    for img in test_images:
        combined_images_by_id.setdefault(img["id"], img)

    combined_images = list(combined_images_by_id.values())
    combined_anns = train_anns + test_anns

    image_annotation_stats("COMBINED", combined_images, combined_anns)

    # ---- Frames on disk ----
    num_frames, _ = count_frames_on_disk(SPLIT_ROOT)

    # ---- Consistency checks ----
    compare_json_vs_frames("TRAIN", len(train_images), num_frames)
    if test_images:
        compare_json_vs_frames("TEST", len(test_images), num_frames)
    compare_json_vs_frames("COMBINED", len(combined_images), num_frames)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
