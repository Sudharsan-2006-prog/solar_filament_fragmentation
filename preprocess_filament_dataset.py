"""
Solar Filament Segmentation Dataset (MAGFiLO / MLEcoFi 2026) Preprocessing Pipeline.

This script performs end-to-end preprocessing of the raw dataset:
1. Parses COCO-format multi-annotator JSON annotations.
2. Performs multi-annotator consensus / majority-voting fusion.
3. Generates high-resolution (2048x2048) and scaled (512x512) semantic & binary masks.
4. Generates filament spine (centerline skeleton) masks.
5. Generates inter-annotator agreement / confidence heatmaps.
6. Performs Contrast-Limited Adaptive Histogram Equalization (CLAHE) for solar limb/fibril enhancement.
7. Produces normalized YOLOv8/v11 segmentation polygon annotations.
8. Creates an observatory-stratified 80/20 train/validation split without data leakage.
9. Exports comprehensive metadata manifests (CSV and JSON).
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm
import pandas as pd
from collections import Counter


# Categories mapping
CATEGORY_MAP = {
    1: {"name": "Left", "description": "Sinistral (left-bearing) filament chirality"},
    2: {"name": "Right", "description": "Dextral (right-bearing) filament chirality"},
    3: {"name": "Unidentifiable", "description": "Filament chirality cannot be determined"},
    4: {"name": "Ambiguous", "description": "Conflicting or ambiguous chirality features"}
}

STATION_MAP = {
    "Bh": "Big Bear Solar Observatory, California, USA",
    "Ch": "Cerro Tololo Inter-American Observatory, Chile",
    "Lh": "Learmonth Solar Observatory, Western Australia",
    "Mh": "Mauna Loa Solar Observatory, Hawaii, USA",
    "Th": "Teide Observatory, Canary Islands, Spain",
    "Uh": "Udaipur Solar Observatory, Rajasthan, India"
}


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess MAGFiLO Solar Filament Segmentation Dataset")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=r"D:\Downloads\filament-segmentation-2026\MAGFiLO_1.0_Kaggle_2026",
        help="Path to raw dataset directory containing train/ and test/"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"D:\Downloads\filament-segmentation-2026\preprocessed",
        help="Directory to save preprocessed dataset"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=512,
        help="Target square image dimension (e.g. 512, 1024)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.20,
        help="Validation split ratio (default: 0.20 for 80/20 split)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split reproducibility"
    )
    return parser.parse_args()


def extract_station_and_date(filename):
    """
    Extract date and GONG station observatory code from filename.
    Pattern: YYYYMMDDHHMMSS<Station>h.jpeg
    Example: 20110109104734Ch.jpeg -> Date: 2011-01-09 10:47:34, Station: Ch
    """
    base = os.path.basename(filename)
    try:
        year = base[0:4]
        month = base[4:6]
        day = base[6:8]
        hour = base[8:10]
        minute = base[10:12]
        second = base[12:14]
        station = base[14:16]
        date_str = f"{year}-{month}-{day} {hour}:{minute}:{second}"
    except Exception:
        station = "Unknown"
        date_str = ""
    return station, date_str


def rasterize_single_annotator(anns, width=2048, height=2048):
    """
    Rasterize an individual annotator's polygons and spines.
    Returns:
        class_mask: uint8 array with values in [0, 4]
        spine_mask: uint8 array with values in [0, 1]
    """
    class_img = Image.new("L", (width, height), 0)
    spine_img = Image.new("L", (width, height), 0)
    
    draw_class = ImageDraw.Draw(class_img)
    draw_spine = ImageDraw.Draw(spine_img)
    
    for ann in anns:
        cat_id = int(ann.get("category_id", 0))
        # Polygons
        for poly in ann.get("segmentation", []):
            if len(poly) >= 6:
                draw_class.polygon(poly, fill=cat_id)
        # Spines (filament backbone skeleton)
        spine = ann.get("spine", [])
        if len(spine) >= 4:
            draw_spine.line(spine, fill=1, width=2)
            
    return np.array(class_img, dtype=np.uint8), np.array(spine_img, dtype=np.uint8)


def fuse_multi_annotator(annotator_anns_list, width=2048, height=2048):
    """
    Fuse annotations from 1 to K annotators.
    
    Consensus Strategy:
    - K = 1: Single annotator ground truth.
    - K = 2: Pixel is filament if at least 1 annotator marked it (union).
    - K = 3: Pixel is filament if at least 2 annotators marked it (majority vote).
    - Agreement map: Count of annotators agreeing on filament presence (0 to K).
    - Multiclass label: Plurality (mode) of class IDs among annotators who marked the pixel.
    - Spine mask: Union of all spines.
    """
    num_annotators = len(annotator_anns_list)
    
    class_masks = []
    spine_masks = []
    
    for anns in annotator_anns_list:
        c_mask, s_mask = rasterize_single_annotator(anns, width, height)
        class_masks.append(c_mask)
        spine_masks.append(s_mask)
        
    binary_masks = [(m > 0).astype(np.uint8) for m in class_masks]
    agreement_map = np.sum(binary_masks, axis=0).astype(np.uint8)
    
    # Consensus threshold
    if num_annotators <= 2:
        consensus_threshold = 1
    else:  # K >= 3
        consensus_threshold = 2
        
    consensus_binary = (agreement_map >= consensus_threshold).astype(np.uint8)
    
    # Consensus multiclass
    consensus_multiclass = np.zeros((height, width), dtype=np.uint8)
    positive_pixels = np.where(consensus_binary > 0)
    
    if len(positive_pixels[0]) > 0:
        stacked_classes = np.stack(class_masks, axis=0) # shape: (K, H, W)
        for y, x in zip(positive_pixels[0], positive_pixels[1]):
            pixel_labels = [stacked_classes[k, y, x] for k in range(num_annotators) if stacked_classes[k, y, x] > 0]
            if pixel_labels:
                # Mode class
                counts = Counter(pixel_labels)
                most_common = counts.most_common()
                # Check for tie between different classes
                if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                    # Tie: if either is Ambiguous (4), use 4; otherwise pick the first
                    consensus_multiclass[y, x] = 4 if 4 in counts else most_common[0][0]
                else:
                    consensus_multiclass[y, x] = most_common[0][0]
                    
    # Spine consensus: union of spines
    consensus_spine = (np.sum(spine_masks, axis=0) > 0).astype(np.uint8)
    
    return consensus_multiclass, consensus_binary, consensus_spine, agreement_map


def extract_yolo_polygons(binary_mask, multiclass_mask, target_w=512, target_h=512):
    """
    Extract normalized polygon contours from mask for YOLOv8/v11-seg format.
    Output lines: <class_id_0_indexed> <x1> <y1> <x2> <y2> ...
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    yolo_lines = []
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:  # filter noise
            continue
        poly = cnt.reshape(-1, 2)
        if len(poly) < 3:
            continue
            
        # Determine class inside this contour
        mask_roi = np.zeros((target_h, target_w), dtype=np.uint8)
        cv2.drawContours(mask_roi, [cnt], -1, 1, -1)
        classes_in_roi = multiclass_mask[mask_roi > 0]
        classes_in_roi = classes_in_roi[classes_in_roi > 0]
        if len(classes_in_roi) == 0:
            continue
        class_mode = Counter(classes_in_roi).most_common(1)[0][0]
        class_yolo = int(class_mode) - 1  # 0-indexed: 0=Left, 1=Right, 2=Unidentifiable, 3=Ambiguous
        
        # Normalize coordinates
        norm_coords = []
        for x, y in poly:
            norm_coords.append(f"{x / target_w:.6f}")
            norm_coords.append(f"{y / target_h:.6f}")
            
        yolo_lines.append(f"{class_yolo} " + " ".join(norm_coords))
        
    return yolo_lines


def apply_clahe(img_gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast-Limited Adaptive Histogram Equalization.
    Enhances contrast of faint chromospheric filaments and fibrils.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_gray)


def main():
    args = parse_args()
    
    train_json_path = os.path.join(args.raw_dir, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    train_img_dir = os.path.join(args.raw_dir, "train", "train_images")
    test_img_dir = os.path.join(args.raw_dir, "test", "test_images")
    
    if not os.path.exists(train_json_path):
        raise FileNotFoundError(f"Training JSON not found at: {train_json_path}")
    if not os.path.exists(train_img_dir):
        raise FileNotFoundError(f"Train images directory not found at: {train_img_dir}")
        
    print(f"=== Starting Solar Filament Segmentation Preprocessing ===")
    print(f"Raw data directory : {args.raw_dir}")
    print(f"Output directory   : {args.output_dir}")
    print(f"Target image size  : {args.target_size}x{args.target_size}")
    print(f"Validation ratio   : {args.val_ratio:.2f}")
    
    # Setup output directories
    out = args.output_dir
    dirs_to_create = [
        # Scaled images
        os.path.join(out, f"images_{args.target_size}", "train"),
        os.path.join(out, f"images_{args.target_size}", "val"),
        os.path.join(out, f"images_{args.target_size}", "test"),
        # CLAHE enhanced images
        os.path.join(out, f"images_clahe_{args.target_size}", "train"),
        os.path.join(out, f"images_clahe_{args.target_size}", "val"),
        os.path.join(out, f"images_clahe_{args.target_size}", "test"),
        # Scaled masks
        os.path.join(out, f"masks_{args.target_size}", "multiclass", "train"),
        os.path.join(out, f"masks_{args.target_size}", "multiclass", "val"),
        os.path.join(out, f"masks_{args.target_size}", "binary", "train"),
        os.path.join(out, f"masks_{args.target_size}", "binary", "val"),
        os.path.join(out, f"masks_{args.target_size}", "spine", "train"),
        os.path.join(out, f"masks_{args.target_size}", "spine", "val"),
        os.path.join(out, f"masks_{args.target_size}", "agreement", "train"),
        os.path.join(out, f"masks_{args.target_size}", "agreement", "val"),
        # Full resolution 2048x2048 masks
        os.path.join(out, "masks_full_2048", "multiclass"),
        os.path.join(out, "masks_full_2048", "binary"),
        os.path.join(out, "masks_full_2048", "spine"),
        # YOLO segmentation format annotations
        os.path.join(out, "labels_yolo", "train"),
        os.path.join(out, "labels_yolo", "val"),
        # Metadata
        os.path.join(out, "metadata"),
    ]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
        
    # 1. Load JSON Annotations
    print("\n[Step 1/5] Loading annotations JSON...")
    with open(train_json_path, "r") as f:
        coco_data = json.load(f)
        
    print(f"Total annotations in JSON: {len(coco_data['annotations'])}")
    print(f"Total image records in JSON: {len(coco_data['images'])}")
    
    # Group annotations by image_id
    ann_by_image_id = {}
    for ann in coco_data["annotations"]:
        ann_by_image_id.setdefault(ann["image_id"], []).append(ann)
        
    # Map filename -> list of image_ids (multi-annotator mapping)
    fn_to_image_ids = {}
    for img_rec in coco_data["images"]:
        fn_to_image_ids.setdefault(img_rec["file_name"], []).append(img_rec["id"])
        
    unique_train_files = sorted(list(fn_to_image_ids.keys()))
    print(f"Unique physical train images: {len(unique_train_files)}")
    
    # 2. Stratified Train / Validation Split by Observatory Station
    print("\n[Step 2/5] Creating stratified Train/Validation split...")
    station_to_files = {}
    for fn in unique_train_files:
        station, _ = extract_station_and_date(fn)
        station_to_files.setdefault(station, []).append(fn)
        
    np.random.seed(args.seed)
    train_files = []
    val_files = []
    
    for station, files in sorted(station_to_files.items()):
        files_shuffled = np.random.permutation(files).tolist()
        n_val = max(1, int(round(len(files_shuffled) * args.val_ratio)))
        val_files.extend(files_shuffled[:n_val])
        train_files.extend(files_shuffled[n_val:])
        print(f"  Station {station} ({STATION_MAP.get(station, 'Unknown')}): {len(files)} total -> {len(files_shuffled)-n_val} train, {n_val} val")
        
    print(f"Total Train images: {len(train_files)} ({len(train_files)/len(unique_train_files)*100:.1f}%)")
    print(f"Total Val images  : {len(val_files)} ({len(val_files)/len(unique_train_files)*100:.1f}%)")
    
    split_map = {fn: "train" for fn in train_files}
    split_map.update({fn: "val" for fn in val_files})
    
    # 3. Process Train and Validation Images and Masks
    print(f"\n[Step 3/5] Processing images and generating consensus masks (target: {args.target_size}x{args.target_size})...")
    manifest_rows = []
    global_class_counts = Counter()
    total_filament_pixels = 0
    total_image_pixels = 0
    
    for fn in tqdm(unique_train_files, desc="Train/Val processing"):
        img_path = os.path.join(train_img_dir, fn)
        split = split_map[fn]
        station, date_str = extract_station_and_date(fn)
        base_name = os.path.splitext(fn)[0]
        
        # Read raw image (2048x2048)
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        h_orig, w_orig = img_gray.shape
        
        # Gather all annotator annotations for this image
        img_ids = fn_to_image_ids[fn]
        annotator_anns = [ann_by_image_id.get(iid, []) for iid in img_ids]
        
        # Multi-annotator consensus fusion at full resolution
        cons_multi_2048, cons_bin_2048, cons_spine_2048, agree_map_2048 = fuse_multi_annotator(
            annotator_anns, width=w_orig, height=h_orig
        )
        
        # Save full resolution 2048x2048 consensus masks
        cv2.imwrite(os.path.join(out, "masks_full_2048", "multiclass", f"{base_name}.png"), cons_multi_2048)
        cv2.imwrite(os.path.join(out, "masks_full_2048", "binary", f"{base_name}.png"), cons_bin_2048 * 255)
        cv2.imwrite(os.path.join(out, "masks_full_2048", "spine", f"{base_name}.png"), cons_spine_2048 * 255)
        
        # Downscale image to target size
        img_resized = cv2.resize(img_gray, (args.target_size, args.target_size), interpolation=cv2.INTER_AREA)
        # Apply CLAHE
        img_clahe = apply_clahe(img_resized)
        
        # Downscale masks with NEAREST interpolation to preserve exact discrete label values
        cons_multi_target = cv2.resize(cons_multi_2048, (args.target_size, args.target_size), interpolation=cv2.INTER_NEAREST)
        cons_bin_target = cv2.resize(cons_bin_2048, (args.target_size, args.target_size), interpolation=cv2.INTER_NEAREST)
        cons_spine_target = cv2.resize(cons_spine_2048, (args.target_size, args.target_size), interpolation=cv2.INTER_NEAREST)
        agree_map_target = cv2.resize(agree_map_2048, (args.target_size, args.target_size), interpolation=cv2.INTER_NEAREST)
        
        # Save scaled images and masks
        img_save_path = os.path.join(out, f"images_{args.target_size}", split, f"{base_name}.png")
        clahe_save_path = os.path.join(out, f"images_clahe_{args.target_size}", split, f"{base_name}.png")
        cv2.imwrite(img_save_path, img_resized)
        cv2.imwrite(clahe_save_path, img_clahe)
        
        multi_mask_path = os.path.join(out, f"masks_{args.target_size}", "multiclass", split, f"{base_name}.png")
        bin_mask_path = os.path.join(out, f"masks_{args.target_size}", "binary", split, f"{base_name}.png")
        spine_mask_path = os.path.join(out, f"masks_{args.target_size}", "spine", split, f"{base_name}.png")
        agree_mask_path = os.path.join(out, f"masks_{args.target_size}", "agreement", split, f"{base_name}.png")
        
        cv2.imwrite(multi_mask_path, cons_multi_target)
        cv2.imwrite(bin_mask_path, cons_bin_target * 255)
        cv2.imwrite(spine_mask_path, cons_spine_target * 255)
        cv2.imwrite(agree_mask_path, agree_map_target)
        
        # YOLO segmentation label export
        yolo_lines = extract_yolo_polygons(cons_bin_target, cons_multi_target, args.target_size, args.target_size)
        yolo_txt_path = os.path.join(out, "labels_yolo", split, f"{base_name}.txt")
        with open(yolo_txt_path, "w") as f_yolo:
            for line in yolo_lines:
                f_yolo.write(line + "\n")
                
        # Compute statistics
        px_filament = int(np.sum(cons_bin_2048 > 0))
        px_total = h_orig * w_orig
        total_filament_pixels += px_filament
        total_image_pixels += px_total
        
        # Count classes present
        unique_classes, counts = np.unique(cons_multi_2048, return_counts=True)
        sample_class_counts = {int(c): int(cnt) for c, cnt in zip(unique_classes, counts) if c > 0}
        for c, cnt in sample_class_counts.items():
            global_class_counts[c] += cnt
            
        classes_present_names = [CATEGORY_MAP[c]["name"] for c in sorted(sample_class_counts.keys())]
        
        manifest_rows.append({
            "filename": fn,
            "split": split,
            "station": station,
            "station_name": STATION_MAP.get(station, "Unknown"),
            "date_captured": date_str,
            "annotator_count": len(img_ids),
            "num_filaments_annotated": sum(len(a) for a in annotator_anns),
            "yolo_objects_count": len(yolo_lines),
            "filament_pixel_count": px_filament,
            "filament_area_pct": round(px_filament / px_total * 100, 4),
            "classes_present": "|".join(classes_present_names),
            "count_left_pixels": sample_class_counts.get(1, 0),
            "count_right_pixels": sample_class_counts.get(2, 0),
            "count_unidentifiable_pixels": sample_class_counts.get(3, 0),
            "count_ambiguous_pixels": sample_class_counts.get(4, 0),
            "rel_path_img": os.path.relpath(img_save_path, out),
            "rel_path_clahe": os.path.relpath(clahe_save_path, out),
            "rel_path_mask_multiclass": os.path.relpath(multi_mask_path, out),
            "rel_path_mask_binary": os.path.relpath(bin_mask_path, out),
            "rel_path_mask_spine": os.path.relpath(spine_mask_path, out),
            "rel_path_yolo_label": os.path.relpath(yolo_txt_path, out)
        })
        
    df_manifest = pd.DataFrame(manifest_rows)
    df_train = df_manifest[df_manifest["split"] == "train"]
    df_val = df_manifest[df_manifest["split"] == "val"]
    
    df_train.to_csv(os.path.join(out, "metadata", "train_manifest.csv"), index=False)
    df_val.to_csv(os.path.join(out, "metadata", "val_manifest.csv"), index=False)
    df_manifest.to_csv(os.path.join(out, "metadata", "full_manifest.csv"), index=False)
    
    # 4. Process Test Images (Unlabeled)
    print("\n[Step 4/5] Processing test images (512x512 & CLAHE)...")
    test_files = sorted(os.listdir(test_img_dir)) if os.path.exists(test_img_dir) else []
    test_rows = []
    
    for fn in tqdm(test_files, desc="Test processing"):
        img_path = os.path.join(test_img_dir, fn)
        station, date_str = extract_station_and_date(fn)
        base_name = os.path.splitext(fn)[0]
        
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
            
        img_resized = cv2.resize(img_gray, (args.target_size, args.target_size), interpolation=cv2.INTER_AREA)
        img_clahe = apply_clahe(img_resized)
        
        save_path = os.path.join(out, f"images_{args.target_size}", "test", f"{base_name}.png")
        clahe_save_path = os.path.join(out, f"images_clahe_{args.target_size}", "test", f"{base_name}.png")
        
        cv2.imwrite(save_path, img_resized)
        cv2.imwrite(clahe_save_path, img_clahe)
        
        test_rows.append({
            "filename": fn,
            "split": "test",
            "station": station,
            "station_name": STATION_MAP.get(station, "Unknown"),
            "date_captured": date_str,
            "rel_path_img": os.path.relpath(save_path, out),
            "rel_path_clahe": os.path.relpath(clahe_save_path, out)
        })
        
    df_test = pd.DataFrame(test_rows)
    df_test.to_csv(os.path.join(out, "metadata", "test_manifest.csv"), index=False)
    
    # 5. Export Summary Statistics JSON
    print("\n[Step 5/5] Compiling dataset statistics and distribution...")
    summary_stats = {
        "dataset_name": "MAGFiLO_1.0_Kaggle_2026",
        "description": "Preprocessed Solar Filament Chirality and Segmentation Dataset",
        "resolution_scaled": f"{args.target_size}x{args.target_size}",
        "resolution_raw": "2048x2048",
        "split_counts": {
            "train": len(df_train),
            "val": len(df_val),
            "test": len(df_test),
            "total_annotated": len(df_manifest)
        },
        "station_distribution_train_val": dict(Counter(df_manifest["station"])),
        "station_distribution_test": dict(Counter(df_test["station"])) if len(df_test) > 0 else {},
        "multi_annotator_breakdown": dict(Counter(df_manifest["annotator_count"])),
        "class_mapping": CATEGORY_MAP,
        "class_pixel_distribution_consensus": {
            CATEGORY_MAP[c]["name"]: global_class_counts[c] for c in sorted(global_class_counts.keys())
        },
        "filament_pixel_coverage_pct": round(total_filament_pixels / total_image_pixels * 100, 4),
        "mean_filament_area_pct_per_image": round(float(df_manifest["filament_area_pct"].mean()), 4),
        "consensus_logic": {
            "1_annotator": "Single ground truth",
            "2_annotators": "Union (at least 1 positive vote)",
            "3_annotators": "Majority vote (at least 2 positive votes)",
            "multiclass_tie_breaker": "Plurality vote; if tied, Ambiguous(4) or first mode"
        }
    }
    
    with open(os.path.join(out, "metadata", "dataset_summary.json"), "w") as f_sum:
        json.dump(summary_stats, f_sum, indent=2)
        
    print("\n========================================================")
    print("SUCCESS: Preprocessing pipeline completed successfully!")
    print(f"Summary JSON : {os.path.join(out, 'metadata', 'dataset_summary.json')}")
    print(f"Train manifest: {os.path.join(out, 'metadata', 'train_manifest.csv')}")
    print(f"Val manifest  : {os.path.join(out, 'metadata', 'val_manifest.csv')}")
    print(f"Test manifest : {os.path.join(out, 'metadata', 'test_manifest.csv')}")
    print("========================================================")


if __name__ == "__main__":
    main()
