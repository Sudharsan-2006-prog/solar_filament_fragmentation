"""
app.py — Interactive U-Net Solar Filament Segmentation Demo UI
"""

import os
import sys
import yaml
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
import streamlit as st
import pandas as pd

# Important: ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet import build_unet
from src.utils import load_checkpoint
from src.metrics import compute_binary_metrics


# =============================================================================
# App Config & UI Setup
# =============================================================================
st.set_page_config(
    page_title="Solar Filament Segmentation — U-Net",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Caching Model & Config
# =============================================================================

@st.cache_resource
def load_project_config():
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

@st.cache_resource
def load_unet_model(checkpoint_path, _cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(_cfg)
    try:
        # load_checkpoint returns epoch, metrics, best_val_dice
        # Ignore the return values, just load the weights in-place
        load_checkpoint(model, checkpoint_path, device=device)
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        return None, device
    model = model.to(device)
    model.eval()
    return model, device

@st.cache_data
def get_validation_manifest(preprocessed_dir):
    manifest_path = Path(preprocessed_dir) / "metadata" / "val_manifest.csv"
    if not manifest_path.exists():
        return None
    return pd.read_csv(manifest_path)

# =============================================================================
# Utility Functions
# =============================================================================

def apply_clahe(img_np):
    """Apply CLAHE to a grayscale numpy image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
    return clahe.apply(img_np)

def preprocess_custom_image(image_pil, use_clahe=True, image_size=512):
    """Process an uploaded image exactly as the model expects."""
    # Convert to grayscale
    img_gray = image_pil.convert("L")
    # Resize
    img_resized = img_gray.resize((image_size, image_size), Image.Resampling.BILINEAR)
    img_np = np.array(img_resized)

    if use_clahe:
        img_np = apply_clahe(img_np)

    # Normalize to [0, 1]
    img_float = img_np.astype(np.float32) / 255.0
    # To tensor (1, 1, H, W)
    tensor = torch.from_numpy(img_float).unsqueeze(0).unsqueeze(0)
    return img_np, tensor

def load_validation_sample(preprocessed_dir, row, image_size=512):
    """Load preprocessed validation image and mask."""
    # The dataframe has 'rel_path_clahe' or 'rel_path_img'
    img_path = Path(preprocessed_dir) / row["rel_path_clahe"]
    
    # We must load the binary mask. It's usually under masks_512/binary/val
    # We derive it from the filename.
    stem = Path(row["rel_path_img"]).stem
    mask_path = Path(preprocessed_dir) / "masks_512" / "binary" / "val" / f"{stem}.png"
    
    # Load image
    img_pil = Image.open(img_path).convert("L")
    img_np = np.array(img_pil)
    img_float = img_np.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_float).unsqueeze(0).unsqueeze(0)
    
    # Load mask
    mask_pil = Image.open(mask_path).convert("L")
    mask_np = np.array(mask_pil)
    # Binary masks are 0, 255 -> convert to 0, 1
    mask_binary = (mask_np > 127).astype(np.uint8)
    mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).long()
    
    return img_np, mask_binary, img_tensor, mask_tensor

def create_overlay(img_np, mask_np, color=(255, 0, 0), alpha=0.4):
    """Create an RGB overlay of the mask on the grayscale image."""
    # img_np is grayscale (0-255 or 0-1)
    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
            
    overlay = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    # mask_np is binary 0, 1
    colored_mask = np.zeros_like(overlay)
    colored_mask[mask_np == 1] = color
    
    # Blend where mask is active
    mask_indices = mask_np == 1
    overlay[mask_indices] = cv2.addWeighted(
        overlay[mask_indices], 1 - alpha, 
        colored_mask[mask_indices], alpha, 0
    )
    return overlay


# =============================================================================
# Main Application
# =============================================================================

def main():
    # ── Header ───────────────────────────────────────────────────────────────
    st.title("Solar Filament Segmentation — U-Net")
    st.subheader("AI-based pixel-level segmentation of solar filaments")
    st.markdown("*Detecting solar filament structures from H-alpha solar observations*")
    st.markdown("---")

    cfg = load_project_config()
    preprocessed_dir = cfg.get("preprocessed_dir", "")
    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / "unet_best.pth"
    
    model, device = load_unet_model(checkpoint_path, cfg)
    
    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Model Information")
        st.write("**Model:** U-Net")
        st.write("**Checkpoint:** Best validation checkpoint — Epoch 19")
        st.write("**Input:** 512 × 512 grayscale")
        st.write(f"**Device:** {device.type.upper() if device.type != 'cpu' else 'CPU'}")
        if device.type == 'cuda':
            st.write(f"*(GPU: {torch.cuda.get_device_name(0)})*")
        st.write("**Dataset:** MAGFiLO 1.0")
        st.write("**Best Validation Dice:** 0.6132")
        st.write("**Best Validation IoU:** 0.4452")
        
        st.markdown("---")
        st.header("Segmentation Threshold")
        threshold = st.slider(
            "Prediction Threshold", 
            min_value=0.1, max_value=0.9, value=0.5, step=0.05
        )
        st.caption("Pixels above this probability are classified as filament.")
        
    # ── Input Section ────────────────────────────────────────────────────────
    
    st.header("Image Input")
    
    mode = st.radio("Select Input Mode:", ["Validation Image", "Upload Image"], horizontal=True)
    
    val_manifest = get_validation_manifest(preprocessed_dir) if preprocessed_dir else None
    
    selected_val_row = None
    uploaded_file = None
    
    if mode == "Validation Image":
        if val_manifest is not None and not val_manifest.empty:
            # Dropdown to select image
            filenames = val_manifest["rel_path_img"].apply(lambda x: Path(x).name).tolist()
            selected_filename = st.selectbox("Select validation image", filenames)
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Random Validation Image"):
                    selected_filename = val_manifest.sample(1).iloc[0]["rel_path_img"]
                    selected_filename = Path(selected_filename).name
            
            # Find the row
            selected_val_row = val_manifest[val_manifest["rel_path_img"].str.endswith(selected_filename)].iloc[0]
            
        else:
            st.error("Validation manifest not found. Please verify the preprocessed dataset path in config.yaml.")
            
    else:
        uploaded_file = st.file_uploader("Upload a solar image", type=["jpg", "jpeg", "png"])
        
    st.markdown("---")
    
    # ── Inference Trigger ────────────────────────────────────────────────────
    
    if st.button("Run U-Net Segmentation", type="primary"):
        if model is None:
            st.error(f"Unable to load the model checkpoint. Please verify that outputs/checkpoints/unet_best.pth exists.")
            return
            
        with st.spinner("Running U-Net inference..."):
            
            has_gt = False
            img_disp = None
            gt_disp = None
            pred_disp = None
            overlay_disp = None
            metrics = None
            prob_map_disp = None
            
            if mode == "Validation Image" and selected_val_row is not None:
                has_gt = True
                try:
                    img_np, mask_binary, img_tensor, mask_tensor = load_validation_sample(preprocessed_dir, selected_val_row)
                    img_disp = img_np
                    gt_disp = mask_binary
                except Exception as e:
                    st.error(f"Error loading validation sample: {e}")
                    return
                    
            elif mode == "Upload Image" and uploaded_file is not None:
                has_gt = False
                try:
                    img_pil = Image.open(uploaded_file)
                    use_clahe = cfg.get("use_clahe", True)
                    img_np, img_tensor = preprocess_custom_image(img_pil, use_clahe=use_clahe)
                    img_disp = img_np
                except Exception as e:
                    st.error(f"Error processing uploaded image: {e}")
                    return
            else:
                st.warning("Please select or upload an image first.")
                return
                
            # Run Inference
            img_tensor = img_tensor.to(device)
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                
            prob_map_disp = probs
            pred_binary = (probs > threshold).astype(np.uint8)
            pred_disp = pred_binary
            
            # Metrics
            if has_gt:
                prob_t = torch.from_numpy(probs).unsqueeze(0).float()
                mask_tensor = mask_tensor.cpu()
                # Use project's binary metric computer
                metrics = compute_binary_metrics(prob_t, mask_tensor, threshold=threshold)
            
            # Overlays
            overlay_disp = create_overlay(img_disp, pred_binary, color=(255, 0, 0))  # Red for prediction

            # ── Results Display ────────────────────────────────────────────────
            st.header("Segmentation Results")
            if has_gt:
                st.success("Validation Image — Ground Truth Available")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_disp, caption="Input Image (CLAHE)", width='stretch', clamp=True)
                with col2:
                    st.image(gt_disp * 255, caption="Ground Truth Mask", width='stretch', clamp=True)
                    
                col3, col4 = st.columns(2)
                with col3:
                    st.image(pred_disp * 255, caption="U-Net Prediction", width='stretch', clamp=True)
                with col4:
                    st.image(overlay_disp, caption="Prediction Overlay (Red = Filament)", width='stretch', channels="RGB")
                    
                # Display metrics
                st.subheader("Current Image Metrics")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Dice", f"{metrics.get('dice', 0):.4f}")
                m2.metric("IoU", f"{metrics.get('iou', 0):.4f}")
                m3.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                m4.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                m5.metric("Pixel Accuracy", f"{metrics.get('pixel_accuracy', 0):.4f}")
                
            else:
                st.info("Prediction Only — Ground Truth Not Available")
                st.caption("Ground truth unavailable for this image.")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(img_disp, caption="Input Image", width='stretch', clamp=True)
                with col2:
                    st.image(pred_disp * 255, caption="Predicted Mask", width='stretch', clamp=True)
                with col3:
                    st.image(overlay_disp, caption="Prediction Overlay (Red = Filament)", width='stretch', channels="RGB")
            
            # ── Advanced Visualization ─────────────────────────────────────────
            with st.expander("Advanced Visualization"):
                st.markdown("**Prediction Probability Map**")
                # Show probability as a heatmap/grayscale
                st.image(prob_map_disp, caption="Probability Heatmap (0.0 to 1.0)", width=512, clamp=True)
                
                st.markdown("**Binary Prediction Mask**")
                st.image(pred_binary * 255, caption=f"Thresholded at {threshold}", width=512, clamp=True)

    st.markdown("---")
    
    # ── Expanders for Info ───────────────────────────────────────────────────
    with st.expander("How the Model Works"):
        st.markdown("""
        **Processing Pipeline:**
        
        Solar Image  
        ↓  
        Preprocessing (Resize to 512×512, CLAHE Enhancement, Grayscale)  
        ↓  
        U-Net Encoder (Feature Extraction)  
        ↓  
        Bottleneck  
        ↓  
        U-Net Decoder (Spatial Reconstruction)  
        ↓  
        Skip Connections (High-resolution feature merging)  
        ↓  
        Pixel-wise Prediction (Sigmoid Probability)  
        ↓  
        Filament Segmentation Mask (Thresholding)
        
        *U-Net is a convolutional encoder-decoder architecture designed for semantic image segmentation. Skip connections transfer spatial information from the encoder to the decoder, helping preserve fine structures such as thin solar filaments.*
        """)
        
    with st.expander("Dataset Information"):
        st.markdown("""
        **MAGFiLO 1.0 (IEEE Big Data Cup 2026)**
        
        - **Raw annotated images:** 707
        - **Training images:** 566
        - **Validation images:** 141
        - **Test images:** 180 (held-out)
        
        **Properties:**
        - **Original resolution:** 2048 × 2048
        - **Model input:** 512 × 512
        - **Task:** Pixel-level semantic segmentation
        - **Domain:** Solar imagery / space weather
        
        *Note: Solar filaments occupy a very small fraction of the image area (typically 0.4%–1.5%), creating a strong foreground-background class imbalance.*
        """)
        
    with st.expander("U-Net Baseline Performance"):
        st.markdown("""
        | Metric | Validation Set Performance |
        |--------|-----------------------------|
        | **Dice** | 0.6132 |
        | **IoU** | 0.4452 |
        | **Precision** | 0.6488 |
        | **Recall** | 0.5868 |
        | **F1** | 0.6132 |
        | **Pixel Accuracy** | 99.71% |
        
        *These are overall validation-set metrics from the best U-Net checkpoint, not metrics for the currently displayed image.*
        """)
        
    with st.expander("Training History"):
        fig_dir = PROJECT_ROOT / "outputs" / "figures"
        loss_fig = fig_dir / "unet_loss_curve.png"
        dice_fig = fig_dir / "unet_dice_curve.png"
        iou_fig = fig_dir / "unet_iou_curve.png"
        
        if loss_fig.exists() and dice_fig.exists() and iou_fig.exists():
            st.image(str(loss_fig), caption="Training vs Validation Loss", width='stretch')
            st.image(str(dice_fig), caption="Training vs Validation Dice", width='stretch')
            st.image(str(iou_fig), caption="Training vs Validation IoU", width='stretch')
        else:
            st.info("Training history figures not found in `outputs/figures/`.")


if __name__ == "__main__":
    main()
