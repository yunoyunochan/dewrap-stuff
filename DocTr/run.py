#!/usr/bin/env python3
"""
Unified Document Rectification Pipeline

Processes a distorted document image through:
- Stage 1: Background removal (segmentation)
- Stage 2: Geometric unwarping (always enabled)
- Stage 3: Illumination correction (optional via --color_correction flag)

Usage:
    python3 run.py --image_path input.jpg --output_path output.png
    python3 run.py --image_path input.jpg --output_path output.png --color_correction

    python3 run.py --image_path S__87040009.jpg --output_path S__87040009-il.jpg --color_correction
    python3 run.py --image_path S__87040009.jpg --output_path S__87040009-geo.jpg

Note: The underlying models (GeoTr, position_encoding) have hardcoded .cuda() calls.
For CPU-only execution, those files would need modification to support device passing.
"""

import argparse
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

# Import local modules
from seg import U2NETP
from GeoTr import GeoTr
from IllTr import IllTr


# ==================== Device Detection ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if not torch.cuda.is_available():
    print("WARNING: CUDA not available. The models have hardcoded .cuda() calls and may fail.")
    print("For CPU support, modify GeoTr.py and position_encoding.py to accept device parameter.")


# ==================== Model Loading Utilities ====================

def reload_segmodel(model, path):
    """Load segmentation model weights, stripping 6-character prefix from keys."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Segmentation model not found: {path}")
    
    state_dict = torch.load(path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[6:]  # remove first 6 characters (e.g., 'model.')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print(f"Loaded segmentation model from {path}")
    return model


def reload_model(model, path):
    """Load GeoTr/IllTr model weights, stripping 7-character prefix from keys."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    
    state_dict = torch.load(path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:]  # remove first 7 characters (e.g., 'module.')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print(f"Loaded model from {path}")
    return model


# ==================== Combined Segmentation + GeoTr Model ====================

class GeoTr_Seg(torch.nn.Module):
    """
    Combined model for background removal and geometric unwarping.
    
    Forward pass:
    1. Segmentation: U2NETP produces binary mask (threshold > 0.5)
    2. Masking: multiply input with mask to remove background
    3. GeoTr: predict backward mapping for geometric correction
    4. Normalize: scale backward map to [-0.99, 0.99] range
    """
    def __init__(self):
        super(GeoTr_Seg, self).__init__()
        self.msk = U2NETP(3, 1)
        self.GeoTr = GeoTr(num_attn_layers=6)

    def forward(self, x):
        # Segmentation
        msk, *_ = self.msk(x)
        msk = (msk > 0.5).float()
        
        # Apply mask to remove background
        x = msk * x
        
        # Geometric correction - get backward mapping
        bm = self.GeoTr(x)
        
        # Normalize backward map: scale to [-0.99, 0.99] range
        # 286.8 is the normalization constant for 288x288 input
        bm = (2 * (bm / 286.8) - 1) * 0.99
        
        return bm


# ==================== Illumination Correction Utilities ====================

def padCropImg(img):
    """
    Crop image into overlapping 128x128 patches with 16-pixel overlap (12.5%).
    
    Args:
        img: BGR uint8 numpy array [H, W, 3]
    
    Returns:
        totalPatch: numpy array [ynum, xnum, 128, 128, 3]
        padH: padded height
        padW: padded width
    """
    patchRes = 128
    overlap = int(patchRes * 0.125)  # 16 pixels
    stride = patchRes - overlap      # 112 pixels
    
    h, w = img.shape[:2]
    
    # Calculate padding needed
    padH = ((h - patchRes) // stride + 1) * stride + patchRes
    padW = ((w - patchRes) // stride + 1) * stride + patchRes
    
    # Pad image with border replication
    imgPad = cv2.copyMakeBorder(img, 0, padH - h, 0, padW - w, cv2.BORDER_REPLICATE)
    
    # Calculate grid dimensions
    ynum = (padH - patchRes) // stride + 1
    xnum = (padW - patchRes) // stride + 1
    
    # Extract patches
    totalPatch = np.zeros([ynum, xnum, patchRes, patchRes, 3], dtype=np.uint8)
    
    for j in range(ynum):
        for i in range(xnum):
            sy = j * stride
            sx = i * stride
            
            # For last row/column, use original image edge instead of padded
            if sy + patchRes > h or sx + patchRes > w:
                sy_img = max(0, h - patchRes)
                sx_img = max(0, w - patchRes)
                totalPatch[j, i] = img[sy_img:sy_img + patchRes, sx_img:sx_img + patchRes]
            else:
                totalPatch[j, i] = imgPad[sy:sy + patchRes, sx:sx + patchRes]
    
    return totalPatch, padH, padW


def illCorrection(net, totalPatch):
    """
    Apply illumination correction to each patch using IllTr model.
    
    Args:
        net: IllTr model
        totalPatch: numpy array [ynum, xnum, 128, 128, 3]
    
    Returns:
        totalResults: numpy array [ynum, xnum, 128, 128, 3] (uint8)
    """
    ynum, xnum = totalPatch.shape[:2]
    patchRes = totalPatch.shape[2]
    
    totalResults = np.zeros(totalPatch.shape, dtype=np.uint8)
    
    for j in range(ynum):
        for i in range(xnum):
            # Get patch and normalize to [0, 1]
            patchImg = totalPatch[j, i].astype(np.float32) / 255.0
            
            # Convert to tensor: HWC -> CHW
            patchImg = torch.from_numpy(patchImg).permute(2, 0, 1).cuda()
            patchImg = patchImg.view(1, 3, patchRes, patchRes)
            
            # Run model
            with torch.no_grad():
                resultImg = net(patchImg)
            
            # Convert back: CHW -> HWC, denormalize
            resultImg = resultImg.permute(0, 2, 3, 1).cpu().numpy()
            resultImg = (resultImg[0] * 255).astype(np.uint8)
            
            totalResults[j, i] = resultImg
    
    return totalResults


def composePatch(totalResults, padH, padW, img):
    """
    Stitch overlapping patches back into a single image with 10-pixel margin blending.
    
    Args:
        totalResults: numpy array [ynum, xnum, 128, 128, 3]
        padH: padded height
        padW: padded width
        img: original image (for getting original dimensions)
    
    Returns:
        resImg: stitched result image [H, W, 3] (uint8)
    """
    patchRes = 128
    overlap = int(patchRes * 0.125)  # 16 pixels
    stride = patchRes - overlap       # 112 pixels
    margin = 10  # Blending margin
    
    h, w = img.shape[:2]
    ynum, xnum = totalResults.shape[:2]
    
    # Initialize result image
    resImg = np.zeros([padH, padW, 3], dtype=np.uint8)
    
    # Place first row/column patches fully
    resImg[:patchRes, :patchRes] = totalResults[0, 0]
    
    # First row
    for i in range(1, xnum):
        sx = i * stride
        resImg[:patchRes, sx + margin:sx + patchRes] = totalResults[0, i, :, margin:]
    
    # First column
    for j in range(1, ynum):
        sy = j * stride
        resImg[sy + margin:sy + patchRes, :patchRes] = totalResults[j, 0, margin:, :]
    
    # Interior patches
    for j in range(1, ynum):
        for i in range(1, xnum):
            sy = j * stride
            sx = i * stride
            resImg[sy + margin:sy + patchRes, sx + margin:sx + patchRes] = \
                totalResults[j, i, margin:, margin:]
    
    # Crop back to original size
    resImg = resImg[:h, :w]
    
    # Set first row to white (artifact fix from original code)
    resImg[0, :, :] = 255
    
    return resImg


def rec_ill(net, img):
    """
    Full illumination correction pipeline: pad -> crop -> correct -> stitch.
    
    Args:
        net: IllTr model
        img: BGR uint8 numpy array from geometric correction
    
    Returns:
        result: RGB uint8 numpy array
    """
    # Crop into patches
    totalPatch, padH, padW = padCropImg(img)
    
    # Apply illumination correction to each patch
    totalResults = illCorrection(net, totalPatch)
    
    # Stitch patches back together
    resImg = composePatch(totalResults, padH, padW, img)
    
    # Convert BGR to RGB for output
    resImg = cv2.cvtColor(resImg, cv2.COLOR_BGR2RGB)
    
    return resImg


# ==================== Main Processing Pipeline ====================

def process_document(image_path, output_path, color_correction=False,
                     seg_model_path='./model_pretrained/seg.pth',
                     geo_model_path='./model_pretrained/geotr.pth',
                     ill_model_path='./model_pretrained/illtr.pth'):
    """
    Process a distorted document image through the full rectification pipeline.
    
    Args:
        image_path: Path to input distorted image
        output_path: Path to save output image
        color_correction: If True, apply illumination correction (Stage 3)
        seg_model_path: Path to segmentation model weights
        geo_model_path: Path to geometric correction model weights
        ill_model_path: Path to illumination correction model weights
    """
    
    # ========== Load Models ==========
    print("\n=== Loading Models ===")
    
    # Segmentation + Geometric correction model
    model_seg_geo = GeoTr_Seg().cuda()
    reload_segmodel(model_seg_geo.msk, seg_model_path)
    reload_model(model_seg_geo.GeoTr, geo_model_path)
    model_seg_geo.eval()
    
    # Illumination correction model (if needed)
    model_ill = None
    if color_correction:
        model_ill = IllTr().cuda()
        reload_model(model_ill, ill_model_path)
        model_ill.eval()
        print("Illumination correction enabled")
    
    # ========== Load and Preprocess Image ==========
    print(f"\n=== Processing Image: {image_path} ===")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    
    # Load original image
    img_np = np.array(Image.open(image_path))[:, :, :3].astype(np.float64) / 255.0
    h_orig, w_orig = img_np.shape[:2]
    print(f"Original image size: {w_orig} x {h_orig}")
    
    # Resize to 288x288 for model input
    im_288 = cv2.resize(img_np, (288, 288))
    
    # Convert to tensor: HWC -> CHW
    im_tensor = torch.from_numpy(im_288).permute(2, 0, 1).float().unsqueeze(0)
    im_tensor = im_tensor.cuda()
    
    # ========== Stage 1 + 2: Segmentation + Geometric Unwarping ==========
    print("\nStage 1+2: Background removal + Geometric unwarping...")
    
    with torch.no_grad():
        # Get backward mapping at 288x288 resolution
        bm = model_seg_geo(im_tensor)  # [1, 2, 288, 288]
    
    # Resize backward map to original resolution
    bm0 = cv2.resize(bm[0, 0].cpu().numpy(), (w_orig, h_orig))
    bm1 = cv2.resize(bm[0, 1].cpu().numpy(), (w_orig, h_orig))
    
    # Smooth backward map
    bm0 = cv2.blur(bm0, (3, 3))
    bm1 = cv2.blur(bm1, (3, 3))
    
    # Stack into grid: [1, H, W, 2]
    lbl = np.stack([bm0, bm1], axis=-1)
    lbl = torch.from_numpy(lbl).unsqueeze(0).float()
    
    # Apply geometric correction via grid sampling
    img_orig_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0)
    
    with torch.no_grad():
        out = F.grid_sample(img_orig_tensor, lbl, align_corners=True)
    
    # Convert to uint8 BGR for further processing
    img_geo = (out[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_geo = cv2.cvtColor(img_geo, cv2.COLOR_RGB2BGR)
    
    print(f"Geometric correction complete: {img_geo.shape}")
    
    # ========== Stage 3: Illumination Correction (Optional) ==========
    if color_correction and model_ill is not None:
        print("\nStage 3: Illumination correction...")
        img_final = rec_ill(model_ill, img_geo)
        print(f"Illumination correction complete: {img_final.shape}")
    else:
        # Convert BGR back to RGB for saving
        img_final = cv2.cvtColor(img_geo, cv2.COLOR_BGR2RGB)
    
    # ========== Save Output ==========
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save as PIL Image (handles various formats)
    img_pil = Image.fromarray(img_final)
    img_pil.save(output_path)
    
    print(f"\n✓ Output saved to: {output_path}")


# ==================== Command Line Interface ====================

def main():
    parser = argparse.ArgumentParser(
        description='Document Rectification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Geometric correction only
  python run.py --image_path distorted/doc.jpg --output_path output/doc_geo.png
  
  # With illumination correction
  python run.py --image_path distorted/doc.jpg --output_path output/doc_full.png --color_correction
        """
    )
    
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input distorted document image')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save output corrected image')
    parser.add_argument('--color_correction', action='store_true',
                        help='Enable illumination correction (Stage 3)')
    parser.add_argument('--seg_model_path', type=str, default='./model_pretrained/seg.pth',
                        help='Path to segmentation model (default: ./model_pretrained/seg.pth)')
    parser.add_argument('--geo_model_path', type=str, default='./model_pretrained/geotr.pth',
                        help='Path to geometric correction model (default: ./model_pretrained/geotr.pth)')
    parser.add_argument('--ill_model_path', type=str, default='./model_pretrained/illtr.pth',
                        help='Path to illumination correction model (default: ./model_pretrained/illtr.pth)')
    
    args = parser.parse_args()
    
    try:
        process_document(
            image_path=args.image_path,
            output_path=args.output_path,
            color_correction=args.color_correction,
            seg_model_path=args.seg_model_path,
            geo_model_path=args.geo_model_path,
            ill_model_path=args.ill_model_path
        )
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
