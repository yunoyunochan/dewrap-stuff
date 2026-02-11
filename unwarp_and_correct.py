#!/usr/bin/env python3
"""
Unwarp and color-correct a single document image
Full pipeline: Background removal + Geometric correction + Illumination correction

Pipeline stages:
  0. Background Removal (U2NETP segmentation) - removes background via masking
  1. Geometric Correction (inv3d-model) - unwarps distorted document
  2. Illumination Correction (DocTr IllTr) - corrects lighting/color issues

Usage:
  # Basic usage (all stages)
  python3 unwarp_and_correct.py --input data/1152226_geo.png

  # Use inv3d model optimized for invoices/forms
  python3 unwarp_and_correct.py --input data/1152226_geo.png --model geotr@inv3d

  # Skip background removal if image already has clean background
  python3 unwarp_and_correct.py --input data/doc.jpg --skip-segmentation

  # Save the intermediate unwarped-only image
  python3 unwarp_and_correct.py --input data/doc.jpg --save-intermediate

  # Use CPU instead of GPU
  python3 unwarp_and_correct.py --input data/1152226.jpg --gpu -1

  # Custom output path
  python3 unwarp_and_correct.py --input data/1152226.jpg --output ./1152226_final.png

  python3 unwarp_and_correct.py --input data/1152226.jpg --output ./1152226_final.png --seg-weights DocTr/model_pretrained/seg.pth

  python3 unwarp_and_correct.py --input data/1152226.jpg --output ./1152226_final.png --skip-segmentation

"""

import cv2
cv2.setNumThreads(0)

import argparse
import os
import sys
from pathlib import Path

import gdown
import numpy as np
import torch
import yaml
from einops import rearrange
from PIL import Image

# Setup paths
script_dir = Path(__file__).parent.resolve()
inv3d_dir = script_dir / "inv3d-model"
doctr_dir = script_dir / "DocTr"

# Add inv3d-model modules to path
sys.path.insert(0, str(inv3d_dir / "src"))

from inv3d_model.models import model_factory
from inv3d_util.image import scale_image
from inv3d_util.load import load_image, save_image, save_npz
from inv3d_util.mapping import apply_map_torch
from inv3d_util.misc import to_numpy_image, to_numpy_map

# Add DocTr modules to path
sys.path.insert(0, str(doctr_dir))

from IllTr import IllTr
from seg import U2NETP

# Load available inv3d models
model_sources = yaml.safe_load((inv3d_dir / "models.yaml").read_text())


def reload_segmodel(model, path):
    """
    Load segmentation model weights, stripping 'model.' prefix (6 characters)
    (from DataParallel wrapping)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Segmentation model weights not found: {path}")
    
    print(f"Loading segmentation weights from: {path}")
    state_dict = torch.load(path, map_location='cpu')
    
    # Strip 'model.' prefix (6 characters) from DataParallel wrapped models
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[6:] if len(k) > 6 else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    return model


def reload_model(model, path):
    """
    Load model weights from checkpoint, stripping 'module.' prefix if present
    (from DataParallel wrapping)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights not found: {path}")
    
    print(f"Loading IllTr weights from: {path}")
    state_dict = torch.load(path, map_location='cpu')
    
    # Strip 'module.' prefix (7 characters) from DataParallel wrapped models
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    return model


def padCropImg(img):
    """
    Split image into overlapping 128x128 patches with 12.5% overlap
    (From DocTr/inference_ill.py - exact implementation to avoid grid artifacts)
    
    Args:
        img: BGR numpy array [H, W, 3] uint8
    
    Returns:
        totalPatch: numpy array [ynum, xnum, 128, 128, 3] uint8
        padH: padded height
        padW: padded width
    """
    H = img.shape[0]
    W = img.shape[1]

    patchRes = 128
    pH = patchRes
    pW = patchRes
    ovlp = int(patchRes * 0.125)  # 16 pixels

    padH = (int((H - patchRes) / (patchRes - ovlp) + 1) * (patchRes - ovlp) + patchRes) - H
    padW = (int((W - patchRes) / (patchRes - ovlp) + 1) * (patchRes - ovlp) + patchRes) - W

    padImg = cv2.copyMakeBorder(img, 0, padH, 0, padW, cv2.BORDER_REPLICATE)

    ynum = int((padImg.shape[0] - pH) / (pH - ovlp)) + 1
    xnum = int((padImg.shape[1] - pW) / (pW - ovlp)) + 1

    totalPatch = np.zeros((ynum, xnum, patchRes, patchRes, 3), dtype=np.uint8)

    for j in range(0, ynum):
        for i in range(0, xnum):
            x = int(i * (pW - ovlp))
            y = int(j * (pH - ovlp))
            
            # Special handling for edge patches - use original image boundaries
            if j == (ynum-1) and i == (xnum-1):
                totalPatch[j, i] = img[-patchRes:, -patchRes:]
            elif j == (ynum-1):
                totalPatch[j, i] = img[-patchRes:, x:int(x + patchRes)]
            elif i == (xnum-1):
                totalPatch[j, i] = img[y:int(y + patchRes), -patchRes:]
            else:
                totalPatch[j, i] = padImg[y:int(y + patchRes), x:int(x + patchRes)]

    return totalPatch, padH, padW


def illCorrection(model, totalPatch, device):
    """
    Run IllTr model on each patch
    (From DocTr/inference_ill.py - exact implementation)
    
    Args:
        model: IllTr model
        totalPatch: numpy array [ynum, xnum, 128, 128, 3] uint8
        device: torch device ('cuda' or 'cpu')
    
    Returns:
        totalResults: numpy array [ynum, xnum, 128, 128, 3] float32 (will be converted to uint8 later)
    """
    totalPatch = totalPatch.astype(np.float32) / 255.0

    ynum = totalPatch.shape[0]
    xnum = totalPatch.shape[1]

    totalResults = np.zeros((ynum, xnum, 128, 128, 3), dtype=np.float32)

    with torch.no_grad():
        for j in range(0, ynum):
            for i in range(0, xnum):
                patchImg = torch.from_numpy(totalPatch[j, i]).permute(2, 0, 1)
                patchImg = patchImg.to(device).view(1, 3, 128, 128)

                output = model(patchImg)
                output = output.permute(0, 2, 3, 1).data.cpu().numpy()[0]

                output = output * 255.0
                output = output.astype(np.uint8)

                totalResults[j, i] = output

    return totalResults


def composePatch(totalResults, padH, padW, img):
    """
    Stitch patches back together with proper overlap handling
    (From DocTr/inference_ill.py - exact implementation to avoid grid artifacts)
    
    Args:
        totalResults: numpy array [ynum, xnum, 128, 128, 3] uint8
        padH: padded height (not used, kept for compatibility)
        padW: padded width (not used, kept for compatibility)
        img: original image for size reference
    
    Returns:
        resImg: BGR numpy array [H, W, 3] uint8
    """
    ynum = totalResults.shape[0]
    xnum = totalResults.shape[1]
    patchRes = totalResults.shape[2]

    ovlp = int(patchRes * 0.125)  # 16 pixels
    step = patchRes - ovlp  # 112 pixels

    # Create result image with same size as original
    resImg = np.zeros_like(img).astype('uint8')

    for j in range(0, ynum):
        for i in range(0, xnum):
            sy = int(j * step)
            sx = int(i * step)
            
            # Different stitching logic for each position
            if j == 0 and i != (xnum-1):
                # First row, not last column
                resImg[sy:(sy + patchRes), sx:(sx + patchRes)] = totalResults[j, i]
            elif i == 0 and j != (ynum-1):
                # First column, not last row - skip top 10 pixels
                resImg[sy+10:(sy + patchRes), sx:(sx + patchRes)] = totalResults[j, i, 10:]
            elif j == (ynum-1) and i == (xnum-1):
                # Last row, last column - skip top-left 10 pixels
                resImg[-patchRes+10:, -patchRes+10:] = totalResults[j, i, 10:, 10:]
            elif j == (ynum-1) and i == 0:
                # Last row, first column - skip top 10 pixels
                resImg[-patchRes+10:, sx:(sx + patchRes)] = totalResults[j, i, 10:]
            elif j == (ynum-1) and i != 0:
                # Last row, not first/last column - skip top-left 10 pixels
                resImg[-patchRes+10:, sx+10:(sx + patchRes)] = totalResults[j, i, 10:, 10:]
            elif i == (xnum-1) and j == 0:
                # Last column, first row - skip left 10 pixels
                resImg[sy:(sy + patchRes), -patchRes+10:] = totalResults[j, i, :, 10:]
            elif i == (xnum-1) and j != 0:
                # Last column, not first row - skip top-left 10 pixels
                resImg[sy+10:(sy + patchRes), -patchRes+10:] = totalResults[j, i, 10:, 10:]
            else:
                # Interior patches - skip top-left 10 pixels
                resImg[sy+10:(sy + patchRes), sx+10:(sx + patchRes)] = totalResults[j, i, 10:, 10:]

    # Fix artifact: set first row to white
    resImg[0, :, :] = 255

    return resImg


def rec_ill(model, img, device):
    """
    Apply illumination correction to an image using IllTr
    
    Args:
        model: IllTr model
        img: BGR numpy array [H, W, 3] uint8
        device: torch device ('cuda' or 'cpu')
    
    Returns:
        result: BGR numpy array [H, W, 3] uint8
    """
    # Split into patches
    totalPatch, padH, padW = padCropImg(img)
    
    # Apply correction to each patch
    totalResults = illCorrection(model, totalPatch, device)
    
    # Stitch patches back together
    result = composePatch(totalResults, padH, padW, img)
    
    return result


def unwarp_and_correct(
    input_path: str,
    output_path: str = None,
    model_name: str = "geotr@doc3d",
    template_path: str = None,
    output_width: int = 1700,
    output_height: int = 2200,
    gpu: int = 0,
    save_flow: bool = False,
    save_intermediate: bool = False,
    illtr_weights: str = None,
    seg_weights: str = None,
    skip_segmentation: bool = False
):
    """
    Unwarp a distorted document image and apply color correction
    
    Args:
        input_path: Path to input warped image
        output_path: Path to save corrected image (default: input_corrected.png)
        model_name: inv3d model to use (geotr@doc3d, geotr@inv3d, etc.)
        template_path: Path to template image (required for template models)
        output_width: Output image width in pixels
        output_height: Output image height in pixels
        gpu: GPU index (-1 for CPU)
        save_flow: Whether to save the backward mapping flow field (.npz)
        save_intermediate: Whether to save the unwarped-only image
        illtr_weights: Path to IllTr weights (default: DocTr/model_pretrained/illtr.pth)
        seg_weights: Path to segmentation weights (default: DocTr/model_pretrained/seg.pth)
        skip_segmentation: Skip background removal segmentation step
    
    Returns:
        Path to color-corrected image
    """
    
    # Setup GPU/CPU device
    if gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    else:
        # Force CPU mode by hiding all GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")
        print("Using device: cpu (GPU disabled)")
    
    
    
    # Validate input
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    
    # Setup output paths
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_corrected{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    intermediate_path = None
    if save_intermediate:
        intermediate_path = input_path.parent / f"{input_path.stem}_unwarped{input_path.suffix}"
    
    # Setup IllTr weights path
    if illtr_weights is None:
        illtr_weights = doctr_dir / "model_pretrained" / "illtr.pth"
    else:
        illtr_weights = Path(illtr_weights)
    
    if not illtr_weights.exists():
        raise FileNotFoundError(f"IllTr weights not found: {illtr_weights}")
    
    # Setup segmentation weights path
    if seg_weights is None:
        seg_weights = doctr_dir / "model_pretrained" / "seg.pth"
    else:
        seg_weights = Path(seg_weights)
    
    if not skip_segmentation and not seg_weights.exists():
        raise FileNotFoundError(f"Segmentation weights not found: {seg_weights}")
    
    # Load original image first for segmentation
    print("=" * 70)
    print("LOADING INPUT IMAGE")
    print("=" * 70)
    print(f"Loading image: {input_path}")
    image_original = load_image(input_path)
    print(f"Original image size: {image_original.shape[1]} x {image_original.shape[0]}")
    
    # STEP 0: Background Removal (Segmentation)
    if not skip_segmentation:
        print("\n" + "=" * 70)
        print("STEP 0: BACKGROUND REMOVAL (Segmentation)")
        print("=" * 70)
        
        # Load segmentation model
        print(f"Loading segmentation model...")
        seg_model = U2NETP(3, 1)
        seg_model = reload_segmodel(seg_model, str(seg_weights))
        seg_model.to(device)
        seg_model.eval()
        
        # Resize image to 288x288 for segmentation
        img_288 = cv2.resize(image_original, (288, 288))
        img_288_tensor = torch.from_numpy(img_288).permute(2, 0, 1).float() / 255.0
        img_288_tensor = img_288_tensor.unsqueeze(0).to(device)
        
        # Run segmentation
        print("Running segmentation...")
        with torch.no_grad():
            msk, *_ = seg_model(img_288_tensor)
            msk = (msk > 0.5).float()  # Binary threshold
        
        # Resize mask to original image size
        msk_resized = torch.nn.functional.interpolate(
            msk, 
            size=(image_original.shape[0], image_original.shape[1]),
            mode='bilinear',
            align_corners=True
        )
        
        # Apply mask to original image
        msk_np = msk_resized[0, 0].cpu().numpy()  # [H, W]
        msk_np = np.stack([msk_np] * 3, axis=-1)  # [H, W, 3]
        image_original = (image_original * msk_np).astype(np.uint8)
        
        print(f"✓ Background removed (mask applied)")
        
        # Free memory
        del seg_model, img_288_tensor, msk, msk_resized
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("\n⚠ Skipping segmentation (background removal)")
    
    print("\n" + "=" * 70)
    print("STEP 1: GEOMETRIC CORRECTION (Unwarping)")
    print("=" * 70)
    
    # Download and load inv3d model
    print(f"Loading inv3d model: {model_name}")
    model_url = model_sources[model_name]
    model_path = Path(
        gdown.cached_download(
            url=model_url, 
            path=inv3d_dir / f"models/{model_name}.ckpt"
        )
    )
    
    unwarp_model = model_factory.load_from_checkpoint(model_name.split("@")[0], model_path)
    unwarp_model.to(device)
    unwarp_model.eval()
    
    # Prepare input image for inv3d model
    image = scale_image(image_original, resolution=unwarp_model.dataset_options["resolution"])
    image = rearrange(image, "h w c -> () c h w")
    image = image.astype("float32") / 255
    image = torch.from_numpy(image).to(device)
    
    model_kwargs = {"image": image}
    
    # Handle template if needed
    if "template" in model_name:
        if template_path is None:
            raise ValueError(f"Model {model_name} requires a template image. Use --template argument.")
        
        template_path = Path(template_path)
        if not template_path.exists():
            raise FileNotFoundError(f"Template image not found: {template_path}")
        
        print(f"Loading template: {template_path}")
        template_original = load_image(template_path)
        template = scale_image(template_original, resolution=unwarp_model.dataset_options["resolution"])
        template = rearrange(template, "h w c -> () c h w")
        template = template.astype("float32") / 255
        template = torch.from_numpy(template).to(device)
        
        model_kwargs["template"] = template
    
    # Run inference
    print("Running geometric correction...")
    with torch.no_grad():
        out_bm = unwarp_model(**model_kwargs).detach().cpu()
    
    # Apply backward mapping to unwarp image
    print("Applying dewarping transformation...")
    image_original = rearrange(image_original, "h w c -> () c h w")
    image_original = image_original.astype("float32") / 255
    image_original = torch.from_numpy(image_original)
    
    unwarped_image = apply_map_torch(
        image=image_original, 
        bm=out_bm, 
        resolution=(output_height, output_width)
    )
    
    # Convert to numpy array [H, W, C] uint8
    unwarped_np = to_numpy_image(unwarped_image)
    
    # Save intermediate result if requested
    if save_intermediate:
        print(f"Saving unwarped image to: {intermediate_path}")
        save_image(intermediate_path, unwarped_np, override=True)
    
    # Optionally save flow field
    if save_flow:
        flow_path = output_path.parent / f"{output_path.stem}_flow.npz"
        print(f"Saving flow field to: {flow_path}")
        save_npz(flow_path, to_numpy_map(out_bm), override=True)
    
    print("✓ Geometric correction complete!")
    
    # Free memory
    del unwarp_model, image, out_bm
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("STEP 2: ILLUMINATION CORRECTION (Color Correction)")
    print("=" * 70)
    
    # Load IllTr model
    print(f"Loading IllTr model...")
    illtr_model = IllTr()
    illtr_model = reload_model(illtr_model, str(illtr_weights))
    illtr_model.to(device)
    illtr_model.eval()
    
    # Convert unwarped image to BGR for IllTr (it expects BGR)
    # unwarped_np is RGB from save_image format
    img_bgr = cv2.cvtColor(unwarped_np, cv2.COLOR_RGB2BGR)
    
    print("Running illumination correction...")
    corrected_bgr = rec_ill(illtr_model, img_bgr, device)
    
    # Convert back to RGB for saving
    corrected_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)
    
    # Save final result
    print(f"Saving color-corrected image to: {output_path}")
    Image.fromarray(corrected_rgb).save(output_path)
    
    print("✓ Color correction complete!")
    print("\n" + "=" * 70)
    print("✓ ALL DONE!")
    print("=" * 70)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Unwarp and color-correct a document image (Background removal + inv3d + DocTr IllTr)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline stages:
  0. Background Removal (U2NETP segmentation) - removes background via masking
  1. Geometric Correction (inv3d-model) - unwarps distorted document  
  2. Illumination Correction (DocTr IllTr) - corrects lighting/color issues

Examples:
  # Basic usage with all stages (default model: geotr@doc3d)
  python3 unwarp_and_correct.py --input data/1152226_geo.png
  
  # Use specific inv3d model for invoices/forms
  python3 unwarp_and_correct.py --input data/1152226_geo.png --model geotr@inv3d
  
  # Skip background removal if image already has clean background
  python3 unwarp_and_correct.py --input data/doc.jpg --skip-segmentation
  
  # Save intermediate unwarped-only image
  python3 unwarp_and_correct.py --input data/doc.jpg --save-intermediate
  
  # Use template-based model
  python3 unwarp_and_correct.py --input data/warped.png --model geotr_template@inv3d --template data/template.png
  
  # Custom output path and resolution
  python3 unwarp_and_correct.py --input data/doc.png --output data/result.png --width 2000 --height 2500
  
  # Use CPU instead of GPU
  python3 unwarp_and_correct.py --input data/doc.png --gpu -1
  
  # Custom model weights
  python3 unwarp_and_correct.py --input data/doc.png --illtr-weights path/to/illtr.pth --seg-weights path/to/seg.pth

Available inv3d models:
  - geotr@doc3d                  : General documents (Doc3D dataset)
  - geotr@inv3d                  : Invoices/forms (Inv3D dataset) - recommended for structured docs
  - geotr_template@inv3d         : Template-based (requires --template)
  - geotr_template_large@inv3d   : Larger template model (best quality, requires --template)
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input warped/distorted image"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save color-corrected image (default: input_corrected.png in same directory)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=list(model_sources.keys()),
        default="geotr@doc3d",
        help="inv3d model to use for unwarping (default: geotr@doc3d)"
    )
    
    parser.add_argument(
        "--template", "-t",
        type=str,
        default=None,
        help="Path to template image (required for template-based models)"
    )
    
    parser.add_argument(
        "--width", "-w",
        type=int,
        default=1700,
        help="Output image width in pixels (default: 1700)"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=2200,
        help="Output image height in pixels (default: 2200)"
    )
    
    parser.add_argument(
        "--gpu", "-g",
        type=int,
        default=0,
        help="GPU index to use (-1 for CPU, default: 0)"
    )
    
    parser.add_argument(
        "--save-flow",
        action="store_true",
        help="Save backward mapping flow field as .npz file"
    )
    
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save the unwarped-only image before color correction"
    )
    
    parser.add_argument(
        "--illtr-weights",
        type=str,
        default=None,
        help="Path to IllTr model weights (default: DocTr/model_pretrained/illtr.pth)"
    )
    
    parser.add_argument(
        "--seg-weights",
        type=str,
        default=None,
        help="Path to segmentation model weights (default: DocTr/model_pretrained/seg.pth)"
    )
    
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Skip background removal segmentation step (use original image as-is)"
    )
    
    args = parser.parse_args()
    
    try:
        unwarp_and_correct(
            input_path=args.input,
            output_path=args.output,
            model_name=args.model,
            template_path=args.template,
            output_width=args.width,
            output_height=args.height,
            gpu=args.gpu,
            save_flow=args.save_flow,
            save_intermediate=args.save_intermediate,
            illtr_weights=args.illtr_weights,
            seg_weights=args.seg_weights,
            skip_segmentation=args.skip_segmentation
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
