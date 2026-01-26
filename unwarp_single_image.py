#!/usr/bin/env python3
"""
Simple script to unwarp a single image using inv3d-model
Usage: python3 unwarp_single_image.py --input data/1152226_geo.png --model geotr@doc3d

python3 unwarp_single_image.py --input data/1152226_geo.png

cd /mnt/e/fusion/image_processing_and_OCR && python3 unwarp_single_image.py --input data/S__87040009.jpg --model geotr@doc3d
cd /mnt/e/fusion/image_processing_and_OCR && python3 unwarp_single_image.py --input data/1152226_geo.png --model geotr@inv3d

"""

import cv2
cv2.setNumThreads(0)

import argparse
import os
import sys
from pathlib import Path

import gdown
import torch
import yaml
from einops import rearrange

# Setup paths
script_dir = Path(__file__).parent.resolve()
project_dir = script_dir / "inv3d-model"
sys.path.insert(0, str(project_dir / "src"))

from inv3d_model.models import model_factory
from inv3d_util.image import scale_image
from inv3d_util.load import load_image, save_image, save_npz
from inv3d_util.mapping import apply_map_torch
from inv3d_util.misc import to_numpy_image, to_numpy_map

# Load available models
model_sources = yaml.safe_load((project_dir / "models.yaml").read_text())


def unwarp_image(
    input_path: str,
    output_path: str = None,
    model_name: str = "geotr@doc3d",
    template_path: str = None,
    output_width: int = 1700,
    output_height: int = 2200,
    gpu: int = 0,
    save_flow: bool = False
):
    """
    Unwarp a single distorted document image
    
    Args:
        input_path: Path to input warped image
        output_path: Path to save unwarped image (default: same dir as input with '_unwarped' suffix)
        model_name: Model to use (geotr@doc3d, geotr@inv3d, geotr_template@inv3d, geotr_template_large@inv3d)
        template_path: Path to template image (required for template models)
        output_width: Output image width in pixels
        output_height: Output image height in pixels
        gpu: GPU index (-1 for CPU)
        save_flow: Whether to save the backward mapping flow field (.npz)
    
    Returns:
        Path to unwarped image
    """
    
    # Setup GPU
    if gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        device = "cuda"
    else:
        device = "cpu"
    
    # Validate input
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    
    # Setup output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_unwarped{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download and load model
    print(f"Loading model: {model_name}")
    model_url = model_sources[model_name]
    model_path = Path(
        gdown.cached_download(
            url=model_url, 
            path=project_dir / f"models/{model_name}.ckpt"
        )
    )
    
    model = model_factory.load_from_checkpoint(model_name.split("@")[0], model_path)
    model.to(device)
    model.eval()
    
    print(f"Processing image: {input_path}")
    
    # Load and prepare input image
    image_original = load_image(input_path)
    image = scale_image(image_original, resolution=model.dataset_options["resolution"])
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
        template = scale_image(template_original, resolution=model.dataset_options["resolution"])
        template = rearrange(template, "h w c -> () c h w")
        template = template.astype("float32") / 255
        template = torch.from_numpy(template).to(device)
        
        model_kwargs["template"] = template
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        out_bm = model(**model_kwargs).detach().cpu()
    
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
    
    # Save unwarped image
    print(f"Saving unwarped image to: {output_path}")
    save_image(output_path, to_numpy_image(unwarped_image), override=True)
    
    # Optionally save flow field
    if save_flow:
        flow_path = output_path.parent / f"{output_path.stem}_flow.npz"
        print(f"Saving flow field to: {flow_path}")
        save_npz(flow_path, to_numpy_map(out_bm), override=True)
    
    print("✓ Done!")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Unwarp a single document image using inv3d-model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default model (geotr@doc3d)
  python unwarp_single_image.py --input data/1152226_geo.png
  
  # Use specific model
  python unwarp_single_image.py --input data/1152226_geo.png --model geotr@inv3d
  
  # Use template-based model
  python unwarp_single_image.py --input data/warped.png --model geotr_template@inv3d --template data/template.png
  
  # Custom output path and resolution
  python unwarp_single_image.py --input data/1152226_geo.png --output data/result.png --width 2000 --height 2500
  
  # Save flow field for analysis
  python unwarp_single_image.py --input data/1152226_geo.png --save-flow
  
  # Use CPU instead of GPU
  python unwarp_single_image.py --input data/1152226_geo.png --gpu -1

Available models:
  - geotr@doc3d                  : General documents (Doc3D dataset)
  - geotr@inv3d                  : Invoices/forms (Inv3D dataset)
  - geotr_template@inv3d         : Template-based (recommended for structured docs)
  - geotr_template_large@inv3d   : Larger template model (best quality)
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
        help="Path to save unwarped image (default: input_unwarped.png in same directory)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=list(model_sources.keys()),
        default="geotr@doc3d",
        help="Model to use for unwarping (default: geotr@doc3d)"
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
    
    args = parser.parse_args()
    
    try:
        unwarp_image(
            input_path=args.input,
            output_path=args.output,
            model_name=args.model,
            template_path=args.template,
            output_width=args.width,
            output_height=args.height,
            gpu=args.gpu,
            save_flow=args.save_flow
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
