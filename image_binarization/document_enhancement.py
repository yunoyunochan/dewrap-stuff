"""
Adaptive Image Enhancement (AIE) for Handwritten Document
Implementation based on paper: "An Efficient Adaptive Image Enhancement Method in Wavelet Domain"

Author: Implementation for document enhancement
Date: 2026-01-22
"""

import cv2
import numpy as np
import pywt
from scipy.ndimage import gaussian_filter
import argparse
from pathlib import Path


class AdaptiveImageEnhancement:
    """
    Adaptive Image Enhancement using MCLAHE and Directional Wavelet Transform
    """

    def __init__(self, wavelet='db4', level=2, clip_limit=2.0, tile_size=8):
        """
        Args:
            wavelet: Wavelet type (default: 'db4' - Daubechies 4)
            level: Decomposition level for DWT
            clip_limit: Clip limit for MCLAHE (Nclip parameter)
            tile_size: Tile size for MCLAHE contextual regions
        """
        self.wavelet = wavelet
        self.level = level
        self.clip_limit = clip_limit
        self.tile_size = tile_size

    def resize_image(self, image, max_size=1200):
        """Resize image if too large while maintaining aspect ratio"""
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    def mclahe(self, image):
        """
        Modified Contrast Limited Adaptive Histogram Equalization
        Based on paper equations (5) and (6)
        """
        # Ensure image is grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Create CLAHE object
        # clipLimit controls histogram clipping threshold
        # tileGridSize is the contextual region size
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_size, self.tile_size)
        )

        # Apply CLAHE
        enhanced = clahe.apply(gray)

        return enhanced

    def dwt_decompose(self, image):
        """
        Discrete Wavelet Transform decomposition
        Decomposes image into LL, LH, HL, HH subbands
        """
        coeffs = pywt.wavedec2(image, self.wavelet, level=self.level)
        return coeffs

    def denoise_subbands(self, coeffs, sigma=1.0):
        """
        Apply Gaussian filtering to denoise high-frequency subbands
        Keep LL subband (approximation) unchanged
        """
        # coeffs[0] is LL (approximation)
        # coeffs[1:] are detail subbands (LH, HL, HH) for each level

        denoised_coeffs = [coeffs[0]]  # Keep LL subband

        # Denoise detail subbands
        for level_coeffs in coeffs[1:]:
            denoised_level = []
            for subband in level_coeffs:  # LH, HL, HH
                # Apply Gaussian filter to reduce noise
                denoised = gaussian_filter(subband, sigma=sigma)
                denoised_level.append(denoised)
            denoised_coeffs.append(tuple(denoised_level))

        return denoised_coeffs

    def dwt_reconstruct(self, coeffs):
        """
        Inverse Discrete Wavelet Transform
        Reconstructs image from wavelet coefficients
        """
        reconstructed = pywt.waverec2(coeffs, self.wavelet)
        return reconstructed

    def binarize(self, image, method='otsu'):
        """
        Convert enhanced image to binary

        Args:
            image: Enhanced grayscale image
            method: 'otsu', 'adaptive', or 'sauvola'
        """
        # Normalize to 0-255 range
        image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if method == 'otsu':
            # Otsu's binarization
            _, binary = cv2.threshold(image_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif method == 'adaptive':
            # Adaptive Gaussian thresholding
            binary = cv2.adaptiveThreshold(
                image_norm, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=15,
                C=10
            )

        elif method == 'sauvola':
            # Sauvola binarization (good for degraded documents)
            binary = self._sauvola_threshold(image_norm)

        else:
            raise ValueError(f"Unknown binarization method: {method}")

        return binary

    def _sauvola_threshold(self, image, window_size=15, k=0.2, R=128):
        """
        Sauvola's binarization method
        Good for documents with variable lighting
        """
        # Calculate local mean and standard deviation
        mean = cv2.boxFilter(image.astype(float), -1, (window_size, window_size))
        sqr_mean = cv2.boxFilter(image.astype(float)**2, -1, (window_size, window_size))
        std = np.sqrt(sqr_mean - mean**2)

        # Sauvola threshold
        threshold = mean * (1 + k * ((std / R) - 1))

        # Apply threshold
        binary = np.where(image > threshold, 255, 0).astype(np.uint8)

        return binary

    def enhance(self, image, output_binary=True, binarization_method='otsu'):
        """
        Complete enhancement pipeline

        Args:
            image: Input image (grayscale or color)
            output_binary: If True, return binary image; else return enhanced grayscale
            binarization_method: Method for binarization ('otsu', 'adaptive', 'sauvola')

        Returns:
            Enhanced image (binary if output_binary=True)
        """
        print("Step 1: Resizing image...")
        resized = self.resize_image(image)

        print("Step 2: Applying MCLAHE...")
        enhanced = self.mclahe(resized)

        print("Step 3: DWT decomposition...")
        coeffs = self.dwt_decompose(enhanced)

        print("Step 4: Denoising subbands...")
        denoised_coeffs = self.denoise_subbands(coeffs)

        print("Step 5: Inverse DWT...")
        reconstructed = self.dwt_reconstruct(denoised_coeffs)

        # Clip to valid range
        reconstructed = np.clip(reconstructed, 0, 255)

        if output_binary:
            print(f"Step 6: Binarization using {binarization_method}...")
            result = self.binarize(reconstructed, method=binarization_method)
        else:
            result = reconstructed.astype(np.uint8)

        print("Enhancement complete!")
        return result


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Adaptive Image Enhancement for Handwritten Documents'
    )
    parser.add_argument('input', type=str, help='Input image path')
    parser.add_argument('output', type=str, help='Output image path')
    parser.add_argument('--wavelet', type=str, default='db4',
                      help='Wavelet type (default: db4)')
    parser.add_argument('--level', type=int, default=2,
                      help='Decomposition level (default: 2)')
    parser.add_argument('--clip-limit', type=float, default=2.0,
                      help='CLAHE clip limit (default: 2.0)')
    parser.add_argument('--tile-size', type=int, default=8,
                      help='CLAHE tile size (default: 8)')
    parser.add_argument('--binarization', type=str, default='otsu',
                      choices=['otsu', 'adaptive', 'sauvola'],
                      help='Binarization method (default: otsu)')
    parser.add_argument('--no-binary', action='store_true',
                      help='Output enhanced grayscale instead of binary')

    args = parser.parse_args()

    # Load image
    print(f"Loading image: {args.input}")
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not load image from {args.input}")
        return

    # Create enhancer
    enhancer = AdaptiveImageEnhancement(
        wavelet=args.wavelet,
        level=args.level,
        clip_limit=args.clip_limit,
        tile_size=args.tile_size
    )

    # Enhance image
    result = enhancer.enhance(
        image,
        output_binary=not args.no_binary,
        binarization_method=args.binarization
    )

    # Save result
    cv2.imwrite(args.output, result)
    print(f"Saved result to: {args.output}")


if __name__ == '__main__':
    main()
    
# python3 document_enhancement.py 10.bmp output.png