"""
Adaptive Image Enhancement (AIE) for Handwritten Document - Streamlit Version
Implementation based on paper: "An Efficient Adaptive Image Enhancement Method in Wavelet Domain"

Author: Implementation for document enhancement
Date: 2026-01-22

Usage: streamlit run app.py
"""

import cv2
import numpy as np
import pywt
from scipy.ndimage import gaussian_filter
import streamlit as st
from PIL import Image


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
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_size, self.tile_size)
        )

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
        denoised_coeffs = [coeffs[0]]  # Keep LL subband

        for level_coeffs in coeffs[1:]:
            denoised_level = []
            for subband in level_coeffs:  # LH, HL, HH
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
        image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if method == 'otsu':
            _, binary = cv2.threshold(image_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                image_norm, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=15,
                C=10
            )

        elif method == 'sauvola':
            binary = self._sauvola_threshold(image_norm)

        else:
            raise ValueError(f"Unknown binarization method: {method}")

        return binary

    def _sauvola_threshold(self, image, window_size=15, k=0.2, R=128):
        """
        Sauvola's binarization method
        Good for documents with variable lighting
        """
        mean = cv2.boxFilter(image.astype(float), -1, (window_size, window_size))
        sqr_mean = cv2.boxFilter(image.astype(float)**2, -1, (window_size, window_size))
        std = np.sqrt(sqr_mean - mean**2)

        threshold = mean * (1 + k * ((std / R) - 1))
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
        resized = self.resize_image(image)
        enhanced = self.mclahe(resized)
        coeffs = self.dwt_decompose(enhanced)
        denoised_coeffs = self.denoise_subbands(coeffs)
        reconstructed = self.dwt_reconstruct(denoised_coeffs)
        reconstructed = np.clip(reconstructed, 0, 255)

        if output_binary:
            result = self.binarize(reconstructed, method=binarization_method)
        else:
            result = reconstructed.astype(np.uint8)

        return result


def main():
    st.set_page_config(
        page_title="Document Enhancement Tool",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 Adaptive Document Image Enhancement")
    st.markdown("### ปรับพารามิเตอร์แบบ real-time")

    # Sidebar parameters
    st.sidebar.header("⚙️ Parameters")

    # Wavelet parameters
    st.sidebar.subheader("🌊 Wavelet Parameters")
    wavelet = st.sidebar.selectbox(
        "Wavelet Type",
        ['db4', 'db2', 'db6', 'db8', 'sym4', 'sym8', 'coif1', 'haar'],
        index=0
    )
    level = st.sidebar.slider("Decomposition Level", 1, 3, 2)

    # CLAHE parameters
    st.sidebar.subheader("📊 CLAHE Parameters")
    clip_limit = st.sidebar.slider("Clip Limit", 1.0, 5.0, 2.0, 0.1)
    tile_size = st.sidebar.slider("Tile Size", 2, 16, 8, 2)

    # Binarization
    st.sidebar.subheader("⚫⚪ Binarization")
    binarization_method = st.sidebar.selectbox(
        "Method",
        ['otsu', 'adaptive', 'sauvola'],
        index=0
    )

    # Output options
    st.sidebar.subheader("📤 Output")
    output_binary = st.sidebar.checkbox("Output Binary", value=True)

    # File uploader
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png', 'bmp'])

    if uploaded_file is not None:
        # Convert uploaded file to opencv image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Store original in session state to avoid re-reading
        if 'original_image' not in st.session_state or st.session_state.get('uploaded_filename') != uploaded_file.name:
            st.session_state['original_image'] = image
            st.session_state['uploaded_filename'] = uploaded_file.name

        image = st.session_state['original_image']

        # Process image (auto-update on parameter change)
        with st.spinner("Processing..."):
            enhancer = AdaptiveImageEnhancement(
                wavelet=wavelet,
                level=level,
                clip_limit=clip_limit,
                tile_size=tile_size
            )

            result = enhancer.enhance(
                image,
                output_binary=output_binary,
                binarization_method=binarization_method
            )

        # Display results in columns
        cols = st.columns(2)

        with cols[0]:
            st.subheader("Original")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

        with cols[1]:
            output_type = "Binary" if output_binary else "Enhanced Grayscale"
            st.subheader(f"{output_type} ({binarization_method.capitalize()})")
            st.image(result, use_container_width=True)

        # Download button
        st.markdown("---")
        _, buffer = cv2.imencode('.png', result)
        st.download_button(
            label="💾 Download Result",
            data=buffer.tobytes(),
            file_name=f"enhanced_{uploaded_file.name}",
            mime="image/png"
        )

    else:
        st.info("👆 Upload an image to get started")

        st.markdown("""
        ### 📖 Quick Guide:

        1. **Upload** document image
        2. **Adjust parameters** in sidebar - output updates automatically!
        3. **Download** when satisfied

        ### 💡 Recommended Settings:

        **For faint handwriting:**
        - Clip Limit: 3.0-4.0
        - Tile Size: 4-6
        - Method: Sauvola

        **For noisy documents:**
        - Clip Limit: 2.0-3.0
        - Tile Size: 6-8
        - Method: Adaptive or Sauvola

        **For clean documents:**
        - Clip Limit: 2.0
        - Tile Size: 8
        - Method: Otsu
        """)


if __name__ == '__main__':
    main()