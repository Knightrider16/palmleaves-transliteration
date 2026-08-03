"""
Image preprocessing utilities for the web app deployment.

This module provides preprocessing functions optimized for web deployment:
- CLAHE contrast enhancement
- Adaptive thresholding for mask generation
- Noise reduction and morphological operations
- Real-ESRGAN 2x upscaling

The preprocessing pipeline mirrors the offline preprocess_pipeline.py.
"""
from __future__ import annotations
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Literal, Optional

# Real-ESRGAN imports
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    print("Warning: Real-ESRGAN not available. Install with: pip install realesrgan basicsr")


PreprocessLevel = Literal["none", "light", "standard", "heavy"]


def enhance_contrast(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        img: Grayscale image
        clip_limit: Threshold for contrast limiting (default: 2.0)
        tile_size: Size of grid for histogram equalization (default: 8)
    
    Returns:
        Enhanced grayscale image
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img)


def create_binary_mask(
    img: np.ndarray,
    block_size: int = 41,
    c_value: int = 7,
    denoise: bool = True
) -> np.ndarray:
    """
    Create a binary mask using adaptive thresholding.
    
    Args:
        img: Grayscale image
        block_size: Size of pixel neighborhood for threshold calculation (must be odd)
        c_value: Constant subtracted from weighted mean
        denoise: Apply Gaussian blur before thresholding
    
    Returns:
        Binary mask (white text on black background)
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if denoise:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    
    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, c_value
    )
    
    return binary


def remove_noise(
    mask: np.ndarray,
    min_area: int = 40,
    min_height: int = 3,
    min_width: int = 3,
    max_area_ratio: float = 0.25
) -> np.ndarray:
    """
    Remove small noise components and very large blobs from binary mask.
    
    Args:
        mask: Binary mask
        min_area: Minimum component area to keep
        min_height: Minimum component height to keep
        min_width: Minimum component width to keep
        max_area_ratio: Maximum component area as ratio of total image area
    
    Returns:
        Cleaned binary mask
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    
    H, W = mask.shape
    max_area = max_area_ratio * H * W
    clean = np.zeros_like(mask)
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if (area >= min_area and 
            h >= min_height and 
            w >= min_width and 
            area <= max_area):
            clean[labels == i] = 255
    
    return clean


def apply_morphology(
    mask: np.ndarray,
    operation: Literal["dilate", "erode", "close", "open"] = "dilate",
    kernel_size: tuple[int, int] = (2, 1),
    iterations: int = 1
) -> np.ndarray:
    """
    Apply morphological operations to the mask.
    
    Args:
        mask: Binary mask
        operation: Type of morphological operation
        kernel_size: Size of the structuring element (width, height)
        iterations: Number of times to apply the operation
    
    Returns:
        Processed binary mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    if operation == "dilate":
        return cv2.dilate(mask, kernel, iterations=iterations)
    elif operation == "erode":
        return cv2.erode(mask, kernel, iterations=iterations)
    elif operation == "close":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation == "open":
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    else:
        return mask


# Global Real-ESRGAN upsampler instance (initialized once)
_UPSAMPLER = None

def get_realesrgan_upsampler():
    """
    Get or initialize the Real-ESRGAN upsampler (singleton pattern).
    
    Returns:
        RealESRGANer instance or None if not available
    """
    global _UPSAMPLER
    
    if not REALESRGAN_AVAILABLE:
        return None
    
    if _UPSAMPLER is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )
        
        # Use relative path from webapp/ to weights/
        model_path = Path(__file__).parent.parent / "weights" / "realesrgan_x4plus.pth"
        
        _UPSAMPLER = RealESRGANer(
            scale=4,
            model_path=str(model_path),
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
            device=device,
        )
    
    return _UPSAMPLER


def realesrgan_upscale(img: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Upscale image using Real-ESRGAN 2x super-resolution.
    
    This matches the offline batch_upscale.py pipeline.
    Falls back to bicubic if Real-ESRGAN is not available.
    
    Args:
        img: Input image (grayscale)
        scale: Upscaling factor (only 2 is supported, uses 4x model with outscale=2)
    
    Returns:
        Upscaled image (grayscale)
    """
    upsampler = get_realesrgan_upsampler()
    
    if upsampler is None:
        # Fallback to bicubic if Real-ESRGAN not available
        print("Warning: Using bicubic fallback (Real-ESRGAN not available)")
        h, w = img.shape[:2]
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    
    # Convert grayscale to BGR for Real-ESRGAN
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Apply Real-ESRGAN
    output, _ = upsampler.enhance(img_bgr, outscale=scale)
    
    # Convert back to grayscale
    output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    
    return output_gray


def lightweight_upscale(img: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Lightweight upscaling using bicubic interpolation.
    This is a fast alternative to Real-ESRGAN for deployment environments.
    
    Args:
        img: Input image
        scale: Upscale factor
    
    Returns:
        Upscaled image
    """
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def preprocess_image(
    img: np.ndarray,
    level: PreprocessLevel = "standard",
    upscale: bool = False,
    upscale_factor: int = 2
) -> np.ndarray:
    """
    Complete preprocessing pipeline for palm leaf images.
    
    Args:
        img: Input image (BGR, grayscale, or binary)
        level: Preprocessing intensity level:
            - "none": Return grayscale only
            - "light": CLAHE only
            - "standard": CLAHE + adaptive threshold + light noise removal
            - "heavy": Full pipeline with aggressive noise removal
        upscale: Whether to apply lightweight upscaling
        upscale_factor: Upscale factor if upscale=True
    
    Returns:
        Preprocessed image (grayscale or binary depending on level)
    """
    # Convert to grayscale if needed
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply upscaling first if requested
    if upscale:
        gray = lightweight_upscale(gray, scale=upscale_factor)
    
    # Apply preprocessing based on level
    if level == "none":
        return gray
    
    # Enhance contrast
    enhanced = enhance_contrast(gray, clip_limit=2.0, tile_size=8)
    
    if level == "light":
        return enhanced
    
    # Create binary mask
    binary = create_binary_mask(enhanced, block_size=41, c_value=7, denoise=True)
    
    if level == "standard":
        # Light noise removal
        clean = remove_noise(binary, min_area=40, min_height=3, min_width=3)
        # Light dilation to connect nearby components
        clean = apply_morphology(clean, "dilate", kernel_size=(2, 1), iterations=1)
        return clean
    
    elif level == "heavy":
        # Aggressive noise removal
        clean = remove_noise(binary, min_area=60, min_height=4, min_width=4, max_area_ratio=0.2)
        # Morphological closing to fill gaps
        clean = apply_morphology(clean, "close", kernel_size=(3, 2), iterations=1)
        # Light dilation
        clean = apply_morphology(clean, "dilate", kernel_size=(2, 1), iterations=1)
        return clean
    
    return gray


def preprocess_from_path(
    image_path: str | Path,
    level: PreprocessLevel = "standard",
    upscale: bool = False,
    upscale_factor: int = 2,
    save_path: Optional[str | Path] = None
) -> np.ndarray:
    """
    Load and preprocess an image from file.
    
    Args:
        image_path: Path to input image
        level: Preprocessing intensity level
        upscale: Whether to apply lightweight upscaling
        upscale_factor: Upscale factor if upscale=True
        save_path: Optional path to save preprocessed image
    
    Returns:
        Preprocessed image
    
    Raises:
        ValueError: If image cannot be read
    """
    img = cv2.imread(str(image_path))
    if img is None:
        # Try grayscale fallback
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
    
    result = preprocess_image(img, level=level, upscale=upscale, upscale_factor=upscale_factor)
    
    if save_path:
        cv2.imwrite(str(save_path), result)
    
    return result


def full_pipeline_with_stages(img: np.ndarray, use_realesrgan: bool = True) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """
    Run the full preprocessing pipeline and return intermediate results.
    
    This matches the offline pipeline from preprocess_pipeline.py:
    1. Original image
    2. CLAHE enhancement (preprocessing)
    3. Upscaled (2x Real-ESRGAN super-resolution)
    4. Sharpened (postprocessing - CLAHE + sharpening kernel)
    5. Binary mask (adaptive threshold)
    6. Cleaned mask (noise removal + morphology)
    
    Args:
        img: Input image (BGR, grayscale, or binary)
        use_realesrgan: Use Real-ESRGAN (True) or bicubic fallback (False)
    
    Returns:
        Tuple of (stages_dict, metadata_dict) where:
        - stages_dict has keys: 'original', 'clahe', 'upscaled', 'sharpened', 'binary', 'cleaned'
        - metadata_dict has 'upscale_method': 'Real-ESRGAN 2x' or 'Bicubic 2x'
    """
    stages = {}
    
    # Convert to grayscale if needed
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    stages['original'] = gray.copy()
    
    # Stage 1: CLAHE enhancement (preprocessing step)
    clahe = enhance_contrast(gray, clip_limit=2.0, tile_size=8)
    stages['clahe'] = clahe.copy()
    
    # Stage 2: Real-ESRGAN 2x upscaling (or bicubic fallback)
    upsampler = get_realesrgan_upsampler()
    metadata = {}
    
    if use_realesrgan and upsampler is not None:
        upscaled = realesrgan_upscale(clahe, scale=2)
        metadata['upscale_method'] = 'Real-ESRGAN 2x'
    else:
        upscaled = lightweight_upscale(clahe, scale=2)
        metadata['upscale_method'] = 'Bicubic 2x'
    stages['upscaled'] = upscaled.copy()
    
    # Stage 3: Postprocessing (CLAHE + sharpening)
    # Apply CLAHE again on upscaled image
    postprocess_clahe = enhance_contrast(upscaled, clip_limit=2.0, tile_size=8)
    # Sharpening kernel
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(postprocess_clahe, -1, sharpen_kernel)
    stages['sharpened'] = sharpened.copy()
    
    # Stage 4: Binary mask (adaptive threshold)
    binary = create_binary_mask(sharpened, block_size=41, c_value=7, denoise=True)
    stages['binary'] = binary.copy()
    
    # Stage 5: Cleaned mask (noise removal + morphology)
    clean = remove_noise(binary, min_area=40, min_height=3, min_width=3)
    clean = apply_morphology(clean, "dilate", kernel_size=(2, 1), iterations=1)
    stages['cleaned'] = clean
    
    return stages, metadata


def is_already_binary(img: np.ndarray, threshold: float = 0.9) -> bool:
    """
    Check if an image is already a binary mask.
    
    Args:
        img: Input image
        threshold: Ratio of pixels at extremes to consider binary
    
    Returns:
        True if image appears to be binary
    """
    if img.ndim != 2:
        return False
    
    hist = np.histogram(img, bins=8)[0]
    edge_pixels = hist[0] + hist[-1]
    return edge_pixels > threshold * img.size
