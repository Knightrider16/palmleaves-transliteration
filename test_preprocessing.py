"""
Test script for webapp preprocessing module.

This script tests the preprocessing functions to ensure they work correctly
before deployment.

Run:
    python test_preprocessing.py
"""
import cv2
import numpy as np
from pathlib import Path
from webapp.preprocess import (
    enhance_contrast,
    create_binary_mask,
    remove_noise,
    apply_morphology,
    lightweight_upscale,
    preprocess_image,
    preprocess_from_path,
    is_already_binary,
)


def test_enhance_contrast():
    """Test CLAHE contrast enhancement."""
    print("\n=== Testing enhance_contrast ===")
    
    # Create test grayscale image with varying brightness
    img = np.random.randint(50, 150, (100, 100), dtype=np.uint8)
    enhanced = enhance_contrast(img)
    
    assert enhanced.shape == img.shape, "Shape mismatch"
    assert enhanced.dtype == np.uint8, "Wrong dtype"
    print("✓ enhance_contrast passed")


def test_create_binary_mask():
    """Test adaptive thresholding."""
    print("\n=== Testing create_binary_mask ===")
    
    img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    mask = create_binary_mask(img)
    
    assert mask.shape == img.shape, "Shape mismatch"
    assert set(np.unique(mask)).issubset({0, 255}), "Not binary"
    print("✓ create_binary_mask passed")


def test_remove_noise():
    """Test noise removal."""
    print("\n=== Testing remove_noise ===")
    
    # Create binary image with small and large components
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (50, 50), (130, 130), 255, -1)  # Large component (80x80=6400 pixels, < 0.25*40000)
    mask[10, 10] = 255  # Single pixel noise
    mask[20:22, 20:22] = 255  # Small noise (4 pixels)
    
    clean = remove_noise(mask, min_area=40, max_area_ratio=0.3)
    
    assert clean.shape == mask.shape, "Shape mismatch"
    assert clean[10, 10] == 0, "Failed to remove single pixel"
    assert np.sum(clean[20:22, 20:22]) == 0, "Failed to remove small noise"
    assert np.sum(clean[80:100, 80:100]) > 0, "Removed large component"
    print("✓ remove_noise passed")


def test_apply_morphology():
    """Test morphological operations."""
    print("\n=== Testing apply_morphology ===")
    
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (40, 40), (60, 60), 255, -1)
    
    dilated = apply_morphology(mask, "dilate", kernel_size=(3, 3), iterations=1)
    eroded = apply_morphology(mask, "erode", kernel_size=(3, 3), iterations=1)
    
    assert np.sum(dilated) > np.sum(mask), "Dilation failed"
    assert np.sum(eroded) < np.sum(mask), "Erosion failed"
    print("✓ apply_morphology passed")


def test_lightweight_upscale():
    """Test upscaling."""
    print("\n=== Testing lightweight_upscale ===")
    
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    upscaled = lightweight_upscale(img, scale=2)
    
    assert upscaled.shape == (200, 200), f"Expected (200, 200), got {upscaled.shape}"
    print("✓ lightweight_upscale passed")


def test_preprocess_image():
    """Test complete preprocessing pipeline."""
    print("\n=== Testing preprocess_image ===")
    
    img = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
    
    # Test all levels
    none_result = preprocess_image(img, level="none")
    light_result = preprocess_image(img, level="light")
    standard_result = preprocess_image(img, level="standard")
    heavy_result = preprocess_image(img, level="heavy")
    
    assert none_result.shape == img.shape, "None level failed"
    assert light_result.shape == img.shape, "Light level failed"
    assert standard_result.shape == img.shape, "Standard level failed"
    assert heavy_result.shape == img.shape, "Heavy level failed"
    
    # Test with upscaling
    upscaled_result = preprocess_image(img, level="standard", upscale=True, upscale_factor=2)
    assert upscaled_result.shape == (400, 600), "Upscaling failed"
    
    print("✓ preprocess_image passed all levels")


def test_is_already_binary():
    """Test binary image detection."""
    print("\n=== Testing is_already_binary ===")
    
    # Create binary image
    binary = np.zeros((100, 100), dtype=np.uint8)
    binary[25:75, 25:75] = 255
    
    # Create grayscale image
    grayscale = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    assert is_already_binary(binary) == True, "Failed to detect binary"
    assert is_already_binary(grayscale) == False, "False positive on grayscale"
    print("✓ is_already_binary passed")


def test_full_pipeline_with_stages():
    """Test the full preprocessing pipeline with stage output."""
    print("\n=== Testing full_pipeline_with_stages ===")
    
    from webapp.preprocess import full_pipeline_with_stages
    
    img = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
    
    stages = full_pipeline_with_stages(img)
    
    # Check all expected stages are present
    expected_stages = ['original', 'clahe', 'upscaled', 'sharpened', 'binary', 'cleaned']
    for stage in expected_stages:
        assert stage in stages, f"Missing stage: {stage}"
        assert stages[stage] is not None, f"Stage {stage} is None"
    
    # Check upscaled dimensions are 2x
    assert stages['upscaled'].shape[0] == img.shape[0] * 2, "Upscaling failed (height)"
    assert stages['upscaled'].shape[1] == img.shape[1] * 2, "Upscaling failed (width)"
    
    # Check binary stage is actually binary
    binary_unique = set(np.unique(stages['binary']))
    assert binary_unique.issubset({0, 255}), "Binary stage is not binary"
    
    print("✓ full_pipeline_with_stages passed")


def test_with_real_image():
    """Test preprocessing with a real image if available."""
    print("\n=== Testing with real image (if available) ===")
    
    from webapp.preprocess import full_pipeline_with_stages
    
    # Try to find a sample image
    sample_dirs = [
        Path("webapp/static/samples"),
        Path("data/original"),
        Path("toy_data/Ancient palm leaf documents"),
    ]
    
    test_image = None
    for sample_dir in sample_dirs:
        if sample_dir.exists():
            images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
            if images:
                test_image = images[0]
                break
    
    if test_image:
        print(f"  Testing with: {test_image.name}")
        try:
            result = preprocess_from_path(
                test_image,
                level="standard",
                upscale=False
            )
            print(f"  ✓ Processed successfully: {result.shape}")
            
            # Test with upscaling
            upscaled = preprocess_from_path(
                test_image,
                level="standard",
                upscale=True,
                upscale_factor=2
            )
            print(f"  ✓ Upscaled successfully: {upscaled.shape}")
            
            # Test full pipeline with stages
            img = cv2.imread(str(test_image))
            stages = full_pipeline_with_stages(img)
            print(f"  ✓ Full pipeline with stages: {len(stages)} stages generated")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    else:
        print("  ⊘ No sample images found, skipping real image test")


def run_all_tests():
    """Run all preprocessing tests."""
    print("=" * 60)
    print("PREPROCESSING MODULE TESTS")
    print("=" * 60)
    
    try:
        test_enhance_contrast()
        test_create_binary_mask()
        test_remove_noise()
        test_apply_morphology()
        test_lightweight_upscale()
        test_preprocess_image()
        test_is_already_binary()
        test_full_pipeline_with_stages()
        test_with_real_image()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nPreprocessing module is ready for deployment!")
        
    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        raise
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ ERROR: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    run_all_tests()
