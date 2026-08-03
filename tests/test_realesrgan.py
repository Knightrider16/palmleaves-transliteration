"""
Quick test to verify Real-ESRGAN integration works in the preprocessing pipeline.
"""
import cv2
import numpy as np
from webapp.preprocess import full_pipeline_with_stages, REALESRGAN_AVAILABLE

def test_realesrgan_available():
    """Test that Real-ESRGAN dependencies are available."""
    print(f"Real-ESRGAN available: {REALESRGAN_AVAILABLE}")
    assert REALESRGAN_AVAILABLE, "Real-ESRGAN not available! Install with: pip install realesrgan basicsr"
    print("✓ Real-ESRGAN dependencies are installed")

def test_realesrgan_upscaling():
    """Test that Real-ESRGAN upscaling works in the pipeline."""
    # Create a small test image (100x50 grayscale)
    test_img = np.random.randint(0, 256, (50, 100), dtype=np.uint8)
    
    print("\nTesting Real-ESRGAN upscaling...")
    print(f"Input image shape: {test_img.shape}")
    
    # Run pipeline with Real-ESRGAN
    stages = full_pipeline_with_stages(test_img, use_realesrgan=True)
    
    # Check that upscaled image is 2x the CLAHE stage size
    clahe_shape = stages['clahe'].shape
    upscaled_shape = stages['upscaled'].shape
    
    print(f"CLAHE stage shape: {clahe_shape}")
    print(f"Upscaled stage shape: {upscaled_shape}")
    
    expected_height = clahe_shape[0] * 2
    expected_width = clahe_shape[1] * 2
    
    assert upscaled_shape[0] == expected_height, f"Height mismatch: {upscaled_shape[0]} != {expected_height}"
    assert upscaled_shape[1] == expected_width, f"Width mismatch: {upscaled_shape[1]} != {expected_width}"
    
    print("✓ Real-ESRGAN upscaling produces correct 2x dimensions")
    
    # Verify all 6 stages are present
    required_stages = ['original', 'clahe', 'upscaled', 'sharpened', 'binary', 'cleaned']
    for stage in required_stages:
        assert stage in stages, f"Missing stage: {stage}"
    
    print(f"✓ All {len(required_stages)} stages present: {', '.join(required_stages)}")

def test_with_real_image():
    """Test with an actual image if available."""
    import os
    
    # Try to find a sample image
    test_paths = [
        "data/original/sample1.png",
        "data/preprocessed/sample1.png",
        "toy_data/Ancient palm leaf documents/sample1.png"
    ]
    
    test_path = None
    for path in test_paths:
        if os.path.exists(path):
            test_path = path
            break
    
    if test_path:
        print(f"\nTesting with real image: {test_path}")
        img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            print(f"Input image shape: {img.shape}")
            stages = full_pipeline_with_stages(img, use_realesrgan=True)
            print(f"Upscaled shape: {stages['upscaled'].shape}")
            print("✓ Real-ESRGAN works with actual palm leaf image")
        else:
            print("⚠ Could not read test image")
    else:
        print("\n⚠ No sample images found for real image test (optional)")

if __name__ == "__main__":
    print("=" * 60)
    print("Real-ESRGAN Integration Test")
    print("=" * 60)
    
    try:
        test_realesrgan_available()
        test_realesrgan_upscaling()
        test_with_real_image()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nReal-ESRGAN is properly integrated into the preprocessing pipeline.")
        print("The webapp will now use 2x Real-ESRGAN super-resolution instead of bicubic.")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        print("\nMake sure Real-ESRGAN is installed:")
        print("  pip install realesrgan basicsr")
        raise
