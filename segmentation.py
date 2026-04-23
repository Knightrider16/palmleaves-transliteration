import cv2
import numpy as np
import os

input_folder = "data/masks_clean_upscaled"
output_folder = "data/characters"
os.makedirs(output_folder, exist_ok=True)

char_id = 0

def sauvola_threshold(gray, window_size=25, k=0.5, R=128):
    gray_f = gray.astype(np.float64)
    mean = cv2.boxFilter(gray_f, -1, (window_size, window_size))
    mean_sq = cv2.boxFilter(gray_f**2, -1, (window_size, window_size))
    std = np.sqrt(np.abs(mean_sq - mean**2))
    threshold = mean * (1 + k * (std / R - 1))
    binary = (gray_f < threshold).astype(np.uint8) * 255
    return binary

def detect_and_remove_punch_holes(binary, num_holes=2, min_area=200):
    """
    Find large near-circular blobs = punch holes.
    Returns cleaned binary and the hole bounding boxes (for debug).
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    mask = np.zeros_like(binary)
    holes_found = 0
    epsilon = 0.4

    for cnt in contours_sorted:
        area = cv2.contourArea(cnt)
        if area < min_area:
            break
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h) if h > 0 else 0
        if (1 - epsilon) < aspect_ratio < (1 + epsilon):
            # Dilate the mask region to erase halo around hole too
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            holes_found += 1
            if holes_found >= num_holes:
                break

    cleaned = cv2.bitwise_xor(binary, mask)
    return cleaned

def crop_to_content(binary, padding=10):
    """
    Crop away the black border edges of the leaf (tapered ends).
    Uses the bounding box of all foreground pixels.
    """
    coords = cv2.findNonZero(binary)
    if coords is None:
        return binary
    x, y, w, h = cv2.boundingRect(coords)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(binary.shape[1] - x, w + 2 * padding)
    h = min(binary.shape[0] - y, h + 2 * padding)
    return binary[y:y+h, x:x+w]

def is_valid_character(char_img, min_ratio=0.08):
    """0/1 ratio: reject blobs that are mostly empty (noise)"""
    black = np.sum(char_img == 0)
    white = np.sum(char_img == 255)
    if white == 0:
        return False
    return (black / float(white)) >= min_ratio

for img_name in sorted(os.listdir(input_folder)):

    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    h_img, w_img = img.shape[:2]

    # --------------------------------------------------
    # 1. Grayscale
    # --------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------
    # 2. Check if image is inverted (white-on-black)
    #    Mean < 127 means mostly dark = inverted
    # --------------------------------------------------
    mean_val = np.mean(gray)
    if mean_val < 127:
        gray = cv2.bitwise_not(gray)  # flip to black-on-white

    # --------------------------------------------------
    # 3. Denoise
    # --------------------------------------------------
    denoised = cv2.fastNlMeansDenoising(gray, h=10,
                                         templateWindowSize=7,
                                         searchWindowSize=21)
    denoised = cv2.medianBlur(denoised, 3)

    # --------------------------------------------------
    # 4. Sauvola adaptive thresholding
    # --------------------------------------------------
    binary = sauvola_threshold(denoised, window_size=25, k=0.5)

    # --------------------------------------------------
    # 5. Crop content (remove tapered black leaf edges)
    # --------------------------------------------------
    binary = crop_to_content(binary, padding=5)

    # --------------------------------------------------
    # 6. Remove punch holes BEFORE anything else
    # --------------------------------------------------
    binary = detect_and_remove_punch_holes(binary, num_holes=2)

    # --------------------------------------------------
    # 7. Remove horizontal lines (leaf fibres)
    # --------------------------------------------------
    h_kernel_len = max(60, w_img // 15)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
    binary = cv2.subtract(binary, horizontal)

    # --------------------------------------------------
    # 8. Light closing to reconnect broken strokes
    # --------------------------------------------------
    kernel_small = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)

    # --------------------------------------------------
    # 9. Connected components with dynamic thresholds
    # --------------------------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    img_area = binary.shape[0] * binary.shape[1]
    min_area = max(120, img_area // 6000)
    max_dim = max(100, min(binary.shape[:2]) // 2)

    chars_this_image = []

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue
        if w < 10 or h < 10:
            continue
        if w > max_dim or h > max_dim:
            continue

        aspect_ratio = w / float(h)
        if aspect_ratio > 6 or aspect_ratio < 0.1:
            continue

        char_img = binary[y:y+h, x:x+w]
        if not is_valid_character(char_img, min_ratio=0.08):
            continue

        chars_this_image.append((x, y, w, h, char_img))

    # Sort left to right (reading order)
    chars_this_image.sort(key=lambda c: c[0])

    for (x, y, w, h, char_img) in chars_this_image:
        size = max(w, h)
        padded = np.zeros((size, size), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = char_img

        char_path = os.path.join(output_folder, f"char_{char_id:06d}.png")
        cv2.imwrite(char_path, padded)
        char_id += 1

print("Segmentation completed!")
print("Characters extracted:", char_id)