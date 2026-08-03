# =============================================================
# PALM LEAF TRANSLITERATION PIPELINE
# For ancient scripts: Vattezhuthu, Malayanma etc.
# =============================================================

import os
import cv2
import csv
import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# =============================================================
# CONFIG — edit these paths
# =============================================================

RAW_IMAGE_DIR   = "data/masks_clean_upscaled"       # your leaf scans
SEGMENTS_DIR    = "data2/segments"         # segmented chars (per image/line)
LABELED_DIR     = "data2/labeled"          # training dataset
MODEL_PATH      = "models/palmleaf_cnn.pth"
LABEL_MAP_PATH  = "models/label_map.json"
TRANSCRIPT_CSV  = "data/labels/labels.csv"
RESULTS_DIR     = "results"

# NOTE: this script is the original per-character segmentation + CNN
# pipeline.  It is preserved for reference but the active pipeline is
# the CRNN/CTC line recognizer in crnn/.  Run crnn/train.py and
# crnn/infer.py instead.

for d in [SEGMENTS_DIR, LABELED_DIR, RESULTS_DIR,
          "models", "rejects/punchholes", "rejects/linenoise"]:
    os.makedirs(d, exist_ok=True)

# =============================================================
# DIGRAPHS — your romanization scheme
# Edit/extend based on what you find in your manuscripts
# =============================================================

DIGRAPHS = [
    'th', 'zh', 'ch', 'sh', 'ng', 'hh', 'nh',
    'aa', 'ee', 'oo', 'ai', 'au', 'rr', 'll',
    'tt', 'nn', 'kk', 'pp', 'mm', 'nj', 'nt',
    'nd', 'mb', 'rt', 'rd', 'lt', 'ld', 'nj'
]
# Sort longest first so tokenizer matches greedily
DIGRAPHS = sorted(set(DIGRAPHS), key=len, reverse=True)

# =============================================================
# PART 1: TRANSCRIPT UTILITIES
# =============================================================

def tokenize(text):
    """
    Split a transcript into akshara tokens.

    The CSV uses '/' as the canonical token separator (e.g. "ka/li/la").
    '[unk]' / '?' are normalized to '?' for downstream code that expects
    a single unknown sentinel.
    """
    if not text or not isinstance(text, str):
        return []
    tokens = []
    for raw in text.split('/'):
        t = raw.strip().lower()
        if not t:
            continue
        if t in ('[unk]', '?'):
            tokens.append('?')
        else:
            tokens.append(t)
    return tokens


def load_transcripts(csv_path):
    """
    Load CSV into a dict:
    { image_id: { line_num: transcript_string } }
    """
    df = pd.read_csv(csv_path)
    df['transcript'] = df['transcript'].fillna('').str.strip()
    df['image']      = df['image'].str.strip()
    df['line']       = df['line'].astype(int)

    transcripts = defaultdict(dict)
    for _, row in df.iterrows():
        transcripts[row['image']][row['line']] = row['transcript']
    return transcripts


def analyze_transcripts(csv_path):
    """
    Print a full analysis of your transcript CSV:
    - character inventory
    - frequency table
    - uncertainty rate
    - lines that need revisiting
    """
    df = pd.read_csv(csv_path)
    df['transcript'] = df['transcript'].fillna('').str.strip()

    all_tokens = []
    for t in df['transcript']:
        all_tokens.extend(tokenize(t))

    counter    = Counter(all_tokens)
    known      = {k: v for k, v in counter.items() if k != '?'}
    unk_count  = counter.get('?', 0)
    total      = sum(counter.values())

    print("=" * 50)
    print("TRANSCRIPT ANALYSIS")
    print("=" * 50)
    print(f"Total tokens        : {total}")
    print(f"Unique units        : {len(known)}")
    print(f"Uncertain (?)       : {unk_count}  "
          f"({unk_count/total*100:.1f}%)")
    print()
    print("Frequency table (sorted):")
    print(f"  {'token':8s} {'count':6s}  {'bar'}")
    for token, count in sorted(known.items(), key=lambda x: -x[1]):
        bar = '█' * min(count // 3, 40)
        print(f"  {token:8s} {count:6d}  {bar}")

    # Lines needing review (contain ?)
    needs_review = []
    for _, row in df.iterrows():
        toks = tokenize(row['transcript'])
        n_unk = toks.count('?')
        if n_unk > 0:
            needs_review.append({
                'image'  : row['image'],
                'line'   : row['line'],
                'n_unk'  : n_unk,
                'pct'    : n_unk / max(len(toks), 1) * 100,
                'text'   : row['transcript']
            })

    if needs_review:
        print(f"\n{len(needs_review)} lines with '?' — save to "
              f"review_needed.csv")
        pd.DataFrame(needs_review).sort_values(
            'pct', ascending=False
        ).to_csv("review_needed.csv", index=False)

    return known


# =============================================================
# PART 2: SEGMENTATION
# =============================================================

def sauvola_threshold(gray, window_size=25, k=0.5, R=128):
    gray_f   = gray.astype(np.float64)
    mean     = cv2.boxFilter(gray_f, -1, (window_size, window_size))
    mean_sq  = cv2.boxFilter(gray_f**2, -1, (window_size, window_size))
    std      = np.sqrt(np.abs(mean_sq - mean**2))
    thresh   = mean * (1 + k * (std / R - 1))
    binary   = (gray_f < thresh).astype(np.uint8) * 255
    return binary


def remove_punch_holes(binary, num_holes=2):
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return binary

    contours_ext = [(i, c) for i, c in enumerate(contours)
                    if hierarchy[0][i][3] == -1]
    contours_ext.sort(key=lambda x: cv2.contourArea(x[1]), reverse=True)

    mask        = np.zeros_like(binary)
    holes_found = 0

    for idx, cnt in contours_ext:
        area = cv2.contourArea(cnt)
        if area < 200:
            break
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 0

        # Circularity
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # Hollow interior check
        has_hole = False
        for j, child_cnt in enumerate(contours):
            if hierarchy[0][j][3] == idx:
                child_area = cv2.contourArea(child_cnt)
                if child_area / max(area, 1) > 0.30:
                    has_hole = True
                    break

        if circularity > 0.70 and has_hole:
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            holes_found += 1
            if holes_found >= num_holes:
                break

    return cv2.bitwise_xor(binary, mask)


def segment_image(image_path, out_dir, image_id):
    """
    Segment a single leaf image into individual character images.
    Saves chars grouped by line:
        out_dir/image_id/line_1/char_00001.png
        out_dir/image_id/line_1/char_00002.png
        out_dir/image_id/line_2/char_00001.png
        ...
    Returns:
        line_segments: { line_num: [(x,y,w,h,char_img), ...] }
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read: {image_path}")
        return {}

    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Inversion check
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(
        gray, h=10, templateWindowSize=7, searchWindowSize=21)
    denoised = cv2.medianBlur(denoised, 3)

    # Threshold
    binary = sauvola_threshold(denoised, window_size=25, k=0.5)

    # Punch hole removal
    binary = remove_punch_holes(binary)

    # Remove horizontal lines
    h_len  = max(60, w_img // 15)
    ker_h  = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    hlines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ker_h)
    binary = cv2.subtract(binary, hlines)

    # Light close
    ker_s  = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ker_s)

    # Connected components
    num_labels, labels, stats, _ = \
        cv2.connectedComponentsWithStats(binary)

    img_area = h_img * w_img
    min_area = max(120, img_area // 6000)
    max_dim  = max(100, min(h_img, w_img) // 2)

    blobs = []
    for i in range(1, num_labels):
        x    = stats[i, cv2.CC_STAT_LEFT]
        y    = stats[i, cv2.CC_STAT_TOP]
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:          continue
        if w < 10 or h < 10:        continue
        if w > max_dim or h > max_dim: continue
        ar = w / float(h)
        if ar > 6 or ar < 0.1:      continue

        char_img = binary[y:y+h, x:x+w]
        blobs.append((x, y, w, h, char_img))

    # ---- Group into lines by y-coordinate ----
    line_segments = group_into_lines(blobs, y_tolerance=h_img // 20)

    # ---- Save ----
    img_out = os.path.join(out_dir, image_id)
    for line_num, chars in line_segments.items():
        line_dir = os.path.join(img_out, f"line_{line_num}")
        os.makedirs(line_dir, exist_ok=True)
        for c_idx, (x, y, w, h, char_img) in enumerate(chars):
            size    = max(w, h)
            padded  = np.zeros((size, size), dtype=np.uint8)
            y_off   = (size - h) // 2
            x_off   = (size - w) // 2
            padded[y_off:y_off+h, x_off:x_off+w] = char_img
            cv2.imwrite(
                os.path.join(line_dir, f"char_{c_idx:05d}.png"),
                padded)

    return line_segments


def group_into_lines(blobs, y_tolerance=20):
    """
    Group character blobs into text lines.
    Returns { line_num(1-indexed): [(x,y,w,h,img),...] }
    sorted left-to-right within each line.
    """
    if not blobs:
        return {}

    blobs_sorted = sorted(blobs, key=lambda b: b[1])  # sort by y
    lines        = []
    current_line = [blobs_sorted[0]]

    for blob in blobs_sorted[1:]:
        avg_y = np.mean([b[1] + b[3]//2 for b in current_line])
        if abs((blob[1] + blob[3]//2) - avg_y) <= y_tolerance:
            current_line.append(blob)
        else:
            lines.append(current_line)
            current_line = [blob]
    lines.append(current_line)

    # Sort each line left to right
    result = {}
    for i, line in enumerate(lines):
        result[i + 1] = sorted(line, key=lambda b: b[0])

    return result


def segment_all_images(image_dir, segments_dir):
    """Run segmentation on all images in image_dir."""
    images = [f for f in os.listdir(image_dir)
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    print(f"Segmenting {len(images)} images...")
    for fname in sorted(images):
        image_id = os.path.splitext(fname)[0]
        image_path = os.path.join(image_dir, fname)
        print(f"  {image_id}...", end=' ')
        segs = segment_image(image_path,
                              segments_dir, image_id)
        total = sum(len(v) for v in segs.values())
        print(f"{len(segs)} lines, {total} chars")


# =============================================================
# PART 3: BUILD LABELED DATASET FROM TRANSCRIPTS
# =============================================================

def build_labeled_dataset(transcripts, segments_dir, labeled_dir,
                           alignment_tolerance=999):
    """
    For each image+line in transcripts:
      - Load segmented chars
      - Tokenize transcript
      - Align segments to tokens (skip '?')
      - Save char image into labeled_dir/<token>/
    
    Note: Uses best-effort alignment. If segments and tokens don't match,
    still uses available segments paired with transcript tokens.
    """
    label_counts = Counter()
    skipped_miss = 0
    used_lines   = 0

    for image_id, lines in transcripts.items():
        for line_num, transcript in lines.items():
            if not transcript:
                continue

            tokens = tokenize(transcript)

            # Load segmented chars for this line
            line_dir = os.path.join(
                segments_dir, image_id, f"line_{line_num}")
            if not os.path.exists(line_dir):
                skipped_miss += 1
                continue

            seg_files = sorted([
                f for f in os.listdir(line_dir)
                if f.endswith('.png')])

            if len(seg_files) == 0:
                continue
            
            # Use best-effort alignment: pair segments with tokens
            used_lines += 1
            for seg_idx, token in enumerate(tokens):
                if token == '?':
                    continue                    # skip unknowns
                if seg_idx >= len(seg_files):
                    break

                img_path = os.path.join(line_dir, seg_files[seg_idx])
                img      = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # Save into label folder
                label_folder = os.path.join(labeled_dir, token)
                os.makedirs(label_folder, exist_ok=True)
                count    = label_counts[token]
                out_path = os.path.join(
                    label_folder, f"{token}_{count:05d}.png")
                cv2.imwrite(out_path, img)
                label_counts[token] += 1

    print("\n=== DATASET BUILT ===")
    print(f"Lines used          : {used_lines}")
    print(f"Aligned samples     : {sum(label_counts.values())}")
    print(f"Skipped (not found) : {skipped_miss}")
    print(f"Unique classes      : {len(label_counts)}")
    
    if label_counts:
        print("\nSamples per class:")
        for label, count in sorted(label_counts.items(),
                                    key=lambda x: -x[1]):
            bar = '█' * min(count // 2, 30)
            print(f"  {label:8s} {count:5d}  {bar}")
    else:
        print("⚠️ WARNING: No training samples created!")

    return dict(label_counts)


# =============================================================
# PART 4: CNN MODEL + DATASET
# =============================================================

class GlyphDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class PalmLeafCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),              # 32x32

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),              # 16x16

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),              # 8x8

            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), # 4x4 regardless of input
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_dataset_splits(labeled_dir, test_size=0.15, min_samples=1):
    """
    Robust version:
    - Works with very small datasets
    - Avoids stratify crashes
    - Handles empty dataset safely
    """

    all_paths  = []
    all_labels = []
    label_map  = {}

    classes = sorted(os.listdir(labeled_dir))
    for cls in classes:
        cls_dir = os.path.join(labeled_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        files = [os.path.join(cls_dir, f)
                 for f in os.listdir(cls_dir)
                 if f.endswith('.png')]

        if len(files) < min_samples:
            print(f"Skipping '{cls}' — only {len(files)} samples")
            continue

        if cls not in label_map:
            label_map[cls] = len(label_map)

        idx = label_map[cls]
        all_paths.extend(files)
        all_labels.extend([idx] * len(files))

    print("\n=== DATASET DEBUG ===")
    print("Total samples:", len(all_paths))
    print("Total classes:", len(label_map))

    # 🚨 Safety check
    if len(all_paths) == 0:
        raise ValueError("❌ No training data found. Check dataset creation step.")

    # Filter out classes with only 1 sample (stratification requires min 2)
    label_counts = Counter(all_labels)
    valid_indices = [i for i, lbl in enumerate(all_labels) if label_counts[lbl] >= 2]
    
    if len(valid_indices) < len(all_paths):
        removed = len(all_paths) - len(valid_indices)
        print(f"⚠️ Removing {removed} samples from classes with only 1 member")
        all_paths = [all_paths[i] for i in valid_indices]
        all_labels = [all_labels[i] for i in valid_indices]
        label_counts = Counter(all_labels)
        print(f"Dataset after filtering: {len(all_paths)} samples, {len(set(all_labels))} classes")

    # 🚨 SMALL DATASET HANDLING
    if len(set(all_labels)) < 2 or len(all_paths) < 10:
        print("⚠️ Small dataset detected — using simple split")

        if len(all_paths) == 0:
            raise ValueError("❌ No training data found. Check dataset creation step.")

        split_idx = max(1, int(0.8 * len(all_paths)))

        paths_train  = all_paths[:split_idx]
        labels_train = all_labels[:split_idx]

        paths_val  = all_paths[split_idx:]
        labels_val = all_labels[split_idx:]

        # fallback
        if len(paths_val) == 0:
            paths_val = paths_train
            labels_val = labels_train

        paths_test  = paths_val
        labels_test = labels_val

    else:
        # Normal case
        if len(all_paths) == 0:
            raise ValueError("❌ No training data found. Check dataset creation step.")
        
        # Check if stratification is safe
        label_counts = Counter(all_labels)
        num_classes = len(label_counts)
        min_class_count = min(label_counts.values()) if label_counts else 0
        test_samples = max(1, int(test_size * len(all_paths)))
        
        # Stratification requires: min_class >= 2 AND test_size >= num_classes
        use_stratify = (min_class_count >= 2 and test_samples >= num_classes)
        
        if not use_stratify:
            if min_class_count < 2:
                print(f"⚠️ Stratification disabled: some classes have <2 samples (min={min_class_count})")
            if test_samples < num_classes:
                print(f"⚠️ Stratification disabled: test_size ({test_samples}) < num_classes ({num_classes})")
        
        paths_train, paths_test, labels_train, labels_test = \
            train_test_split(all_paths, all_labels,
                             test_size=test_size,
                             stratify=all_labels if use_stratify else None,
                             random_state=42)

        # For val split, check min samples in training set
        min_val_samples = max(1, int(0.1 * len(labels_train)))
        use_stratify_val = (len(set(labels_train)) >= 2 and min_val_samples >= len(set(labels_train)))
        
        paths_train, paths_val, labels_train, labels_val = \
            train_test_split(paths_train, labels_train,
                             test_size=0.1,
                             stratify=labels_train if use_stratify_val else None,
                             random_state=42)

    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(8),
        transforms.RandomAffine(0, translate=(0.05, 0.05), shear=5),
        transforms.ToTensor(),
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_ds = GlyphDataset(paths_train, labels_train, transform_train)
    val_ds   = GlyphDataset(paths_val,   labels_val,   transform_eval)
    test_ds  = GlyphDataset(paths_test,  labels_test,  transform_eval)

    print(f"\nDataset splits:")
    print(f"  Train : {len(train_ds)}")
    print(f"  Val   : {len(val_ds)}")
    print(f"  Test  : {len(test_ds)}")
    print(f"  Classes: {len(label_map)}")

    return train_ds, val_ds, test_ds, label_map


def train_model(labeled_dir, model_path, label_map_path,
                epochs=40, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if not os.listdir(labeled_dir):
        raise ValueError("❌ LABELED_DIR is empty. Dataset creation failed.")

    train_ds, val_ds, test_ds, label_map = \
        build_dataset_splits(labeled_dir)

    # Adjust batch size for small datasets
    actual_batch_size = min(batch_size, max(1, len(train_ds) // 2))
    if actual_batch_size < batch_size:
        print(f"⚠️ Adjusted batch_size from {batch_size} to {actual_batch_size} (dataset size: {len(train_ds)})")

    train_loader = DataLoader(train_ds, batch_size=actual_batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=actual_batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=actual_batch_size,
                              shuffle=False, num_workers=0)

    num_classes = len(label_map)
    model       = PalmLeafCNN(num_classes).to(device)
    optimizer   = optim.Adam(model.parameters(), lr=lr,
                              weight_decay=1e-4)
    scheduler   = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True)
    criterion   = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0
        train_correct = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item()
            train_correct += (out.argmax(1) == labels).sum().item()

        # ---- Validate ----
        model.eval()
        val_correct = 0
        val_loss    = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out       = model(imgs)
                val_loss += criterion(out, labels).item()
                val_correct += (out.argmax(1) == labels).sum().item()

        train_acc = train_correct / max(len(train_ds), 1) * 100
        val_acc   = val_correct   / max(len(val_ds), 1)   * 100
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{epochs}  "
              f"train_acc={train_acc:.1f}%  "
              f"val_acc={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved best model (val_acc={val_acc:.1f}%)")

    # ---- Test ----
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            test_correct += (out.argmax(1) == labels).sum().item()
    test_acc = test_correct / max(len(test_ds), 1) * 100
    print(f"\nFinal test accuracy: {test_acc:.1f}%")

    # Save label map
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"Label map saved: {label_map_path}")

    return model, label_map


# =============================================================
# PART 5: INFERENCE
# =============================================================

def load_model(model_path, label_map_path):
    with open(label_map_path) as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}
    model = PalmLeafCNN(len(label_map))
    model.load_state_dict(torch.load(model_path,
                                      map_location='cpu',
                                      weights_only=False))
    model.eval()
    return model, idx_to_label


def classify_char(img, model, idx_to_label, transform):
    """
    Classify a single character image.
    Returns (predicted_label, confidence_score).
    """
    pil = Image.fromarray(img).convert('L')
    tensor = transform(pil).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)
        conf, pred = probs.max(1)
    return idx_to_label[pred.item()], conf.item()


def transliterate_image(image_path, model, idx_to_label,
                         segments_dir=None, image_id=None,
                         confidence_threshold=0.5):
    """
    Full pipeline: image → transliterated text.
    
    If segments_dir and image_id are given, uses pre-saved segments.
    Otherwise re-segments the image on the fly.
    
    Returns list of lines, each line a list of
    (token, confidence) tuples.
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Load or recompute segments
    if segments_dir and image_id:
        img_seg_dir = os.path.join(segments_dir, image_id)
        if not os.path.exists(img_seg_dir):
            return []
        line_dirs   = sorted([
            d for d in os.listdir(img_seg_dir)
            if d.startswith('line_')],
            key=lambda x: int(x.split('_')[1]))
        
        line_segments = {}
        for ld in line_dirs:
            ln  = int(ld.split('_')[1])
            ldir = os.path.join(img_seg_dir, ld)
            files = sorted([f for f in os.listdir(ldir)
                             if f.endswith('.png')])
            segs = []
            for f in files:
                img = cv2.imread(os.path.join(ldir, f),
                                  cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    segs.append((0, 0, img.shape[1],
                                 img.shape[0], img))
            line_segments[ln] = segs
    else:
        line_segments = segment_image(
            image_path, "/tmp/seg_temp",
            os.path.splitext(os.path.basename(image_path))[0])

    results = []
    for line_num in sorted(line_segments.keys()):
        line_result = []
        for (x, y, w, h, char_img) in line_segments[line_num]:
            label, conf = classify_char(
                char_img, model, idx_to_label, transform)
            if conf < confidence_threshold:
                label = '?'   # low confidence → mark unknown
            line_result.append((label, conf))
        results.append(line_result)

    return results


def results_to_text(results):
    """Convert results list to plain text string."""
    lines = []
    for line in results:
        lines.append(''.join(token for token, conf in line))
    return '\n'.join(lines)


# =============================================================
# PART 6: EVALUATION
# =============================================================

def evaluate(transcripts, results_by_image):
    """
    Compare pipeline output against ground truth CSV.
    Skips '?' positions in ground truth.
    Reports per-character accuracy and per-image accuracy.
    """
    total_correct = 0
    total_known   = 0
    per_image     = {}

    for image_id, lines in transcripts.items():
        if image_id not in results_by_image:
            continue

        img_correct = 0
        img_known   = 0

        for line_num, transcript in lines.items():
            if line_num not in results_by_image[image_id]:
                continue

            gt_tokens   = tokenize(transcript)
            pred_tokens = [t for t, c in
                           results_by_image[image_id][line_num]]

            for i, gt in enumerate(gt_tokens):
                if gt == '?':
                    continue
                img_known   += 1
                total_known += 1
                if i < len(pred_tokens) and pred_tokens[i] == gt:
                    img_correct   += 1
                    total_correct += 1

        if img_known > 0:
            per_image[image_id] = img_correct / img_known * 100

    overall = total_correct / max(total_known, 1) * 100

    print("\n=== EVALUATION ===")
    print(f"Overall accuracy : {overall:.1f}%  "
          f"({total_correct}/{total_known})")
    print("\nPer-image accuracy:")
    for img_id, acc in sorted(per_image.items(),
                               key=lambda x: -x[1]):
        bar = '█' * int(acc // 5)
        print(f"  {img_id:20s} {acc:5.1f}%  {bar}")

    return overall, per_image


def save_results(results_by_image, transcripts, out_dir):
    """
    Save side-by-side comparison of prediction vs ground truth
    for each image.
    """
    os.makedirs(out_dir, exist_ok=True)
    for image_id, lines in results_by_image.items():
        out_path = os.path.join(out_dir, f"{image_id}_result.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"IMAGE: {image_id}\n")
            f.write("=" * 60 + "\n")
            for line_num, preds in sorted(lines.items()):
                pred_str = ''.join(t for t, c in preds)
                gt_str   = transcripts.get(
                    image_id, {}).get(line_num, "N/A")
                f.write(f"\nLine {line_num}:\n")
                f.write(f"  PRED: {pred_str}\n")
                f.write(f"  GT  : {gt_str}\n")

                # Mark mismatches
                gt_toks   = tokenize(gt_str) if gt_str != "N/A" else []
                pred_toks = [t for t, c in preds]
                diff = []
                for i, (p, g) in enumerate(
                        zip(pred_toks, gt_toks)):
                    if g == '?':
                        diff.append('_')
                    elif p == g:
                        diff.append('✓')
                    else:
                        diff.append('✗')
                f.write(f"  CHK : {''.join(diff)}\n")
        print(f"Saved: {out_path}")


def create_synthetic_dataset(labeled_dir, num_samples=20):
    """
    Create synthetic training data for demo purposes.
    Generates random character images for common tokens.
    """
    print("\n🔧 Creating synthetic dataset...")
    synthetic_tokens = list('aeiountkpmshjy')  # Common phonemes
    
    os.makedirs(labeled_dir, exist_ok=True)
    
    for token in synthetic_tokens:
        label_dir = os.path.join(labeled_dir, token)
        os.makedirs(label_dir, exist_ok=True)
        
        for i in range(num_samples):
            # Create random character image
            img = np.random.randint(200, 256, (64, 64), dtype=np.uint8)
            # Add some pattern to make it look more realistic
            y, x = np.ogrid[:64, :64]
            mask = ((x - 32)**2 + (y - 32)**2) <= np.random.randint(10, 25)
            img[mask] = np.random.randint(50, 150)
            
            out_path = os.path.join(label_dir, f"{token}_{i:05d}.png")
            cv2.imwrite(out_path, img)
    
    print(f"✓ Created {num_samples} samples × {len(synthetic_tokens)} tokens = {num_samples * len(synthetic_tokens)} images")


# =============================================================
# MAIN — run the full pipeline
# =============================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("STEP 0: Analyze transcripts")
    print("="*60)
    analyze_transcripts(TRANSCRIPT_CSV)

    print("\n" + "="*60)
    print("STEP 1: Segment all images")
    print("="*60)
    segment_all_images(RAW_IMAGE_DIR, SEGMENTS_DIR)

    print("\n" + "="*60)
    print("STEP 2: Build labeled dataset from transcripts")
    print("="*60)
    transcripts = load_transcripts(TRANSCRIPT_CSV)
    build_labeled_dataset(transcripts, SEGMENTS_DIR, LABELED_DIR)
    
    # Check if dataset was created, if not create synthetic fallback
    print("\n🔍 Checking labeled dataset...")
    dataset_size = 0
    for root, dirs, files in os.walk(LABELED_DIR):
        num_files = len(files)
        if num_files > 0:
            print(f"{root} -> {num_files} files")
        dataset_size += num_files
    
    if dataset_size == 0:
        print("⚠️ No real labeled dataset created! Using synthetic data for demo...")
        create_synthetic_dataset(LABELED_DIR, num_samples=30)
    else:
        print(f"✓ Dataset ready: {dataset_size} images")

    print("\n" + "="*60)
    print("STEP 3: Train CNN")
    print("="*60)
    model, label_map = train_model(
        LABELED_DIR, MODEL_PATH, LABEL_MAP_PATH,
        epochs=10, batch_size=8)

    print("\n" + "="*60)
    print("STEP 4: Run inference + evaluate")
    print("="*60)
    model, idx_to_label = load_model(MODEL_PATH, LABEL_MAP_PATH)

    results_by_image = {}
    for image_id, lines in transcripts.items():
        results_by_image[image_id] = {}
        for line_num in lines:
            r = transliterate_image(
                None, model, idx_to_label,
                segments_dir=SEGMENTS_DIR,
                image_id=image_id)
            if line_num <= len(r):
                results_by_image[image_id][line_num] = r[line_num-1]

    evaluate(transcripts, results_by_image)
    save_results(results_by_image, transcripts, RESULTS_DIR)

    print("\nDone. Check results/ folder.")