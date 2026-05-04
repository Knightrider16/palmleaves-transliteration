"""
Script 2 (Improved) — Character Clustering with Better Noise Filtering
======================================================================
Improvements over original:
  - Stricter pre-filtering: punch holes, line noise, specks, low-fill
  - Solidity filter before feature extraction
  - Tighter HDBSCAN parameters for cleaner clusters
  - Reads from data/characters_named/ (Script A output)

Usage:
    python script_2_clustering.py

Inputs:
    data/characters_named/   — from Script A

Outputs:
    data/clusters/
    cluster_plot.png
    cluster_assignments.csv  — filename → cluster label (for Script B)
"""

import os
import shutil
import numpy as np
import cv2
import umap
import hdbscan
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.morphology import skeletonize

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
IMG_FOLDER     = "data/characters_named"
CLUSTER_FOLDER = "data/clusters"
os.makedirs(CLUSTER_FOLDER, exist_ok=True)
os.makedirs("rejects/punchholes", exist_ok=True)
os.makedirs("rejects/linenoise",  exist_ok=True)
os.makedirs("rejects/specks",     exist_ok=True)


# ─────────────────────────────────────────────────────────────
# PRE-FILTERING  (runs before feature extraction)
# Three independent filters, each with its own rejection folder
# ─────────────────────────────────────────────────────────────

def load_binary(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img.shape[0], img.shape[1], binary


def is_punch_hole(h, w, binary, circularity_thresh=0.70):
    """
    Punch holes: large + circular + hollow interior.
    All three conditions must be true.
    """
    if h * w < 1000:
        return False

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    cnt       = max(contours, key=cv2.contourArea)
    area      = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return False

    circularity = (4 * np.pi * area) / (perimeter ** 2)
    if circularity < circularity_thresh:
        return False

    # Hollow check
    c_all, hier = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return False
    for i, he in enumerate(hier[0]):
        if he[3] >= 0:
            ha = cv2.contourArea(c_all[i])
            pa = cv2.contourArea(c_all[he[3]])
            if pa > 0 and ha / pa > 0.25:
                return True
    return False


def is_line_noise(h, w, aspect_thresh=4.5):
    """Extreme aspect ratio = horizontal/vertical line fragment."""
    if h == 0 or w == 0:
        return False
    ratio = w / float(h)
    return ratio > aspect_thresh or ratio < (1.0 / aspect_thresh)


def is_speck(h, w, binary, max_area=100):
    """Tiny blobs — too small to be a real character."""
    return (h * w) < max_area


def has_low_fill(binary, h, w, min_fill=0.07):
    """Nearly-empty bounding box — ghost blob."""
    fg = np.sum(binary == 255)
    return (fg / float(h * w)) < min_fill if h * w > 0 else True


def has_low_solidity(binary, min_solidity=0.15):
    """Broken/fragmentary stroke."""
    cnts, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return True
    cnt  = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    hull = cv2.contourArea(cv2.convexHull(cnt))
    if hull == 0:
        return True
    return (area / hull) < min_solidity


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# HOG + Zernike moments + skeleton features + density grid
# (same as original, kept intact)
# ─────────────────────────────────────────────────────────────

def compute_zernike_moments(binary, radius=32, degree=8):
    size  = radius * 2
    img   = cv2.resize(binary, (size, size))
    y_idx, x_idx = np.mgrid[:size, :size]
    cx, cy = size // 2, size // 2
    x_n   = (x_idx - cx) / float(radius)
    y_n   = (y_idx - cy) / float(radius)
    rho   = np.sqrt(x_n**2 + y_n**2)
    theta = np.arctan2(y_n, x_n)
    mask  = rho <= 1.0
    img_norm = img.astype(float) / 255.0

    features = []
    for n in range(degree + 1):
        for m in range(-n, n + 1, 2):
            R = np.zeros_like(rho)
            for s in range((n - abs(m)) // 2 + 1):
                import math
                coeff = ((-1)**s * math.factorial(n - s)) / (
                    math.factorial(s) *
                    math.factorial((n + abs(m)) // 2 - s) *
                    math.factorial((n - abs(m)) // 2 - s))
                R += coeff * (rho ** (n - 2 * s))
            Z_real = R * np.cos(m * theta)
            Z_imag = R * np.sin(m * theta)
            A_real = np.sum(img_norm * Z_real * mask)
            A_imag = np.sum(img_norm * Z_imag * mask)
            features.append(np.sqrt(A_real**2 + A_imag**2))
    return np.array(features)


def compute_skeleton_features(binary):
    skel_input = (binary > 127).astype(np.uint8)
    skeleton   = skeletonize(skel_input).astype(np.uint8)
    total_len  = np.sum(skeleton)

    endpoints  = 0
    junctions  = 0
    padded     = np.pad(skeleton, 1, mode='constant')
    ys, xs     = np.where(skeleton > 0)
    for y, x in zip(ys, xs):
        nb = padded[y:y+3, x:x+3].copy()
        nb[1, 1] = 0
        nc = np.sum(nb)
        if nc == 1:   endpoints += 1
        elif nc >= 3: junctions += 1

    h, w = skeleton.shape
    grid = []
    for r in range(4):
        for c in range(4):
            zone = skeleton[r*h//4:(r+1)*h//4, c*w//4:(c+1)*w//4]
            grid.append(np.sum(zone) / max(zone.size, 1))

    return np.array([
        total_len / max(h * w, 1),
        endpoints / max(total_len, 1),
        junctions / max(total_len, 1),
    ] + grid)


def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img    = cv2.resize(img, (64, 64))
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # HOG
    hog = cv2.HOGDescriptor(
        (64,64), (16,16), (8,8), (8,8), 9)
    hog_features = hog.compute(binary).flatten()

    # Zernike
    try:
        zernike = compute_zernike_moments(binary, radius=32, degree=8)
    except Exception:
        zernike = np.zeros(45)

    # Skeleton
    try:
        skel = compute_skeleton_features(binary)
    except Exception:
        skel = np.zeros(19)

    # Density grid 8x8
    density = []
    for r in range(8):
        for c in range(8):
            zone = binary[r*8:(r+1)*8, c*8:(c+1)*8]
            density.append(np.sum(zone > 0) / 64.0)
    density = np.array(density)

    # Fill + aspect
    fill   = np.sum(binary > 0) / (64 * 64)
    coords = cv2.findNonZero(binary)
    aspect = 1.0
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        aspect = w / float(h) if h > 0 else 1.0

    return np.concatenate(
        [hog_features, zernike, skel, density, [fill, aspect]])


# ─────────────────────────────────────────────────────────────
# PROCESS ALL IMAGES
# ─────────────────────────────────────────────────────────────

print("Pre-filtering and extracting features...")

embeddings   = []
image_paths  = []
skip_counts  = {"punch_hole": 0, "line_noise": 0,
                "speck": 0, "low_fill": 0, "low_solidity": 0}

all_files = sorted([
    f for f in os.listdir(IMG_FOLDER)
    if f.lower().endswith(('.png','.jpg','.jpeg'))
    and f != "character_index.csv"
])

for img_name in all_files:
    path = os.path.join(IMG_FOLDER, img_name)
    h, w, binary = load_binary(path)
    if binary is None:
        continue

    # ── Pre-filters ───────────────────────────────────────────
    if is_speck(h, w, binary):
        skip_counts["speck"] += 1
        shutil.copy(path, "rejects/specks")
        continue

    if is_line_noise(h, w):
        skip_counts["line_noise"] += 1
        shutil.copy(path, "rejects/linenoise")
        continue

    if is_punch_hole(h, w, binary):
        skip_counts["punch_hole"] += 1
        shutil.copy(path, "rejects/punchholes")
        continue

    if has_low_fill(binary, h, w):
        skip_counts["low_fill"] += 1
        continue

    if has_low_solidity(binary):
        skip_counts["low_solidity"] += 1
        continue

    features = extract_features(path)
    if features is None:
        continue

    embeddings.append(features)
    image_paths.append(path)

print(f"\nPre-filter rejections:")
for reason, count in skip_counts.items():
    print(f"  {reason:15s}: {count}")
print(f"\nCharacters kept for clustering: {len(embeddings)}")

embeddings = np.array(embeddings)
print(f"Feature shape: {embeddings.shape}")

# ─────────────────────────────────────────────────────────────
# PCA + UMAP + HDBSCAN
# ─────────────────────────────────────────────────────────────

print("\nNormalizing + PCA...")
scaler          = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

pca             = PCA(n_components=0.95, svd_solver='full')
embeddings_pca  = pca.fit_transform(embeddings_scaled)
print(f"PCA: {embeddings.shape[1]} → {embeddings_pca.shape[1]} components")

print("Running UMAP (2D for visualization)...")
embedding_2d = umap.UMAP(
    n_neighbors=10, min_dist=0.05,
    metric='euclidean', n_components=2, random_state=42
).fit_transform(embeddings_pca)

print("Running UMAP (15D for clustering)...")
embedding_nd = umap.UMAP(
    n_neighbors=10, min_dist=0.0,
    metric='euclidean', n_components=15, random_state=42
).fit_transform(embeddings_pca)

print("Clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,       # raised from 4 — fewer tiny noisy clusters
    min_samples=3,            # raised from 2 — more conservative
    cluster_selection_epsilon=0.15,
    cluster_selection_method='leaf',
    metric='euclidean'
)
labels = clusterer.fit_predict(embedding_nd)

num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"\nClusters found : {num_clusters}")
print(f"Noise points   : {np.sum(labels == -1)}")
print(f"Clustered      : {np.sum(labels >= 0)}")

# ─────────────────────────────────────────────────────────────
# VISUALIZE
# ─────────────────────────────────────────────────────────────

plt.figure(figsize=(14, 10))
scatter = plt.scatter(
    embedding_2d[:, 0], embedding_2d[:, 1],
    c=labels, cmap='Spectral', s=6, alpha=0.8)
plt.colorbar(scatter, label='Cluster ID')
plt.title(f"Character Clusters  "
          f"(clusters={num_clusters}, noise={np.sum(labels==-1)})")
plt.tight_layout()
plt.savefig("cluster_plot.png", dpi=150)
print("Saved: cluster_plot.png")

# ─────────────────────────────────────────────────────────────
# SAVE CLUSTERS + CONTACT SHEETS + ASSIGNMENT CSV
# ─────────────────────────────────────────────────────────────

print("Saving cluster folders...")
for label in set(labels):
    folder = os.path.join(
        CLUSTER_FOLDER,
        "noise" if label == -1 else f"cluster_{label:03d}")
    os.makedirs(folder, exist_ok=True)

for i, label in enumerate(labels):
    folder = os.path.join(
        CLUSTER_FOLDER,
        "noise" if label == -1 else f"cluster_{label:03d}")
    shutil.copy(image_paths[i], folder)


def make_contact_sheet(paths, out_path, thumb=48, cols=20):
    if not paths:
        return
    rows  = (len(paths) + cols - 1) // cols
    sheet = np.zeros((rows * thumb, cols * thumb), dtype=np.uint8)
    for idx, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (thumb, thumb))
        r, c = idx // cols, idx % cols
        sheet[r*thumb:(r+1)*thumb, c*thumb:(c+1)*thumb] = img
    cv2.imwrite(out_path, sheet)


print("Generating contact sheets...")
for label in sorted(set(labels)):
    indices = [i for i, l in enumerate(labels) if l == label]
    paths   = [image_paths[i] for i in indices]
    name    = ("noise_sheet.png" if label == -1
                else f"cluster_{label:03d}_sheet.png")
    make_contact_sheet(paths, os.path.join(CLUSTER_FOLDER, name))

# Save cluster assignments CSV — needed by Script B
assign_path = "cluster_assignments.csv"
with open(assign_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "cluster"])
    writer.writeheader()
    for i, label in enumerate(labels):
        cluster_name = ("noise" if label == -1
                        else f"cluster_{label:03d}")
        writer.writerow({
            "filename": os.path.basename(image_paths[i]),
            "cluster":  cluster_name,
        })

print(f"Saved: {assign_path}")
print("\nDone. Check cluster_plot.png and data/clusters/")