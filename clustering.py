import os
import shutil
import numpy as np
from PIL import Image
import cv2
import umap
import hdbscan
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ----------------------------------------
# CONFIG
# ----------------------------------------
IMG_FOLDER = "data/characters"
CLUSTER_FOLDER = "data/clusters"
os.makedirs(CLUSTER_FOLDER, exist_ok=True)

# ----------------------------------------
# PUNCH HOLE DETECTION — improved
# Uses size + circularity + hollow interior
# ----------------------------------------

def get_image_stats(img_path):
    """Returns (h, w, binary) for reuse across filters."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img.shape[0], img.shape[1], binary


def is_punch_hole(img_path, circularity_thresh=0.75):
    """
    Punch holes are:
    1. Large relative to typical characters
    2. Highly circular
    3. Hollow — large inner contour (the hole itself)

    We check all three. Circular *letters* fail condition 3
    because they are strokes, not filled rings.
    """
    result = get_image_stats(img_path)
    if result is None:
        return False
    h, w, binary = result

    # Condition 1: Must be large enough to be a punch hole
    # Characters are typically small; punch holes span
    # a significant portion of the image height
    img_area = h * w
    if img_area < 1000:   # too small to be a punch hole
        return False

    # External contours
    contours_ext, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_ext:
        return False

    cnt = max(contours_ext, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        return False

    # Condition 2: Circularity check
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    if circularity < circularity_thresh:
        return False

    # Condition 3: Hollow interior check
    # A punch hole has a large inner void.
    # Use RETR_CCOMP to find holes inside the contour.
    contours_all, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return False

    # Find child contours (interior holes)
    # hierarchy[0][i] = [next, prev, child, parent]
    has_large_hole = False
    for i, h_entry in enumerate(hierarchy[0]):
        parent = h_entry[3]
        if parent >= 0:  # this contour is inside another
            hole_area = cv2.contourArea(contours_all[i])
            parent_area = cv2.contourArea(contours_all[parent])
            if parent_area > 0:
                # Hole takes up >30% of the parent = hollow
                if hole_area / parent_area > 0.30:
                    has_large_hole = True
                    break

    return has_large_hole


def is_line_noise(img_path, aspect_thresh=5.0):
    result = get_image_stats(img_path)
    if result is None:
        return False
    h, w, _ = result
    if h == 0 or w == 0:
        return False
    ratio = w / float(h)
    return ratio > aspect_thresh or ratio < (1.0 / aspect_thresh)


# ----------------------------------------
# FEATURE EXTRACTION — improved
# HOG + Zernike moments + skeleton features
# + spatial density grid
# ----------------------------------------

def compute_zernike_moments(binary, radius=32, degree=8):
    """
    Zernike moments capture topology and fine structure
    better than Hu moments. They are orthogonal and
    rotation-invariant — ideal for character discrimination.
    """
    # Resize to 2*radius square
    size = radius * 2
    img = cv2.resize(binary, (size, size))

    # Build coordinate grid centered at image center
    y_idx, x_idx = np.mgrid[:size, :size]
    cx, cy = size // 2, size // 2
    x_n = (x_idx - cx) / float(radius)
    y_n = (y_idx - cy) / float(radius)
    rho = np.sqrt(x_n**2 + y_n**2)
    theta = np.arctan2(y_n, x_n)

    # Only use pixels within unit circle
    mask = rho <= 1.0
    img_norm = img.astype(float) / 255.0

    features = []
    for n in range(degree + 1):
        for m in range(-n, n + 1, 2):
            # Zernike radial polynomial
            R = np.zeros_like(rho)
            for s in range((n - abs(m)) // 2 + 1):
                coeff = ((-1)**s * np.math.factorial(n - s)) / (
                    np.math.factorial(s) *
                    np.math.factorial((n + abs(m)) // 2 - s) *
                    np.math.factorial((n - abs(m)) // 2 - s)
                )
                R += coeff * (rho ** (n - 2 * s))

            # Zernike basis function
            Z_real = R * np.cos(m * theta)
            Z_imag = R * np.sin(m * theta)

            # Moment = integral over unit circle
            A_real = np.sum(img_norm * Z_real * mask)
            A_imag = np.sum(img_norm * Z_imag * mask)
            A_mag = np.sqrt(A_real**2 + A_imag**2)
            features.append(A_mag)

    return np.array(features)


def compute_skeleton_features(binary):
    """
    Skeletonize the character and extract:
    - Total skeleton length (stroke complexity)
    - Number of endpoints (stroke count proxy)
    - Number of junction points (topology)
    - Skeleton density in 4x4 grid zones

    These features directly capture stroke structure —
    what actually differentiates similar-looking characters.
    """
    # Skeletonize
    from skimage.morphology import skeletonize
    skel_input = (binary > 127).astype(np.uint8)
    skeleton = skeletonize(skel_input).astype(np.uint8)

    total_length = np.sum(skeleton)

    # Detect endpoints and junctions using hit-or-miss
    # Endpoint: pixel with exactly 1 neighbor
    # Junction: pixel with 3+ neighbors
    endpoints = 0
    junctions = 0

    padded = np.pad(skeleton, 1, mode='constant')
    ys, xs = np.where(skeleton > 0)
    for y, x in zip(ys, xs):
        neighborhood = padded[y:y+3, x:x+3].copy()
        neighborhood[1, 1] = 0
        neighbor_count = np.sum(neighborhood)
        if neighbor_count == 1:
            endpoints += 1
        elif neighbor_count >= 3:
            junctions += 1

    # Skeleton density grid (4x4 zones)
    h, w = skeleton.shape
    grid_features = []
    rows, cols = 4, 4
    for r in range(rows):
        for c in range(cols):
            y0 = r * h // rows
            y1 = (r + 1) * h // rows
            x0 = c * w // cols
            x1 = (c + 1) * w // cols
            zone = skeleton[y0:y1, x0:x1]
            grid_features.append(np.sum(zone) / max(zone.size, 1))

    return np.array([total_length / max(h * w, 1),
                     endpoints / max(total_length, 1),
                     junctions / max(total_length, 1)] + grid_features)


def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.resize(img, (64, 64))
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # HOG
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
                             cell_size, nbins)
    hog_features = hog.compute(binary).flatten()

    # Zernike moments (finer shape topology)
    try:
        zernike = compute_zernike_moments(binary, radius=32, degree=8)
    except Exception:
        zernike = np.zeros(45)

    # Skeleton features (stroke structure)
    try:
        skel = compute_skeleton_features(binary)
    except Exception:
        skel = np.zeros(19)

    # Spatial density grid (8x8)
    zone_size = 8
    density = []
    for r in range(8):
        for c in range(8):
            zone = binary[r*zone_size:(r+1)*zone_size,
                          c*zone_size:(c+1)*zone_size]
            density.append(np.sum(zone > 0) / (zone_size ** 2))
    density = np.array(density)

    # Fill + aspect
    fill = np.sum(binary > 0) / (64 * 64)
    coords = cv2.findNonZero(binary)
    aspect = 1.0
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        aspect = w / float(h) if h > 0 else 1.0

    meta = np.array([fill, aspect])

    return np.concatenate([hog_features, zernike, skel, density, meta])


# ----------------------------------------
# PROCESS ALL IMAGES
# ----------------------------------------
print("Extracting features...")

embeddings = []
image_paths = []
skipped_holes = 0
skipped_noise = 0

for img_name in sorted(os.listdir(IMG_FOLDER)):
    path = os.path.join(IMG_FOLDER, img_name)
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    if is_punch_hole(path):
        skipped_holes += 1
        os.makedirs("rejects/punchholes", exist_ok=True)
        shutil.copy(path, "rejects/punchholes")
        continue

    if is_line_noise(path):
        skipped_noise += 1
        os.makedirs("rejects/linenoise", exist_ok=True)
        shutil.copy(path, "rejects/linenoise")
        continue

    features = extract_features(path)
    if features is None:
        continue

    embeddings.append(features)
    image_paths.append(path)

print(f"Punch holes filtered: {skipped_holes}")
print(f"Line noise filtered:  {skipped_noise}")
print(f"Characters kept:      {len(embeddings)}")

embeddings = np.array(embeddings)
print("Feature shape:", embeddings.shape)

# ----------------------------------------
# NORMALIZE + PCA
# Reduce feature dims before UMAP —
# PCA removes redundant dimensions and
# speeds up UMAP significantly
# ----------------------------------------
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# Keep enough components to explain 95% variance
pca = PCA(n_components=0.95, svd_solver='full')
embeddings_pca = pca.fit_transform(embeddings_scaled)
print(f"PCA reduced to {embeddings_pca.shape[1]} components")

# ----------------------------------------
# UMAP
# ----------------------------------------
print("Running UMAP (2D for viz)...")
reducer_2d = umap.UMAP(
    n_neighbors=8,
    min_dist=0.05,
    metric='euclidean',
    n_components=2,
    random_state=42
)
embedding_2d = reducer_2d.fit_transform(embeddings_pca)

print("Running UMAP (15D for clustering)...")
reducer_nd = umap.UMAP(
    n_neighbors=8,
    min_dist=0.0,
    metric='euclidean',
    n_components=15,
    random_state=42
)
embedding_nd = reducer_nd.fit_transform(embeddings_pca)

# ----------------------------------------
# HDBSCAN
# ----------------------------------------
print("Clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=4,
    min_samples=2,
    cluster_selection_epsilon=0.2,
    cluster_selection_method='leaf',  # finer clusters than 'eom'
    metric='euclidean'
)
labels = clusterer.fit_predict(embedding_nd)

num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Clusters found: {num_clusters}")
print(f"Noise points:   {np.sum(labels == -1)}")
print(f"Clustered:      {np.sum(labels >= 0)}")

# ----------------------------------------
# VISUALIZE
# ----------------------------------------
plt.figure(figsize=(14, 10))
scatter = plt.scatter(
    embedding_2d[:, 0],
    embedding_2d[:, 1],
    c=labels,
    cmap='Spectral',
    s=6,
    alpha=0.8
)
plt.colorbar(scatter, label='Cluster ID')
plt.title(f"Character Clusters  (n_clusters={num_clusters}, "
          f"noise={np.sum(labels==-1)})")
plt.tight_layout()
plt.savefig("cluster_plot.png", dpi=150)
plt.show()

# ----------------------------------------
# SAVE CLUSTERS + CONTACT SHEETS
# ----------------------------------------
print("Saving clusters...")

for label in set(labels):
    folder = os.path.join(CLUSTER_FOLDER,
                          "noise" if label == -1 else f"cluster_{label:03d}")
    os.makedirs(folder, exist_ok=True)

for i, label in enumerate(labels):
    folder = os.path.join(CLUSTER_FOLDER,
                          "noise" if label == -1 else f"cluster_{label:03d}")
    shutil.copy(image_paths[i], folder)


def make_contact_sheet(image_list, out_path, thumb_size=48, cols=20):
    n = len(image_list)
    if n == 0:
        return
    rows = (n + cols - 1) // cols
    sheet = np.zeros((rows * thumb_size, cols * thumb_size), dtype=np.uint8)
    for idx, p in enumerate(image_list):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (thumb_size, thumb_size))
        r, c = idx // cols, idx % cols
        sheet[r*thumb_size:(r+1)*thumb_size,
              c*thumb_size:(c+1)*thumb_size] = img
    cv2.imwrite(out_path, sheet)


print("Generating contact sheets...")
for label in sorted(set(labels)):
    indices = [i for i, l in enumerate(labels) if l == label]
    paths = [image_paths[i] for i in indices]
    name = "noise_sheet.png" if label == -1 else f"cluster_{label:03d}_sheet.png"
    make_contact_sheet(paths, os.path.join(CLUSTER_FOLDER, name))

print("Done. Check cluster_plot.png and contact sheets.")