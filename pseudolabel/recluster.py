"""
Re-cluster the char crops using the CNN embeddings produced by
embed_chars.py.  Far purer clusters than the HOG+Zernike+skeleton
pipeline because the features were trained to discriminate Malayalam
glyphs.

Pipeline:
    1. L2-normalize the 512-D embeddings
    2. Project to 30-D via UMAP (preserves local + global structure)
    3. Cluster with HDBSCAN

Usage:
    python -m pseudolabel.recluster
    python -m pseudolabel.recluster --min-cluster-size 15

Outputs:
    cluster_assignments_v2.csv      (filename, cluster)
    data/embeddings/cluster_embedding_2d.npy   (for visualization)
    data/embeddings/cluster_plot_v2.png         (scatter coloured by cluster)
"""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path

import hdbscan
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.preprocessing import normalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb",   default="data/embeddings/char_embeddings.npy")
    ap.add_argument("--names", default="data/embeddings/char_filenames.txt")
    ap.add_argument("--out",   default="cluster_assignments_v2.csv")
    ap.add_argument("--min-cluster-size", type=int, default=8)
    ap.add_argument("--min-samples",      type=int, default=3)
    ap.add_argument("--umap-n",           type=int, default=30,
                    help="UMAP target dim for clustering")
    ap.add_argument("--method",           default="eom",
                    choices=["leaf", "eom"])
    ap.add_argument("--epsilon",          type=float, default=0.2)
    args = ap.parse_args()

    print(f"Loading embeddings from {args.emb}")
    X = np.load(args.emb)
    print(f"  shape: {X.shape}")
    with open(args.names, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    assert len(names) == X.shape[0], "filename / embedding count mismatch"

    print("L2-normalising features...")
    X = normalize(X, norm="l2")

    print(f"UMAP -> {args.umap_n} dims for clustering...")
    Xn = umap.UMAP(
        n_neighbors=15, min_dist=0.0,
        n_components=args.umap_n, metric="cosine", random_state=42
    ).fit_transform(X)

    print("UMAP -> 2 dims for visualization...")
    X2 = umap.UMAP(
        n_neighbors=15, min_dist=0.05,
        n_components=2, metric="cosine", random_state=42
    ).fit_transform(X)

    print(f"HDBSCAN (min_cluster_size={args.min_cluster_size}, "
          f"min_samples={args.min_samples})...")
    cl = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method=args.method,
        cluster_selection_epsilon=args.epsilon,
    )
    labels = cl.fit_predict(Xn)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    print(f"  Clusters: {n_clusters}")
    print(f"  Noise   : {n_noise} ({n_noise / len(labels) * 100:.1f}%)")

    print(f"Saving {args.out} ...")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "cluster"])
        w.writeheader()
        for name, lbl in zip(names, labels):
            cl_name = "noise" if lbl == -1 else f"cluster_{lbl:03d}"
            w.writerow({"filename": name, "cluster": cl_name})

    # Visualize
    out_dir = os.path.dirname(args.emb) or "."
    np.save(os.path.join(out_dir, "cluster_embedding_2d.npy"), X2)
    plt.figure(figsize=(12, 9))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="Spectral", s=4, alpha=0.6)
    plt.title(f"CNN-feature clusters: {n_clusters} clusters, "
              f"{n_noise} noise / {len(labels)}")
    plt.tight_layout()
    plt.savefig(os.path.join("results", "cluster_plot_v2.png"), dpi=150)
    print(f"Plot: results/cluster_plot_v2.png")


if __name__ == "__main__":
    main()
