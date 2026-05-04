"""
Build pseudo-labels for every char crop.

Inputs:
  data/characters_named/character_index.csv  — char positions per image
  cluster_assignments_v2.csv                 — CNN-based cluster ids
  data/labels/labels.csv                     — gold transcripts

Algorithm:

  Per labeled image:
    1. Sort the chars left-to-right (by x)
    2. Concatenate all CSV tokens for the image in line order
    3. Map token i to char round(i × n_chars / n_tokens)
    4. Boundary tokens (first/last 2 of each line) get downgraded
       confidence — they're the most likely to misalign

  Across labeled chars:
    5. For each cluster, count how many times each token won a vote
       (only high/medium-confidence votes count)
    6. Cluster's pseudo-label = majority token, with confidence based
       on vote share

  Propagation:
    7. Every char in a labeled cluster inherits that cluster's label
    8. Chars in unlabeled or noise clusters are tagged [unk]

Outputs:
  data/pseudo_labeled_v2/character_labels.csv
  data/pseudo_labeled_v2/cluster_labels.csv
  data/pseudo_labeled_v2/summary.txt
"""
from __future__ import annotations
import argparse
import csv
import os
from collections import defaultdict, Counter
from pathlib import Path


def tokenize(transcript: str) -> list[str]:
    if not transcript:
        return []
    return [t.strip() for t in transcript.split("/") if t.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char-index",
                    default="data/characters_named/character_index.csv")
    ap.add_argument("--clusters",
                    default="cluster_assignments_v2.csv")
    ap.add_argument("--labels",
                    default="data/labels/labels.csv")
    ap.add_argument("--out-dir",
                    default="data/pseudo_labeled_v2")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load char index (positions) ----
    char_index = []
    with open(args.char_index, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["x"] = int(r["x"])
            r["line_num"] = int(r["line_num"])
            r["char_num"] = int(r["char_num"])
            char_index.append(r)
    print(f"Char index entries: {len(char_index)}")

    # Group by image, sort by x within image
    image_to_chars: dict[str, list[dict]] = defaultdict(list)
    for c in char_index:
        image_to_chars[c["image_id"]].append(c)
    for img_id in image_to_chars:
        image_to_chars[img_id].sort(key=lambda c: c["x"])

    # ---- Load clusters ----
    filename_to_cluster: dict[str, str] = {}
    with open(args.clusters, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            filename_to_cluster[r["filename"]] = r["cluster"]
    print(f"Clustered chars: {len(filename_to_cluster)}")
    n_clusters = len(set(filename_to_cluster.values()) - {"noise"})
    print(f"Distinct clusters (excl. noise): {n_clusters}")

    # ---- Load gold labels ----
    csv_lines = []
    with open(args.labels, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["line"] = int(r["line"].strip())
            csv_lines.append(r)
    print(f"Labeled lines: {len(csv_lines)}")

    # ---- Build per-image token sequence ----
    image_tokens: dict[str, list[dict]] = defaultdict(list)
    for r in csv_lines:
        img = r["image"].strip()
        line = r["line"]
        conf = r["confidence"].strip()
        toks = tokenize(r["transcript"])
        n = len(toks)
        for i, tok in enumerate(toks):
            is_boundary = i < 2 or i >= n - 2
            image_tokens[img].append({
                "token":      tok,
                "csv_conf":   conf,
                "line":       line,
                "boundary":   is_boundary,
                "is_unk":     tok == "[unk]",
            })

    # ---- Proportional alignment + cluster vote tally ----
    char_label: dict[str, dict] = {}
    align_stats = []

    for img_id, tok_seq in image_tokens.items():
        chars = image_to_chars.get(img_id, [])
        n_tok = len(tok_seq)
        n_chr = len(chars)
        if n_chr == 0:
            print(f"  [skip] {img_id}: 0 chars extracted")
            continue
        ratio = n_chr / n_tok if n_tok > 0 else 0
        diff_pct = abs(n_tok - n_chr) / max(n_tok, 1) * 100
        status = ("good" if diff_pct <= 10 else
                  "ok"   if diff_pct <= 25 else
                  "poor" if diff_pct <= 40 else "bad")
        align_stats.append({
            "image_id": img_id, "n_tokens": n_tok, "n_chars": n_chr,
            "diff_pct": round(diff_pct, 1), "status": status,
        })
        print(f"  {img_id}: {n_tok} tokens / {n_chr} chars  "
              f"({diff_pct:.1f}% diff, {status})")

        if status == "bad":
            continue

        used = set()
        for i, info in enumerate(tok_seq):
            ci = min(round(i * ratio), n_chr - 1)
            while ci in used and ci < n_chr - 1:
                ci += 1
            used.add(ci)
            fname = chars[ci]["filename"]

            csv_conf = info["csv_conf"]
            if info["boundary"] or status == "poor":
                eff = "low"
            elif status == "ok" and csv_conf == "high":
                eff = "medium"
            else:
                eff = csv_conf

            char_label[fname] = {
                "label":     info["token"],
                "image_id":  img_id,
                "line":      info["line"],
                "csv_conf":  csv_conf,
                "eff_conf":  eff,
                "is_unk":    info["is_unk"],
                "boundary":  info["boundary"],
            }

    real_n = sum(1 for v in char_label.values() if not v["is_unk"])
    print(f"\n  Direct labels (real)  : {real_n}")
    print(f"  [unk] labels         : {sum(1 for v in char_label.values() if v['is_unk'])}")

    # ---- Vote per cluster ----
    print("\nVoting per cluster...")
    cluster_votes: dict[str, Counter] = defaultdict(Counter)
    for fname, info in char_label.items():
        if info["is_unk"]:
            continue
        if info["eff_conf"] not in ("high", "medium"):
            continue
        cl = filename_to_cluster.get(fname)
        if cl and cl != "noise":
            cluster_votes[cl][info["label"]] += 1

    cluster_label: dict[str, dict] = {}
    for cl, votes in cluster_votes.items():
        total = sum(votes.values())
        top, top_n = votes.most_common(1)[0]
        score = top_n / total
        confidence = ("high"   if score >= 0.70 else
                      "medium" if score >= 0.50 else "low")
        cluster_label[cl] = {
            "label":       top,
            "vote_score":  round(score, 3),
            "total_votes": total,
            "confidence":  confidence,
            "top_2":       votes.most_common(2),
        }
    high_n = sum(1 for v in cluster_label.values() if v["confidence"] == "high")
    med_n  = sum(1 for v in cluster_label.values() if v["confidence"] == "medium")
    low_n  = sum(1 for v in cluster_label.values() if v["confidence"] == "low")
    print(f"  Clusters labeled : {len(cluster_label)} / {n_clusters}")
    print(f"  high (≥70%)      : {high_n}")
    print(f"  medium (50-70%)  : {med_n}")
    print(f"  low (<50%)       : {low_n}")

    # ---- Propagate ----
    print("\nPropagating cluster labels...")
    output_rows = []
    for c in char_index:
        fname = c["filename"]
        cl = filename_to_cluster.get(fname, "unknown")

        if fname in char_label and not char_label[fname]["is_unk"]:
            info = char_label[fname]
            label, ltype, conf = info["label"], "direct", info["eff_conf"]
        elif cl in cluster_label:
            ci = cluster_label[cl]
            label, ltype, conf = ci["label"], "pseudo", ci["confidence"]
        else:
            label, ltype, conf = "[unk]", "unlabeled", "none"

        output_rows.append({
            "filename":   fname,
            "image_id":   c["image_id"],
            "line_num":   c["line_num"],
            "char_num":   c["char_num"],
            "x":          c["x"],
            "cluster":    cl,
            "label":      label,
            "label_type": ltype,
            "confidence": conf,
        })

    # ---- Save ----
    chr_csv = os.path.join(args.out_dir, "character_labels.csv")
    with open(chr_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "filename","image_id","line_num","char_num","x",
            "cluster","label","label_type","confidence"])
        w.writeheader()
        w.writerows(output_rows)
    print(f"\nSaved: {chr_csv}")

    cl_csv = os.path.join(args.out_dir, "cluster_labels.csv")
    with open(cl_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "cluster", "label", "vote_score", "total_votes", "confidence"])
        w.writeheader()
        for cl, info in sorted(cluster_label.items()):
            w.writerow({
                "cluster": cl, "label": info["label"],
                "vote_score": info["vote_score"],
                "total_votes": info["total_votes"],
                "confidence": info["confidence"],
            })
    print(f"Saved: {cl_csv}")

    # ---- Summary ----
    direct = sum(1 for r in output_rows if r["label_type"] == "direct")
    pseudo = sum(1 for r in output_rows if r["label_type"] == "pseudo")
    unlab  = sum(1 for r in output_rows if r["label_type"] == "unlabeled")
    usable = sum(1 for r in output_rows
                 if r["label"] != "[unk]"
                 and r["confidence"] in ("high", "medium"))
    summary = f"""
=== PSEUDO-LABEL v2 SUMMARY ===

Total characters       : {len(output_rows)}

Label sources:
  Direct (CSV)         : {direct}
  Pseudo (cluster)     : {pseudo}
  Unlabeled            : {unlab}

Usable for training
  (non-[unk], high/med): {usable}

Cluster confidence breakdown:
  high   : {high_n}
  medium : {med_n}
  low    : {low_n}

Alignment per image:
""" + "\n".join(
    f"  {s['image_id']}: {s['n_tokens']} tokens / "
    f"{s['n_chars']} chars — {s['diff_pct']}% diff ({s['status']})"
    for s in align_stats)

    with open(os.path.join(args.out_dir, "summary.txt"), "w",
              encoding="utf-8") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
