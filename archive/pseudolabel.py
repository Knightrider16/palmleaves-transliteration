"""
Script B (v4) — Pseudo-Label Pipeline with Proportional Alignment
==================================================================
Since line detection fails for horizontal manuscript strips,
this script uses SEQUENTIAL PROPORTIONAL ALIGNMENT:

  All tokens in an image are concatenated in CSV order.
  All characters in that image are sorted left-to-right.
  Token[i] is mapped to char[ round(i × n_chars / n_tokens) ]

This works because:
  - Characters are read left to right
  - Your CSV transcribes left to right
  - Total counts are close (~8% difference acceptable)

Alignment confidence is higher for CSV lines marked "high"
and for tokens far from line boundaries (boundary chars
are most likely to be misaligned due to count mismatch).

Usage:
    python script_B_pseudolabel.py

Inputs:
    data/characters_named/character_index.csv
    cluster_assignments.csv
    dataset.csv

Outputs:
    data/pseudo_labeled/character_labels.csv
    data/pseudo_labeled/line_dataset.csv
    data/pseudo_labeled/summary.txt
"""

import os
import csv
import math
from collections import defaultdict, Counter

CHAR_INDEX_CSV     = "data/characters_named/character_index.csv"
CLUSTER_ASSIGN_CSV = "cluster_assignments.csv"
DATASET_CSV        = "data/labels/labels.csv"
OUTPUT_FOLDER      = "data/pseudo_labeled"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────────────────────

def tokenize(transcript):
    if not transcript or not isinstance(transcript, str):
        return []
    return [t.strip() for t in transcript.split("/") if t.strip()]


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

print("Loading character index...")
char_index = []
with open(CHAR_INDEX_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        row["line_num"] = int(row["line_num"])
        row["char_num"] = int(row["char_num"])
        row["x"]        = int(row["x"])
        char_index.append(row)
print(f"  Characters: {len(char_index)}")

# Group by image_id, sorted by x position (left to right)
image_to_chars = defaultdict(list)
for c in char_index:
    image_to_chars[c["image_id"]].append(c)
for img_id in image_to_chars:
    image_to_chars[img_id].sort(key=lambda c: c["x"])

print("Loading cluster assignments...")
filename_to_cluster = {}
with open(CLUSTER_ASSIGN_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        filename_to_cluster[row["filename"]] = row["cluster"]
print(f"  Clustered: {len(filename_to_cluster)}")

print("Loading dataset CSV...")
csv_lines = []
with open(DATASET_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        row["line"] = int(row["line"].strip())
        csv_lines.append(row)
print(f"  Labeled lines: {len(csv_lines)}")


# ─────────────────────────────────────────────────────────────
# BUILD PER-IMAGE TOKEN SEQUENCE
# Concatenate all lines for each image in CSV order
# Each token carries its source line's confidence
# ─────────────────────────────────────────────────────────────

# image_id → [ (token, csv_confidence, line_num, position_in_line,
#               line_length, is_boundary) ]
image_token_seq = defaultdict(list)

for row in csv_lines:
    image_id   = row["image"].strip()
    line_num   = row["line"]
    confidence = row["confidence"].strip()
    transcript = row["transcript"].strip()
    tokens     = tokenize(transcript)
    n          = len(tokens)

    for i, token in enumerate(tokens):
        # Boundary = first or last 2 tokens of a line
        # These are most likely to be misaligned
        is_boundary = (i < 2 or i >= n - 2)

        image_token_seq[image_id].append({
            "token":          token,
            "csv_confidence": confidence,
            "line_num":       line_num,
            "is_boundary":    is_boundary,
            "is_unk":         token == "[unk]",
        })


# ─────────────────────────────────────────────────────────────
# PROPORTIONAL ALIGNMENT
# Map token[i] → char[ round(i × n_chars / n_tokens) ]
# ─────────────────────────────────────────────────────────────

print("\nProportional alignment per image...")

char_label_map = {}   # filename → label info
alignment_stats = []

for image_id, token_seq in image_token_seq.items():
    chars   = image_to_chars.get(image_id, [])
    n_tok   = len(token_seq)
    n_chr   = len(chars)

    if n_chr == 0:
        print(f"  [skip] {image_id}: no characters extracted")
        alignment_stats.append({
            "image_id": image_id,
            "n_tokens": n_tok, "n_chars": n_chr,
            "ratio": 0, "status": "no_chars"
        })
        continue

    ratio = n_chr / n_tok if n_tok > 0 else 0
    diff_pct = abs(n_tok - n_chr) / max(n_tok, 1) * 100

    status = ("good"    if diff_pct <= 10 else
              "ok"      if diff_pct <= 25 else
              "poor"    if diff_pct <= 40 else "bad")

    alignment_stats.append({
        "image_id": image_id,
        "n_tokens": n_tok, "n_chars": n_chr,
        "ratio": round(ratio, 3), "diff_pct": round(diff_pct, 1),
        "status": status
    })

    print(f"  {image_id}: {n_tok} tokens → {n_chr} chars "
          f"({diff_pct:.1f}% diff, {status})")

    # Skip images where alignment is too unreliable
    if status == "bad":
        print(f"    → Skipping (>40% mismatch)")
        continue

    # Proportional mapping
    used_char_indices = set()
    for i, tok_info in enumerate(token_seq):
        # Map token position proportionally to char position
        char_idx = min(round(i * ratio), n_chr - 1)

        # Avoid assigning same char twice — shift forward if needed
        while char_idx in used_char_indices and char_idx < n_chr - 1:
            char_idx += 1
        used_char_indices.add(char_idx)

        filename = chars[char_idx]["filename"]

        # Determine effective confidence
        # Downgrade boundary tokens and poor-alignment images
        csv_conf = tok_info["csv_confidence"]
        if tok_info["is_boundary"] or status == "poor":
            eff_conf = "low"
        elif status == "ok" and csv_conf == "high":
            eff_conf = "medium"
        else:
            eff_conf = csv_conf

        char_label_map[filename] = {
            "label":          tok_info["token"],
            "source_image":   image_id,
            "source_line":    tok_info["line_num"],
            "csv_confidence": csv_conf,
            "eff_confidence": eff_conf,
            "is_unk":         tok_info["is_unk"],
            "is_boundary":    tok_info["is_boundary"],
        }

real_n = sum(1 for v in char_label_map.values() if not v["is_unk"])
unk_n  = sum(1 for v in char_label_map.values() if v["is_unk"])
print(f"\n  Real labels assigned : {real_n}")
print(f"  [unk] labels         : {unk_n}")
print(f"  Unmatched characters : {len(char_index) - len(char_label_map)}")


# ─────────────────────────────────────────────────────────────
# VOTE PER CLUSTER
# ─────────────────────────────────────────────────────────────

print("\nVoting per cluster...")

cluster_votes = defaultdict(Counter)
for filename, info in char_label_map.items():
    if info["is_unk"]:
        continue
    # Only high/medium confidence labels vote
    if info["eff_confidence"] not in ("high", "medium"):
        continue
    cluster = filename_to_cluster.get(filename)
    if cluster and cluster != "noise":
        cluster_votes[cluster][info["label"]] += 1

cluster_label = {}
for cluster, votes in cluster_votes.items():
    total            = sum(votes.values())
    top_label, top_n = votes.most_common(1)[0]
    score            = top_n / total
    confidence       = ("high"   if score >= 0.70 else
                        "medium" if score >= 0.50 else "low")
    cluster_label[cluster] = {
        "label":       top_label,
        "vote_score":  round(score, 3),
        "total_votes": total,
        "confidence":  confidence,
        "top_2":       votes.most_common(2),
    }

high_n   = sum(1 for v in cluster_label.values() if v["confidence"] == "high")
medium_n = sum(1 for v in cluster_label.values() if v["confidence"] == "medium")
low_n    = sum(1 for v in cluster_label.values() if v["confidence"] == "low")
print(f"  Clusters labeled      : {len(cluster_label)}")
print(f"  High  (≥70% vote)     : {high_n}")
print(f"  Medium (50–70%)       : {medium_n}")
print(f"  Low   (<50%)          : {low_n}")

# Print top clusters for inspection
print("\n  Top 20 clusters by vote count:")
top_clusters = sorted(cluster_label.items(),
                      key=lambda x: x[1]["total_votes"], reverse=True)[:20]
for cl, info in top_clusters:
    print(f"    {cl}: '{info['label']}' "
          f"({info['vote_score']*100:.0f}% of {info['total_votes']} votes) "
          f"| top2={info['top_2']}")


# ─────────────────────────────────────────────────────────────
# ASSIGN PSEUDO-LABELS TO ALL CHARACTERS
# ─────────────────────────────────────────────────────────────

print("\nAssigning pseudo-labels...")
output_rows = []

for c in char_index:
    filename = c["filename"]
    cluster  = filename_to_cluster.get(filename, "unknown")

    if filename in char_label_map and not char_label_map[filename]["is_unk"]:
        info       = char_label_map[filename]
        label      = info["label"]
        label_type = "direct"
        confidence = info["eff_confidence"]

    elif cluster in cluster_label:
        cl         = cluster_label[cluster]
        label      = cl["label"]
        label_type = "pseudo"
        confidence = cl["confidence"]

    else:
        label      = "[unk]"
        label_type = "unlabeled"
        confidence = "none"

    output_rows.append({
        "filename":   filename,
        "image_id":   c["image_id"],
        "line_num":   c["line_num"],
        "char_num":   c["char_num"],
        "x":          c["x"],
        "cluster":    cluster,
        "label":      label,
        "label_type": label_type,
        "confidence": confidence,
    })


# ─────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────

char_csv = os.path.join(OUTPUT_FOLDER, "character_labels.csv")
with open(char_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["filename","image_id","line_num","char_num",
                        "x","cluster","label","label_type","confidence"])
    writer.writeheader()
    writer.writerows(output_rows)
print(f"\nSaved: {char_csv}")

line_csv = os.path.join(OUTPUT_FOLDER, "line_dataset.csv")
with open(line_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["image","line","confidence","transcript"])
    writer.writeheader()
    for row in csv_lines:
        if row["confidence"] in ("high", "medium"):
            writer.writerow({
                "image":      row["image"],
                "line":       row["line"],
                "confidence": row["confidence"],
                "transcript": row["transcript"],
            })

# Cluster label reference CSV — useful for manual review
cluster_csv = os.path.join(OUTPUT_FOLDER, "cluster_labels.csv")
with open(cluster_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["cluster","label","vote_score","total_votes","confidence"])
    writer.writeheader()
    for cluster, info in sorted(cluster_label.items()):
        writer.writerow({
            "cluster":     cluster,
            "label":       info["label"],
            "vote_score":  info["vote_score"],
            "total_votes": info["total_votes"],
            "confidence":  info["confidence"],
        })
print(f"Saved: {cluster_csv}")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────

direct_n    = sum(1 for r in output_rows if r["label_type"] == "direct")
pseudo_n    = sum(1 for r in output_rows if r["label_type"] == "pseudo")
unlabeled_n = sum(1 for r in output_rows if r["label_type"] == "unlabeled")
usable      = sum(1 for r in output_rows
                  if r["label"] != "[unk]"
                  and r["confidence"] in ("high", "medium"))

summary = f"""
=== PSEUDO-LABEL PIPELINE SUMMARY ===

Total characters              : {len(output_rows)}

Label sources:
  Direct (from your CSV)      : {direct_n}
  Pseudo (cluster vote)       : {pseudo_n}
  Unlabeled                   : {unlabeled_n}

Usable for training
(label != [unk], high/medium confidence):
                              : {usable}

Alignment quality per image:
""" + "\n".join(
    f"  {r['image_id']}: {r['n_tokens']} tokens / "
    f"{r['n_chars']} chars — {r.get('diff_pct','?')}% diff ({r['status']})"
    for r in alignment_stats
) + f"""

Output files:
  {char_csv}
  {line_csv}
  {cluster_csv}

Next step:
  Review cluster_labels.csv to verify pseudo-labels look correct.
  Then use character_labels.csv to train your character classifier.
"""

summary_path = os.path.join(OUTPUT_FOLDER, "summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary)

print(summary)
print(f"Summary saved: {summary_path}")