"""Analyze V-JEPA2 embeddings as preference control. Run with system python."""

import json
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from pathlib import Path

cache = Path("cache/preference")
pairs = json.loads((cache / "selected_pairs.json").read_text())
embeddings = json.loads((cache / "jepa_embeddings.json").read_text())

by_image = {r["image"]: r for r in embeddings if "error" not in r}
print(f"{len(by_image)} images with embeddings")

X_pref, X_nonpref = [], []
for pair in pairs:
    p = by_image.get(pair["preferred_image"])
    n = by_image.get(pair["nonpreferred_image"])
    if p and n:
        X_pref.append(p["embedding"])
        X_nonpref.append(n["embedding"])

X_pref = np.array(X_pref)
X_nonpref = np.array(X_nonpref)
print(f"{len(X_pref)} pairs, embedding dim: {X_pref.shape[1]}")

diff = X_pref - X_nonpref

# Test 1: Mean preference direction (LOO to avoid circularity)
acc_direction_correct = 0
for i in range(len(diff)):
    # Compute mean direction from all OTHER pairs
    mask = np.ones(len(diff), dtype=bool)
    mask[i] = False
    mean_dir = diff[mask].mean(axis=0)
    norm = np.linalg.norm(mean_dir)
    if norm > 0:
        mean_dir /= norm
        score = diff[i] @ mean_dir
        if score > 0:
            acc_direction_correct += 1
acc_direction = acc_direction_correct / len(diff) * 100

# Test 2: LOO-CV logistic regression (binary: is pref or nonpref?)
# Create balanced dataset: half pref-nonpref (label 1), half nonpref-pref (label 0)
X_cls = np.vstack([diff, -diff])  # (200, dim)
y_cls = np.array([1]*len(diff) + [0]*len(diff))

# Shuffle
rng = np.random.RandomState(42)
idx = rng.permutation(len(X_cls))
X_cls, y_cls = X_cls[idx], y_cls[idx]

from sklearn.model_selection import cross_val_score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cls)
clf = LogisticRegression(C=0.01, max_iter=1000)
cv_scores = cross_val_score(clf, X_scaled, y_cls, cv=10, scoring="accuracy")
acc_loocv = cv_scores.mean() * 100

# Test 3: L2 norm
norm_deltas = np.linalg.norm(X_pref, axis=1) - np.linalg.norm(X_nonpref, axis=1)
t_norm, p_norm = stats.ttest_1samp(norm_deltas, 0)
acc_norm = (norm_deltas > 0).mean() * 100

print(f"\n{'='*60}")
print(f"V-JEPA2 CONTROL EXPERIMENT")
print(f"{'='*60}")
print(f"Mean preference direction accuracy: {acc_direction:.1f}%")
print(f"LOO-CV logistic regression accuracy: {acc_loocv:.1f}%")
print(f"L2 norm accuracy: {acc_norm:.1f}% (t={t_norm:.2f}, p={p_norm:.4f})")
print()
print(f"Compare with:")
print(f"  TRIBE DMN+Limbic (no training): 75.0%")
print(f"  CLIP score (no training):       50.0%")
print()
if acc_loocv >= 75:
    print("RESULT: V-JEPA2 alone matches TRIBE. Brain mapping may not add signal.")
elif acc_loocv >= 65:
    print("RESULT: V-JEPA2 captures some preference. TRIBE adds modest signal via brain mapping.")
else:
    print("RESULT: V-JEPA2 alone is weak. TRIBE's brain mapping adds substantial signal.")
