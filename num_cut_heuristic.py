import numpy as np
from kneed import KneeLocator

Z_THRESHOLD = 2.0
TOP_K = 2

def detect_big_gaps(scores, z_threshold=2, top_k=1):
    if len(scores) < 2:
        return len(scores)

    diffs = np.diff(scores)  # score[i] - score[i+1]
    abs_diffs = np.abs(diffs)

    mean = np.mean(abs_diffs)
    std = np.std(abs_diffs)

    if std == 0:
        return len(scores)

    z_scores = (abs_diffs - mean) / std

    # Indices des plus gros écarts "anormaux"
    indices_significatifs = np.where(z_scores > z_threshold)[0]

    # Optionnel : garder les top_k plus gros z-scores (même s’ils ne dépassent pas le seuil)
    if len(indices_significatifs) == 0 and top_k > 0:
        indices_significatifs = np.argsort(z_scores)[-top_k:]

    # +1 car diff[i] est entre score[i] et score[i+1]
    return max([int(i)+1 for i in indices_significatifs])

def knee_method(scores):
    total = len(scores)

    x = list(range(len(scores)))

    try:
        knee = KneeLocator(x, scores, curve='convex', direction='decreasing')
        knee_idx = knee.knee if knee.knee is not None else len(scores)
    except:
        knee_idx = len(scores)
    
    return knee_idx

def num_cut_heuristic(scores):
    sorted_scores = sorted(scores, reverse=True)
    return max(
        detect_big_gaps(sorted_scores, z_threshold=Z_THRESHOLD, top_k=TOP_K),
        knee_method(sorted_scores)
    )