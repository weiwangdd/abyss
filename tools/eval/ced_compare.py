#!/usr/bin/env python3
"""Compare two logits files written by ced_check: per row the largest difference, the KL divergence and the argmax."""
import sys

import numpy as np


def main():
    a_path, b_path, n_vocab = sys.argv[1], sys.argv[2], int(sys.argv[3])
    a = np.fromfile(a_path, dtype=np.float32).reshape(-1, n_vocab).astype(np.float64)
    b = np.fromfile(b_path, dtype=np.float32).reshape(-1, n_vocab).astype(np.float64)
    if a.shape != b.shape:
        print(f"FAIL shapes differ: {a.shape} {b.shape}")
        return 1

    def logp(x):
        x = x - x.max(axis=1, keepdims=True)
        return x - np.log(np.exp(x).sum(axis=1, keepdims=True))

    la, lb = logp(a), logp(b)
    kl = (np.exp(la) * (la - lb)).sum(axis=1)
    diff = np.abs(a - b).max(axis=1)
    same = (a.argmax(axis=1) == b.argmax(axis=1))
    first = int(np.argmin(same)) if not same.all() else -1
    print(f"rows {a.shape[0]}  max|diff| {diff.max():.3g}  mean KL {kl.mean():.3g}  max KL {kl.max():.3g}  "
          f"argmax same {int(same.sum())}/{len(same)}" + (f"  first differs at row {first}" if first >= 0 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
