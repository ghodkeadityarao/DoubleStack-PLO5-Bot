"""Offline: build data/pt_table.npy, a (630 pairs x 7140 triples) table of dense hand-rank ids
(uint16, higher = better; 0 = pair and triple share a card). Lets numpy score
'2 hole cards + 3 board cards' hands with a single gather."""
import os, sys, time
from itertools import combinations
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluator import get_table

t0 = time.time()
T = get_table()
pairs = list(combinations(range(36), 2))
triples = list(combinations(range(36), 3))
pm = [(1 << a) | (1 << b) for a, b in pairs]
tm = [(1 << a) | (1 << b) | (1 << c) for a, b, c in triples]
scores = np.zeros((len(pairs), len(triples)), dtype=np.int64)
for i, p in enumerate(pm):
    row = [T[p | t] if not (p & t) else 0 for t in tm]
    scores[i] = row
uniq = np.unique(scores[scores > 0])
print("distinct scores", len(uniq))
dense = np.zeros(scores.shape, dtype=np.uint16)
mask = scores > 0
dense[mask] = (np.searchsorted(uniq, scores[mask]) + 1).astype(np.uint16)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "pt_table.npy")
np.save(out, dense)
print("saved", dense.shape, time.time() - t0)
