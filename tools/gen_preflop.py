"""Offline: Monte Carlo preflop equity (vs a random hand) for every suit-canonical
5-card hole hand. Writes data/preflop_eq.json  {packed_canonical_hand: equity}.

    python tools/gen_preflop.py [samples_per_hand] [workers]
"""
import json, os, random, sys, time
from itertools import combinations, permutations
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluator import get_table, omaha_score  # noqa: E402

PERMS = list(permutations(range(4)))


def canon(hand):
    best = None
    for p in PERMS:
        t = tuple(sorted(((c >> 2) << 2) | p[c & 3] for c in hand))
        if best is None or t < best:
            best = t
    return best


def pack(t):
    v = 0
    for c in t:
        v = (v << 6) | c
    return v


def work(args):
    hand, n, seed = args
    rng = random.Random(seed)
    rest = [c for c in range(36) if c not in hand]
    wins = 0.0
    for _ in range(n):
        s = rng.sample(rest, 10)
        board = s[5:]
        a = omaha_score(hand, board)
        b = omaha_score(s[:5], board)
        wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return pack(hand), wins / n


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    get_table()
    t0 = time.time()
    reps = set()
    for h in combinations(range(36), 5):
        reps.add(canon(h))
    reps = sorted(reps)
    print(len(reps), "canonical hands", time.time() - t0, flush=True)
    jobs = [(h, n, i) for i, h in enumerate(reps)]
    out = {}
    with Pool(workers) as pool:
        for k, (key, eq) in enumerate(pool.imap_unordered(work, jobs, chunksize=50)):
            out[key] = round(eq, 4)
            if k % 2000 == 0:
                print(k, time.time() - t0, flush=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "preflop_eq.json"), "w") as f:
        json.dump(out, f)
    print("done", len(out), time.time() - t0)
