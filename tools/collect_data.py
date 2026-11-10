"""Collect training rows for the neural-network advisor.

The bot plays duplicate matches against the practice opponents (tools/opponents.py)
with random exploration moves mixed in. Every decision is stored as
(features, action taken, chips won or lost in that hand / 80).

    python tools/collect_data.py OUT.npz [--seeds 60] [--deals 500] [--eps 0.25] [--policy WEIGHTS.npz]

Without --policy the rule-based bot plays alone (first round). With --policy the
given network also overrides moves, so later rounds see the situations the
network itself reaches.
"""
import argparse
import os
import random
import sys
import types
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

OPPONENTS = ["tight", "loose", "nit", "maniac", "bluffer", "trapper", "pre3bet", "randagg", "station",
             "pot", "half", "exploiter", "exploiter2", "stealer", "tilt", "foldy"]

LOG: list = []


def make_collector(eps, rng):
    import advisor
    import bot_core
    from game import Action

    def new_half(config, memory):
        bot_core.init_memory(memory)
        memory["pending"] = []

    def choose_action(obs, memory):
        prev = memory["raised_this_hand"]
        a = bot_core.choose(obs, memory)
        eq = memory.get("last_eq")
        if eq is not None:
            if rng.random() < eps:
                a = rng.choice(obs.legal_actions)
            memory["pending"].append((advisor.feats(obs, memory, eq), advisor.A_ORDER.index(a), advisor.legal_mask(obs)))
        memory["raised_this_hand"] = prev or a in (Action.RAISE_HALF, Action.RAISE_POT)
        return a

    def on_hand_end(result, memory):
        bot_core.record_hand_end(result, memory)
        for f, ai, m in memory["pending"]:
            LOG.append((f, ai, m, result.my_net / 80.0))
        memory["pending"] = []

    return types.SimpleNamespace(new_half=new_half, choose_action=choose_action, on_hand_end=on_hand_end)


def job(args):
    opp, seed, deals, eps, policy = args
    import advisor
    import selfplay
    if policy:
        d = np.load(policy)
        advisor._W = {k: d[k] for k in d.files}
    else:
        advisor._W = None                      # rule-based bot only
    LOG.clear()
    selfplay.match(make_collector(eps, random.Random(seed)), selfplay.get("tools/opponents.py:" + opp), deals, str(seed))
    return (np.stack([r[0] for r in LOG]), np.array([r[1] for r in LOG]),
            np.stack([r[2] for r in LOG]), np.array([r[3] for r in LOG], dtype=np.float32))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--seeds", type=int, default=60, help="matches per opponent")
    p.add_argument("--deals", type=int, default=500, help="deals per half")
    p.add_argument("--eps", type=float, default=0.25, help="probability of a random exploration move")
    p.add_argument("--policy", default=None, help="network weights (.npz) that also choose moves")
    p.add_argument("--base-seed", type=int, default=1000)
    p.add_argument("--workers", type=int, default=min(9, os.cpu_count() or 1))
    a = p.parse_args()
    jobs = [(o, a.base_seed + s, a.deals, a.eps, a.policy) for o in OPPONENTS for s in range(a.seeds)]
    parts = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(job, jobs):
            parts.append(r)
    X, A, M, Y = (np.concatenate([p_[i] for p_ in parts]) for i in range(4))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, X=X, A=A, M=M, Y=Y)
    print(f"saved {len(X)} rows to {a.out} (mean result {Y.mean():+.4f})")


if __name__ == "__main__":
    main()
