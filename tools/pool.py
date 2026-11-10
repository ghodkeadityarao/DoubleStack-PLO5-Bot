"""Run the bot (optionally with PARAMS overrides) against the practice opponents.

    python tools/pool.py ['{"param": value}'] [deals] [seeds]
    e.g. python tools/pool.py '' 600 1,2

Prints bb/deal against each opponent (duplicate matches), the mean and the minimum.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

OPPS = ["random", "station", "pot", "half", "foldy", "tight", "loose", "nit", "maniac", "randagg", "overfold",
        "bluffer", "lag", "trapper", "sizer", "pre3bet", "rivbluff", "valuestation", "tilt",
        "exploiter", "exploiter2", "stealer"]


def run(args):
    over, opp, deals, seed = args
    import bot_core
    import selfplay
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(bot_core.PARAMS.get(k), dict):
            bot_core.PARAMS[k].update({int(a): b for a, b in v.items()})
        else:
            bot_core.PARAMS[k] = v
    m, se, A, B = selfplay.match(selfplay.get("agent"), selfplay.get("tools/opponents.py:" + opp), deals, seed)
    return opp, m, se, A.time, A.errors


def evaluate(over, deals=600, seeds=("1", "2"), opps=OPPS, workers=None):
    jobs = [(over, o, deals, s) for o in opps for s in seeds]
    res = {}
    with ProcessPoolExecutor(workers or min(9, os.cpu_count() or 1)) as ex:
        for opp, m, se, t, e in ex.map(run, jobs):
            res.setdefault(opp, []).append((m, se, t, e))
    return {o: sum(r[0] for r in res[o]) / len(res[o]) for o in opps}, res


if __name__ == "__main__":
    over = json.loads(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else {}
    deals = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    seeds = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("1", "2")
    out, res = evaluate(over, deals, seeds)
    for o, v in out.items():
        se = (sum(r[1] ** 2 for r in res[o]) ** 0.5) / len(res[o])
        print(f"  {o:13s} {v:+7.2f} +/-{1.96 * se:.2f}  time={max(r[2] for r in res[o]):.1f}s errors={sum(r[3] for r in res[o])}")
    vals = list(out.values())
    print(f"mean {sum(vals) / len(vals):+.3f}   min {min(vals):+.3f}")
