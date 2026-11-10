"""
bot_core.py - strategy engine used by agent.py.

Pieces:
  * preflop equity table (data/preflop_eq.json, built offline by tools/gen_preflop.py)
  * Monte-Carlo equity vs a random opponent hand for flop / turn / river
    (river uses an exact "pair score" trick so it is cheap)
  * a simple opponent model (aggression, fold-to-raise) learned within a half
  * an equity-vs-pot-odds betting policy with a strict time budget

Everything is wrapped so that choose() always returns a legal action.
"""

from __future__ import annotations

import json
import os
import random
import time
from itertools import combinations

import numpy as np

from evaluator import get_table, omaha_score
from game import Action, Street

TABLE = get_table()
HERE = os.path.dirname(os.path.abspath(__file__))

_PAIRS5 = tuple(combinations(range(5), 2))
_TRIPLES5 = tuple(combinations(range(5), 3))
_BIT = [1 << c for c in range(36)]

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
PARAMS = {
    # bank management (seconds, per half)
    "bank": 8.0,
    "bank_use": 0.5,
    "hard_use": 0.85,          # stop simulating entirely beyond this fraction of the bank          # fraction of the bank we allow ourselves to use
    "overhead": 0.0004,        # assumed per-decision overhead outside our code
    "max_samples": 900,
    "min_samples": 50,
    # opponent-range shrink applied to our equity when facing a bet
    "shrink": {0: 0.05, 1: 0.07, 2: 0.09, 3: 0.11},   # legacy (unused when learn_shrink)
    "learn_shrink": True,
    "lam": {0: 0.95, 1: 1.1, 2: 1.2, 3: 1.3},          # shrink = lam * (opp_bet_eq - 0.5)
    "prior_eq": {0: 0.58, 1: 0.62, 2: 0.64, 3: 0.66}, # prior for opp's equity when it bets
    "prior_w": 6.0,
    "call_margin": 0.03,
    "raise_pot_eq": 0.74,
    "raise_half_eq": 0.68,
    "bet_pot_eq": 0.68,
    "bet_half_eq": 0.57,
    "folder_thr": 0.70,        # exploit opponents that fold to raises more often than this
    "semibluff": 0.20,         # probability of a half-pot semi-bluff with a draw-ish hand
    "pre_open_pct": 0.50,      # preflop raise when equity >= this percentile of hands
    "pre_fold_eq": 0.0,
    "pre_sb_call_eq": 0.34,    # SB completes the blind only with at least this equity
    "pre_extra": 0.0,          # extra equity needed to continue preflop vs a raise
}

# ---------------------------------------------------------------------------
# Preflop table
# ---------------------------------------------------------------------------
_PERMS = None
_PRE = None
_PRE_SORTED = None


def _load_pre():
    global _PRE, _PRE_SORTED, _PERMS
    if _PRE is not None:
        return
    from itertools import permutations
    _PERMS = list(permutations(range(4)))
    path = os.path.join(HERE, "data", "preflop_eq.json")
    try:
        with open(path) as f:
            raw = json.load(f)
        _PRE = {int(k): v for k, v in raw.items()}
        _PRE_SORTED = np.sort(np.array(list(_PRE.values())))
    except Exception:
        _PRE = {}
        _PRE_SORTED = np.array([0.5])


def preflop_equity(hole) -> float:
    """Equity vs a random hand (table lookup). Falls back to 0.5 if unknown."""
    _load_pre()
    best = None
    for p in _PERMS:
        t = tuple(sorted(((c >> 2) << 2) | p[c & 3] for c in hole))
        if best is None or t < best:
            best = t
    v = 0
    for c in best:
        v = (v << 6) | c
    return _PRE.get(v, 0.5)


def preflop_percentile(eq: float) -> float:
    _load_pre()
    return float(np.searchsorted(_PRE_SORTED, eq) / max(1, len(_PRE_SORTED)))


# ---------------------------------------------------------------------------
# Postflop equity
# ---------------------------------------------------------------------------

def _score_hand_masks(hole_masks, tri_masks):
    t = TABLE
    return max([t[h | m] for h in hole_masks for m in tri_masks])


def mc_equity(hole, board, samples: int, rng: random.Random) -> float:
    """Equity vs a random opponent hand with random runout (flop/turn)."""
    used = set(hole) | set(board)
    rest = [c for c in range(36) if c not in used]
    nb = 5 - len(board)
    hm = [_BIT[hole[i]] | _BIT[hole[j]] for i, j in _PAIRS5]
    t = TABLE
    sample = rng.sample
    wins = 0.0
    for _ in range(samples):
        s = sample(rest, nb + 5)
        full = list(board) + s[:nb]
        opp = s[nb:]
        bb = [_BIT[c] for c in full]
        tm = [bb[i] | bb[j] | bb[k] for i, j, k in _TRIPLES5]
        om = [_BIT[opp[i]] | _BIT[opp[j]] for i, j in _PAIRS5]
        a = max([t[h | m] for h in hm for m in tm])
        b = max([t[h | m] for h in om for m in tm])
        if a > b:
            wins += 1.0
        elif a == b:
            wins += 0.5
    return wins / samples


# ---- vectorised scorer: PT[pair_id, triple_id] = dense hand rank (see tools/gen_pt_table.py)
_PT = None
_PID = None
_TID = None
_NT = 7140
_TRI_IDX = np.array(_TRIPLES5, dtype=np.int64)       # (10,3)
_PAIR_IDX = np.array(_PAIRS5, dtype=np.int64)        # (10,2)


def _load_pt():
    global _PT, _PID, _TID
    if _PT is not None:
        return
    path = os.path.join(HERE, "data", "pt_table.npy")
    pt = np.load(path)
    assert pt.shape == (630, _NT)
    pid = np.zeros((36, 36), dtype=np.int64)
    for k, (a, b) in enumerate(combinations(range(36), 2)):
        pid[a, b] = pid[b, a] = k
    tid = np.zeros((36, 36, 36), dtype=np.int64)
    for k, (a, b, c) in enumerate(combinations(range(36), 3)):
        for x, y, z in ((a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)):
            tid[x, y, z] = k
    _PT = pt.reshape(-1).astype(np.int32)
    _PID, _TID = pid, tid


def np_equity(hole, board, samples: int, nprng) -> float:
    """Vectorised Monte-Carlo equity vs a random hand (any street)."""
    _load_pt()
    used = set(hole) | set(board)
    rest = np.array([c for c in range(36) if c not in used], dtype=np.int64)
    nb = 5 - len(board)
    sel = np.argsort(nprng.random((samples, len(rest))), axis=1)[:, : nb + 5]
    cards = rest[sel]                                           # (S, nb+5)
    if nb:
        fb = np.concatenate([np.tile(np.array(board, dtype=np.int64), (samples, 1)), cards[:, :nb]], axis=1)
    else:
        fb = np.tile(np.array(board, dtype=np.int64), (samples, 1))
    opp = cards[:, nb:]
    tri = _TID[fb[:, _TRI_IDX[:, 0]], fb[:, _TRI_IDX[:, 1]], fb[:, _TRI_IDX[:, 2]]]   # (S,10)
    hole_a = np.array(hole, dtype=np.int64)
    my_pairs = _PID[hole_a[_PAIR_IDX[:, 0]], hole_a[_PAIR_IDX[:, 1]]]                 # (10,)
    opp_pairs = _PID[opp[:, _PAIR_IDX[:, 0]], opp[:, _PAIR_IDX[:, 1]]]                # (S,10)
    tri3 = tri[:, None, :] 
    mine = np.take(_PT, my_pairs[None, :, None] * _NT + tri3).max(axis=(1, 2))
    theirs = np.take(_PT, opp_pairs[:, :, None] * _NT + tri3).max(axis=(1, 2))
    return float((np.sum(mine > theirs) + 0.5 * np.sum(mine == theirs)) / samples)


_RIVER_PAIR_CACHE: dict = {}


def river_opp_scores(hole, board, samples: int, nprng) -> tuple[int, np.ndarray]:
    """My exact score and `samples` random opponent hand scores (river)."""
    my = omaha_score(hole, board)
    used = set(hole) | set(board)
    rest = [c for c in range(36) if c not in used]
    n = len(rest)  # 26 + ... = 31 on the river
    bb = [_BIT[c] for c in board]
    tm = [bb[i] | bb[j] | bb[k] for i, j, k in _TRIPLES5]
    t = TABLE
    idx = np.zeros((n, n), dtype=np.int32)
    pm = []
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[i, j] = idx[j, i] = k
            k += 1
            pm.append(_BIT[rest[i]] | _BIT[rest[j]])
    ps = np.array([max([t[p | m] for m in tm]) for p in pm], dtype=np.int64)
    sel = np.argsort(nprng.random((samples, n)), axis=1)[:, :5]
    cols = [idx[sel[:, a], sel[:, b]] for a, b in _PAIRS5]
    opp = ps[np.stack(cols, axis=1)].max(axis=1)
    return my, opp


def river_equity(hole, board, samples, nprng) -> float:
    my, opp = river_opp_scores(hole, board, samples, nprng)
    return float((np.sum(my > opp) + 0.5 * np.sum(my == opp)) / len(opp))


# ---------------------------------------------------------------------------
# Opponent model
# ---------------------------------------------------------------------------

def init_memory(memory: dict, seed=None, num_hands: int = 1000) -> None:
    memory.clear()
    memory.update({
        "rng": random.Random(seed if seed is not None else 12345),
        "nprng": np.random.default_rng(seed if seed is not None else 12345),
        "num_hands": max(1, int(num_hands)),
        "t_used": 0.0,
        "decisions": 0,
        "hands_seen": 0,
        "cur_hand": -1,
        "seen_hist": 0,
        "raised_this_hand": False,
        # opponent stats (with priors)
        "opp_agg": 3.5, "opp_agg_n": 10.0,          # raises / (raises+calls+checks) postflop
        "opp_pre_raise": 2.0, "opp_pre_n": 8.0,     # preflop raise freq
        "fold_to_raise": 3.0, "fold_to_raise_n": 8.0,
        "sd_hands": 0,
        "opp_raise_streets": [],
        "bet_eq_sum": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}, "bet_eq_n": {0: 0, 1: 0, 2: 0, 3: 0},
        "time_exhausted": False,
    })


def _update_opp_actions(obs, memory):
    """Scan newly visible history for opponent actions."""
    if memory["cur_hand"] != obs.hand_index:
        memory["cur_hand"] = obs.hand_index
        memory["seen_hist"] = 0
        memory["raised_this_hand"] = False
        memory["opp_raise_streets"] = []
    hist = obs.action_history
    for a in hist[memory["seen_hist"]:]:
        if a.is_me:
            if a.action in (Action.RAISE_HALF, Action.RAISE_POT):
                memory["raised_this_hand"] = True
            continue
        is_raise = a.action in (Action.RAISE_HALF, Action.RAISE_POT)
        if is_raise and int(a.street) not in memory["opp_raise_streets"]:
            memory["opp_raise_streets"].append(int(a.street))
        if a.street == Street.PREFLOP:
            memory["opp_pre_n"] += 1
            memory["opp_pre_raise"] += 1.0 if is_raise else 0.0
        else:
            memory["opp_agg_n"] += 1
            memory["opp_agg"] += 1.0 if is_raise else 0.0
    memory["seen_hist"] = len(hist)


def record_hand_end(result, memory) -> None:
    memory["hands_seen"] += 1
    if memory["raised_this_hand"]:
        memory["fold_to_raise_n"] += 1
        if (not result.went_to_showdown) and result.winner == "me":
            memory["fold_to_raise"] += 1
    if result.went_to_showdown:
        memory["sd_hands"] += 1
        try:
            _learn_showdown(result, memory)
        except Exception:
            pass
    memory["raised_this_hand"] = False
    memory["opp_raise_streets"] = []


def _learn_showdown(result, memory):
    """Opponent's cards are known: measure how strong its hand was when it bet."""
    streets = memory["opp_raise_streets"]
    opp = result.opp_hole
    if not streets or opp is None:
        return
    t0 = time.perf_counter()
    st = max(streets)
    if st == 0:
        eq = preflop_equity(opp)
    else:
        k = {1: 3, 2: 4, 3: 5}[st]
        eq = np_equity(tuple(opp), tuple(result.board[:k]), 300, memory["nprng"])
    memory["bet_eq_sum"][st] += eq
    memory["bet_eq_n"][st] += 1
    memory["t_used"] += time.perf_counter() - t0


def opp_bet_strength(memory, street) -> float:
    """Estimated avg equity-vs-random of the opponent's hands when it bets on `street`."""
    P = PARAMS
    st = int(street)
    # pool with neighbouring streets (postflop) for more data
    if st == 0:
        pool = [0]
    else:
        pool = [1, 2, 3]
    n = sum(memory["bet_eq_n"][p] for p in pool)
    sm = sum(memory["bet_eq_sum"][p] for p in pool)
    pr = P["prior_eq"][st]
    w = P["prior_w"]
    return (sm + pr * w) / (n + w)


# ---------------------------------------------------------------------------
# Budgeting
# ---------------------------------------------------------------------------

_F0 = 4.0e-5      # nominal fixed cost of one np_equity call (s) on a fast machine
_C0 = 1.2e-6      # nominal cost per sample (s)


def _sample_budget(memory, street) -> int:
    """How many MC samples we can afford for this decision (0 = out of time).

    `slow` is a running estimate of how much slower this machine is than the
    nominal one, so the same code adapts to a slow / shared CPU.  We always
    run at least `min_samples` while the bank has not nearly run out.
    """
    P = PARAMS
    hard_left = P["bank"] * P["hard_use"] - memory["t_used"]
    if hard_left <= 0.0:
        return 0
    slow = memory.get("slow", 1.0)
    allowed = P["bank"] * P["bank_use"]
    left = allowed - memory["t_used"]
    hands_done = max(1, memory["hands_seen"])
    dec_per_hand = max(1.5, min(4.0, (memory["decisions"] + 3) / hands_done))
    hands_left = max(1, memory.get("num_hands", 1000) - memory["hands_seen"])
    per = left / (hands_left * dec_per_hand) - P["overhead"] * slow if left > 0 else 0.0
    n = int((per / slow - _F0) / _C0)
    return max(P["min_samples"], min(P["max_samples"], n))


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def _estimate_equity(obs, memory):
    street = obs.street
    hole = obs.hole_cards
    if street == Street.PREFLOP:
        return preflop_equity(hole)
    n = _sample_budget(memory, street)
    if n <= 0:
        memory["time_exhausted"] = True
        return None
    t0 = time.perf_counter()
    eq = np_equity(hole, obs.board, n, memory["nprng"])
    dt = time.perf_counter() - t0
    ratio = dt / (_F0 + _C0 * n)
    ratio = min(max(ratio, 0.5), 200.0)
    memory["slow"] = 0.85 * memory.get("slow", ratio) + 0.15 * ratio
    return eq


def _fallback_action(obs):
    legal = obs.legal_actions
    if obs.to_call == 0:
        return Action.CHECK_CALL
    # without an estimate: call cheap, fold expensive
    if obs.to_call <= max(2, 0.18 * obs.pot):
        return Action.CHECK_CALL
    return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL


def _advise(obs, memory, act, eq):
    """Let the neural-net advisor override the rule-based action (see advisor.py)."""
    try:
        import advisor
        new = advisor.advise(obs, memory, act, eq)
        return new if new in obs.legal_actions else act
    except Exception:
        return act


def choose(obs, memory) -> Action:
    t0 = time.perf_counter()
    memory["last_eq"] = None
    try:
        act = _choose(obs, memory)
        if memory.get("last_eq") is not None:
            act = _advise(obs, memory, act, memory["last_eq"])
    except Exception:
        act = _fallback_action(obs)
    if act not in obs.legal_actions:
        act = _fallback_action(obs)
        if act not in obs.legal_actions:
            act = Action.CHECK_CALL
    if act in (Action.RAISE_HALF, Action.RAISE_POT):
        memory["raised_this_hand"] = True      # the opponent may fold before we see it in the history
    memory["t_used"] += time.perf_counter() - t0
    memory["decisions"] += 1
    return act


def _choose(obs, memory) -> Action:
    P = PARAMS
    _update_opp_actions(obs, memory)
    rng = memory["rng"]
    legal = obs.legal_actions
    pot, c = obs.pot, obs.to_call
    street = obs.street
    can_pot = Action.RAISE_POT in legal
    can_half = Action.RAISE_HALF in legal
    can_raise = can_pot or can_half

    eq = _estimate_equity(obs, memory)
    memory["last_eq"] = eq
    if eq is None:
        return _fallback_action(obs)

    agg = memory["opp_agg"] / memory["opp_agg_n"]
    fold_rate = memory["fold_to_raise"] / memory["fold_to_raise_n"]
    exploit_folder = memory["fold_to_raise_n"] >= 20 and fold_rate > P["folder_thr"]
    facing_bet = c > 0 and not (street == Street.PREFLOP and obs.opp_bet <= obs.big_blind and obs.raises_this_street == 0)

    # --- opponent range shrink when they put in a real bet/raise
    shrink = 0.0
    if facing_bet:
        if P["learn_shrink"]:
            shrink = max(0.0, P["lam"][int(street)] * (opp_bet_strength(memory, street) - 0.5))
        else:
            shrink = P["shrink"][int(street)] * (1.25 - min(1.0, agg))
        shrink *= (1.0 + 0.35 * max(0, obs.raises_this_street - 1))
    eff = eq - shrink

    # ---------------- preflop ----------------
    if street == Street.PREFLOP:
        pct = preflop_percentile(eq)
        if exploit_folder and not facing_bet and can_raise:
            return Action.RAISE_POT                    # opponent folds too much: steal
        if c == 0:                                     # BB option
            if can_raise and pct >= P["pre_open_pct"] + 0.1:
                return Action.RAISE_POT if pct >= 0.9 else Action.RAISE_HALF
            return Action.CHECK_CALL
        if not facing_bet:                             # SB completing / raising vs the BB
            if can_raise and pct >= P["pre_open_pct"]:
                return Action.RAISE_POT if pct >= 0.85 else Action.RAISE_HALF
            if eq >= P["pre_sb_call_eq"] or c == 0:
                return Action.CHECK_CALL
            return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL
        need = c / (pot + c)
        if eff < need + P["call_margin"] + P["pre_extra"]:
            return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL
        if can_pot and eff >= P["raise_pot_eq"] - 0.1:
            return Action.RAISE_POT
        if can_half and eff >= P["raise_half_eq"] - 0.1:
            return Action.RAISE_HALF
        return Action.CHECK_CALL

    # ---------------- postflop ----------------
    if c > 0:
        need = c / (pot + c)
        if eff < need + P["call_margin"]:
            # occasional bluff-raise against folders
            if can_half and fold_rate > 0.45 and rng.random() < 0.08:
                return Action.RAISE_HALF
            return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL
        if can_pot and eff >= P["raise_pot_eq"]:
            return Action.RAISE_POT
        if can_half and eff >= P["raise_half_eq"]:
            return Action.RAISE_HALF
        return Action.CHECK_CALL

    # no bet to face: value bet / semi-bluff / check
    if exploit_folder and can_half and eff < P["bet_half_eq"] and street != Street.RIVER:
        return Action.RAISE_HALF
    if exploit_folder and can_half and eff < P["bet_half_eq"] and street == Street.RIVER and rng.random() < 0.6:
        return Action.RAISE_HALF
    if can_pot and eff >= P["bet_pot_eq"]:
        # trap vs very aggressive opponents with a monster
        if eff >= 0.82 and agg > 0.55 and street != Street.RIVER and rng.random() < 0.5:
            return Action.CHECK_CALL
        return Action.RAISE_POT
    if can_half and eff >= P["bet_half_eq"]:
        return Action.RAISE_HALF
    if can_half and street != Street.RIVER and 0.36 <= eq < P["bet_half_eq"] and fold_rate > 0.25 \
            and rng.random() < P["semibluff"]:
        return Action.RAISE_HALF
    return Action.CHECK_CALL


# Load the lookup tables at import time (the server must answer within 15 s of
# starting, and the first decision should not pay for loading).
_load_pre()
_load_pt()
