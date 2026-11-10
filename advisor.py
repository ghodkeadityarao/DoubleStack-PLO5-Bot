"""
advisor.py - small neural network that second-guesses the heuristic bot.

The network (data/policy_net.npz, trained offline by self-play, see README)
predicts the expected chip result of each of the four actions. The bot only
overrides its rule-based action when the network predicts a clear gain.
Inference is plain numpy (about 30 microseconds); any failure falls back
to the heuristic action.
"""
import os
import numpy as np
import bot_core
from game import Action, Street

NF = 26
A_ORDER = [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT]


def feats(obs, mem, eq):
    st = int(obs.street)
    c, pot = obs.to_call, obs.pot
    need = c / (pot + c) if c > 0 else 0.0
    facing = 1.0 if (c > 0 and not (st == 0 and obs.opp_bet <= obs.big_blind and obs.raises_this_street == 0)) else 0.0
    agg = mem["opp_agg"] / mem["opp_agg_n"]
    fold = mem["fold_to_raise"] / mem["fold_to_raise_n"]
    pre = mem["opp_pre_raise"] / mem["opp_pre_n"]
    bs = bot_core.opp_bet_strength(mem, st) - 0.5
    opp_raises_hand = sum(1 for a in obs.action_history if (not a.is_me) and a.action in (Action.RAISE_HALF, Action.RAISE_POT))
    opp_raised_street = 1.0 if any((not a.is_me) and int(a.street) == st and a.action in (Action.RAISE_HALF, Action.RAISE_POT) for a in obs.action_history) else 0.0
    pct = bot_core.preflop_percentile(eq) if st == 0 else eq
    f = np.zeros(NF, dtype=np.float32)
    f[st] = 1.0
    f[4] = eq
    f[5] = pot / 80.0
    f[6] = c / 80.0
    f[7] = need
    f[8] = obs.my_stack / 80.0
    f[9] = obs.opp_stack / 80.0
    f[10] = 1.0 if obs.is_button else 0.0
    f[11] = obs.raises_this_street / 4.0
    f[12] = obs.my_bet / 80.0
    f[13] = obs.opp_bet / 80.0
    f[14] = facing
    f[15] = agg
    f[16] = fold
    f[17] = pre
    f[18] = bs
    f[19] = opp_raised_street
    f[20] = opp_raises_hand / 4.0
    f[21] = 1.0 if (Action.RAISE_POT in obs.legal_actions) else 0.0
    f[22] = pct
    f[23] = min(1.0, mem["hands_seen"] / 500.0)
    f[24] = eq - need if c > 0 else eq - 0.5
    f[25] = (eq - bs) if facing else eq
    return f


def legal_mask(obs):
    return np.array([1.0 if a in obs.legal_actions else 0.0 for a in A_ORDER], dtype=np.float32)


MARGIN = 0.03
_W = None


def _load():
    global _W
    try:
        d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "policy_net.npz"))
        _W = {k: d[k] for k in d.files}
    except Exception:
        _W = None


_load()


def advise(obs, memory, action, eq):
    """Return the action to play: `action` unless the network predicts a clear gain."""
    if _W is None or eq is None:
        return action
    try:
        f = feats(obs, memory, eq)
        h = np.maximum(f @ _W["W1"] + _W["b1"], 0)
        h = np.maximum(h @ _W["W2"] + _W["b2"], 0)
        q = h @ _W["W3"] + _W["b3"]
        q = np.where(legal_mask(obs) > 0, q, -1e9)
        best = int(np.argmax(q))
        ai = A_ORDER.index(action)
        if best != ai and q[best] - q[ai] > MARGIN:
            return A_ORDER[best]
    except Exception:
        pass
    return action
