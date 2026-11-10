"""Practice opponents for in-process matches. make(name) -> object with the 3 bot functions."""
import random, types, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from game import Action
import bot_core
from game import Street


def _mk(choose, name):
    m = types.SimpleNamespace()
    m.new_half = lambda config, memory: memory.update(rng=random.Random(7), nprng=__import__("numpy").random.default_rng(7))
    m.choose_action = choose
    m.on_hand_end = lambda r, memory: None
    return m


def _rand(obs, mem): return mem["rng"].choice(obs.legal_actions)
def _station(obs, mem): return Action.CHECK_CALL
def _pot(obs, mem): return Action.RAISE_POT if Action.RAISE_POT in obs.legal_actions else Action.CHECK_CALL
def _half(obs, mem): return Action.RAISE_HALF if Action.RAISE_HALF in obs.legal_actions else Action.CHECK_CALL
def _foldy(obs, mem): return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD


def _eq(obs, mem, n=60):
    if obs.street == Street.PREFLOP:
        return bot_core.preflop_equity(obs.hole_cards)
    return bot_core.np_equity(obs.hole_cards, obs.board, 500, mem["nprng"])


def make_eqbot(call_margin, raise_eq, bet_eq, bluff=0.0):
    def f(obs, mem):
        eq = _eq(obs, mem)
        legal = obs.legal_actions
        c, pot = obs.to_call, obs.pot
        if c > 0:
            need = c / (pot + c)
            if eq < need + call_margin:
                return Action.FOLD
            if eq > raise_eq and Action.RAISE_POT in legal:
                return Action.RAISE_POT
            return Action.CHECK_CALL
        if eq > bet_eq and Action.RAISE_POT in legal:
            return Action.RAISE_POT
        if bluff and Action.RAISE_HALF in legal and mem["rng"].random() < bluff:
            return Action.RAISE_HALF
        return Action.CHECK_CALL
    return f


FACTORY = {
    "random": _rand, "station": _station, "pot": _pot, "half": _half, "foldy": _foldy,
    "tight": make_eqbot(0.05, 0.70, 0.64),
    "loose": make_eqbot(0.0, 0.62, 0.55, 0.2),
    "nit": make_eqbot(0.10, 0.75, 0.70),
    "maniac": make_eqbot(-0.05, 0.5, 0.5, 0.5),
}


def make(name):
    return _mk(FACTORY[name], name)


def _randagg(obs, mem):
    r = mem["rng"].random(); legal = obs.legal_actions
    if r < 0.45 and Action.RAISE_POT in legal: return Action.RAISE_POT
    if r < 0.60 and Action.RAISE_HALF in legal: return Action.RAISE_HALF
    if obs.to_call > 0 and r > 0.85: return Action.FOLD
    return Action.CHECK_CALL


def _overfold(obs, mem):
    # calls small bets, folds to anything big; bets only with strong hands
    eq = _eq(obs, mem)
    if obs.to_call > 0:
        return Action.CHECK_CALL if obs.to_call <= 0.3 * obs.pot and eq > .4 else Action.FOLD
    return Action.RAISE_HALF if eq > 0.6 and Action.RAISE_HALF in obs.legal_actions else Action.CHECK_CALL


FACTORY["randagg"] = _randagg
FACTORY["overfold"] = _overfold


# ---------------- more varied styles ----------------
def make_style(pre_raise=0.0, bet_eq=0.62, bluff=0.0, call_margin=0.03, raise_eq=0.72,
               trap=False, half_for_mid=False, bluff_river=None, wide_call=0.0, n=250):
    def f(obs, mem):
        legal = obs.legal_actions
        c, pot = obs.to_call, obs.pot
        r = mem["rng"].random
        if obs.street == Street.PREFLOP:
            eq = bot_core.preflop_equity(obs.hole_cards)
            pct = bot_core.preflop_percentile(eq)
            if pre_raise and pct >= 1 - pre_raise and Action.RAISE_POT in legal:
                return Action.RAISE_POT
            if c > 0:
                need = c / (pot + c)
                return Action.CHECK_CALL if eq > need + call_margin - wide_call else Action.FOLD
            return Action.CHECK_CALL
        eq = bot_core.np_equity(obs.hole_cards, obs.board, n, mem["nprng"])
        if c > 0:
            need = c / (pot + c)
            if trap and eq > 0.75 and Action.RAISE_POT in legal:
                return Action.RAISE_POT
            if eq > raise_eq and Action.RAISE_POT in legal:
                return Action.RAISE_POT
            if eq < need + call_margin - wide_call:
                if bluff and Action.RAISE_POT in legal and r() < bluff * 0.3:
                    return Action.RAISE_POT
                return Action.FOLD
            return Action.CHECK_CALL
        # nothing to call
        if trap and eq > 0.75:
            return Action.CHECK_CALL
        if eq > bet_eq:
            if half_for_mid and eq < 0.72 and Action.RAISE_HALF in legal:
                return Action.RAISE_HALF
            if Action.RAISE_POT in legal:
                return Action.RAISE_POT
        b = bluff
        if bluff_river is not None and obs.street == Street.RIVER:
            b = bluff_river
        if b and r() < b and Action.RAISE_HALF in legal:
            return Action.RAISE_HALF if r() < 0.6 else Action.RAISE_POT
        return Action.CHECK_CALL
    return f


FACTORY["bluffer"] = make_style(bluff=0.4, call_margin=0.0, wide_call=0.05)
FACTORY["lag"] = make_style(pre_raise=0.7, bet_eq=0.46, bluff=0.25, call_margin=0.0, raise_eq=0.65, wide_call=0.05)
FACTORY["trapper"] = make_style(trap=True, bet_eq=0.8, raise_eq=0.8)
FACTORY["sizer"] = make_style(half_for_mid=True, bet_eq=0.55, bluff=0.1)
FACTORY["pre3bet"] = make_style(pre_raise=0.95, bet_eq=0.7, call_margin=0.05)
FACTORY["rivbluff"] = make_style(bluff=0.0, bluff_river=0.45, bet_eq=0.6)
FACTORY["valuestation"] = make_style(bet_eq=0.62, call_margin=-0.5, raise_eq=0.99)


def _tilt(obs, mem):
    # alternates style every 100 hands: maniac then nit
    phase = (obs.hand_index // 100) % 2
    return FACTORY["maniac"](obs, mem) if phase == 0 else FACTORY["nit"](obs, mem)


FACTORY["tilt"] = _tilt


# ---------------- adaptive exploiters ----------------
def _mk_exploiter(fold_thr=0.40, call_thr=0.0):
    base = make_style(bet_eq=0.62, call_margin=0.03, raise_eq=0.72)

    def new_half(config, memory):
        memory.update(rng=random.Random(11), nprng=__import__("numpy").random.default_rng(11),
                      raised=False, my_raises=3.0, my_wins=1.0)

    def choose(obs, mem):
        folds = mem["my_wins"] / mem["my_raises"]
        legal = obs.legal_actions
        if folds > fold_thr:                       # opponent folds to our raises: bluff relentlessly
            if Action.RAISE_POT in legal and obs.street != Street.PREFLOP:
                mem["raised"] = True
                return Action.RAISE_POT if mem["rng"].random() < 0.5 else (Action.RAISE_HALF if Action.RAISE_HALF in legal else Action.RAISE_POT)
            if Action.RAISE_POT in legal and obs.street == Street.PREFLOP:
                mem["raised"] = True
                return Action.RAISE_POT
        a = base(obs, mem)
        if a in (Action.RAISE_HALF, Action.RAISE_POT):
            mem["raised"] = True
        return a

    def on_hand_end(res, mem):
        if mem.get("raised"):
            mem["my_raises"] += 1
            if (not res.went_to_showdown) and res.winner == "me":
                mem["my_wins"] += 1
        mem["raised"] = False
    return types.SimpleNamespace(new_half=new_half, choose_action=choose, on_hand_end=on_hand_end)


def _stealer(obs, mem):
    legal = obs.legal_actions
    if obs.to_call == 0 and Action.RAISE_HALF in legal:
        return Action.RAISE_HALF
    eq = _eq(obs, mem, 200)
    need = obs.to_call / (obs.pot + obs.to_call)
    return Action.CHECK_CALL if eq > need else Action.FOLD


_MAKE_OLD = make


def make(name):
    if name == "exploiter":
        return _mk_exploiter(0.40)
    if name == "exploiter2":
        return _mk_exploiter(0.30)
    return _MAKE_OLD(name)


FACTORY["stealer"] = _stealer
