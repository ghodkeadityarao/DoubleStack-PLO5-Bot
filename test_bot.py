"""Tests for the bot logic (bot_core.py, advisor.py). Run: python -m pytest -q"""
import random

import numpy as np

import advisor
import bot_core
from evaluator import cards_from_str
from game import Action, Game, make_deal


def play_random_hands(n, seed="t"):
    """Bot sits in seat 0 against a random opponent; returns every (obs, action) the bot played."""
    rng = random.Random(seed)
    mem = {}
    bot_core.init_memory(mem)
    played = []
    for i in range(n):
        g = Game(make_deal(seed, i), button_seat=i % 2, hand_index=i)
        while not g.is_terminal:
            seat = g.to_act
            obs = g.observation(seat)
            if seat == 0:
                a = bot_core.choose(obs, mem)
                played.append((obs, a))
            else:
                a = rng.choice(obs.legal_actions)
            g.apply(a)
        bot_core.record_hand_end(g.hand_result(0), mem)
    return played


def test_bot_always_plays_a_legal_action():
    played = play_random_hands(150)
    assert played
    assert all(a in obs.legal_actions for obs, a in played)


def test_advisor_weights_are_loaded():
    assert advisor._W is not None
    assert advisor._W["W1"].shape[0] == advisor.NF


def test_advisor_without_weights_keeps_the_rule_based_action(monkeypatch):
    monkeypatch.setattr(advisor, "_W", None)
    obs, a = play_random_hands(5)[0]
    assert advisor.advise(obs, {}, a, 0.5) == a


def test_advisor_never_returns_an_illegal_action():
    mem = {}
    bot_core.init_memory(mem)
    for obs, a in play_random_hands(60):
        b = advisor.advise(obs, mem, a, 0.55)
        assert b in obs.legal_actions


def test_preflop_equity_is_a_probability_and_ignores_suit_names():
    hand = cards_from_str("Ah Kh Qd Jd Tc")
    swapped = cards_from_str("As Ks Qc Jc Td")       # same hand with suits renamed
    e = bot_core.preflop_equity(hand)
    assert 0.2 < e < 0.8
    assert abs(e - bot_core.preflop_equity(swapped)) < 1e-9


def test_vectorised_equity_agrees_with_the_exact_river_calculation():
    hole = cards_from_str("Ah Kh Qd Jd Tc")
    board = cards_from_str("7c 8d 9h Ks 6s")
    rng = np.random.default_rng(1)
    a = bot_core.np_equity(hole, board, 6000, rng)
    b = bot_core.river_equity(hole, board, 6000, rng)
    assert abs(a - b) < 0.04


def test_init_memory_clears_previous_state():
    mem = {"junk": 1}
    bot_core.init_memory(mem)
    assert "junk" not in mem and mem["decisions"] == 0
