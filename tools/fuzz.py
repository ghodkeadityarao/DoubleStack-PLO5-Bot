"""Fuzz test: random game states; the bot must always return a legal action quickly and never raise.

    python tools/fuzz.py [hands]
"""
import os, random, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from game import Game, make_deal, Action
import bot_core

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
rng = random.Random(99)
bad = 0; worst = 0.0; calls = 0; tot = 0.0
mem = {}
bot_core.init_memory(mem)
for i in range(N):
    g = Game(make_deal("fuzz", i), button_seat=i % 2, hand_index=i)
    # play random actions (biased to raise) until a random stop point; bot acts at every step for both seats
    while not g.is_terminal:
        seat = g.to_act
        obs = g.observation(seat)
        if rng.random() < 0.1:
            bot_core.init_memory(mem)           # also test fresh memory mid-hand
        t = time.perf_counter()
        try:
            a = bot_core.choose(obs, mem)
        except Exception as e:
            print("EXC", repr(e)); bad += 1; a = obs.legal_actions[0]
        dt = time.perf_counter() - t
        worst = max(worst, dt); tot += dt; calls += 1
        if a not in obs.legal_actions:
            print("ILLEGAL", a, obs.legal_actions); bad += 1; a = obs.legal_actions[0]
        # opponent-like chaos: sometimes override with random legal action to reach odd states
        if rng.random() < 0.6:
            a = rng.choice(obs.legal_actions)
        g.apply(a)
print(f"hands={N} decisions={calls} bad={bad} worst={worst*1000:.1f}ms avg={tot/calls*1000:.2f}ms")
