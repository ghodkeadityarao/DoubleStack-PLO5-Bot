# DoubleStack

A poker bot for heads-up Short Deck 5-card Pot-Limit Omaha, served as a small web server that a referee program calls once per decision.

**Game in short:** 36-card deck (6 to Ace), 5 hole cards, 5 board cards. A hand uses exactly 2 hole cards and exactly 3 board cards. A flush beats a full house and three of a kind beats a straight. Stacks are 80 chips (40 big blinds), blinds 1/2. Each decision is one of: fold, check/call, raise half pot, raise pot.

## How it works

- **Preflop:** looks up the hand's equity against a random hand in a precomputed table (`data/preflop_eq.json`).
- **Flop, turn, river:** estimates equity by simulating random opponent hands and board runouts with numpy, using a precomputed score table (`data/pt_table.npy`). It continues when equity beats the pot odds, `c / (pot + c)`, plus a margin, and raises with strong hands.
- **Opponent model:** learns during a match how strong the opponent's hands are when it bets (from showdowns) and how often it folds to raises, and adjusts.
- **Advisor:** a small neural network (`advisor.py`, `data/policy_net.npz`) predicts the expected chip result of each action. It overrides the rule-based move only when it predicts a clear gain.
- **Time:** it measures its own speed and sizes the simulations to stay within the time bank. Any failure falls back to a legal action.

## Files

| Path | Purpose |
|---|---|
| `agent.py` | Web server (routes used by the judge) |
| `bot_core.py` | Equity simulation, opponent model, decision rules |
| `advisor.py` | Neural-network advisor (numpy inference) |
| `data/` | Lookup tables and network weights used at runtime |
| `game.py`, `evaluator.py`, `judge_engine.py` | Game rules, hand scorer, local referee |
| `baseline-bots/` | Simple opponents (random, calling station, always half pot, always pot) |
| `test_agent.py`, `test_bot.py` | Tests for the server routes and the bot logic |
| `docker/`, `tools/check_build.sh`, `tools/smoke_test.py` | Container build and smoke test |
| `tools/` | Evaluation and training scripts (see below) |

## Setup

Python 3.10+.

```
pip install -r requirements-dev.txt
```

## Run and test

```
python -m pytest -q                          # unit tests
python agent.py                              # the bot, port 5008
python baseline-bots/baseline_random.py      # an opponent, port 5009 (second terminal)
python judge_engine.py --seed 1              # referee: plays a full match (third terminal)
./tools/check_build.sh                       # container build and smoke test (needs Docker)
```

The judge plays 1,000 deals twice with the seats swapped, so both bots hold the same cards. A match is a win only if the 95% confidence interval of the average result is entirely above zero; otherwise it is a draw.

## Evaluation tools

These play matches in-process, without servers:

```
python tools/selfplay.py agent tools/opponents.py:tight 1000 1   # one match
python tools/pool.py '' 600 1,2                                  # all practice opponents
python tools/fuzz.py 3000                                        # random states: moves must be legal
python tools/play_human.py                                       # play against the bot in the terminal
```

## Rebuilding the data files

```
python tools/gen_preflop.py                  # data/preflop_eq.json (about 10 minutes)
python tools/gen_pt_table.py                 # data/pt_table.npy
```

The advisor network is trained in rounds (training needs `pip install torch`; the bot itself does not):

```
python tools/collect_data.py runs/r1.npz                  # rule-based bot plays, with random exploration
python tools/train_policy.py runs/net1.npz runs/r1.npz
python tools/collect_data.py runs/r2.npz --eps 0.15 --policy runs/net1.npz
python tools/train_policy.py data/policy_net.npz runs/r1.npz runs/r2.npz
```

## Results and limits

Measured locally in duplicate matches of 1,000 deals:

- Wins against all four baseline bots (about +7 to +16 bb/deal) and against the 22 practice opponents in `tools/opponents.py` (mean about +7, smallest +1.5).
- It spends about 1 to 2 seconds of the 8-second per-half time bank, with no errors.
- Against strong bots that mix their bets, it is roughly even or behind. It loses most chips in large all-in pots and to small bets it folds to.
- The practice opponents are simple bots written for testing, so these numbers say little about how it compares with stronger bots.
