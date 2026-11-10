<h1 align="center">DoubleStack</h1>

<p align="center">
A heads-up Short Deck 5-card Pot-Limit Omaha poker bot:<br>
Monte Carlo equity, a simple opponent model and a small neural-network advisor.
</p>

<p align="center">
<img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white">
<img alt="NumPy" src="https://img.shields.io/badge/numpy-inference-013243?logo=numpy&logoColor=white">
<img alt="FastAPI" src="https://img.shields.io/badge/server-FastAPI-009688?logo=fastapi&logoColor=white">
<img alt="Docker" src="https://img.shields.io/badge/container-Docker-2496ED?logo=docker&logoColor=white">
</p>

The bot runs as a small web server. A referee program (`judge_engine.py`) deals the cards, asks the bot for each move, enforces the rules and scores the match.

## The game

| | |
|---|---|
| Deck | 36 cards, 6 to Ace |
| Cards | 5 hole cards per player, 5 board cards |
| Hand | exactly 2 hole cards + exactly 3 board cards |
| Ranking changes | a flush beats a full house; three of a kind beats a straight |
| Stacks and blinds | 80 chips (40 big blinds), blinds 1 and 2, stacks reset every hand |
| Moves | fold, check/call, raise half pot, raise pot |

A match is two halves of 1,000 deals on the same cards with the seats swapped, so both bots hold the same hands. It counts as a win only if the 95% confidence interval of the average result is entirely above zero; otherwise it is a draw.

## How it decides

```
 game state
     |
     +-- preflop? ------> equity from a precomputed table of all starting hands
     |
     +-- otherwise ------> Monte Carlo equity (numpy, about 1.5 microseconds per sample)
                                |
                                v
              opponent model: how strong are its bets, how often does it fold
                                |
                                v
        rule-based move: continue if equity beats the pot odds c / (pot + c) plus a margin,
        raise with strong hands, occasional semi-bluffs, more steals against opponents that fold a lot
                                |
                                v
        neural-network advisor: override only when it predicts a clear gain
                                |
                                v
                        legal action (always)
```

- **Time:** the bot measures its own speed and sizes the simulations to stay inside the per-half time bank (it uses about 1 to 2 seconds of 8).
- **Safety:** any error falls back to a legal action, and the opponent model and advisor are optional layers.

## Quick start

Python 3.10 or newer.

```bash
git clone https://github.com/ghodkeadityarao/DoubleStack-PLO5-Bot.git
cd DoubleStack-PLO5-Bot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
```

Play a full match against a baseline bot (three terminals):

```bash
python agent.py                              # the bot, port 5008
python baseline-bots/baseline_random.py      # an opponent, port 5009
python judge_engine.py --seed 1              # the referee plays 2 x 1,000 deals
```

Example output:

```
agent1 = TwoStack / DoubleStack (http://localhost:5008)
agent2 = Baseline / RandomBot (http://localhost:5009)
agent1: +10.536 bb/deal  (std 22.89, stderr 0.724)
RESULT: AGENT1_WIN
```

The names in the first lines are set by `PARTICIPANT` and `AGENT_NAME` in `agent.py`.

Container check (needs Docker): `./tools/check_build.sh` builds the image with restricted resources and runs a smoke test.

## Project layout

```
agent.py              web server (routes used by the judge)
bot_core.py           equity simulation, opponent model, decision rules
advisor.py            neural-network advisor (numpy inference)
data/                 lookup tables and network weights used at runtime
game.py               game rules              evaluator.py     hand scoring
judge_engine.py       local referee           baseline-bots/   simple opponents
test_agent.py         server route tests      test_bot.py      bot logic tests
docker/               container recipes       tools/           scripts, see below
```

## Tools

These play matches in-process, without servers:

```bash
python tools/selfplay.py agent tools/opponents.py:tight 1000 1   # one duplicate match
python tools/pool.py '' 600 1,2                                  # all 22 practice opponents
python tools/fuzz.py 3000                                        # random states: moves must be legal
python tools/play_human.py                                       # play against the bot in the terminal
```

## Rebuilding the data files

```bash
python tools/gen_preflop.py        # data/preflop_eq.json (about 10 minutes)
python tools/gen_pt_table.py        # data/pt_table.npy (reproduces the committed file exactly)
```

The advisor network is trained in rounds. Training needs `pip install torch`; running the bot does not.

```bash
python tools/collect_data.py runs/r1.npz                       # rule-based bot plays, with random exploration
python tools/train_policy.py runs/net1.npz runs/r1.npz
python tools/collect_data.py runs/r2.npz --eps 0.15 --policy runs/net1.npz
python tools/train_policy.py data/policy_net.npz runs/r1.npz runs/r2.npz
```

## Results and limits

Measured locally in duplicate matches of 1,000 deals:

| Opponents | Result |
|---|---|
| The four baseline bots | wins, about +7 to +16 bb/deal |
| The 22 practice opponents in `tools/opponents.py` | wins against all, mean about +7, smallest +1.5 bb/deal |
| Time and errors | about 1 to 2 s of the 8 s bank per half, no errors |

Limits:

- Against strong bots that mix their bets, it is roughly even or behind. It loses most chips in large all-in pots and to small bets it folds to.
- The practice opponents are simple bots written for testing, so these numbers say little about how it compares with stronger bots.
- The network was trained only against those practice opponents.

## Credits

`game.py`, `evaluator.py`, `judge_engine.py`, `test_agent.py`, `baseline-bots/`, `docker/`, `tools/check_build.sh` and `tools/smoke_test.py` come from the starter kit supplied with the project. They are used as provided, with only comments reworded.
