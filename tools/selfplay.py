"""In-process duplicate match runner (no HTTP): a fast way to compare bots.

    python tools/selfplay.py <botA> <botB> [deals] [seed]

A bot is "agent" (agent.py in the repo root) or "tools/opponents.py:<name>".
Prints bb/deal for A with a 95% confidence interval and the time each bot used.
"""
import importlib.util, math, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from game import Action, AgentConfig, Game, make_deal, default_action, BIG_BLIND  # noqa: E402


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Bot:
    def __init__(self, mod):
        self.mod = mod
        self.mem = {}
        self.time = 0.0
        self.errors = 0

    def new_half(self):
        self.mem = {}
        self.mod.new_half(AgentConfig(), self.mem)

    def act(self, obs):
        t = time.perf_counter()
        try:
            a = self.mod.choose_action(obs, self.mem)
            if a not in obs.legal_actions:
                raise ValueError("illegal %r" % (a,))
        except Exception as e:  # noqa
            self.errors += 1
            if self.errors <= 3:
                import traceback; traceback.print_exc()
            a = default_action(obs.to_call)
        self.time += time.perf_counter() - t
        return a

    def end(self, res):
        t = time.perf_counter()
        try:
            self.mod.on_hand_end(res, self.mem)
        except Exception:
            self.errors += 1
            import traceback; traceback.print_exc()
        self.time += time.perf_counter() - t


def play_half(bots, seed, deals, swap, results):
    """bots[0] sits in seat `0 if not swap else 1`. results[i] += net of bot A."""
    seats = (0, 1) if not swap else (1, 0)  # seat of bot A, bot B
    a, b = bots
    a.new_half(); b.new_half()
    by_seat = {seats[0]: a, seats[1]: b}
    nets = []
    for i in range(deals):
        deal = make_deal(seed, i)
        g = Game(deal, button_seat=i % 2, hand_index=i)
        while not g.is_terminal:
            s = g.to_act
            g.apply(by_seat[s].act(g.observation(s)))
        for s, bot in by_seat.items():
            bot.end(g.hand_result(s))
        nets.append(g.result().nets[seats[0]])
    return nets


def match(modA, modB, deals=1000, seed=1):
    A, B = Bot(modA), Bot(modB)
    n1 = play_half((A, B), seed, deals, False, None)
    n2 = play_half((A, B), seed, deals, True, None)
    tot = [(x + y) / BIG_BLIND for x, y in zip(n1, n2)]
    m = sum(tot) / len(tot)
    var = sum((t - m) ** 2 for t in tot) / (len(tot) - 1)
    se = math.sqrt(var / len(tot))
    return m, se, A, B


def get(spec):
    if spec == "agent":
        return load_module(os.path.join(ROOT, "agent.py"), "agent_mod")
    path, name = spec.split(":")
    mod = load_module(os.path.join(ROOT, path), "opp_" + name)
    return mod.make(name)


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    deals = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    seed = sys.argv[4] if len(sys.argv) > 4 else "1"
    m, se, A, B = match(get(a), get(b), deals, seed)
    lo, hi = m - 1.96 * se, m + 1.96 * se
    verdict = "A WINS" if lo > 0 else ("B WINS" if hi < 0 else "CHOP")
    print(f"{a} vs {b}: {m:+.3f} bb/deal  CI[{lo:+.3f},{hi:+.3f}]  {verdict}  "
          f"timeA={A.time:.1f}s timeB={B.time:.1f}s errA={A.errors} errB={B.errors}")
