"""Play against the bot yourself, hand by hand, in the terminal.

    python tools/play_human.py [seed]

Keys:  f = fold   c = check/call   h = raise half pot   p = raise pot   q = quit
"""
import importlib.util, os, random, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from game import Action, AgentConfig, Game, make_deal, BIG_BLIND  # noqa: E402
from evaluator import card_to_str, category_name, omaha_score     # noqa: E402

SUIT = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}


def show(cards):
    out = []
    for c in cards:
        s = card_to_str(c)
        out.append(s[0] + SUIT[s[1]])
    return "  ".join(out) if out else "(none yet)"


def load_bot():
    spec = importlib.util.spec_from_file_location("agent_mod", os.path.join(ROOT, "agent.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def describe(rec, me_seat):
    who = "You" if rec.seat == me_seat else "Bot"
    name = {Action.FOLD: "folds", Action.CHECK_CALL: "checks/calls", Action.RAISE_HALF: "raises (half pot)",
            Action.RAISE_POT: "raises (pot)"}[rec.action]
    return f"  {who} {name}" + (f"  (+{rec.amount} chips, street bet {rec.street_total})" if rec.amount else "")


def main():
    seed = sys.argv[1] if len(sys.argv) > 1 else str(random.randrange(10**6))
    bot = load_bot()
    mem = {}
    bot.new_half(AgentConfig(), mem)
    me, botseat = 0, 1
    total = 0
    i = 0
    print("\nHeads-up Short Deck 5-card Omaha. You start every hand with 80 chips (40 big blinds).")
    print("Make your hand from EXACTLY 2 of your 5 cards + EXACTLY 3 of the 5 board cards.")
    print("Ranking: straight flush > four kind > FLUSH > full house > THREE OF A KIND > straight > two pair > pair\n")
    while True:
        deal = make_deal(seed, i)
        button = i % 2
        g = Game(deal, button_seat=button, hand_index=i)
        print("=" * 60)
        print(f"Hand {i+1}   You are the {'BUTTON (small blind, act first preflop)' if button == me else 'BIG BLIND'}")
        shown = 0
        street_seen = None
        while not g.is_terminal:
            seat = g.to_act
            obs = g.observation(seat)
            if street_seen != obs.street:
                street_seen = obs.street
                print(f"\n--- {obs.street.name} ---   Board: {show(g.deal.board[:len(obs.board)])}")
            if seat == botseat:
                a = bot.choose_action(obs, mem)
                g.apply(a)
                print(describe(g.history[-1], me))
                continue
            print(f"\n  Your cards: {show(obs.hole_cards)}")
            print(f"  Pot {obs.pot}   To call {obs.to_call}   Your stack {obs.my_stack}   Bot stack {obs.opp_stack}")
            opts = ["c=" + ("check" if obs.to_call == 0 else f"call {obs.to_call}")]
            if Action.FOLD in obs.legal_actions: opts.insert(0, "f=fold")
            if Action.RAISE_HALF in obs.legal_actions: opts.append(f"h=raise half (to {obs.raise_to[Action.RAISE_HALF]})")
            if Action.RAISE_POT in obs.legal_actions: opts.append(f"p=raise pot (to {obs.raise_to[Action.RAISE_POT]})")
            while True:
                k = input("  Your move [" + ", ".join(opts) + ", q=quit]: ").strip().lower()
                if k == "q":
                    print(f"\nFinished. Net result: {total/BIG_BLIND:+.1f} big blinds over {i} hands.")
                    return
                act = {"f": Action.FOLD, "c": Action.CHECK_CALL, "h": Action.RAISE_HALF, "p": Action.RAISE_POT}.get(k)
                if act in obs.legal_actions:
                    break
                print("  That move isn't allowed right now.")
            g.apply(act)
            print(describe(g.history[-1], me))
        res = g.result()
        net = res.nets[me]
        total += net
        if res.went_to_showdown:
            print(f"\n*** SHOWDOWN ***   Board: {show(deal.board)}")
            for s, nm in ((me, "You"), (botseat, "Bot")):
                sc = omaha_score(deal.hole_cards[s], deal.board)
                print(f"  {nm}: {show(deal.hole_cards[s])}  ->  {category_name(sc)}")
        print(f"\n>>> You {'won' if net > 0 else 'lost' if net < 0 else 'split'} {abs(net)} chips ({net/BIG_BLIND:+.1f} bb).   "
              f"Session total: {total/BIG_BLIND:+.1f} bb after {i+1} hands")
        bot.on_hand_end(g.hand_result(botseat), mem)
        i += 1


if __name__ == "__main__":
    main()
