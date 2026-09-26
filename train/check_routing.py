"""Fast routing check without running the model: for every trained phrasing
and every eval question, does the gate (or the calculator) pick the right
fact? Catches ambiguous or colliding phrasings before hours of training.

    make -C host && python train/check_routing.py --model firmware/data/model.bin
"""

import argparse
import sys

import engine
from common import canonical, load_eval, load_facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="small", choices=["small", "large"],
                    help="which fact set: small (2M model, 4 MB boards) or large (13M, 16 MB ESP32-S3)")
    ap.add_argument("--bin", default="host/tinyai")
    ap.add_argument("--model", default="firmware/data/model.bin")
    ap.add_argument("--eval", nargs="*", default=None, help="default: the tier's eval files")
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()
    facts = load_facts(args.tier)
    key2ans = {canonical(qs): a for qs, a in facts}

    def route(qs):
        out = engine.ask(args.bin, args.model, qs, "--gate")
        return [o[1:] if o.startswith("=") else ("I don't know." if o == "-" else key2ans.get(o, "?"))
                for o in out]

    failed = False
    for name, rows in [("eval", load_eval(args.tier, args.eval)),
                       ("trained", [(q, a) for qs, a in facts for q in qs])]:
        got = route([q for q, _ in rows])
        bad = [(q, w, g) for (q, w), g in zip(rows, got) if g != w]
        refused = sum(g == "I don't know." for _, _, g in bad)
        print(f"{name}: {len(rows) - len(bad)}/{len(rows)} routed right "
              f"({len(bad) - refused} to a wrong answer, {refused} refused)")
        for q, w, g in bad[:args.show]:
            print(f"    {q!r}: want {w!r}, got {g!r}")
        failed |= bool(bad)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
