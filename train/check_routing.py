"""Fast routing check without running the model: for every trained phrasing
and every eval question, does the gate (or the calculator) pick the right
fact? Catches ambiguous or colliding phrasings before hours of training.

    make -C host && python train/check_routing.py --model firmware/data/model.bin
"""

import argparse
import subprocess
import sys

from common import canonical, load_facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="host/tinyai")
    ap.add_argument("--model", default="firmware/data/model.bin")
    ap.add_argument("--eval", default="data/eval.tsv")
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()
    facts = load_facts()
    key2ans = {canonical(qs): a for qs, a in facts}

    def route(qs):
        out = subprocess.run([args.bin, args.model, "--gate"], input="\n".join(qs) + "\n",
                             capture_output=True, text=True, check=True).stdout.split("\n")[:len(qs)]
        return [o[1:] if o.startswith("=") else ("I don't know." if o == "-" else key2ans.get(o, "?"))
                for o in out]

    failed = False
    for name, rows in [("eval", [tuple(l.rstrip("\n").split("\t")) for l in open(args.eval)
                                 if l.strip() and not l.startswith("#")]),
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
