"""Score the exported model by running the real C engine (host build).

    make -C host && python train/evaluate.py

Checks:
  1. every question in facts.tsv (and its paraphrases) -> exact answer
  2. data/eval.tsv held-out paraphrases, refusals and arithmetic
  3. directness: no answer may exceed the hard cap or open with filler
"""

import argparse
import subprocess
import sys

from common import MAX_A, load_facts

FILLER = ("sure", "well", "so ", "the answer", "i think", "great question", "it is", "as an")


def run(binary, model, questions):
    out = subprocess.run([binary, model], input="\n".join(questions) + "\n",
                         capture_output=True, text=True, check=True).stdout
    return out.rstrip("\n").split("\n")


def report(name, rows):
    ok = sum(got == want for _, want, got in rows)
    print(f"{name}: {ok}/{len(rows)}")
    for q, want, got in rows:
        if got != want:
            print(f"    {q!r}: want {want!r}, got {got!r}")
    return ok, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="host/tinyai")
    ap.add_argument("--model", default="firmware/data/model.bin")
    ap.add_argument("--facts", default="data/facts.tsv")
    ap.add_argument("--eval", default="data/eval.tsv")
    ap.add_argument("--min-heldout", type=float, default=0.0,
                    help="exit 1 if held-out accuracy falls below this")
    args = ap.parse_args()

    pairs = [(q, a) for qs, a in load_facts(args.facts) for q in qs]
    got = run(args.bin, args.model, [q for q, _ in pairs])
    report("trained questions", [(q, a, g) for (q, a), g in zip(pairs, got)])

    held = [tuple(l.rstrip("\n").split("\t")) for l in open(args.eval) if l.strip() and not l.startswith("#")]
    got_h = run(args.bin, args.model, [q for q, _ in held])
    ok, n = report("held-out", [(q, a, g) for (q, a), g in zip(held, got_h)])

    bad = [g for g in got + got_h if len(g) > MAX_A or g.lower().startswith(FILLER)]
    print(f"directness violations: {len(bad)}")
    for g in bad:
        print(f"    {g!r}")
    if bad or ok / n < args.min_heldout:
        sys.exit(1)


if __name__ == "__main__":
    main()
