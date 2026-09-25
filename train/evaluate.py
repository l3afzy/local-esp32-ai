"""Score the exported model by running the real C engine (host build).

    make -C host && python train/evaluate.py

Checks:
  1. every question in facts.tsv (and its paraphrases) -> exact answer
  2. data/eval.tsv held-out paraphrases, refusals and arithmetic
  3. directness: no answer may exceed the hard cap or open with filler
  4. the gate index (built in Python) parses every phrasing exactly like the
     C gate does
"""

import argparse
import re
import subprocess
import sys

import gate_index
from common import MAX_A, load_facts, normalize

FILLER = re.compile(r"^(sure|well|so|okay|ok|the answer|i think|great question|it is|as an)\b", re.I)


def run(binary, model, questions, *flags):
    out = subprocess.run([binary, model, *flags], input="\n".join(questions) + "\n",
                         capture_output=True, text=True, check=True).stdout
    lines = out.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # only the final newline; empty answers stay as rows
    assert len(lines) == len(questions), f"{len(lines)} answers for {len(questions)} questions"
    return lines


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
    ap.add_argument("--eval", default="data/eval.tsv")
    ap.add_argument("--min-heldout", type=float, default=0.0,
                    help="exit 1 if held-out accuracy falls below this")
    args = ap.parse_args()

    phrasings = sorted({normalize(q) for qs, _ in load_facts() for q in qs})
    c_side = run(args.bin, args.model, phrasings, "--words")
    py_side = [" ".join([str(gate_index.QTYPES.index(gate_index.qtype(p)))] +
                        gate_index.content_words(p)) for p in phrasings]
    drift = [(p, c, py) for p, c, py in zip(phrasings, c_side, py_side) if c != py]
    print(f"gate index matches C parser: {len(phrasings) - len(drift)}/{len(phrasings)}")
    for p, c, py in drift[:10]:
        print(f"    {p!r}: C {c!r}, Python {py!r}")

    pairs = [(q, a) for qs, a in load_facts() for q in qs]
    got = run(args.bin, args.model, [q for q, _ in pairs])
    report("trained questions", [(q, a, g) for (q, a), g in zip(pairs, got)])

    held = [tuple(l.rstrip("\n").split("\t")) for l in open(args.eval) if l.strip() and not l.startswith("#")]
    got_h = run(args.bin, args.model, [q for q, _ in held])
    ok, n = report("held-out", [(q, a, g) for (q, a), g in zip(held, got_h)])

    bad = [g for g in got + got_h if len(g) > MAX_A or FILLER.match(g)]
    print(f"directness violations: {len(bad)}")
    for g in bad:
        print(f"    {g!r}")
    if bad or drift or ok / n < args.min_heldout:
        sys.exit(1)


if __name__ == "__main__":
    main()
