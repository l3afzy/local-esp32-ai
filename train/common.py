"""Shared text rules. Every function here has a twin in firmware/lib/tinyai/ —
change one, change both, or the device sees text the model never trained on."""

import glob
import os

# Token ids. 0 ends an answer, 1 separates question from answer,
# 2..96 are printable ASCII 32..126.
EOS, SEP = 0, 1
VOCAB_SIZE = 2 + 95
PROMPT_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789 '+-*/.,%^()=:")
MAX_Q = 64  # question chars kept (the tail, if longer)
MAX_A = 48  # answer chars, hard cap: short answers are the point
# Knowledge tiers. "small" is what the 2M int4 model knows; "max4mb" adds 91
# more survival facts for the 8.7M ternary model (the most parameters that fit
# a 4 MB ESP32); "large" adds a bigger Wikidata pull for 16 MB boards (parked).
# Order matters: fact indices follow it.
_BASE = ["data/facts.tsv", "data/facts_generated.tsv", "data/facts_survival.tsv"]
TIERS = {
    "small": _BASE + ["data/facts_wikidata.tsv"],
    "max4mb": _BASE + ["data/facts_survival_more.tsv", "data/facts_wikidata.tsv"],
    "large": _BASE + ["data/facts_survival_more.tsv", "data/large/facts_wikidata.tsv"],
}
FACT_FILES = "small"
# Facts about named things (people, films, places). The gate needs every word
# of one of their phrasings, so "obama" can't reach "michelle obama"; short
# names ("einstein") are listed as aliases only where they are unambiguous.
STRICT_FILES = {"facts_wikidata.tsv"}
# Advice (survival): people describe their situation, so the gate lets a
# question add one word of context ("how to survive in *extreme* heat").
LENIENT_FILES = {"facts_survival.tsv", "facts_survival_more.tsv"}


def encode_char(c):
    o = ord(c)
    if 32 <= o <= 126:
        return o - 30
    return 2  # space


def decode_tok(t):
    return "" if t < 2 else chr(t + 30)


def normalize(q):
    """Lowercase, drop unknown chars, collapse spaces, strip end punctuation."""
    out, prev_space = [], True
    for c in q.lower():
        if c not in PROMPT_CHARS:
            c = " "
        if c == " ":
            if prev_space:
                continue
            prev_space = True
        else:
            prev_space = False
        out.append(c)
    s = "".join(out).rstrip(" ?!.,")
    return s[-MAX_Q:].lstrip()


def encode_example(q, a):
    """Returns (tokens, n_prompt). Loss applies only after n_prompt."""
    p = [encode_char(c) for c in normalize(q)] + [SEP]
    return p + [encode_char(c) for c in a[:MAX_A]] + [EOS], len(p)


EVAL_FILES = {"small": ["data/eval.tsv"], "max4mb": ["data/eval.tsv", "data/eval_more.tsv"],
              "large": ["data/eval.tsv", "data/eval_more.tsv"]}


def load_eval(tier="small", paths=None):
    """Held-out [(question, expected answer)]. For a tier with several files, a
    question in a later file replaces the same question in an earlier one."""
    rows = {}
    for path in paths or EVAL_FILES[tier]:
        for line in open(path, encoding="utf-8"):
            if line.strip() and not line.startswith("#"):
                q, a = line.rstrip("\n").split("\t")
                rows.pop(q, None)
                rows[q] = a
    return list(rows.items())


def canonical(questions):
    """The one phrasing the model learns for a fact: the shortest. The gate
    maps every accepted phrasing to it, and on the device each prompt
    character costs a full forward pass, so shorter keys answer faster."""
    return min((normalize(q) for q in questions if not q.startswith("~")), key=lambda q: (len(q), q))


def load_facts(pattern=FACT_FILES, strict=None, lenient=None):
    """[(questions, answer)]. Any phrasing is accepted; the model is trained on
    canonical(questions) only. A phrasing starting with "~" is an alias: the
    gate accepts it but it is never the key. If `strict` / `lenient` are sets,
    the index of every fact from a STRICT_FILES / LENIENT_FILES file is added."""
    facts = []
    paths = TIERS[pattern] if pattern in TIERS else sorted(glob.glob(pattern))
    for path in paths:
        name = os.path.basename(path)
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            qs, a = line.split("\t")
            for flags, files in ((strict, STRICT_FILES), (lenient, LENIENT_FILES)):
                if flags is not None and name in files:
                    flags.add(len(facts))
            facts.append(([q.strip() for q in qs.split("|")], a.strip()))
    return facts
