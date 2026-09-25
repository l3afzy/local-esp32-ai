"""Shared text rules. Every function here has a twin in firmware/lib/tinyai/ —
change one, change both, or the device sees text the model never trained on."""

import glob

# Token ids. 0 ends an answer, 1 separates question from answer,
# 2..96 are printable ASCII 32..126.
EOS, SEP = 0, 1
VOCAB_SIZE = 2 + 95
PROMPT_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789 '+-*/.,%^()=:")
MAX_Q = 64  # question chars kept (the tail, if longer)
MAX_A = 48  # answer chars, hard cap: short answers are the point
FACT_FILES = "data/facts*.tsv"


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


def canonical(questions):
    """The one phrasing the model learns for a fact: the shortest. The gate
    maps every accepted phrasing to it, and on the device each prompt
    character costs a full forward pass, so shorter keys answer faster."""
    return min((normalize(q) for q in questions), key=lambda q: (len(q), q))


def load_facts(pattern=FACT_FILES):
    """[(questions, answer)]. Any phrasing is accepted; the model is trained on
    canonical(questions) only."""
    facts = []
    for path in sorted(glob.glob(pattern)):
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            qs, a = line.split("\t")
            facts.append(([q.strip() for q in qs.split("|")], a.strip()))
    return facts
