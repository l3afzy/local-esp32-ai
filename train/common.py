"""Shared text rules. Every function here has a twin in firmware/lib/tinyai/ —
change one, change both, or the device sees text the model never trained on."""

import random
import re

# Token ids. 0 ends an answer, 1 separates question from answer,
# 2..96 are printable ASCII 32..126.
EOS, SEP = 0, 1
VOCAB_SIZE = 2 + 95
PROMPT_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789 '+-*/.,%^()=")
MAX_Q = 64  # question chars kept (the tail, if longer)
MAX_A = 48  # answer chars, hard cap: short answers are the point


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


def load_facts(path):
    facts = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        qs, a = line.split("\t")
        facts.append(([q.strip() for q in qs.split("|")], a.strip()))
    return facts


# Noise that real people add. The model must look past it.
PREFIXES = ["", "", "", "hey ", "tell me ", "do you know ", "can you tell me ",
            "quick question ", "please tell me ", "i want to know ", "so ",
            "ok ", "i was wondering ", "yo "]
SUFFIXES = ["", "", "", " please", " exactly", " roughly", " approximately",
            " again", " quickly"]
CONTRACTIONS = [("what is", "what's"), ("what is", "whats"), ("who is", "who's"),
                ("how many", "how many"), ("do you", "do u"), ("you", "u")]
KEYS = "qwertyuiopasdfghjklzxcvbnm"


def typo(s, rng):
    if len(s) < 6:
        return s
    i = rng.randrange(1, len(s) - 1)
    op = rng.randrange(4)
    if op == 0:
        return s[:i] + s[i + 1:]                      # drop
    if op == 1:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]    # swap
    if op == 2:
        return s[:i] + s[i] + s[i:]                   # double
    return s[:i] + rng.choice(KEYS) + s[i + 1:]       # wrong key


def augment(q, rng):
    s = q
    if rng.random() < 0.3:
        a, b = rng.choice(CONTRACTIONS)
        s = s.replace(a, b, 1)
    if rng.random() < 0.15:
        s = re.sub(r"^(what|who|how|where|when|which) (is|are|was|does|do) ", "", s)
    s = rng.choice(PREFIXES) + s + rng.choice(SUFFIXES)
    if rng.random() < 0.5:
        s = s + rng.choice(["?", "??", "", "."])
    if rng.random() < 0.2:
        s = s.capitalize()
    n = rng.choice([0, 0, 0, 1, 1, 2])
    for _ in range(n):
        s = typo(s, rng)
    return s
