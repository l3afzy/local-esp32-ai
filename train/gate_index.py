"""Builds the knowledge gate's index at export time.

Instead of storing every accepted phrasing as text and re-parsing all of them
for every question, the device gets:

    a sorted dictionary of the content words that appear in any phrasing
    per phrasing: its fact, question type, length, and content-word ids

so a query fuzzy-matches each of its words against the dictionary once and
then scores thousands of phrasings with integer compares.

The word rules below mirror firmware/lib/tinyai/gate.c (content_words and
qtype). train/evaluate.py checks the two agree on every phrasing.
"""

import struct

MAX_WORDS, MAX_WLEN = 16, 24

STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "do", "does", "did", "of", "in", "on",
    "at", "to", "for", "from", "by", "with", "and", "or", "what", "whats", "who", "whos", "how",
    "when", "where", "which", "why", "tell", "me", "please", "you", "u", "your", "can", "could",
    "would", "will", "i", "im", "want", "know", "wondering", "question", "quick", "hey", "hi",
    "yo", "ok", "so", "exactly", "roughly", "approximately", "about", "again", "quickly",
    "much", "many", "there", "it", "its", "that", "this", "we", "my", "have", "has", "called",
    "just", "like", "um", "uh", "hmm", "now", "actually", "really", "some", "any", "all",
    "should", "need", "needs", "must", "supposed", "ought", "make", "makes", "whens", "wheres",
    "hows", "whys", "gonna", "wanna", "take", "takes", "once", "after",
}
SYNONYMS = {
    "begin": "start", "began": "start", "begins": "start", "started": "start", "starts": "start",
    "ended": "end", "ends": "end", "finish": "end", "finished": "end", "biggest": "largest",
    "quickest": "fastest", "temp": "temperature", "temps": "temperature",
    "refrigerator": "fridge", "cooking": "cook", "cooked": "cook", "boiling": "boil",
    "boiled": "boil", "baking": "bake", "baked": "bake", "defrost": "thaw", "reboot": "restart",
    "hrs": "hours", "hr": "hours", "mins": "minutes", "stay": "last", "keep": "last",
    "replace": "change", "isnt": "not", "arent": "not", "doesnt": "not", "dont": "not",
    "cant": "not", "wont": "not", "opened": "open", "opening": "open", "spaghetti": "pasta",
    "noodles": "pasta", "detector": "alarm", "detectors": "alarms",
}
HOW = ["many", "much", "long", "far", "fast", "old", "big", "tall", "heavy", "hot", "cold",
       "deep", "high", "often", "smart"]
QTYPES = ["", "what", "who", "when", "where", "why", "how"] + [f"how {h}" for h in HOW]


def _alnum(c):
    return "a" <= c <= "z" or "0" <= c <= "9"


def content_words(s):
    words, i = [], 0
    while i < len(s) and len(words) < MAX_WORDS:
        while i < len(s) and not _alnum(s[i]):
            i += 1
        w = []
        while i < len(s) and (_alnum(s[i]) or s[i] == "'"):
            if s[i] != "'" and len(w) < MAX_WLEN - 1:
                w.append(s[i])
            i += 1
        w = "".join(w)
        if w and SYNONYMS.get(w, w) not in STOP:
            words.append(SYNONYMS.get(w, w))
    return words


def qtype(s):
    toks = [t for t in s.split(" ") if t]
    for i, raw in enumerate(toks):
        w = raw.replace("'", "")[: MAX_WLEN - 1]
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if w in ("what", "which") and nxt == "year":
            return "when"  # "what year did ww2 end" asks when
        if w in ("what", "whats", "which"):
            return "what"
        if w in ("who", "whos", "whose"):
            return "who"
        if w in ("when", "where", "why", "whens", "wheres", "whys"):
            return w.rstrip("s") if w != "whys" else "why"
        if w in ("how", "hows"):
            return f"how {nxt}" if nxt in HOW else "how"
    return ""


def _pad4(b):
    return b + b"\0" * (-len(b) % 4)


def build(known):
    """known: [(normalized phrasing, fact index)] -> index bytes (see the file layout in export.py)."""
    parsed = [(content_words(q), QTYPES.index(qtype(q)), len(q), f, q) for q, f in known]
    vocab = sorted({w for words, *_ in parsed for w in words})
    wid = {w: i for i, w in enumerate(vocab)}
    assert len(vocab) < 65536

    blob = "\0".join(vocab).encode() + b"\0"
    offsets, off = [], 0
    for w in vocab:
        offsets.append(off)
        off += len(w) + 1
    ids = [wid[w] for words, *_ in parsed for w in words]
    small_talk = "\0".join(q for words, _, _, _, q in parsed if not words).encode() + b"\0"

    out = [struct.pack("<II", len(vocab), len(blob)), _pad4(blob),
           struct.pack(f"<{len(vocab)}I", *offsets),
           _pad4(bytes(len(w) for w in vocab)),
           _pad4(struct.pack(f"<{len(parsed)}H", *(f for *_, f, _ in parsed))),
           _pad4(bytes(t for _, t, _, _, _ in parsed)),
           _pad4(bytes(len(words) for words, *_ in parsed)),
           _pad4(bytes(min(n, 255) for _, _, n, _, _ in parsed)),
           struct.pack("<I", len(ids)), _pad4(struct.pack(f"<{len(ids)}H", *ids)),
           struct.pack("<I", len(small_talk)), _pad4(small_talk)]
    return b"".join(out), len(vocab)
