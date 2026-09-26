"""Quantize a checkpoint and write the files the device needs.

    python train/export.py

Writes:
    firmware/data/model.bin        the model; the host build loads it and the firmware
                                   embeds it in flash (firmware/embed_model.py)

File layout, version 4 (int4 weights), 5 (ternary) or 6 (int2), little endian,
every block padded to 4 bytes:
    u32 magic "TAI2", version, vocab, ctx, dim, layers, heads, kv_heads,
        hidden, n_facts, n_known
    tok, pos      int8 [rows*cols] + f32 [rows] scales
    per layer:    n1 f32[dim], wq wk wv wo, n2 f32[dim], w1 w2 (quantized)
                  int4 = [rows*cols/2] bytes, two weights per byte, low
                  nibble first, stored +8; then f16 [rows*cols/32] scales
                  int2 = [rows*cols/4] bytes, four weights per byte, first in
                  the lowest 2 bits, stored +2 (value = stored - 1.5); then
                  f16 [rows*cols/32] scales
                  ternary = [rows*ceil(cols/5)] bytes, five weights per byte
                  in base 3, first in the lowest digit, stored +1; then
                  f16 [rows] scales
    norm          f32[dim]
    gate index    see train/gate_index.py: word dictionary, phrasings grouped
                  by fact (u8 count per fact), per phrasing its question type
                  (+0x80: every word must match; +0x40: one extra question
                  word is allowed), word count, length and word ids
    keys          each fact's canonical key (shortest phrasing), coded: u32
                  n_extra, u16 offsets and a u32-sized blob of extra words, then
                  u32 bytes of codes (see encode_keys)
    errata        u32 count, u16 fact indices (sorted), then their answers as
                  strings: facts the int4 model answers wrong, found by
                  running the C engine (host/tinyai) on every fact's key
"""

import argparse
import os
import struct
import sys

import numpy as np
import torch

import engine
import gate_index
from common import canonical, load_facts, normalize
from model import GROUP, quantize_int2, quantize_int4, quantize_ternary


def pad4(b):
    return b + b"\0" * (-len(b) % 4)


def q8(w):
    w = w.detach().float().numpy()
    scale = np.abs(w).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
    return pad4(q.tobytes()) + scale.astype("<f4").tobytes()


def q4(w):
    """int4 with the exact rounding QAT trained against."""
    rows, cols = w.shape
    g = w.detach().float().reshape(rows, cols // GROUP, GROUP)
    scale = (g.abs().amax(-1, keepdim=True) / 7).clamp(min=6.2e-5).half().float()
    q = torch.clamp(torch.round(g / scale), -7, 7).to(torch.int16).reshape(rows, cols)
    assert torch.equal((q.reshape(rows, cols // GROUP, GROUP) * scale).reshape(rows, cols),
                       quantize_int4(w.detach().float()))
    n = (q + 8).numpy().astype(np.uint8)
    packed = (n[:, 0::2] | (n[:, 1::2] << 4)).astype(np.uint8)
    return pad4(packed.tobytes()) + pad4(scale.reshape(-1).numpy().astype("<f2").tobytes())


def q2(w):
    """int2 with the exact rounding QAT trained against: 4 weights per byte,
    first in the lowest 2 bits, stored +2; then one fp16 scale per 32."""
    rows, cols = w.shape
    g = w.detach().float().reshape(rows, cols // GROUP, GROUP)
    scale = (g.abs().mean(-1, keepdim=True) * 1.2).clamp(min=6.2e-5).half().float()
    q = torch.clamp(torch.floor(g / scale), -2, 1)
    assert torch.equal(((q + 0.5) * scale).reshape(rows, cols), quantize_int2(w.detach().float()))
    u = (q + 2).to(torch.uint8).reshape(rows, cols // 4, 4).numpy()
    packed = (u[..., 0] | (u[..., 1] << 2) | (u[..., 2] << 4) | (u[..., 3] << 6)).astype(np.uint8)
    return pad4(packed.tobytes()) + pad4(scale.reshape(-1).numpy().astype("<f2").tobytes())


def qt(w):
    """Ternary: 5 weights per byte in base 3 (first weight in the lowest digit,
    stored +1), each row padded to whole bytes, then one fp16 scale per row."""
    rows, cols = w.shape
    w = w.detach().float()
    scale = w.abs().mean(-1, keepdim=True).clamp(min=6.2e-5).half().float()
    t = torch.clamp(torch.round(w / scale), -1, 1)
    assert torch.equal(t * scale, quantize_ternary(w))
    stride = -(-cols // 5)
    u = np.zeros((rows, stride * 5), np.uint8)
    u[:, :cols] = (t + 1).numpy().astype(np.uint8)
    u = u.reshape(rows, stride, 5).astype(np.uint16)
    packed = (u[..., 0] + 3 * u[..., 1] + 9 * u[..., 2] + 27 * u[..., 3] + 81 * u[..., 4]).astype(np.uint8)
    return pad4(packed.tobytes()) + pad4(scale.reshape(-1).numpy().astype("<f2").tobytes())


def f32(t):
    return t.detach().float().numpy().astype("<f4").tobytes()


def strings(items):
    blob = "\0".join(items).encode("ascii") + b"\0"
    return struct.pack("<I", len(blob)) + pad4(blob)


def encode_keys(keys, wid):
    """Keys (each fact's shortest phrasing, the model's prompt) as codes: a byte
    below 0x80 is a literal character; 0x80 | hi, lo is word hi << 8 | lo, from
    the gate's dictionary or, after it, from `extras` (other frequent words).
    A word gets a space before it unless it starts the key. Keys end in 0."""
    count = {}
    for k in keys:
        for t in k.split(" "):
            if t not in wid:
                count[t] = count.get(t, 0) + 1
    # a word worth coding: saves (len - 2) bytes each time, costs len + 3 once
    extras = sorted((t for t, n in count.items() if n * (len(t) - 2) > len(t) + 3),
                    key=lambda t: -count[t] * (len(t) - 2))
    code = dict(wid)
    for t in extras:
        code[t] = len(code)
    assert len(code) < 32768
    out = bytearray()
    for k in keys:
        for i, t in enumerate(k.split(" ")):
            c = code.get(t)
            if c is not None and len(t) > 2:
                out += bytes([0x80 | c >> 8, c & 0xFF])
            else:
                out += (b" " if i else b"") + t.encode("ascii")
        out += b"\0"
    return extras, bytes(out)


def decode_keys(c, wid, extras):
    """Mirror of tai_fact in tinyai.c, to check the codes round-trip. A code's
    second byte can be 0, so keys are parsed, not split on 0."""
    words = sorted(wid, key=wid.get) + extras
    keys, s, i = [], "", 0
    while i < len(c):
        if c[i] >= 0x80:
            s += (" " if s else "") + words[(c[i] & 0x7F) << 8 | c[i + 1]]
            i += 2
        elif c[i]:
            s += chr(c[i])
            i += 1
        else:
            keys.append(s)
            s, i = "", i + 1
    return keys


def errata(fixes):
    """[(fact index, answer)] -> the errata block (see the layout above)."""
    return (struct.pack("<I", len(fixes)) + pad4(struct.pack(f"<{len(fixes)}H", *(i for i, _ in fixes))) +
            strings([a for _, a in fixes]))


ask = engine.ask  # the C engine, on all CPU cores


def write(args, data):
    with open(args.bin, "wb") as f:
        f.write(data)
    if not args.header:
        return
    with open(args.header, "w") as f:
        f.write("// Generated by train/export.py. Do not edit.\n#pragma once\n#include <stdint.h>\n\n")
        f.write(f"static const unsigned int model_data_len = {len(data)};\n")
        f.write("static const uint8_t model_data[] __attribute__((aligned(4))) = {\n")
        for i in range(0, len(data), 32):
            f.write(",".join(str(x) for x in data[i:i + 32]) + ",\n")
        f.write("};\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="small", choices=["small", "max4mb", "large"],
                    help="fact set: small (2M int4), max4mb (8.7M ternary) or large (16 MB boards)")
    ap.add_argument("--ckpt", default="train/ckpt.pt")
    ap.add_argument("--bin", default="firmware/data/model.bin")
    ap.add_argument("--header", default=None, help="also write the model as a C array (optional; the firmware embeds the .bin)")
    ap.add_argument("--host", default="host/tinyai", help="C engine used to find errata")
    ap.add_argument("--no-check", action="store_true", help="skip the errata check")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    c, sd = ck["cfg"], ck["model"]
    strict, lenient = set(), set()
    facts = load_facts(args.tier, strict=strict, lenient=lenient)
    keys = [canonical(qs) for qs, _ in facts]
    known = {}
    for i, (qs, _) in enumerate(facts):
        for q in qs:
            known.setdefault(normalize(q), i)  # first fact wins a shared phrasing
    assert len(facts) < 65536

    kind = c.get("weights", "int4")
    qw, version = {"int4": (q4, 4), "ternary": (qt, 5), "int2": (q2, 6)}[kind]
    out = [struct.pack("<11I", 0x32494154, version, c["vocab"], c["ctx"], c["dim"], c["layers"],
                       c["heads"], c["kv_heads"], c["hidden"], len(facts), len(known)),
           q8(sd["tok.weight"]), q8(sd["pos.weight"])]
    for l in range(c["layers"]):
        p = f"blocks.{l}."
        out.append(f32(sd[p + "n1.w"]))
        out += [qw(sd[p + n + ".weight"]) for n in ("wq", "wk", "wv", "wo")]
        out.append(f32(sd[p + "n2.w"]))
        out += [qw(sd[p + n + ".weight"]) for n in ("w1", "w2")]
    out.append(f32(sd["norm.w"]))
    index, wid = gate_index.build(list(known.items()), len(facts), strict, lenient)
    out.append(index)
    extras, codes = encode_keys(keys, wid)
    assert decode_keys(codes, wid, extras) == keys
    ext = "\0".join(extras).encode("ascii") + b"\0"
    offs, o = [], 0
    for e in extras:
        offs.append(o)
        o += len(e) + 1
    assert o < 65536
    out += [struct.pack("<I", len(extras)), pad4(struct.pack(f"<{len(extras)}H", *offs)),
            struct.pack("<I", len(ext)), pad4(ext), struct.pack("<I", len(codes)), pad4(codes)]
    n_words = len(wid)
    base = b"".join(out)

    data = base + errata([])
    write(args, data)
    fixes = []
    if not args.no_check:
        if not os.path.exists(args.host):
            sys.exit(f"{args.host} not found: build it with `make -C host`, or pass --no-check")
        # the real C engine answers every fact's key, exactly as on the device
        got = ask(args.host, args.bin, keys)
        fixes = [(i, a) for i, ((_, a), g) in enumerate(zip(facts, got)) if g != a]
        if fixes:
            data = base + errata(fixes)
            write(args, data)
            still = [i for (i, a), g in zip(fixes, ask(args.host, args.bin, [keys[i] for i, _ in fixes])) if g != a]
            assert not still, f"errata did not take: {[keys[i] for i in still[:5]]}"
        print(f"C engine: {len(facts) - len(fixes)}/{len(facts)} facts exact from the model; "
              f"{len(fixes)} stored as errata")
        for i, a in fixes[:20]:
            print(f"    {keys[i]!r} -> {got[i]!r}, stored {a!r}")
    n_params = sum(v.numel() for v in sd.values())
    print(f"{n_params:,} params -> {len(data):,} bytes ({8 * len(data) / n_params:.2f} bits/param "
          f"incl. text), {len(facts)} facts, {len(known)} phrasings, {n_words} indexed words, "
          f"gate index {len(index):,} bytes")


if __name__ == "__main__":
    main()
