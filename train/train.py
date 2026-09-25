"""Train the tiny GPT on direct question -> answer pairs.

    python train/train.py                # several hours on a 4-core CPU
    python train/train.py --steps 300    # smoke test
    python train/train.py --resume       # continue from train/ckpt.pt.partial

The model sees one canonical key per fact: its shortest phrasing. On the
device the knowledge gate maps whatever the user typed to that key first, so
no capacity is spent on typos and rephrasings — it all goes into facts.

Loss is on answer characters only: the model never learns to produce a
preamble or a sign-off. The last part of training is quantization-aware,
so the int4 model on the chip answers like the float model here."""

import argparse
import math
import random
import time

import torch
import torch.nn.functional as F

from common import EOS, SEP, canonical, decode_tok, encode_char, encode_example, load_facts, normalize
from model import Config, TinyGPT


def make_batch(examples, cfg):
    T = min(cfg.ctx, max(len(t) for t, _ in examples) - 1)
    x = torch.zeros(len(examples), T, dtype=torch.long)
    y = torch.full((len(examples), T), -100, dtype=torch.long)
    for i, (toks, n_prompt) in enumerate(examples):
        toks = toks[: T + 1]
        x[i, : len(toks) - 1] = torch.tensor(toks[:-1])
        y[i, n_prompt - 1 : len(toks) - 1] = torch.tensor(toks[n_prompt:])
    return x, y


def batches(examples, size, rng):
    """Endless batches of similar-length examples: shuffle, sort each window
    of 50 batches by length, cut into batches, shuffle the batches. Padding
    to the longest example then wastes little compute."""
    while True:
        order = examples[:]
        rng.shuffle(order)
        window = size * 50
        out = []
        for i in range(0, len(order), window):
            chunk = sorted(order[i:i + window], key=lambda e: len(e[0]))
            out += [chunk[j:j + size] for j in range(0, len(chunk), size)]
        rng.shuffle(out)
        yield from out


@torch.no_grad()
def answer(model, q, cfg, max_new=48):
    toks = [encode_char(c) for c in normalize(q)] + [SEP]
    out = []
    for _ in range(max_new):
        if len(toks) >= cfg.ctx:
            break
        t = int(model(torch.tensor([toks]))[0, -1].argmax())
        if t in (EOS, SEP):
            break
        toks.append(t)
        out.append(decode_tok(t))
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="train/ckpt.pt")
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--qat-from", type=float, default=0.5, help="fraction of steps before QAT starts")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--resume", action="store_true", help="continue from <out>.partial")
    ap.add_argument("--check-every", type=int, default=2000, help="sampled exact-match check")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    facts = load_facts()
    examples = [encode_example(canonical(qs), a) for qs, a in facts]
    cfg = Config(dim=args.dim, layers=args.layers, hidden=3 * args.dim)
    too_long = [canonical(qs) for (qs, _), (t, _) in zip(facts, examples) if len(t) > cfg.ctx]
    assert not too_long, f"key + answer exceed the {cfg.ctx}-token context: {too_long[:3]}"
    model = TinyGPT(cfg)
    print(f"{len(facts)} facts, {sum(p.numel() for p in model.parameters()):,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    warm, qat_step = 200, int(args.steps * args.qat_from)
    partial, start = args.out + ".partial", 1
    if args.resume:
        ck = torch.load(partial, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        rng.setstate(ck["rng"])
        start = ck["step"] + 1
        print(f"resumed at step {start}")
    probe = random.Random(0).sample(facts, min(500, len(facts)))
    feed = batches(examples, min(args.batch, len(examples)), rng)
    t0 = time.time()
    for step in range(start, args.steps + 1):
        model.set_qat(step >= qat_step)
        if step == qat_step:
            print("quantization-aware training on")
        lr = args.lr * min(step / warm, 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        lr = max(lr, args.lr * 0.02)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = make_batch(next(feed), cfg)
        loss = F.cross_entropy(model(x).view(-1, cfg.vocab), y.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == args.steps:
            print(f"step {step:5d}  loss {loss.item():.4f}  lr {lr:.2e}  {time.time() - t0:.0f}s", flush=True)
        if step % args.check_every == 0 and step < args.steps:
            model.eval()
            ok = sum(answer(model, canonical(qs), cfg) == a for qs, a in probe)
            model.train()
            print(f"step {step:5d}  sampled exact match {ok}/{len(probe)}", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "rng": rng.getstate(),
                        "step": step}, partial)

    model.eval()
    wrong = [(canonical(qs), a) for qs, a in facts if answer(model, canonical(qs), cfg) != a]
    print(f"exact match (int4 weights): {len(facts) - len(wrong)}/{len(facts)}")
    for q, a in wrong[:20]:
        print(f"    {q!r} -> {answer(model, q, cfg)!r}, want {a!r}")
    torch.save({"cfg": cfg.__dict__, "model": model.state_dict()}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
