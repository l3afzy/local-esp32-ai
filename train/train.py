"""Train the tiny GPT on direct question -> answer pairs.

    python train/train.py                # ~15 min on a 4-core CPU
    python train/train.py --steps 500    # smoke test

Loss is computed on answer characters only. The model never learns to
produce a question, a preamble or a sign-off — only the answer, then EOS."""

import argparse
import math
import random
import time

import torch
import torch.nn.functional as F

from common import EOS, SEP, augment, decode_tok, encode_char, encode_example, load_facts, normalize
from model import Config, TinyGPT


def batch(facts, cfg, rng, n):
    x = torch.zeros(n, cfg.ctx, dtype=torch.long)
    y = torch.full((n, cfg.ctx), -100, dtype=torch.long)
    for i in range(n):
        qs, a = rng.choice(facts)
        q = rng.choice(qs)
        if rng.random() < 0.7:
            q = augment(q, rng)
        toks, n_prompt = encode_example(q, a)
        toks = toks[: cfg.ctx + 1]
        x[i, : len(toks) - 1] = torch.tensor(toks[:-1])
        y[i, n_prompt - 1 : len(toks) - 1] = torch.tensor(toks[n_prompt:])
    return x, y


@torch.no_grad()
def answer(model, q, cfg, max_new=48):
    toks = [encode_char(c) for c in normalize(q)] + [SEP]
    out = []
    for _ in range(max_new):
        logits = model(torch.tensor([toks[-cfg.ctx:]]))[0, -1]
        t = int(logits.argmax())
        if t == EOS or len(toks) >= cfg.ctx:
            break
        toks.append(t)
        out.append(decode_tok(t))
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/facts.tsv")
    ap.add_argument("--out", default="train/ckpt.pt")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    facts = load_facts(args.data)
    cfg = Config(dim=args.dim, layers=args.layers, hidden=3 * args.dim)
    model = TinyGPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{len(facts)} facts, {n_params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    warm = 200
    t0 = time.time()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(step / warm, 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch(facts, cfg, rng, args.batch)
        loss = F.cross_entropy(model(x).view(-1, cfg.vocab), y.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == args.steps:
            print(f"step {step:5d}  loss {loss.item():.4f}  lr {lr:.2e}  {time.time() - t0:.0f}s", flush=True)

    model.eval()
    exact = sum(answer(model, qs[0], cfg) == a for qs, a in facts)
    print(f"exact match on canonical questions: {exact}/{len(facts)}")
    torch.save({"cfg": cfg.__dict__, "model": model.state_dict()}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
