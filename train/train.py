"""Train the tiny GPT on direct question -> answer pairs.

    python train/train.py                # ~15 min on a 4-core CPU
    python train/train.py --steps 300    # smoke test

The model sees one canonical question per fact. On the device the knowledge
gate maps whatever the user typed to that canonical question first, so no
capacity is spent on typos and rephrasings — it all goes into facts.

Loss is on answer characters only: the model never learns to produce a
preamble or a sign-off. The last part of training is quantization-aware,
so the int4 model on the chip answers like the float model here."""

import argparse
import math
import random
import time

import torch
import torch.nn.functional as F

from common import EOS, SEP, decode_tok, encode_char, encode_example, load_facts, normalize
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
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--qat-from", type=float, default=0.5, help="fraction of steps before QAT starts")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    facts = load_facts()
    examples = [encode_example(qs[0], a) for qs, a in facts]
    cfg = Config(dim=args.dim, layers=args.layers, hidden=3 * args.dim)
    model = TinyGPT(cfg)
    print(f"{len(facts)} facts, {sum(p.numel() for p in model.parameters()):,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    warm, qat_step = 200, int(args.steps * args.qat_from)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if step == qat_step:
            model.set_qat(True)
            print("quantization-aware training on")
        lr = args.lr * min(step / warm, 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        lr = max(lr, args.lr * 0.02)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = make_batch(rng.sample(examples, min(args.batch, len(examples))), cfg)
        loss = F.cross_entropy(model(x).view(-1, cfg.vocab), y.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == args.steps:
            print(f"step {step:5d}  loss {loss.item():.4f}  lr {lr:.2e}  {time.time() - t0:.0f}s", flush=True)

    model.eval()
    wrong = [(qs[0], a) for qs, a in facts if answer(model, qs[0], cfg) != a]
    print(f"exact match (int4 weights): {len(facts) - len(wrong)}/{len(facts)}")
    for q, a in wrong[:20]:
        print(f"    {q!r} -> {answer(model, q, cfg)!r}, want {a!r}")
    torch.save({"cfg": cfg.__dict__, "model": model.state_dict()}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
