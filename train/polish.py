"""Fix the last few facts a trained model gets wrong, without disturbing the rest.

    python train/polish.py                       # updates train/ckpt.pt in place

After training, a handful of the ~25K facts can still come out wrong on the
int4 weights. This keeps training at a small learning rate on batches where
those facts are oversampled, then re-checks every fact (so a fix can't break
another one unnoticed), and repeats until all are right or it gives up."""

import argparse
import random

import torch
import torch.nn.functional as F

from common import canonical, encode_example, load_facts
from model import Config, TinyGPT
from train import answer, batches, make_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="train/ckpt.pt")
    ap.add_argument("--out", default=None, help="default: overwrite --ckpt")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200, help="steps per round")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--copies", type=int, default=16, help="copies of each wrong fact per batch")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = Config(**ck["cfg"])
    model = TinyGPT(cfg)
    model.load_state_dict(ck["model"])
    model.set_qat(True)  # polish the int4 model the device runs
    facts = load_facts()
    keys = [canonical(qs) for qs, _ in facts]
    examples = [encode_example(k, a) for k, (_, a) in zip(keys, facts)]

    def wrong():
        model.eval()
        bad = [i for i, k in enumerate(keys) if answer(model, k, cfg) != facts[i][1]]
        model.train()
        return bad

    bad = wrong()
    print(f"before: {len(facts) - len(bad)}/{len(facts)} exact", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = random.Random(args.seed)
    feed = batches(examples, args.batch, rng)
    for r in range(args.rounds):
        if not bad:
            break
        focus = [examples[i] for i in bad] * args.copies
        for _ in range(args.steps):
            picked = rng.sample(focus, min(len(focus), args.batch // 2))
            x, y = make_batch(next(feed)[: args.batch - len(picked)] + picked, cfg)
            loss = F.cross_entropy(model(x).view(-1, cfg.vocab), y.view(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        bad = wrong()
        print(f"round {r + 1}: {len(facts) - len(bad)}/{len(facts)} exact", flush=True)
    model.eval()
    for i in bad[:20]:
        print(f"    {keys[i]!r} -> {answer(model, keys[i], cfg)!r}, want {facts[i][1]!r}")
    out = args.out or args.ckpt
    torch.save({"cfg": ck["cfg"], "model": model.state_dict()}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
