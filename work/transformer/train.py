"""
train.py — train the hand-built Transformer on a paper-like translation task:
date normalization.  Free-form human dates -> ISO 8601.

    "24th Jan 2001"    -> "2001-01-24"
    "March 3, 1998"    -> "1998-03-03"
    "07/06/2013"       -> "2013-07-06"  (interpreted D/M/Y)

This is genuine sequence-to-sequence translation with different source/target
lengths and vocabularies-in-spirit, exercising the encoder, the masked decoder
self-attention, and encoder-decoder cross-attention. Self-contained: dates are
generated procedurally, no dataset download.
"""
import math
import random
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformer import Transformer, causal_mask, run_verification

SEED = 1
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# --------------------------------------------------------------------------
# Data: procedural human dates -> ISO
# --------------------------------------------------------------------------
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
MON_ABBR = [m[:3] for m in MONTHS]

def ordinal(n):
    if 10 <= n % 100 <= 20: suf = "th"
    else: suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"

def make_example(return_fmt=False):
    y = random.randint(1950, 2025)
    mo = random.randint(1, 12)
    # pick a valid day
    dmax = [31,29,31,30,31,30,31,31,30,31,30,31][mo-1]
    d = random.randint(1, dmax)
    fmt = random.choice(range(5))
    if fmt == 0:   src = f"{ordinal(d)} {MON_ABBR[mo-1]} {y}"
    elif fmt == 1: src = f"{MONTHS[mo-1]} {d}, {y}"
    elif fmt == 2: src = f"{d:02d}/{mo:02d}/{y}"          # D/M/Y
    elif fmt == 3: src = f"{MON_ABBR[mo-1]} {ordinal(d)}, {y}"
    else:          src = f"{d} {MONTHS[mo-1]} {y}"
    tgt = f"{y:04d}-{mo:02d}-{d:02d}"
    if return_fmt:
        return src, tgt, fmt
    return src, tgt

# --------------------------------------------------------------------------
# Character-level vocab (shared, so embedding tying is meaningful — §3.4)
# --------------------------------------------------------------------------
PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
chars = sorted(set("".join(
    [c for _ in range(2000) for c in "".join(make_example())]) +
    "0123456789-/, abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
vocab = [PAD, BOS, EOS] + chars
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}
V = len(vocab)
SRC_LEN, TGT_LEN = 20, 12  # "YYYY-MM-DD" = 10 chars + BOS/EOS margin

def encode_src(s):
    ids = [stoi[c] for c in s][:SRC_LEN]
    return ids + [stoi[PAD]] * (SRC_LEN - len(ids))

def encode_tgt(s):
    ids = [stoi[BOS]] + [stoi[c] for c in s] + [stoi[EOS]]
    ids = ids[:TGT_LEN]
    return ids + [stoi[PAD]] * (TGT_LEN - len(ids))

def batch(n):
    S, T = [], []
    for _ in range(n):
        s, t = make_example()
        S.append(encode_src(s)); T.append(encode_tgt(t))
    return torch.tensor(S), torch.tensor(T)

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
device = "cpu"
model = Transformer(V, d_model=64, num_heads=4, d_ff=256,
                    num_layers=2, dropout=0.1, max_len=64).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"vocab={V}  params={n_params:,}")

opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
loss_fn = nn.CrossEntropyLoss(ignore_index=stoi[PAD], label_smoothing=0.1)  # §5.4

# §5.3 Noam warmup schedule: lr = d_model^-0.5 * min(step^-0.5, step*warmup^-1.5)
D_MODEL = 64
WARMUP = 800
def noam_lr(step):
    step = max(step, 1)
    return D_MODEL ** -0.5 * min(step ** -0.5, step * WARMUP ** -1.5)
sched = torch.optim.lr_scheduler.LambdaLR(opt, noam_lr)

def src_pad_mask(src):
    return (src != stoi[PAD]).unsqueeze(1).unsqueeze(2)   # [B,1,1,n]

STEPS = 6000
BATCH = 64
losses = []
lrs = []
model.train()
for step in range(1, STEPS + 1):
    src, tgt = batch(BATCH)
    src, tgt = src.to(device), tgt.to(device)
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    tm = causal_mask(tgt_in.size(1)).to(device)
    sm = src_pad_mask(src)
    logits = model(src, tgt_in, sm, tm)
    loss = loss_fn(logits.reshape(-1, V), tgt_out.reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    losses.append(loss.item()); lrs.append(sched.get_last_lr()[0])
    if step % 500 == 0 or step == 1:
        print(f"step {step:4d}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}")

# --------------------------------------------------------------------------
# Greedy decode
# --------------------------------------------------------------------------
@torch.no_grad()
def translate(s):
    model.eval()
    src = torch.tensor([encode_src(s)])
    sm = src_pad_mask(src)
    enc = model.encode(src, sm)
    ids = [stoi[BOS]]
    for _ in range(TGT_LEN):
        tgt = torch.tensor([ids])
        tm = causal_mask(tgt.size(1))
        dec = model.decode(tgt, enc, tm, sm)
        logits = dec @ model.embed.weight.T
        nxt = logits[0, -1].argmax().item()
        if nxt == stoi[EOS]: break
        ids.append(nxt)
    return "".join(itos[i] for i in ids[1:])

# --------------------------------------------------------------------------
# Evaluate exact-match accuracy on fresh samples
# --------------------------------------------------------------------------
random.seed(999)
N_EVAL = 500
correct = 0
samples = []
fmt_correct = {}
fmt_total = {}
for i in range(N_EVAL):
    s, t, fmt = make_example(return_fmt=True)
    pred = translate(s)
    ok = (pred == t)
    correct += ok
    fmt_total[fmt] = fmt_total.get(fmt, 0) + 1
    fmt_correct[fmt] = fmt_correct.get(fmt, 0) + int(ok)
    if i < 12:
        samples.append((s, t, pred, ok))
acc = correct / N_EVAL
print(f"\nExact-match accuracy on {N_EVAL} unseen dates: {acc*100:.1f}%")
FMT_NAMES = {0: "24th Jan 2001", 1: "March 3, 1998", 2: "05/08/2012 (D/M/Y)",
             3: "Jan 3rd, 1998", 4: "3 March 1998"}
print("\nPer-format accuracy:")
per_format = []
for f in sorted(fmt_total):
    a = fmt_correct[f] / fmt_total[f]
    per_format.append({"format": FMT_NAMES[f], "n": fmt_total[f], "acc": round(a, 4)})
    print(f"  {FMT_NAMES[f]:22s} {a*100:5.1f}%  (n={fmt_total[f]})")
print("\nSample translations (source -> prediction  [gold]):")
for s, t, p, ok in samples:
    print(f"  {'OK ' if ok else 'XX '} {s:24s} -> {p:12s}  [{t}]")

# --------------------------------------------------------------------------
# Loss curve (smoothed)
# --------------------------------------------------------------------------
def smooth(x, k=30):
    x = np.array(x)
    if len(x) < k: return x
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[k:] - c[:-k]) / k

plt.figure(figsize=(7, 3.2))
ax1 = plt.gca()
ax1.plot(losses, color="#cbd5e1", lw=0.8, label="raw loss")
sm_loss = smooth(losses, 40)
ax1.plot(range(len(sm_loss)), sm_loss, color="#0ea5e9", lw=1.8, label="smoothed loss")
ax1.set_xlabel("training step"); ax1.set_ylabel("cross-entropy loss")
ax1.grid(alpha=0.2)
ax2 = ax1.twinx()
ax2.plot(lrs, color="#f59e0b", lw=1.2, ls="--", label="lr (§5.3 warmup)")
ax2.set_ylabel("learning rate", color="#f59e0b")
ax2.tick_params(axis="y", labelcolor="#f59e0b")
lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=8, loc="upper right")
plt.title(f"Date-normalization Transformer — {n_params:,} params, {acc*100:.1f}% exact-match")
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=140)
print("\nsaved loss_curve.png")

# --------------------------------------------------------------------------
# Dump results as JSON for the blog page
# --------------------------------------------------------------------------
results = {
    "vocab_size": V, "params": n_params, "steps": STEPS,
    "final_loss": round(float(np.mean(losses[-50:])), 4),
    "accuracy": round(acc, 4),
    "warmup": WARMUP, "peak_lr": round(max(lrs), 5),
    "per_format": per_format,
    "samples": [{"src": s, "gold": t, "pred": p, "ok": bool(ok)} for s, t, p, ok in samples],
    "loss_first": round(losses[0], 4),
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved results.json")

# save checkpoint + vocab so analyze.py can reload without retraining
torch.save({"model": model.state_dict(), "stoi": stoi, "itos": itos,
            "SRC_LEN": SRC_LEN, "TGT_LEN": TGT_LEN, "V": V}, "checkpoint.pt")
print("saved checkpoint.pt")
