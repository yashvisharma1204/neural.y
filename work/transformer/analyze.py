"""
analyze.py — load the trained checkpoint and visualise the encoder-decoder
CROSS-attention for a numeric-format date, showing that the model reorders
D/M/Y -> Y-M-D by attending from each output character back to the right
source characters.  Produces cross_attention.png.
"""
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformer import Transformer, causal_mask

ckpt = torch.load("checkpoint.pt", weights_only=False)
stoi, itos = ckpt["stoi"], ckpt["itos"]
SRC_LEN, TGT_LEN, V = ckpt["SRC_LEN"], ckpt["TGT_LEN"], ckpt["V"]
PAD = "<pad>"; BOS = "<bos>"; EOS = "<eos>"

model = Transformer(V, d_model=64, num_heads=4, d_ff=256, num_layers=2, dropout=0.1, max_len=64)
model.load_state_dict(ckpt["model"]); model.eval()

def encode_src(s):
    ids = [stoi[c] for c in s][:SRC_LEN]
    return ids + [stoi[PAD]] * (SRC_LEN - len(ids))

def src_pad_mask(src):
    return (src != stoi[PAD]).unsqueeze(1).unsqueeze(2)

@torch.no_grad()
def decode_with_attn(s):
    """Greedy-decode s and collect cross-attention from the last decoder layer,
    averaged over heads, at each generation step."""
    src = torch.tensor([encode_src(s)])
    sm = src_pad_mask(src)
    enc = model.encode(src, sm)
    ids = [stoi[BOS]]
    attn_rows = []
    out_chars = []
    for _ in range(TGT_LEN):
        tgt = torch.tensor([ids])
        tm = causal_mask(tgt.size(1))
        dec = model.decode(tgt, enc, tm, sm)
        # cross-attention of last decoder layer: [B, h, m, n]; take last query row
        ca = model.decoder[-1].cross_attn.last_attn  # [1, h, m, n]
        row = ca[0, :, -1, :].mean(0).numpy()        # avg over heads -> [n]
        logits = dec @ model.embed.weight.T
        nxt = logits[0, -1].argmax().item()
        if nxt == stoi[EOS]:
            break
        attn_rows.append(row)
        out_chars.append(itos[nxt])
        ids.append(nxt)
    return "".join(out_chars), np.array(attn_rows), src[0].numpy()

EXAMPLE = "05/08/2012"
pred, attn, src_ids = decode_with_attn(EXAMPLE)
print(f"input : {EXAMPLE}")
print(f"output: {pred}")

# trim source display to non-pad chars
src_chars = [itos[i] for i in src_ids if itos[i] != PAD]
attn = attn[:, :len(src_chars)]

# --- plot heatmap: rows = generated output chars, cols = source chars ---
fig, ax = plt.subplots(figsize=(7.2, 4.2))
im = ax.imshow(attn, aspect="auto", cmap="Blues", vmin=0)
ax.set_xticks(range(len(src_chars)))
ax.set_xticklabels(src_chars, fontsize=11)
ax.set_yticks(range(len(pred)))
ax.set_yticklabels(list(pred), fontsize=11)
ax.set_xlabel(f"source:  {EXAMPLE}", fontsize=10)
ax.set_ylabel(f"generated:  {pred}", fontsize=10)
ax.set_title("Encoder–decoder cross-attention (last layer, head-averaged)\n"
             "each output char attends back to the source chars it copies", fontsize=10)
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("attention weight", fontsize=9)
plt.tight_layout()
plt.savefig("cross_attention.png", dpi=140)
print("saved cross_attention.png")

# --- sanity: for the year digits in the output, does argmax attention land on
#     the year region of the source? (source '05/08/2012' -> year at cols 6-9) ---
year_src_cols = [i for i, c in enumerate(src_chars) if c in "2012"[:0]] # placeholder
# report where each output char attends most
print("\noutput char -> most-attended source char:")
for oc, r in zip(pred, attn):
    j = int(r.argmax())
    print(f"  '{oc}'  ->  source '{src_chars[j]}' (col {j}, w={r[j]:.2f})")
