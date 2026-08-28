"""
transformer.py — a paper-faithful encoder-decoder Transformer, hand-built.

Every layer here corresponds to a section of the "Attention Is All You Need"
blog series:
  MultiHeadAttention   -> §3.2 (study3.html)
  PositionwiseFFN      -> §3.3 (study4.html)
  token embedding+tie  -> §3.4 (study5.html)
  PositionalEncoding   -> §3.5 (study6.html)
  EncoderLayer         -> §3.1 encoder (study1.html)
  DecoderLayer         -> §3.1 decoder (study2.html)

Nothing calls nn.Transformer / nn.MultiheadAttention — all built from nn.Linear.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# §3.2  Scaled Dot-Product + Multi-Head Attention
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(Q, K, V, mask=None):
    """softmax(QK^T / sqrt(d_k)) V  — the core primitive from §3.2.1."""
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)      # [B, h, m, n]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return attn @ V, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.h, self.d_k = num_heads, d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q = self.W_q(q).view(B, -1, self.h, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(B, -1, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(B, -1, self.h, self.d_k).transpose(1, 2)
        out, attn = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# §3.3  Position-wise Feed-Forward Network
# ---------------------------------------------------------------------------
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


# ---------------------------------------------------------------------------
# §3.5  Positional Encoding  (fixed sinusoids, zero learned params)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# §3.1  Encoder / Decoder layers (post-norm, exactly as in the paper)
# ---------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        a = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(a))
        f = self.ffn(x)
        x = self.norm2(x + self.dropout(f))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)   # masked
        self.cross_attn = MultiHeadAttention(d_model, num_heads)  # cross
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, enc_out, tgt_mask, src_mask=None):
        a = self.self_attn(y, y, y, tgt_mask)
        y = self.norm1(y + self.dropout(a))
        c = self.cross_attn(y, enc_out, enc_out, src_mask)
        y = self.norm2(y + self.dropout(c))
        f = self.ffn(y)
        y = self.norm3(y + self.dropout(f))
        return y


# ---------------------------------------------------------------------------
# Full model with tied embeddings (§3.4)
# ---------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, d_ff=256,
                 num_layers=2, dropout=0.1, max_len=64):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)          # tied matrix E
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def _embed(self, ids):
        return self.pos(self.embed(ids) * math.sqrt(self.d_model))  # §3.4 scaling

    def encode(self, src, src_mask=None):
        x = self._embed(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, tgt_mask, src_mask=None):
        y = self._embed(tgt)
        for layer in self.decoder:
            y = layer(y, enc_out, tgt_mask, src_mask)
        return y

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc = self.encode(src, src_mask)
        dec = self.decode(tgt, enc, tgt_mask, src_mask)
        logits = dec @ self.embed.weight.T          # tied pre-softmax projection (§3.4)
        return logits


def causal_mask(size):
    """Lower-triangular mask — 1 where allowed (j<=i), 0 where masked (§3.2.3)."""
    return torch.tril(torch.ones(size, size)).bool().unsqueeze(0).unsqueeze(0)


# ===========================================================================
# NumPy forward-pass mirror — proves the math, no autograd.
# Loads weights straight out of the trained PyTorch module.
# ===========================================================================
def np_softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def np_layernorm(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


def np_linear(x, W, b):
    return x @ W.T + b


def np_mha(mod, q, k, v, mask=None):
    """NumPy re-implementation of MultiHeadAttention using the module's weights."""
    W = {n: p.detach().numpy() for n, p in mod.named_parameters()}
    h, d_k = mod.h, mod.d_k
    B = q.shape[0]
    Q = np_linear(q, W["W_q.weight"], W["W_q.bias"])
    K = np_linear(k, W["W_k.weight"], W["W_k.bias"])
    V = np_linear(v, W["W_v.weight"], W["W_v.bias"])
    def split(t):
        return t.reshape(B, -1, h, d_k).transpose(0, 2, 1, 3)
    Q, K, V = split(Q), split(K), split(V)
    scores = Q @ K.transpose(0, 1, 3, 2) / math.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    attn = np_softmax(scores, axis=-1)
    out = attn @ V
    out = out.transpose(0, 2, 1, 3).reshape(B, -1, h * d_k)
    return np_linear(out, W["W_o.weight"], W["W_o.bias"])


def np_ffn(mod, x):
    W = {n: p.detach().numpy() for n, p in mod.named_parameters()}
    h = np.maximum(0, np_linear(x, W["w1.weight"], W["w1.bias"]))   # ReLU
    return np_linear(h, W["w2.weight"], W["w2.bias"])


def np_encode(model, src):
    """NumPy mirror of the full encoder stack (dropout off = identity)."""
    E = model.embed.weight.detach().numpy()
    pe = model.pos.pe.detach().numpy()
    x = E[src] * math.sqrt(model.d_model)
    x = x + pe[:, : x.shape[1]]
    for layer in model.encoder:
        a = np_mha(layer.self_attn, x, x, x)
        g1 = layer.norm1.weight.detach().numpy(); b1 = layer.norm1.bias.detach().numpy()
        x = np_layernorm(x + a, g1, b1)
        f = np_ffn(layer.ffn, x)
        g2 = layer.norm2.weight.detach().numpy(); b2 = layer.norm2.bias.detach().numpy()
        x = np_layernorm(x + f, g2, b2)
    return x


# ===========================================================================
# idea 5 — verification: shape tables + causal mask, checked numerically.
# ===========================================================================
def run_verification(verbose=True):
    torch.manual_seed(0)
    d_model, heads, d_ff, layers, V = 64, 4, 256, 2, 30
    B, m, n = 2, 7, 9
    model = Transformer(V, d_model, heads, d_ff, layers).eval()

    checks = []

    # -- shape flow: MHA preserves [B, *, d_model] --
    with torch.no_grad():
        x = torch.randn(B, n, d_model)
        mha = MultiHeadAttention(d_model, heads).eval()
        out = mha(x, x, x)
        checks.append(("MHA output shape == input [B,n,d]", tuple(out.shape) == (B, n, d_model)))

        # -- FFN preserves shape --
        ffn = PositionwiseFFN(d_model, d_ff).eval()
        checks.append(("FFN output shape preserved", tuple(ffn(x).shape) == (B, n, d_model)))

        # -- positional encoding adds, keeps shape --
        pe = PositionalEncoding(d_model).eval()
        checks.append(("PosEnc preserves shape", tuple(pe(x).shape) == (B, n, d_model)))

        # -- cross-attention: Q from tgt(m), K/V from src(n) -> weights m x n --
        q = torch.randn(B, m, d_model)
        Bq = q.size(0)
        Qh = mha.W_q(q).view(Bq, -1, heads, d_model // heads).transpose(1, 2)
        Kh = mha.W_k(x).view(Bq, -1, heads, d_model // heads).transpose(1, 2)
        _, attn = scaled_dot_product_attention(Qh, Kh,
                    mha.W_v(x).view(Bq, -1, heads, d_model // heads).transpose(1, 2))
        checks.append(("cross-attn weights are m x n", tuple(attn.shape) == (B, heads, m, n)))

        # -- full model logits are [B, m, |V|] --
        src = torch.randint(0, V, (B, n))
        tgt = torch.randint(0, V, (B, m))
        logits = model(src, tgt, None, causal_mask(m))
        checks.append(("model logits == [B,m,|V|]", tuple(logits.shape) == (B, m, V)))

        # -- causal mask actually zeroes future positions --
        cm = causal_mask(m)
        s = torch.randn(1, heads, m, m)
        _, a2 = scaled_dot_product_attention(
            torch.randn(1, heads, m, d_model // heads),
            torch.randn(1, heads, m, d_model // heads),
            torch.randn(1, heads, m, d_model // heads), cm)
        upper = a2[0, 0].triu(1)                       # strictly-future entries
        checks.append(("causal mask: future weights == 0", torch.allclose(upper, torch.zeros_like(upper))))
        checks.append(("causal mask: rows still sum to 1", torch.allclose(a2.sum(-1), torch.ones(1, heads, m))))

        # -- PE relative-shift rotation identity (§3.5) --
        # PE(pos+k) should be a fixed linear (rotation) map of PE(pos), independent of pos.
        peb = PositionalEncoding(d_model).pe[0].numpy()   # [max_len, d]
        k_off = 5
        # for dim pair (2i,2i+1), rotation by omega_i * k
        d = d_model
        ok_rot = True
        for i in range(0, 6, 1):
            omega = 10000 ** (-2 * i / d)
            ck, sk = math.cos(omega * k_off), math.sin(omega * k_off)
            for pos in [3, 10, 20]:
                s0, c0 = peb[pos, 2 * i], peb[pos, 2 * i + 1]
                s1_pred = ck * s0 + sk * c0
                c1_pred = -sk * s0 + ck * c0
                s1_true, c1_true = peb[pos + k_off, 2 * i], peb[pos + k_off, 2 * i + 1]
                if not (abs(s1_pred - s1_true) < 1e-4 and abs(c1_pred - c1_true) < 1e-4):
                    ok_rot = False
        checks.append(("PE relative-shift is a pos-independent rotation", ok_rot))

        # -- NumPy forward mirror matches PyTorch encoder to ~1e-5 --
        model_eval = model.eval()
        torch_enc = model_eval.encode(src).detach().numpy()
        numpy_enc = np_encode(model_eval, src.numpy())
        max_diff = float(np.abs(torch_enc - numpy_enc).max())
        checks.append((f"NumPy encoder == PyTorch (max|Δ|={max_diff:.2e})", max_diff < 1e-4))

    if verbose:
        print("=" * 64)
        print("VERIFICATION — shape tables, causal mask, PE identity, NumPy mirror")
        print("=" * 64)
        for name, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
        print("=" * 64)
    assert all(ok for _, ok in checks), "a verification check failed!"
    return checks, max_diff


if __name__ == "__main__":
    run_verification()
