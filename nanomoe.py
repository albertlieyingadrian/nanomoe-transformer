"""
nanomoe: the smallest readable DeepSeek-style Mixture-of-Experts transformer.

A dense transformer sends every token through the same feed-forward network,
like a restaurant with one kitchen. A Mixture-of-Experts (MoE) model has a food
court: many small kitchens (experts) and a host (the router) who sends each
token to the top_k kitchens that suit it best, plus one shared kitchen that
cooks for everyone. Many more weights in total; the same work per token.

    tokens -> embed -> [ attention (MLA) -> MoE ] x n_layer -> logits
                                             |
               router: softmax scores, top_k + 1 shared expert
               balance: aux loss, or a selection-only bias (DeepSeek-V3)

One file, PyTorch only. Train, compare, sample and look at the routing:

    python nanomoe.py --mode compare --steps 1000   # dense vs MoE at the same active parameters
    python nanomoe.py --mode train                  # train an MoE on Tiny Shakespeare
    python nanomoe.py --mode sample --out out/model.pt
    python nanomoe.py --mode heatmap --out out/model.pt
"""
import argparse
import json
import math
import os
import time
import urllib.request
from dataclasses import asdict, dataclass, fields, replace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    n_embd: int = 128            # width of every token's vector
    n_head: int = 4              # attention heads
    n_layer: int = 4             # transformer blocks
    block_size: int = 128        # context length in characters
    attn: str = "mla"            # "mla" (latent attention, small KV cache) or "mha" (plain)
    kv_rank: int = 32            # MLA: width of the cached key/value latent
    rope_dim: int = 16           # MLA: per-head width of the decoupled RoPE slice
    ffn: str = "moe"             # "moe" or "dense"
    n_routed: int = 8            # routed experts per MoE layer
    top_k: int = 2               # routed experts each token uses
    n_shared: int = 1            # experts every token uses (no routing)
    expert_hidden: int = 128     # hidden width of one expert
    dense_hidden: int = 0        # dense FFN width; 0 = (top_k + n_shared) * expert_hidden (same active params)
    fine: int = 1                # fine-grained split: m times more experts, each 1/m as wide (DeepSeekMoE)
    capacity_factor: float = 0.0  # 0 = no limit; else each expert takes at most cf * fair share, the rest drop
    balance: str = "bias"        # "bias" (aux-loss-free), "aux" (load-balancing loss) or "none"
    aux_coef: float = 0.01
    bias_rate: float = 0.01      # how far each router bias moves per step
    router_skew: float = 0.0     # fixed head start for expert 0 (to provoke routing collapse)
    mtp: int = 0                 # 1 = also predict the token after next (DeepSeek-V3 multi-token prediction)
    mtp_coef: float = 0.3
    dropout: float = 0.0         # residual dropout (0 = the course's model; 0.2 for long runs on 1 MB of text)
    batch_size: int = 32
    steps: int = 3000
    lr: float = 3e-3
    seed: int = 1337
    device: str = "auto"         # auto picks cuda, then mps, then cpu
    out: str = "out/model.pt"
    variants: str = "dense,moe-bias,moe-aux,moe-none"  # --mode compare
    vocab: str = ""              # filled in from the data; saved with the checkpoint


# ----------------------------------------------------------------------------- model

class RMSNorm(nn.Module):
    """Scale each vector to unit root-mean-square (LayerNorm without the mean and bias)."""
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


def rope(x):
    """Rotary position embedding on x (B, T, heads, D): rotate each feature pair by an
    angle proportional to the position, so q.k depends only on the distance."""
    T, D = x.shape[1], x.shape[-1]
    freqs = 1.0 / (10000 ** (torch.arange(0, D, 2, device=x.device).float() / D))
    ang = torch.arange(T, device=x.device).float()[:, None] * freqs
    cos, sin = ang.cos()[None, :, None], ang.sin()[None, :, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


def causal_softmax(s):
    T = s.shape[-1]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=s.device), 1)
    return F.softmax(s.masked_fill(mask, float("-inf")), -1)


class Attention(nn.Module):
    """Plain multi-head attention. A KV cache stores 2 * n_embd numbers per token."""
    def __init__(self, c):
        super().__init__()
        self.h = c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.cache_floats = 2 * c.n_embd

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = (t.view(B, T, self.h, -1).transpose(1, 2) for t in self.qkv(x).split(C, 2))
        att = causal_softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]))
        return self.proj((att @ v).transpose(1, 2).reshape(B, T, C))


class MLA(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2). Cache one small latent c_kv per
    token (a zip file) and rebuild every head's keys and values from it. Position
    lives in a separate RoPE slice: per-head for queries, one shared key for all
    heads, because rotating the latent itself would break the caching trick."""
    def __init__(self, c):
        super().__init__()
        self.h, self.d, self.r = c.n_head, c.n_embd // c.n_head, c.rope_dim
        self.w_dkv = nn.Linear(c.n_embd, c.kv_rank, bias=False)    # cached
        self.kv_norm = RMSNorm(c.kv_rank)
        self.w_uk = nn.Linear(c.kv_rank, c.n_embd, bias=False)
        self.w_uv = nn.Linear(c.kv_rank, c.n_embd, bias=False)
        self.w_kr = nn.Linear(c.n_embd, c.rope_dim, bias=False)    # cached, shared by heads
        self.w_q = nn.Linear(c.n_embd, c.n_head * (self.d + c.rope_dim), bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.cache_floats = c.kv_rank + c.rope_dim

    def scores(self, x):
        B, T, _ = x.shape
        H, D, R = self.h, self.d, self.r
        ckv = self.kv_norm(self.w_dkv(x))
        k = torch.cat([self.w_uk(ckv).view(B, T, H, D),
                       rope(self.w_kr(x).view(B, T, 1, R)).expand(B, T, H, R)], -1)
        v = self.w_uv(ckv).view(B, T, H, D)
        q = self.w_q(x).view(B, T, H, D + R)
        q = torch.cat([q[..., :D], rope(q[..., D:])], -1)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        return q @ k.transpose(-2, -1) / math.sqrt(D + R), v

    def forward(self, x):
        B, T, C = x.shape
        s, v = self.scores(x)
        return self.proj((causal_softmax(s) @ v).transpose(1, 2).reshape(B, T, C))


class Expert(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x)). The gate is a dimmer on every wire. 3*d*h weights."""
    def __init__(self, d, h):
        super().__init__()
        self.gate = nn.Linear(d, h, bias=False)
        self.up = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Router(nn.Module):
    """probs = softmax(W x); pick top_k of (probs + bias); weight by the chosen probs,
    renormalised. The bias only steers the choice and never gets a gradient."""
    def __init__(self, d, n, k):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(d, n, bias=False)
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("skew", torch.zeros(n))

    def forward(self, x):
        probs = F.softmax(self.proj(x) + self.skew, -1)
        idx = (probs + self.bias).topk(self.k, -1).indices
        w = probs.gather(-1, idx)
        if self.k > 1:
            w = w / w.sum(-1, keepdim=True)
        return idx, w, probs

    @torch.no_grad()
    def update_bias(self, counts, rate):
        """DeepSeek-V3's auxiliary-loss-free balancing: a thermostat. Busy experts'
        bias goes down a notch, idle ones' goes up."""
        counts = counts.float()
        self.bias += rate * torch.sign(counts.mean() - counts)


def balance_loss(probs, idx, n):
    """Switch / DeepSeek aux loss n * sum_i f_i * P_i: 1.0 when even, n/k when collapsed."""
    f = torch.bincount(idx.flatten(), minlength=n).float() / idx.numel()
    return n * (f * probs.mean(0)).sum()


class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.router = Router(c.n_embd, c.n_routed, c.top_k)
        self.experts = nn.ModuleList(Expert(c.n_embd, c.expert_hidden) for _ in range(c.n_routed))
        # n_shared experts that see every token are the same as one wider expert
        self.shared = Expert(c.n_embd, c.n_shared * c.expert_hidden) if c.n_shared else None
        self.aux_loss, self.counts, self.idx = torch.zeros(()), None, None

    def forward(self, x):
        B, T, d = x.shape
        flat = x.reshape(-1, d)
        N, E, k = flat.shape[0], self.c.n_routed, self.c.top_k
        idx, w, probs = self.router(flat)
        cf = self.c.capacity_factor
        cap = math.ceil(cf * N * k / E) if cf > 0 else N
        out = torch.zeros_like(flat)
        for e in range(E):
            tok, slot = (idx == e).nonzero(as_tuple=True)  # dispatch: who chose expert e
            tok, slot = tok[:cap], slot[:cap]               # capacity: first come, first served
            if len(tok):                                    # combine: add back, scaled by the gate
                out.index_add_(0, tok, w[tok, slot, None] * self.experts[e](flat[tok]))
        if self.shared is not None:
            out = out + self.shared(flat)
        self.aux_loss = balance_loss(probs, idx, E)
        self.counts = torch.bincount(idx.flatten(), minlength=E).detach()
        self.idx = idx.detach()
        return out.view(B, T, d)


def dense_width(c):
    return c.dense_hidden or (c.top_k + c.n_shared) * c.expert_hidden


def apply_fine(c):
    """DeepSeekMoE fine-grained experts: split each expert into m narrower ones and
    route to m times as many. Same weights, same active weights, far more combinations."""
    m = c.fine
    if m <= 1:
        return c
    return replace(c, n_routed=c.n_routed * m, top_k=c.top_k * m, n_shared=c.n_shared * m,
                   expert_hidden=c.expert_hidden // m, fine=1)


class Block(nn.Module):
    def __init__(self, c, ffn=None):
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.n_embd), RMSNorm(c.n_embd)
        self.attn = MLA(c) if c.attn == "mla" else Attention(c)
        self.ffn = MoE(c) if (ffn or c.ffn) == "moe" else Expert(c.n_embd, dense_width(c))
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.n1(x)))
        return x + self.drop(self.ffn(self.n2(x)))


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        c = apply_fine(c)
        self.c = c
        V = len(c.vocab)
        self.tok = nn.Embedding(V, c.n_embd)
        self.pos = nn.Embedding(c.block_size, c.n_embd) if c.attn == "mha" else None
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, V, bias=False)
        if c.mtp:  # depth-1 MTP: [final state at t ; embedding of token t+1] -> block -> token t+2
            self.mtp_norm_h, self.mtp_norm_e = RMSNorm(c.n_embd), RMSNorm(c.n_embd)
            self.mtp_mix = nn.Linear(2 * c.n_embd, c.n_embd, bias=False)
            self.mtp_block = Block(c, ffn="dense")
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.moes():
            m.router.skew[0] = c.router_skew

    def moes(self):
        return [b.ffn for b in self.blocks if isinstance(b.ffn, MoE)]

    def forward(self, idx, targets=None):
        x = self.tok(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        self.lm_loss = loss.detach()
        if self.c.ffn == "moe" and self.c.balance == "aux":
            loss = loss + self.c.aux_coef * sum(m.aux_loss for m in self.moes())
        if self.c.mtp:
            h = self.mtp_mix(torch.cat([self.mtp_norm_h(x[:, :-1]), self.mtp_norm_e(self.tok(targets[:, :-1]))], -1))
            mtp = F.cross_entropy(self.head(self.norm(self.mtp_block(h))).flatten(0, 1), targets[:, 1:].flatten())
            loss = loss + self.c.mtp_coef * mtp
        return logits, loss


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def active_params(model):
    """Weights one token uses: all but the routed experts it skipped (and the training-only MTP)."""
    c = model.c
    skipped = sum((c.n_routed - c.top_k) * n_params(m.experts[0]) for m in model.moes())
    mtp = sum(n_params(getattr(model, a)) for a in ("mtp_norm_h", "mtp_norm_e", "mtp_mix", "mtp_block")
              if hasattr(model, a))
    return n_params(model) - skipped - mtp


def load_stats(counts):
    """max_load: busiest expert's share * E (1 = even). entropy: normalised (1 = even)."""
    E = counts.numel()
    f = counts.float() / counts.sum().clamp(min=1)
    nz = f[f > 0]
    return {"max_load": (f.max() * E).item(), "entropy": (-(nz * nz.log()).sum() / math.log(E)).item(),
            "dead": int((f < 0.1 / E).sum())}


# ----------------------------------------------------------------------------- data

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def shakespeare(path="data/tinyshakespeare.txt"):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return open(path, encoding="utf-8").read()


def get_data(c):
    text = shakespeare()
    vocab = "".join(sorted(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    data = torch.tensor([stoi[ch] for ch in text])
    n = int(0.9 * len(data))
    return vocab, data[:n], data[n:]


def get_batch(data, c, gen, device):
    ix = torch.randint(len(data) - c.block_size - 1, (c.batch_size,), generator=gen)
    x = torch.stack([data[i:i + c.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + c.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ----------------------------------------------------------------------------- train

def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def evaluate(model, data, device, batches=20, seed=0):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    losses, counts = [], [0 for _ in model.moes()]
    for _ in range(batches):
        x, y = get_batch(data, model.c, gen, device)
        losses.append(F.cross_entropy(model(x)[0].flatten(0, 1), y.flatten()).item())
        counts = [c + m.counts.cpu() for c, m in zip(counts, model.moes(), strict=True)]
    model.train()
    return sum(losses) / len(losses), [load_stats(c) for c in counts]


def train(c, log=True):
    device = pick_device(c.device)
    torch.set_float32_matmul_precision("high")  # TF32 matmuls on recent GPUs; no effect on CPU
    torch.manual_seed(c.seed)
    vocab, tr, va = get_data(c)
    c = replace(c, vocab=vocab)
    model = GPT(c).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=0.01, betas=(0.9, 0.95))
    warm = max(1, c.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * (0.1 + 0.45 * (1 + math.cos(math.pi * min(s, c.steps) / c.steps))))
    gen, t0, curve = torch.Generator().manual_seed(c.seed), time.time(), []
    for step in range(c.steps + 1):
        if step % max(1, c.steps // 30) == 0 or step == c.steps:
            val, loads = evaluate(model, va, device, batches=50 if step == c.steps else 10)
            curve.append((step, round(val, 4)))
            if log and (step % max(1, c.steps // 10) == 0 or step == c.steps):
                ml = " ".join(f"{s['max_load']:.2f}" for s in loads)
                print(f"step {step:5d}  val {val:.3f}" + (f"  max load {ml}" if ml else "")
                      + f"  {time.time() - t0:.0f}s", flush=True)
        if step == c.steps:
            break
        x, y = get_batch(tr, c, gen, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if c.ffn == "moe" and c.balance == "bias":
            for m in model.moes():
                m.router.update_bias(m.counts, c.bias_rate)
    os.makedirs(os.path.dirname(c.out) or ".", exist_ok=True)
    torch.save({"config": asdict(c), "model": model.state_dict()}, c.out)
    return model, {"val": val, "loads": loads, "params": n_params(model), "active": active_params(model),
                   "best": min(v for _, v in curve), "seconds": round(time.time() - t0, 1), "curve": curve}


def variant(c, name):
    """dense = same active params as the MoE; moe-<balance>[-fine] = an MoE with that balancing."""
    if name == "dense":
        return replace(c, ffn="dense")
    if name == "dense-wide":  # same total params as the MoE
        return replace(c, ffn="dense", dense_hidden=(c.n_routed + c.n_shared) * c.expert_hidden)
    parts = name.split("-")
    return replace(c, ffn="moe", balance=parts[1], fine=4 if "fine" in parts else c.fine)


def compare(c):
    results = {}
    base = os.path.splitext(c.out)[0]
    for name in c.variants.split(","):
        print(f"--- {name}")
        _, r = train(replace(variant(c, name), out=f"{base}_{name}.pt"))
        results[name] = r
    print(f"\n{'variant':14s} {'total':>10s} {'active':>10s} {'val loss':>9s} {'max load':>9s} {'time':>7s}")
    for name, r in results.items():
        ml = f"{sum(s['max_load'] for s in r['loads']) / len(r['loads']):.2f}" if r["loads"] else "-"
        print(f"{name:14s} {r['params']:10d} {r['active']:10d} {r['val']:9.3f} {ml:>9s} {r['seconds']:6.0f}s")
    with open(f"{base}_compare.json", "w") as f:
        json.dump({"config": asdict(c), "results": results}, f, indent=1)
    return results


# ----------------------------------------------------------------------------- look inside

CLASSES = ["space", "newline", "vowel", "consonant", "upper", "punct"]


def char_class(ch):
    if ch == " ":
        return 0
    if ch == "\n":
        return 1
    if ch.isupper():
        return 4
    if ch in "aeiou":
        return 2
    return 3 if ch.isalpha() else 5


def norm_mi(a, b):
    """Mutual information I(a; b) / H(b): how much the token decides its expert (0 = not at all, 1 = fully)."""
    nb = int(b.max()) + 1
    p = torch.bincount(a * nb + b, minlength=(int(a.max()) + 1) * nb).float().view(-1, nb)
    p = p / p.sum()
    pa, pb = p.sum(1, keepdim=True), p.sum(0, keepdim=True)
    nz = p > 0
    mi = (p[nz] * (p[nz] / (pa @ pb)[nz]).log()).sum()
    hb = -(pb[pb > 0] * pb[pb > 0].log()).sum()
    return (mi / hb).item()


@torch.no_grad()
def routing(model, data, device, batches=20):
    """Top-1 and top-k experts per token, per layer, on validation windows."""
    gen = torch.Generator().manual_seed(0)
    model.eval()
    toks, idxs = [], [[] for _ in model.moes()]
    for _ in range(batches):
        x, _ = get_batch(data, model.c, gen, device)
        model(x)
        toks.append(x.flatten().cpu())
        for i, m in enumerate(model.moes()):
            idxs[i].append(m.idx.cpu())
    return torch.cat(toks), [torch.cat(i) for i in idxs]


def heatmap(c, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = path or os.path.splitext(c.out)[0] + "_heatmap.png"
    ck = torch.load(c.out, map_location="cpu", weights_only=False)
    cc = Config(**ck["config"])
    device = pick_device(c.device)
    model = GPT(cc).to(device)
    model.load_state_dict(ck["model"])
    _, _, va = get_data(cc)
    toks, idxs = routing(model, va, device)
    torch.manual_seed(0)
    _, fresh = routing(GPT(cc).to(device), va, device)
    cls = torch.tensor([char_class(cc.vocab[t]) for t in toks.tolist()])
    E = model.c.n_routed
    fig, axes = plt.subplots(1, len(idxs), figsize=(3.2 * len(idxs), 2.8), squeeze=False)
    report = []
    for i, (ax, idx) in enumerate(zip(axes[0], idxs, strict=True)):
        table = torch.zeros(len(CLASSES), E)
        for s in range(idx.shape[1]):
            table.index_put_((cls, idx[:, s]), torch.ones(len(cls)), accumulate=True)
        table = table / table.sum(1, keepdim=True)
        mi = norm_mi(toks, idx[:, 0])
        shuffled = norm_mi(toks, idx[torch.randperm(len(idx)), 0])
        untrained = norm_mi(toks, fresh[i][:, 0])
        report.append({"layer": i, "mi": round(mi, 3), "shuffled": round(shuffled, 3),
                       "untrained": round(untrained, 3), "table": table.tolist()})
        print(f"layer {i}: token -> expert {mi:.3f} of H(expert) "
              f"(shuffled {shuffled:.3f}, untrained model {untrained:.3f})")
        for k, name in enumerate(CLASSES):
            print(f"  {name:9s} " + " ".join(f"{v:.2f}" for v in table[k].tolist()))
        im = ax.imshow(table, cmap="viridis", vmin=0, aspect="auto")
        ax.set_yticks(range(len(CLASSES)), CLASSES if i == 0 else [""] * len(CLASSES))
        ax.set_xticks(range(E))
        ax.set_xlabel("expert")
        ax.set_title(f"layer {i}  (MI {mi:.2f})", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("share of each character class's routing slots, per expert", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")
    return report


@torch.no_grad()
def sample(c, prompt="\n", n=400, temperature=0.8):
    ck = torch.load(c.out, map_location="cpu", weights_only=False)
    cc = Config(**ck["config"])
    model = GPT(cc)
    model.load_state_dict(ck["model"])
    model.eval()
    stoi = {ch: i for i, ch in enumerate(cc.vocab)}
    idx = torch.tensor([[stoi[ch] for ch in prompt]])
    gen = torch.Generator().manual_seed(c.seed)
    for _ in range(n):
        logits = model(idx[:, -cc.block_size:])[0][:, -1] / temperature
        idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1, generator=gen)], 1)
    text = "".join(cc.vocab[i] for i in idx[0].tolist())
    print(text)
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="train", choices=["train", "compare", "sample", "heatmap"])
    for f in fields(Config):
        if f.name != "vocab":
            p.add_argument(f"--{f.name}", type=f.type, default=f.default)
    args = vars(p.parse_args())
    mode = args.pop("mode")
    c = Config(**args)
    if mode == "train":
        train(c)
    elif mode == "compare":
        compare(c)
    elif mode == "sample":
        sample(c)
    else:
        heatmap(c)


if __name__ == "__main__":
    main()
