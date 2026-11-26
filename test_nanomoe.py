"""Tiny checks. Run with `python test_nanomoe.py` or `pytest`."""
import os
import tempfile

import torch

import nanomoe as nm

VOCAB = "".join(chr(i) for i in range(32, 97))


def tiny(**kw):
    base = dict(vocab=VOCAB, n_embd=32, n_head=2, n_layer=2, block_size=16, kv_rank=8, rope_dim=4,
                n_routed=4, top_k=2, n_shared=1, expert_hidden=16, batch_size=4, device="cpu")
    base.update(kw)
    return nm.Config(**base)


def test_shapes_and_loss():
    torch.manual_seed(0)
    for attn in ("mla", "mha"):
        for ffn in ("moe", "dense"):
            m = nm.GPT(tiny(attn=attn, ffn=ffn, mtp=1))
            x = torch.randint(len(VOCAB), (3, 16))
            logits, loss = m(x, x)
            assert logits.shape == (3, 16, len(VOCAB)) and torch.isfinite(loss)


def test_active_params_match():
    # The dense baseline is as wide as top_k + n_shared experts: the same work per token.
    moe, dense = nm.GPT(tiny()), nm.GPT(tiny(ffn="dense"))
    # MoE stores (n_routed - top_k) extra experts per layer, plus a router
    assert nm.n_params(moe) - nm.n_params(dense) == 2 * ((4 - 2) * 3 * 32 * 16 + 32 * 4)
    assert abs(nm.active_params(moe) - nm.active_params(dense)) / nm.active_params(dense) < 0.02


def test_dispatch_matches_reference():
    torch.manual_seed(0)
    moe = nm.MoE(tiny(n_shared=0))
    torch.nn.init.normal_(moe.router.proj.weight, std=0.5)
    x = torch.randn(1, 10, 32)
    with torch.no_grad():
        fast = moe(x)[0]
        idx, w, _ = moe.router(x[0])
        slow = torch.stack([sum(w[t, j] * moe.experts[idx[t, j]](x[0, t]) for j in range(2)) for t in range(10)])
    assert (fast - slow).abs().max() < 1e-5


def test_aux_loss_bounds():
    N, E, k = 64, 8, 2
    even = nm.balance_loss(torch.full((N, E), 1 / E), torch.arange(N * k).view(N, k) % E, E)
    probs = torch.zeros(N, E)
    probs[:, :k] = 1 / k
    collapsed = nm.balance_loss(probs, torch.arange(k).expand(N, k), E)
    assert abs(even.item() - 1.0) < 1e-6 and abs(collapsed.item() - E / k) < 1e-6


def test_bias_balancing_fixes_a_skewed_router():
    torch.manual_seed(0)
    r = nm.Router(32, 8, 2)
    r.skew[0] = 2.0
    gen = torch.Generator().manual_seed(0)

    def max_load():
        idx = r(torch.randn(2048, 32, generator=gen))[0]
        return nm.load_stats(torch.bincount(idx.flatten(), minlength=8))["max_load"], idx

    before, _ = max_load()
    for _ in range(200):
        _, idx = max_load()
        r.update_bias(torch.bincount(idx.flatten(), minlength=8), 0.01)
    after, _ = max_load()
    assert before > 3.0 and after < 1.6
    assert not any(p is r.bias for p in r.parameters())  # the bias never gets a gradient


def test_mla_cache_and_causality():
    c = tiny()
    mla, mha = nm.MLA(c), nm.Attention(c)
    assert mla.cache_floats < mha.cache_floats
    x = torch.randn(1, 8, 32)
    x2 = x.clone()
    x2[0, -1] += 10
    with torch.no_grad():
        assert (mla(x2) - mla(x))[0, :-1].abs().max() < 1e-6


def test_train_smoke():
    with tempfile.TemporaryDirectory() as d:
        c = tiny(steps=30, out=os.path.join(d, "m.pt"), vocab="", batch_size=8, block_size=32)
        _, r = nm.train(c, log=False)
        assert r["val"] < 4.2 and os.path.exists(c.out)  # ln(65) = 4.17: below guessing after 30 steps


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
