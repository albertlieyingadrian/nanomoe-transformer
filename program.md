# program.md

A short brief for running small experiments with `nanomoe.py`. Read it before a session. One idea per run, one line
per result in `results.tsv`.

## The question

At the same work per token (active parameters), does a Mixture-of-Experts model beat a dense one? And what keeps
its router honest: an auxiliary loss, DeepSeek-V3's selection-only bias, or nothing?

## Ground rules

- Change one thing per run. Keep `--seed` fixed unless you are measuring noise.
- Compare models at matched active parameters (`dense` vs `moe-*` in `--mode compare` does this for you). A model
  with more active parameters doing better is not news.
- The metric is validation loss (the last line, and the best along the curve: Tiny Shakespeare is 1 MB, so the
  bigger models start to overfit). For routing, the max load (1.0 = even, E = everything on one expert).
- Run at least two seeds before you believe a gap smaller than about 0.02.
- Budget: a CPU run at the default size takes a few minutes per 1,000 steps; shrink `--n_embd`, `--n_layer` or
  `--block_size` before adding anything.

## What to try

| knob | flag | what you learn |
|---|---|---|
| balancing | `--balance bias / aux / none` | which keeps the load even, and what each costs in loss |
| collapse | `--router_skew 3 --balance none` | a lopsided start stays lopsided without balancing; `bias` repairs it |
| bias step | `--bias_rate 0.001 / 0.01 / 0.05` | too slow never catches up; too fast makes the load flicker |
| aux weight | `--balance aux --aux_coef 0.001 / 0.01 / 0.1` | balance bought with loss |
| fine-grained | `--fine 4` | 4x the experts, each 1/4 as wide, top-8: same weights, more combinations |
| shared expert | `--n_shared 0 --top_k 3` | the same active parameters with no expert every token sees |
| capacity | `--capacity_factor 1.0 / 1.25` | how many tokens get dropped, and what it costs |
| MTP | `--mtp 1` | a second head on the token after next: does the main loss move? |
| attention | `--attn mha` | MLA caches far fewer numbers per token; does the loss notice? |
| total params | `--variants dense,dense-wide,moe-bias` | dense-wide has the MoE's total parameters, at 3x the work per token |

## Ideas past the flags

- Sigmoid router scores (DeepSeek-V3) instead of softmax.
- Expert-level ablation: zero one expert and measure the loss on the tokens it used to get.
- Token-choice vs expert-choice routing: let each expert pick its top tokens instead.
