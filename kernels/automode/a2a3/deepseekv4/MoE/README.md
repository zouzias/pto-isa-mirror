# MoE — DeepSeek-V4 Mixture-of-Experts (with hash routing)

Auto-mode A3 split of the `Gate` / `Expert` / `MoE` modules
([model.py:547](../../../../../deepseek/model.py),
[model.py:588](../../../../../deepseek/model.py),
[model.py:610](../../../../../deepseek/model.py)).

> The pre-existing top-level [MoE/](../../MoE/) and [MoEv2/](../../MoEv2/)
> families already cover the **score-based** routing pipeline
> (router_matmul → moe_topk → scatter → expert_ffn → gather). This
> `deepseekv4/MoE/` scaffold adds the **hash-routing** branch as a separate
> leaf and re-organizes the gate into per-stage leaves matching the
> DeepSeek-V4 model exactly.

## Reference

```python
# Gate.forward: model.py:565-585
scores = linear(x.float(), self.weight.float())       # -- gate_logits (cube, FP32!)
if score_func == "softmax":      scores = scores.softmax(dim=-1)        # -- gate_softmax (vector)
elif score_func == "sigmoid":    scores = scores.sigmoid()
else:                            scores = F.softplus(scores).sqrt()
original_scores = scores

if self.bias is not None:
    scores = scores + self.bias
if self.hash:                                          # -- gate_hash_routing (vector)
    indices = self.tid2eid[input_ids]                  #    pure gather by token id
else:                                                  # -- gate_score_topk (vector)
    indices = scores.topk(self.topk, dim=-1)[1]
weights = original_scores.gather(1, indices)
if score_func != "softmax":
    weights /= weights.sum(dim=-1, keepdim=True)
weights *= self.route_scale

# MoE.forward: model.py:630-645
counts = bincount(indices.flatten(), minlength=n_routed_experts).tolist()
# -- scatter (vector) reorders tokens by destination expert
for i in routed_expert_range:
    idx, top = torch.where(indices == i)
    y[idx] += expert(x[idx], weights[idx, top, None])  # -- expert_ffn (combined kernel)
y += self.shared_experts(x)                            # -- shared_expert_ffn (combined kernel)
# -- gather (vector) merges per-expert outputs back into token order
```

## Leaves

| Leaf | HW | Op | Notes |
|------|----|----|-------|
| [gate_logits/](gate_logits/) | cube | `scores = X @ W^T` (FP32 weight, FP32 input — see model.py:566) | dim=4096, n_routed_experts=8 |
| [gate_softmax/](gate_softmax/) | vector | softmax / sigmoid / sqrt(softplus) score activation (3 variants behind a build flag) | model.py:567-572 |
| [gate_score_topk/](gate_score_topk/) | vector | `scores + bias → topk → gather original_scores` | model.py:574-581 |
| [gate_hash_routing/](gate_hash_routing/) | vector | `indices = tid2eid[input_ids]` (pure gather; no topk) — taken when `layer_id < n_hash_layers` | model.py:559, 577-578 |
| [scatter/](scatter/) | vector | token → expert reorder via per-expert counts | mirrors [MoEv2/scatter/](../../MoEv2/scatter/) |
| [expert_ffn/](expert_ffn/) | hybrid (cube + vector inside one kernel) | SwiGLU FFN: `silu(W1·x) * W3·x → W2·(…)` with optional swiglu_limit clamp | mirrors [MoEv2/expert_ffn/](../../MoEv2/expert_ffn/); FP4 weight optional |
| [shared_expert_ffn/](shared_expert_ffn/) | hybrid | always-on expert (no routing); same FFN as `expert_ffn` but for a single expert with all tokens | model.py:628, 644 |
| [gather/](gather/) | vector | expert output → token-position recombine | mirrors [MoEv2/gather/](../../MoEv2/gather/) |

## Hash routing as its own leaf — rationale

The hash branch (model.py:559, 577-578) has a fundamentally different I/O
contract than the score branch:

- **Score branch:** consumes `scores [T, E_routed]`, calls topk along E,
  gathers from `original_scores` — vector reduce + sort.
- **Hash branch:** consumes `input_ids [T]` and a static lookup table
  `tid2eid [vocab, n_activated_experts]` — pure gather, no reduce, no sort.

Treating them as one kernel would force a branch on `self.hash` that the
auto-mode compiler cannot fold cleanly. Keeping them separate matches the
"one-by-one not fused" directive and lets each be tuned for its access
pattern.

## Family shape contract

Generated header: `build/generated_cases.h` (see [scripts/generate_cases.py](scripts/generate_cases.py)).
Case tuple: `T, DIM, INTER_DIM, N_ROUTED, N_ACTIVATED, VOCAB`.

Defaults from [ModelArgs](../../../../../deepseek/model.py):

- `dim=4096`, `moe_inter_dim=4096`, `n_routed_experts=8`,
  `n_activated_experts=2`, `n_shared_experts=1`, `vocab_size=129280`,
  `score_func="sqrtsoftplus"`, `route_scale=1.0`, `swiglu_limit=0`.

## Out of scope

- TP / EP sharding (`world_size > 1`, `dist.all_reduce`). Single-rank only
  in this scaffold.
- `expert_dtype="fp4"` weight loading (the `Expert.__init__` accepts FP4 via
  `torch.float4_e2m1fn_x2`); leaf `expert_ffn/` README notes this as an
  `Assumption` to test under.
