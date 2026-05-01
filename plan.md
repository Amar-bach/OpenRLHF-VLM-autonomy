# Plan: Implementing TinyLoRA in OpenRLHF

Based on the paper "Learning to Reason in 13 Parameters" (arXiv 2602.04118) by John X. Morris et al.

## Paper Summary

TinyLoRA replaces standard LoRA's two trainable matrices (A, B) with a much smaller parameterization:

```
W' = W + U * Sigma * (sum_i v_i * P_i) * V^T
```

- **U, Sigma, V**: from truncated SVD of W (rank r) -- **frozen**
- **P_i**: fixed random matrices in R^{r x r} -- **frozen**
- **v_i**: trainable scalars (i = 1..u) -- **only trainable params**
- **u**: "trainable projection dimension" (can be as low as 1)

Weight tying factor `n_tie` controls how many modules share a single `v` vector.
With full tying across all 7 modules x N layers, total params = just `u`.

The paper applies this to all 7 linear layers per transformer block (q, k, v, o, up, down, gate).
For 13 params on Qwen2.5-7B: likely u=1 with partial tying (e.g., 13 groups across ~28 layers x 7 modules = 196 modules).

Key finding: This only works well with RL (GRPO), not SFT. SFT needs 100-1000x more params for equivalent performance.

## Implementation Plan

### Phase 1: TinyLoRA Module (`openrlhf/models/tinylora.py`)

Create a new module that wraps a frozen `nn.Linear` with the TinyLoRA parameterization.

```python
class TinyLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int, u: int, v_shared: nn.Parameter = None):
        # 1. Compute truncated SVD of base_linear.weight (rank r)
        #    U_r, S_r, V_r = torch.svd_lowrank(W, q=rank)
        # 2. Register U_r, S_r, V_r as frozen buffers
        # 3. Generate u fixed random P_i matrices (r x r), register as frozen buffers
        # 4. If v_shared is provided, use it (weight tying); else create nn.Parameter(torch.zeros(u))
        # 5. Freeze base_linear.weight
```

The forward pass computes:
```python
def forward(self, x):
    base_out = F.linear(x, self.weight, self.bias)  # frozen base
    # delta = U @ diag(S) @ (sum_i v[i] * P[i]) @ V^T
    P_sum = sum(self.v[i] * self.P[i] for i in range(self.u))
    delta_weight = self.U @ torch.diag(self.S) @ P_sum @ self.V.T
    return base_out + F.linear(x, delta_weight)
```

### Phase 2: Model Integration (`openrlhf/models/actor.py`, `openrlhf/models/model.py`)

Add a new code path alongside the existing LoRA path. When `--tinylora` is enabled:

1. **Skip PEFT entirely** -- load the base model normally
2. Freeze all parameters
3. Apply `TinyLoRALinear` wrapper to target modules (q, k, v, o, up, down, gate projections)
4. Handle weight tying: group modules according to `n_tie` and share `v` parameters

Changes needed:
- `actor.py:102-116` -- add `elif tinylora:` branch after the existing `if lora_rank > 0:` block
- `model.py:116-128` -- same pattern for critic/reward models
- New helper function `apply_tinylora(model, rank, u, n_tie, seed)` that walks the model and replaces target linears

### Phase 3: CLI Arguments (`openrlhf/cli/train_ppo_ray.py`)

Add new arguments:
```
--tinylora                  # enable TinyLoRA mode
--tinylora_rank INT         # SVD truncation rank r (default: 16)
--tinylora_u INT            # trainable projection dim u (default: 1)
--tinylora_n_tie INT        # weight tying factor (default: 0 = full tying)
--tinylora_seed INT         # seed for fixed random P matrices (default: 42)
```

These get threaded through `strategy.args` to Actor/Critic constructors (same pattern as existing `lora_rank`).

### Phase 4: vLLM Weight Sync (`openrlhf/trainer/ray/ppo_actor.py`)

The existing `broadcast_to_vllm()` method merges LoRA into base weights before broadcasting. TinyLoRA needs the same pattern:

- `ppo_actor.py:322-406` -- extend `is_peft` logic to handle TinyLoRA
- Before broadcast: compute `delta_W` for each TinyLoRA module and add it to base weight
- After broadcast: subtract it back (or just recompute from `v` params which are tiny)
- Alternatively, add `merge()` and `unmerge()` methods to TinyLoRALinear that modify `weight` in-place

### Phase 5: Checkpoint Save/Load

TinyLoRA state is trivially small (13-200 scalars). Need to handle:

- **Save**: Only save the `v` parameters + metadata (rank, u, n_tie, seed for P matrices)
- **Load**: Recompute SVD and P matrices from the base model + seed, then load `v`
- Modify `save_model()` in `ppo_actor.py:530-538` to save TinyLoRA adapter separately
- The SVD computation is deterministic given the same base model, and P matrices are deterministic given the seed

### Phase 6: Training Script

Create `examples/scripts/slurm_train_ppo_qwen25_7b_gsm8k_tinylora.sh`:

Key hyperparameters from the paper (GRPO on GSM8K):
- No KL penalty (`--init_kl_coef 0`)
- 3 epochs (`--max_epochs 3`)
- 44 samples per problem (`--n_samples_per_prompt 44`)
- Batch size 64 (`--train_batch_size 64`)
- Max gen length 4096 (`--generate_max_len 4096`)
- LR sweep: {1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 1e-4, 2e-4}
- `--tinylora --tinylora_rank 16 --tinylora_u 1 --tinylora_n_tie 0`

Note: The paper uses GRPO, but this codebase uses PPO. The core idea (tiny trainable params + RL signal) should transfer, but results may differ. Consider also implementing a GRPO trainer if you want exact replication.

## Key Design Decisions

1. **SVD computation cost**: Computing truncated SVD for all ~196 linear layers at init is expensive but one-time. For Qwen2.5-7B with rank=16 this should take ~1-2 minutes. Cache the SVD results.

2. **Memory**: The SVD factors (U, S, V) are stored as frozen buffers. For rank r=16 and hidden_dim d=3584: each module stores U(d x r) + S(r) + V(d x r) + u * P(r x r) -- about 230KB per module in bf16. Across 196 modules that's ~45MB total overhead, negligible.

3. **Gradient flow**: Only `v` parameters have `requires_grad=True`. DeepSpeed ZeRO should handle this gracefully since there are so few params. May want to use ZeRO stage 0 or 1 since sharding 13 params is pointless.

4. **Weight tying implementation**: Create a dict mapping tie-group-id -> shared `nn.Parameter`. When applying TinyLoRA, modules in the same tie group get the same parameter tensor.

5. **Compatibility with existing LoRA path**: Keep the existing PEFT LoRA code completely untouched. TinyLoRA is a separate path triggered by `--tinylora`. The two should be mutually exclusive (`assert not (lora_rank > 0 and tinylora)`).

## Files to Modify

| File | Change |
|------|--------|
| `openrlhf/models/tinylora.py` | **NEW** -- TinyLoRALinear module + apply_tinylora() helper |
| `openrlhf/models/actor.py` | Add tinylora init path (~10 lines) |
| `openrlhf/models/model.py` | Add tinylora init path for critic/reward (~10 lines) |
| `openrlhf/cli/train_ppo_ray.py` | Add CLI args (~5 lines) |
| `openrlhf/trainer/ray/ppo_actor.py` | Extend broadcast_to_vllm merge/unmerge + save logic |
| `examples/scripts/slurm_train_ppo_qwen25_7b_gsm8k_tinylora.sh` | **NEW** -- training script |

## Open Questions

1. **GRPO vs PPO**: The paper uses GRPO. This codebase has PPO. Should we add a GRPO trainer, or test with PPO first? PPO should also work since the key insight is about RL vs SFT, not the specific RL algorithm.

2. **Random matrix distribution for P_i**: The paper says "fixed random" but doesn't specify the distribution. Gaussian N(0, 1/r) is a reasonable default (preserves scale). Need to verify or experiment.

3. **SVD vs random init for U, V**: The paper uses SVD of the pretrained weight. An alternative (used in LoRA-XS) is random orthogonal matrices. SVD is more principled but slower to initialize.

4. **Critic model**: Should the critic also use TinyLoRA, or full fine-tuning / standard LoRA? The paper focuses on the policy; the critic is separate. Recommend standard LoRA or full FT for the critic.
