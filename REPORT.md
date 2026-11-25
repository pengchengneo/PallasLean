# PallasLean Verification Report (Target: TPU v7x)

sglang-jax `b3e3e16`, sglang-torchtpu `1f79f6bfbb`, maxtext `55ed4cffb`, target `TPU v7x (TPU7x)`, 83.2 s total (Lean 4 certificate check `PallasLean/Certificates.lean`: **PROVED** in 59.3 s).


## sglang.kv_page_write

| status | obligation | detail |
|---|---|---|
| PROVED | `in-bounds[pages]` | blocks [16, 1, 1, 1] |
| PROVED | `in-bounds[kv]` | blocks [4096, 1, 1, 1] |
| PROVED | `in-bounds[kv_out]` | blocks [4096, 1, 1, 1] |
| PROVED | `prefetch-bounds[0]` | 2 reads |
| N/A | `parallel-race[kv_out]` | no parallel dims |
| N/A | `contiguity[kv_out]` | unconditional full overwrite: scatter semantics (Lean Pipeline.overwrite_is_scatter) |
| N/A | `coverage[kv_out]` | aliased to an input: unvisited blocks keep their value |
| PROVED | `termination` | bounded grid [16] |
| PROVED | `v7x-mosaic-lower` | TPU7x TC module @_kernel (vector.load=2) |
| PROVED | `v7x-vmem-budget` | VMEM 192.0 KiB / 64 MiB, SMEM 64 B / 1024 KiB (Lean PallasLean.fitsV7xMemory) |
| WARN | `unused-input[kv]` | block is DMA'd HBM->VMEM every step but never read; pass it with memory_space=pl.ANY |

## sglang.qk_norm_rope

| status | obligation | detail |
|---|---|---|
| PROVED | `in-bounds[qkv]` | blocks [2, 2] |
| PROVED | `in-bounds[cos_sin]` | blocks [2, 1] |
| PROVED | `in-bounds[q_w]` | blocks [1] |
| PROVED | `in-bounds[k_w]` | blocks [1] |
| PROVED | `in-bounds[q_out]` | blocks [2, 1] |
| PROVED | `in-bounds[k_out]` | blocks [2, 1] |
| N/A | `parallel-race[q_out]` | no parallel dims |
| N/A | `parallel-race[k_out]` | no parallel dims |
| N/A | `contiguity[q_out]` | unconditional full overwrite: scatter semantics (Lean Pipeline.overwrite_is_scatter) |
| PROVED | `coverage[q_out]` |  |
| N/A | `contiguity[k_out]` | unconditional full overwrite: scatter semantics (Lean Pipeline.overwrite_is_scatter) |
| PROVED | `coverage[k_out]` |  |
| PROVED | `termination` | bounded grid [2] |
| PROVED | `v7x-mosaic-lower` | TPU7x TC module @fused_qk_norm_rope (math.rsqrt=2, arith.extf=5, arith.truncf=2, vector.load=6) |
| PROVED | `v7x-vmem-budget` | VMEM 1664.5 KiB / 64 MiB, SMEM 0 B / 1024 KiB (Lean PallasLean.fitsV7xMemory) |
| PROVED | `C:rmsnorm-rope-exact` | q_out and k_out match RMSNorm + NeoX RoPE spec; V cols >= 768 untouched (Lean RoPERMSNorm) |

## sglang.kimi_attn_res_decode

| status | obligation | detail |
|---|---|---|
| PROVED | `in-bounds[prefix_a]` | blocks [1, 1] |
| PROVED | `in-bounds[prefix_b]` | blocks [1, 1] |
| PROVED | `in-bounds[bank]` | blocks [1, 1, 1] |
| PROVED | `in-bounds[cw]` | blocks [1, 1] |
| PROVED | `in-bounds[w]` | blocks [1, 1] |
| PROVED | `in-bounds[normed_out]` | blocks [1, 1] |
| PROVED | `in-bounds[prefix_out]` | blocks [1, 1] |
| N/A | `parallel-race[normed_out]` | no parallel dims |
| N/A | `parallel-race[prefix_out]` | no parallel dims |
| N/A | `contiguity[normed_out]` | unconditional full overwrite: scatter semantics (Lean Pipeline.overwrite_is_scatter) |
| PROVED | `coverage[normed_out]` |  |
| N/A | `contiguity[prefix_out]` | unconditional full overwrite: scatter semantics (Lean Pipeline.overwrite_is_scatter) |
| PROVED | `coverage[prefix_out]` |  |
| PROVED | `termination` | bounded grid [1] |
| PROVED | `v7x-mosaic-lower` | TPU7x TC module @kimi_attn_res_decode_nvb4 (math.exp=5, math.rsqrt=6, arith.extf=8, arith.truncf=3, vector.load=10) |
| PROVED | `v7x-vmem-budget` | VMEM 25.0 KiB / 64 MiB, SMEM 0 B / 1024 KiB (Lean PallasLean.fitsV7xMemory) |
| PROVED | `C:prefix-add` | prefix_out == prefix_a + prefix_b across all token rows and channels |
| PROVED | `C:bank-mixture` | normed_out mixes live bank rows [0, 4) + prefix and ignores dead bank rows [4, 8) (Lean kimi_attn_res_exact) |

## maxtext.ragged_mqa

| status | obligation | detail |
|---|---|---|
| PROVED | `in-bounds[q]` | blocks [8, 1, 1] |
| PROVED | `in-bounds[k]` | blocks [8, 16, 1] |
| PROVED | `in-bounds[v]` | blocks [8, 16, 1] |
| PROVED | `in-bounds[o]` | blocks [8, 1, 1] |
| PROVED | `in-bounds[m]` | blocks [8, 1, 1] |
| PROVED | `in-bounds[l]` | blocks [8, 1, 1] |
| PROVED | `prefetch-bounds[0]` | 3 reads |
| PROVED | `parallel-race[o]` | parallel dims [0] |
| PROVED | `parallel-race[m]` | parallel dims [0] |
| PROVED | `parallel-race[l]` | parallel dims [0] |
| PROVED | `contiguity[o]` |  |
| PROVED | `init-at-start[o]` |  |
| PROVED | `no-midrun-reset[o]` |  |
| PROVED | `coverage[o]` |  |
| PROVED | `contiguity[m]` |  |
| PROVED | `init-at-start[m]` |  |
| PROVED | `no-midrun-reset[m]` |  |
| PROVED | `coverage[m]` |  |
| PROVED | `contiguity[l]` |  |
| PROVED | `init-at-start[l]` |  |
| PROVED | `no-midrun-reset[l]` |  |
| PROVED | `coverage[l]` |  |
| PROVED | `termination` | bounded grid [8, 16] |
| PROVED | `v7x-mosaic-lower` | TPU7x TC module @ragged_flash_attention_kernel (tpu.matmul=2, math.exp=3, arith.extf=2, vector.load=12, scf.if=2) |
| PROVED | `v7x-vmem-budget` | VMEM 288.0 KiB / 64 MiB, SMEM 32 B / 1024 KiB (Lean PallasLean.fitsV7xMemory) |
| PROVED | `C:step[3, 5]` | l' and acc'*l' match the osState recurrence (rows [0, 7], cols [0, 127]) |
| PROVED | `C:step@init[3, 0]` | l' and acc'*l' match the osState recurrence (rows [0, 7], cols [0, 127]) |
| PROVED | `C:score[t=0]` | cut == scale*QK^T + mask on every grid point reaching it |
| PROVED | `C:skip-sound` | skipped tiles contain no kept (row, col) |

## maxtext.ragged_mqa[requires_len>=256]

| status | obligation | detail |
|---|---|---|
| PROVED | `in-bounds[q]` | blocks [8, 1, 1] |
| PROVED | `in-bounds[k]` | blocks [8, 16, 1] |
| PROVED | `in-bounds[v]` | blocks [8, 16, 1] |
| PROVED | `in-bounds[o]` | blocks [8, 1, 1] |
| PROVED | `in-bounds[m]` | blocks [8, 1, 1] |
| PROVED | `in-bounds[l]` | blocks [8, 1, 1] |
| PROVED | `prefetch-bounds[0]` | 3 reads |
| PROVED | `parallel-race[o]` | parallel dims [0] |
| PROVED | `parallel-race[m]` | parallel dims [0] |
| PROVED | `parallel-race[l]` | parallel dims [0] |
| PROVED | `contiguity[o]` |  |
| PROVED | `init-at-start[o]` |  |
| PROVED | `no-midrun-reset[o]` |  |
| PROVED | `coverage[o]` |  |
| PROVED | `contiguity[m]` |  |
| PROVED | `init-at-start[m]` |  |
| PROVED | `no-midrun-reset[m]` |  |
| PROVED | `coverage[m]` |  |
| PROVED | `contiguity[l]` |  |
| PROVED | `init-at-start[l]` |  |
| PROVED | `no-midrun-reset[l]` |  |
| PROVED | `coverage[l]` |  |
| PROVED | `termination` | bounded grid [8, 16] |
| PROVED | `v7x-mosaic-lower` | TPU7x TC module @ragged_flash_attention_kernel (tpu.matmul=2, math.exp=3, arith.extf=2, vector.load=12, scf.if=2) |
| PROVED | `v7x-vmem-budget` | VMEM 288.0 KiB / 64 MiB, SMEM 32 B / 1024 KiB (Lean PallasLean.fitsV7xMemory) |

## maxtext.megablox_gmm[no_existing_out]

| status | obligation | detail |
|---|---|---|
| PROVED | `in-bounds[lhs]` | blocks [2, 2] |
| PROVED | `in-bounds[rhs]` | blocks [4, 2, 2] |
| PROVED | `in-bounds[out]` | blocks [2, 2] |
| PROVED | `prefetch-bounds[1]` | 2 reads |
| PROVED | `prefetch-bounds[2]` | 2 reads |
| PROVED | `prefetch-bounds[3]` | 1 reads |
| PROVED | `parallel-race[out]` | parallel dims [0] |
| PROVED | `contiguity[out]` |  |
| FAILED | `init-at-start[out]` | accumulator read without init at run start — counterexample `{"grid": [0, 0, 0], "pred": [0, 0, 0], "prefetch1": {"0": 1}, "prefetch2": {"0": 0}, "prefetch3": {"0": 0}}` |
| PROVED | `no-midrun-reset[out]` |  |
| PROVED | `coverage[out]` |  |
| PROVED | `init-at-start[scratch0]` |  |
| PROVED | `no-midrun-reset[scratch0]` |  |
| PROVED | `termination` | bounded grid [2, dynamic, 2] |
| PROVED | `v7x-mosaic-lower` | TPU7x TC module @kernel (tpu.matmul=2, vector.load=12, scf.if=2) |
| PROVED | `v7x-vmem-budget` | VMEM 320.0 KiB / 64 MiB, SMEM 64 B / 1024 KiB (Lean PallasLean.fitsV7xMemory) |

## maxtext.megablox_gmm[with_existing_out]

| status | obligation | detail |
|---|---|---|
| PROVED | `in-bounds[lhs]` | blocks [2, 2] |
| PROVED | `in-bounds[rhs]` | blocks [4, 2, 2] |
| PROVED | `in-bounds[existing_out]` | blocks [2, 2] |
| PROVED | `in-bounds[out]` | blocks [2, 2] |
| PROVED | `prefetch-bounds[1]` | 2 reads |
| PROVED | `prefetch-bounds[2]` | 5 reads |
| PROVED | `prefetch-bounds[3]` | 1 reads |
| PROVED | `parallel-race[out]` | parallel dims [0] |
| PROVED | `contiguity[out]` |  |
| PROVED | `init-at-start[out]` |  |
| PROVED | `no-midrun-reset[out]` |  |
| N/A | `coverage[out]` | aliased to an input: unvisited blocks keep their value |
| PROVED | `init-at-start[scratch0]` |  |
| PROVED | `no-midrun-reset[scratch0]` |  |
| PROVED | `termination` | bounded grid [2, dynamic, 2] |
| PROVED | `v7x-mosaic-lower` | TPU7x TC module @kernel (tpu.matmul=2, vector.load=14, scf.if=3) |
| PROVED | `v7x-vmem-budget` | VMEM 448.0 KiB / 64 MiB, SMEM 64 B / 1024 KiB (Lean PallasLean.fitsV7xMemory) |
| PROVED | `C:gmm-step@init[0, 0, 0]` | k_i=0 zeroes acc_scratch, adds LHS@RHS, and seeds out from existing_out |
| PROVED | `C:gmm-step@final[0, 1, 1]` | k_i=K-1 updates rows in [group_start, group_end) with acc + LHS@RHS and preserves prior group rows (Lean GroupedMM) |

## sglang.flash[vision,single-step]

| status | obligation | detail |
|---|---|---|
| PROVED | `in-bounds[q]` | blocks [1, 2, 8, 1] |
| PROVED | `in-bounds[k]` | blocks [1, 2, 1, 1] |
| PROVED | `in-bounds[v]` | blocks [1, 2, 1, 1] |
| PROVED | `in-bounds[o]` | blocks [1, 2, 8, 1] |
| PROVED | `parallel-race[o]` | parallel dims [0, 1, 2] |
| N/A | `contiguity[o]` | unconditional full overwrite: scatter semantics (Lean Pipeline.overwrite_is_scatter) |
| PROVED | `coverage[o]` |  |
| PROVED | `termination` | bounded grid [1, 2, 8, 1] |
| PROVED | `v7x-mosaic-lower` | TPU7x TC module @_flash_attention_kernel (tpu.matmul=2, math.exp=1, arith.truncf=2, vector.load=4) |
| PROVED | `v7x-vmem-budget` | VMEM 1152.0 KiB / 64 MiB, SMEM 0 B / 1024 KiB (Lean PallasLean.fitsV7xMemory) |

## sglang.flash[causal,multi-block]

| status | obligation | detail |
|---|---|---|
| PROVED | `in-bounds[q]` | blocks [1, 2, 4, 1] |
| PROVED | `in-bounds[k]` | blocks [1, 2, 4, 1] |
| PROVED | `in-bounds[v]` | blocks [1, 2, 4, 1] |
| PROVED | `in-bounds[o]` | blocks [1, 2, 4, 1] |
| PROVED | `parallel-race[o]` | parallel dims [0, 1, 2] |
| PROVED | `contiguity[o]` |  |
| PROVED | `write-at-end[o]` |  |
| PROVED | `coverage[o]` |  |
| PROVED | `init-at-start[scratch0]` |  |
| PROVED | `no-midrun-reset[scratch0]` |  |
| PROVED | `init-at-start[scratch1]` |  |
| PROVED | `no-midrun-reset[scratch1]` |  |
| PROVED | `init-at-start[scratch2]` |  |
| PROVED | `no-midrun-reset[scratch2]` |  |
| PROVED | `termination` | bounded grid [1, 2, 4, 4], scan_lengths=[1] |
| PROVED | `v7x-mosaic-lower` | TPU7x TC module @_flash_attention_kernel (tpu.matmul=2, math.exp=2, arith.truncf=2, vector.load=16, scf.if=3) |
| PROVED | `v7x-vmem-budget` | VMEM 448.0 KiB / 64 MiB, SMEM 0 B / 1024 KiB (Lean PallasLean.fitsV7xMemory) |
| PROVED | `C:step[0, 1, 3, 1]` | l' and acc'*l' match the osState recurrence (rows [0, 127], cols [0, 127]) |
| PROVED | `C:step[0, 1, 3, 3]` | l' and acc'*l' match the osState recurrence (rows [0, 127], cols [0, 127]) |
| PROVED | `C:step@init[0, 1, 3, 0]` | l' and acc'*l' match the osState recurrence (rows [0, 127], cols [0, 127]) |
| PROVED | `C:score[t=0]` | cut == scale*QK^T + mask on every grid point reaching it |
| PROVED | `C:skip-sound` | skipped tiles contain no kept (row, col) |

## sglang.flash[causal,ab,q_len=200]

| status | obligation | detail |
|---|---|---|
| PROVED | `in-bounds[q]` | blocks [1, 2, 2, 1] |
| PROVED | `in-bounds[k]` | blocks [1, 2, 4, 1] |
| PROVED | `in-bounds[v]` | blocks [1, 2, 4, 1] |
| FAILED | `in-bounds[ab]` | block index out of range — counterexample `{"grid": [0, 0, 1, 2], "block_index": [0, 0, 2, 0], "num_blocks": [1, 2, 2, 4]}` |
| PROVED | `in-bounds[o]` | blocks [1, 2, 2, 1] |
| PROVED | `parallel-race[o]` | parallel dims [0, 1, 2] |
| PROVED | `contiguity[o]` |  |
| PROVED | `write-at-end[o]` |  |
| PROVED | `coverage[o]` |  |
| PROVED | `init-at-start[scratch0]` |  |
| PROVED | `no-midrun-reset[scratch0]` |  |
| PROVED | `init-at-start[scratch1]` |  |
| PROVED | `no-midrun-reset[scratch1]` |  |
| PROVED | `init-at-start[scratch2]` |  |
| PROVED | `no-midrun-reset[scratch2]` |  |
| PROVED | `termination` | bounded grid [1, 2, 2, 4], scan_lengths=[1] |
| PROVED | `v7x-mosaic-lower` | TPU7x TC module @_flash_attention_kernel (tpu.matmul=2, math.exp=2, arith.truncf=2, vector.load=17, scf.if=3) |
| PROVED | `v7x-vmem-budget` | VMEM 576.0 KiB / 64 MiB, SMEM 0 B / 1024 KiB (Lean PallasLean.fitsV7xMemory) |

## sglang.flash[causal,block_q=129]

| status | obligation | detail |
|---|---|---|
| PROVED | `in-bounds[q]` | blocks [1, 2, 1, 1] |
| PROVED | `in-bounds[k]` | blocks [1, 2, 2, 1] |
| PROVED | `in-bounds[v]` | blocks [1, 2, 2, 1] |
| PROVED | `in-bounds[o]` | blocks [1, 2, 1, 1] |
| PROVED | `parallel-race[o]` | parallel dims [0, 1, 2] |
| PROVED | `contiguity[o]` |  |
| PROVED | `write-at-end[o]` |  |
| PROVED | `coverage[o]` |  |
| PROVED | `init-at-start[scratch0]` |  |
| PROVED | `no-midrun-reset[scratch0]` |  |
| PROVED | `init-at-start[scratch1]` |  |
| PROVED | `no-midrun-reset[scratch1]` |  |
| PROVED | `init-at-start[scratch2]` |  |
| PROVED | `no-midrun-reset[scratch2]` |  |
| PROVED | `termination` | bounded grid [1, 2, 1, 2], scan_lengths=[1] |
| PROVED | `v7x-mosaic-lower` | TPU7x TC module @_flash_attention_kernel (tpu.matmul=2, math.exp=2, arith.truncf=2, vector.load=16, scf.if=3) |
| PROVED | `v7x-vmem-budget` | VMEM 450.5 KiB / 64 MiB, SMEM 0 B / 1024 KiB (Lean PallasLean.fitsV7xMemory) |
| PROVED | `C:score[t=0]` | cut == scale*QK^T + mask on every grid point reaching it |
| FAILED | `C:skip-sound` | a skipped tile contains a kept entry — counterexample `{"grid": [0, 0, 0, 1], "row": 128, "col": 128}` |

## sglang_jax.megablox_gmm[quant_b2,existing_out]

| status | obligation | detail |
|---|---|---|
| PROVED | `in-bounds[lhs]` | blocks [2, 4] |
| PROVED | `in-bounds[rhs]` | blocks [4, 4, 2] |
| FAILED | `in-bounds[rhs_scale]` | block index out of range — counterexample `{"grid": [0, 0, 1], "prefetch1": {"0": 2}, "prefetch2": {"0": 0}, "prefetch3": {"0": 0}, "block_index": [2, -1, 0, 0], "num_blocks": [4, 2, 1, 2]}` |
| PROVED | `in-bounds[existing_out]` | blocks [2, 2] |
| PROVED | `in-bounds[out]` | blocks [2, 2] |
| PROVED | `prefetch-bounds[1]` | 3 reads |
| PROVED | `prefetch-bounds[2]` | 5 reads |
| PROVED | `prefetch-bounds[3]` | 2 reads |
| PROVED | `parallel-race[out]` | parallel dims [0] |
| PROVED | `contiguity[out]` |  |
| PROVED | `init-at-start[out]` |  |
| PROVED | `no-midrun-reset[out]` |  |
| N/A | `coverage[out]` | aliased to an input: unvisited blocks keep their value |
| PROVED | `init-at-start[scratch0]` |  |
| PROVED | `no-midrun-reset[scratch0]` |  |
| PROVED | `termination` | bounded grid [2, dynamic, 4] |
| PROVED | `v7x-mosaic-lower` | TPU7x TC module @gmm-g_4-m_256-k_512-n_256-tm_128-tk_128-tn_128 (tpu.matmul=2, vector.load=14, scf.if=3) |
| PROVED | `v7x-vmem-budget` | VMEM 449.0 KiB / 64 MiB, SMEM 64 B / 1024 KiB (Lean PallasLean.fitsV7xMemory) |

## sglang_jax.megablox_gmm[quant_b1,existing_out]

| status | obligation | detail |
|---|---|---|
| PROVED | `in-bounds[lhs]` | blocks [2, 2] |
| PROVED | `in-bounds[rhs]` | blocks [4, 2, 2] |
| FAILED | `in-bounds[rhs_scale]` | block index out of range — counterexample `{"grid": [0, 0, 1], "prefetch1": {"0": 0}, "prefetch2": {"0": 0}, "prefetch3": {"0": 0}, "block_index": [0, -1, 0, 0], "num_blocks": [4, 1, 1, 2]}` |
| PROVED | `in-bounds[existing_out]` | blocks [2, 2] |
| PROVED | `in-bounds[out]` | blocks [2, 2] |
| PROVED | `prefetch-bounds[1]` | 3 reads |
| PROVED | `prefetch-bounds[2]` | 5 reads |
| PROVED | `prefetch-bounds[3]` | 2 reads |
| PROVED | `parallel-race[out]` | parallel dims [0] |
| PROVED | `contiguity[out]` |  |
| PROVED | `init-at-start[out]` |  |
| PROVED | `no-midrun-reset[out]` |  |
| N/A | `coverage[out]` | aliased to an input: unvisited blocks keep their value |
| PROVED | `init-at-start[scratch0]` |  |
| PROVED | `no-midrun-reset[scratch0]` |  |
| PROVED | `termination` | bounded grid [2, dynamic, 2] |
| PROVED | `v7x-mosaic-lower` | TPU7x TC module @gmm-g_4-m_256-k_256-n_256-tm_128-tk_128-tn_128 (tpu.matmul=2, vector.load=14, scf.if=3) |
| PROVED | `v7x-vmem-budget` | VMEM 449.0 KiB / 64 MiB, SMEM 64 B / 1024 KiB (Lean PallasLean.fitsV7xMemory) |

## sglang_jax.update_kv_cache

| status | obligation | detail |
|---|---|---|
| PROVED | `in-bounds[new_kv]` | blocks [1, 1, 1] |
| PROVED | `in-bounds[kv_cache]` | blocks [1, 1, 1] |
| PROVED | `in-bounds[kv_out]` | blocks [1, 1, 1] |
| N/A | `parallel-race[kv_out]` | no parallel dims |
| N/A | `contiguity[kv_out]` | explicit HBM ref (memory_space=any): no VMEM output pipeline |
| PROVED | `write-at-end[kv_out]` | explicit HBM DMA write (memory_space=any) |
| N/A | `coverage[kv_out]` | aliased to an input: unvisited blocks keep their value |
| PROVED | `init-at-start[scratch0]` | per-step staging buffer: written before read on every grid step |
| N/A | `no-midrun-reset[scratch0]` | per-step staging buffer (not a cross-step accumulator) |
| PROVED | `termination` | bounded grid [dynamic], dma_pairs=16 |
| PROVED | `v7x-mosaic-lower` | TPU7x TC module @kv_cache_update_kernel (tpu.enqueue_dma=16, tpu.wait_dma=16) |
| PROVED | `v7x-vmem-budget` | VMEM 256.0 KiB / 64 MiB, SMEM 96 B / 1024 KiB (Lean PallasLean.fitsV7xMemory) |

## Level D: TPU v7x Floating-Point Rounding Bounds

exp model: ue = 7.76e-06, TPU-measured

| kernel | requires | f32->bf16 casts in body | eps (abs) | spec_tol | status |
|---|---|---|---|---|---|
| sglang.flash[vision,single-step] | |q|,|k|,|v|<=1, kv=1024, d=128, scale=1/8 | [(128, 1024), (128, 128)] | 0.00848 | 0.01 | PROVED |
| sglang.flash[causal,multi-block] | |q|,|k|,|v|<=1, kv=512 (4x128), d=128, scale=1/8 | [(128, 128), (128, 128)] | 0.00451 | 0.01 | PROVED |
| maxtext.ragged_mqa | |q|<=1/8,|k|,|v|<=1, T=4096 (16x256), d=128 | [] | 0.000859 | 0.001 | PROVED |
| sglang.qk_norm_rope | |qkv|<=1, |w|<=1, rms>=1, hd=128 | [(256, 512), (256, 256)] | 0.00783 | 0.02 | PROVED |
| maxtext.megablox_gmm[with_existing_out] | |lhs|<=1, |rhs|<=1/16, K=256 (2x128), acc=f32 | [] | 0.000122 | 0.0002 | PROVED |
| sglang_jax.megablox_gmm[quant_b1,existing_out] | |lhs|<=1, |rhs|<=1/16, K=256 (2x128), quant_blocks=1, acc=f32 | [] | 0.000122 | 0.0002 | PROVED |
