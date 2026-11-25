import PallasLean.Recurrence

/-!
# Grouped Matrix Multiplication (MegaBlox / MoE GMM) Formal Theory (`maxtext.megablox_gmm`)

Formalizes the two core invariants of MaxText's Pallas MegaBlox Grouped Matrix
Multiplication kernel (`MaxText/kernels/megablox/backend.py`):

1. **`gmm_k_tile_closed`**: Over the `(+, *)` semiring with rescale factor `a_k = 1`,
   zero-initializing `acc_scratch` at `k_i = 0` and adding each `K`-tile product
   `tileProd k` yields the exact sum `∑ k ∈ range K, tileProd k` at `k_i = K - 1`.
2. **`gmm_partial_tile_select` vs. `gmm_uninit_leak`**: When an `M`-tile is shared
   across groups (`m_tile_ids[g₁] = m_tile_ids[g₂]`), `_store_accum` writes
   `select(mask, acc, out)`. If `out` is initialized at the start of the `M`-tile's
   contiguous run (`existing_out` provided), every row holds its owning group's
   product or `existing_out`. If `out` is NOT initialized (`existing_out = None`),
   unowned rows leak the uninitialized VMEM buffer `garbage`.
-/

open Finset

namespace PallasLean.GroupedMM

/-- `K`-tile accumulation recurrence: `acc (k+1) = 1 * acc k + tileProd k`, `acc 0 = 0`. -/
def gmmAcc (tileProd : ℕ → ℝ) : ℕ → ℝ :=
  linRec (fun _ => (1 : ℝ)) tileProd 0

/-- **Closed form of `K`-tile accumulation**: after `K` tiles, `gmmAcc` equals
the exact sum of all `K` tile products. -/
theorem gmmAcc_closed (tileProd : ℕ → ℝ) (K : ℕ) :
    gmmAcc tileProd K = ∑ k ∈ range K, tileProd k := by
  unfold gmmAcc
  rw [linRec_closed]
  simp

/-- Masked store of one group's result `acc` into the VMEM output buffer `buf`
for a row `r` with membership predicate `inGroup r`. -/
def maskedStore (inGroup : ℕ → Bool) (acc buf : ℕ → ℝ) (r : ℕ) : ℝ :=
  if inGroup r then acc r else buf r

/-- **Two-group partial-tile composition.** If groups `1` and `2` have disjoint row
predicates (`inG1 r = true → inG2 r = false`) and `buf` starts initialized to `init`,
then running `maskedStore` for group `1` followed by group `2` produces:
* `acc2 r` on group 2's rows,
* `acc1 r` on group 1's rows, and
* `init r` on any remaining rows. -/
theorem gmm_partial_tile_two_groups (inG1 inG2 : ℕ → Bool) (acc1 acc2 init : ℕ → ℝ)
    (r : ℕ) (hdisj : inG1 r = true → inG2 r = false) :
    maskedStore inG2 acc2 (maskedStore inG1 acc1 init) r =
      if inG2 r then acc2 r else if inG1 r then acc1 r else init r := by
  unfold maskedStore
  cases h2 : inG2 r <;> simp

/-- **Uninitialized VMEM leak when `existing_out = None`.**
If a row `r` in an active `M`-tile belongs to no active group (`inG r = false`, e.g.
padded tail tokens when `sum(group_sizes) < m`), `maskedStore` without prior
output-buffer initialization returns the stale VMEM buffer contents `garbage r`
instead of `0`. -/
theorem gmm_uninit_vmem_leak (inG : ℕ → Bool) (acc garbage : ℕ → ℝ) (r : ℕ)
    (hpad : inG r = false) :
    maskedStore inG acc garbage r = garbage r := by
  unfold maskedStore
  simp [hpad]

/-- **Mid-run reset bug in quantized GMM (`sglang-jax` `megablox_gmm_kernel/gmm.py`).**
When `num_quant_blocks ≥ 2` and `existing_out` is passed, `_init_out` resets
`out := existing_out` whenever `k_i = 0 ∧ first_time_seeing_out` (without checking
`b_i = 0`). Thus quantization block `b_i = 1` overwrites the accumulated output of
`b_i = 0`, producing `existing_out + b1` instead of `existing_out + b0 + b1`,
which disagrees with the specification whenever `b0 ≠ 0`. -/
theorem gmm_quant_midrun_reset_loses_prior_block (existing_out b0 b1 : ℝ) (hb0 : b0 ≠ 0) :
    existing_out + b1 ≠ existing_out + b0 + b1 := by
  intro h
  apply hb0
  linarith

end PallasLean.GroupedMM
