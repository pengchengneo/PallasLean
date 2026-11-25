import Mathlib

/-!
# Causal block skipping

`flash_attention` skips grid block `(r, c)` (q rows `[r bq, (r+1) bq)`, kv
columns `[c bk, (c+1) bk)`) unless `below_or_on_diag`:
`(r + 1) * bq - 1 > c * bk`. Skipping is sound iff every entry of a skipped
block is causally masked, `col > row`.

The test is strict, so the corner case `(r+1) bq - 1 = c bk` (an on-diagonal
entry) is skipped. That entry is real; it cannot occur when `8 | bq` and
`8 | bk`, which Mosaic's tiling enforces whenever `bq < q_seq_len`.
-/

namespace PallasLean.CausalSkip

/-- The kernel's run predicate, in ℕ (all operands are non-negative). -/
def runs (r bq c bk : ℕ) : Prop := (r + 1) * bq > c * bk + 1

instance (r bq c bk : ℕ) : Decidable (runs r bq c bk) := by unfold runs; infer_instance

/-- **Sound under the tiling constraint.** -/
theorem skip_sound {r bq c bk : ℕ} (hq : 8 ∣ bq) (hk : 8 ∣ bk) (hskip : ¬ runs r bq c bk)
    (i j : ℕ) (hi : i < bq) (hj : j < bk) : c * bk + j > r * bq + i := by
  obtain ⟨a, rfl⟩ := hq
  obtain ⟨b, rfl⟩ := hk
  unfold runs at hskip
  have e1 : (r + 1) * (8 * a) = 8 * (r * a) + 8 * a := by ring
  have e2 : c * (8 * b) = 8 * (c * b) := by ring
  have e3 : r * (8 * a) = 8 * (r * a) := by ring
  rw [e1, e2] at hskip
  rw [e2, e3]
  omega

/-- **Unsound without it.** `bq = 129` (a single q block covering
`q_seq_len = 129`, allowed because it equals the full dimension) and
`bk = 128`: block `(0, 1)` is skipped, yet row 128 attends column 128. -/
theorem skip_unsound_odd_block_q :
    ¬ runs 0 129 1 128 ∧ ¬ (1 * 128 + 0 > 0 * 129 + 128) := by decide

end PallasLean.CausalSkip
