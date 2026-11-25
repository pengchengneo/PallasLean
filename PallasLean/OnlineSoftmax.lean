import PallasLean.Recurrence

/-!
# Online softmax, shift-invariant form

Blocks are `[c k, c (k+1))` for an arbitrary monotone boundary sequence `c`,
so one theorem covers fixed tiles, two-level tiling (grid block x `pl.loop`
sub-block) and ragged tails. `skip k` marks blocks the kernel never runs
(causal block skipping, ragged early exit).

The kernel's running max `mu` is an arbitrary shift here: correctness does not
depend on its value. The max only matters for the rounding analysis
(`le_runMax`), where it keeps every exponent non-positive.
-/

open Finset

noncomputable section

namespace PallasLean

variable (c : ℕ → ℕ) (skip : ℕ → Prop) [DecidablePred skip]

/-- Sum of `f` over the processed (non-skipped) blocks `k < n`. -/
def procSum (f : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ k ∈ range n, if skip k then 0 else ∑ j ∈ Ico (c k) (c (k + 1)), f j

/-- Contribution of block `k` at shift `m`, weighted by `f` (`f = 1` for `l`). -/
def blk (x f : ℕ → ℝ) (m : ℝ) (k : ℕ) : ℝ :=
  if skip k then 0 else ∑ j ∈ Ico (c k) (c (k + 1)), Real.exp (x j - m) * f j

/-- Unnormalized online-softmax state, a `(+, *)` linear recurrence:
`S (k+1) = exp (mu k - mu (k+1)) * S k + blk (mu (k+1)) k`, `S 0 = 0`.
`f = 1` gives the denominator `l`; `f = v` gives `acc * l`. -/
def osState (x f mu : ℕ → ℝ) : ℕ → ℝ :=
  linRec (fun k => Real.exp (mu k - mu (k + 1))) (fun k => blk c skip x f (mu (k + 1)) k) 0

/-- The semiring closed form, with the rescale coefficients telescoped. -/
theorem osState_closed (x f mu : ℕ → ℝ) (n : ℕ) :
    osState c skip x f mu n =
      Real.exp (-mu n) * procSum c skip (fun j => Real.exp (x j) * f j) n := by
  rw [osState, linRec_closed, mul_zero, zero_add, procSum, mul_sum]
  refine sum_congr rfl fun k hk => ?_
  rw [mem_range] at hk
  rw [prod_exp_telescope mu (show k + 1 ≤ n by omega), blk]
  split_ifs
  · simp
  · rw [mul_sum, mul_sum]
    refine sum_congr rfl fun j _ => ?_
    rw [← mul_assoc, ← mul_assoc, ← Real.exp_add, ← Real.exp_add]
    ring_nf

/-- **Online softmax is exact in real arithmetic**, for every tiling, skip
pattern and choice of running shift. -/
theorem online_softmax_exact (x v mu : ℕ → ℝ) (n : ℕ) :
    osState c skip x v mu n / osState c skip x (fun _ => 1) mu n =
      procSum c skip (fun j => Real.exp (x j) * v j) n /
        procSum c skip (fun j => Real.exp (x j) * 1) n := by
  rw [osState_closed, osState_closed, mul_div_mul_left _ _ (Real.exp_pos _).ne']

/-- **Block skipping is sound** when every skipped block holds only excluded
indices: the processed sum of a kept-only function equals the full sum. -/
theorem procSum_skip_sound (keep : ℕ → Prop) [DecidablePred keep] (hc : Monotone c)
    (h0 : c 0 = 0) (f : ℕ → ℝ) (n : ℕ)
    (hskip : ∀ k < n, skip k → ∀ j ∈ Ico (c k) (c (k + 1)), ¬ keep j) :
    procSum c skip (fun j => if keep j then f j else 0) n =
      ∑ j ∈ range (c n), if keep j then f j else 0 := by
  induction n with
  | zero => simp [procSum, h0]
  | succ n ih =>
    rw [procSum, sum_range_succ, ← procSum,
      ih (fun k hk => hskip k (by omega)),
      ← sum_range_add_sum_Ico _ (hc (Nat.le_succ n))]
    congr 1
    split_ifs with hs
    · exact (sum_eq_zero fun j hj => by simp [hskip n (by omega) hs j hj]).symm
    · rfl

/-- Weighted averages with non-negative weights and `|v| ≤ V` stay in `[-V, V]`. -/
theorem abs_wsum_le (s : Finset ℕ) (w v : ℕ → ℝ) (V : ℝ) (hw : ∀ j ∈ s, 0 ≤ w j)
    (hv : ∀ j ∈ s, |v j| ≤ V) : |∑ j ∈ s, w j * v j| ≤ V * ∑ j ∈ s, w j := by
  calc |∑ j ∈ s, w j * v j| ≤ ∑ j ∈ s, |w j * v j| := abs_sum_le_sum_abs _ _
    _ ≤ ∑ j ∈ s, V * w j := sum_le_sum fun j hj => by
        rw [abs_mul, abs_of_nonneg (hw j hj), mul_comm]
        exact mul_le_mul_of_nonneg_right (hv j hj) (hw j hj)
    _ = V * ∑ j ∈ s, w j := (mul_sum _ _ _).symm

/-- **Finite mask values.** Kernels mask with a large negative constant instead
of `-inf`, so in real arithmetic masked entries keep a tiny weight `b`.
Against the exclusion spec `A / B` the error is at most `2 V b / B`. -/
theorem mask_approx {A B a b V : ℝ} (hB : 0 < B) (hb : 0 ≤ b) (hA : |A| ≤ V * B)
    (ha : |a| ≤ V * b) : |(A + a) / (B + b) - A / B| ≤ 2 * V * b / B := by
  have hBb : 0 < B + b := by linarith
  have hV : 0 ≤ V := by
    have := (abs_nonneg A).trans hA
    exact nonneg_of_mul_nonneg_left this hB
  rw [div_sub_div _ _ hBb.ne' hB.ne', abs_div, abs_of_pos (mul_pos hBb hB),
    div_le_div_iff₀ (mul_pos hBb hB) hB]
  have key : |(A + a) * B - (B + b) * A| ≤ 2 * V * b * B := by
    have : (A + a) * B - (B + b) * A = a * B - A * b := by ring
    rw [this]
    calc |a * B - A * b| ≤ |a * B| + |A * b| := abs_sub _ _
      _ = |a| * B + |A| * b := by rw [abs_mul, abs_mul, abs_of_pos hB, abs_of_nonneg hb]
      _ ≤ V * b * B + V * B * b := by gcongr
      _ = 2 * V * b * B := by ring
  calc |(A + a) * B - (B + b) * A| * B ≤ 2 * V * b * B * B := by gcongr
    _ ≤ 2 * V * b * (B + b * 1) * B := by
        have : 0 ≤ 2 * V * b * (b * 1) * B := by positivity
        nlinarith
    _ = 2 * V * b * ((B + b) * B) := by ring

/-! ## Kernel step forms

Production kernels keep the normalized accumulator `acc = S_v / S_1` rather than
`S_v`. These lemmas show each kernel's literal update preserves `acc * l = S_v`;
the frontend checks that the traced jaxpr body computes exactly these terms. -/

/-- sglang / vllm `flash_attention` multi-block step:
`l' = sum p + alpha l`, `acc' = acc * (alpha l / l') + (p @ v) / l'`. -/
theorem sglang_fa_step {acc l alpha ps o l' : ℝ} (hl : l' = ps + alpha * l) (h0 : l' ≠ 0) :
    (acc * (alpha * l * (1 / l')) + o * (1 / l')) * l' = alpha * (acc * l) + o := by
  subst hl; field_simp

/-- maxtext `ragged_flash_attention_kernel` step:
`l' = alpha l + beta l_c`, `o' = (l alpha o + beta (s_c @ v)) / l'`. -/
theorem maxtext_ragged_step {o l alpha beta lc oc l' : ℝ} (_hl : l' = alpha * l + beta * lc)
    (h0 : l' ≠ 0) : ((l * alpha * o + beta * oc) / l') * l' = alpha * (o * l) + beta * oc := by
  rw [div_mul_cancel₀ _ h0]; ring

/-- The ragged kernel re-centres block scores at the block max `mc` first;
`beta = exp (mc - mn)` moves them to the running max `mn`. -/
theorem recentre (x mc mn : ℝ) : Real.exp (mc - mn) * Real.exp (x - mc) = Real.exp (x - mn) := by
  rw [← Real.exp_add]; ring_nf

/-- sglang `kimi_attn_residual` decode mixture (`decode_finish`):
shifting all `n` bank+prefix scores `s j` by any shift `m` (`exp(s j - m)`)
and dividing the weighted sum `∑ j, exp(s j - m) * v j` by `∑ j, exp(s j - m)`
equals the exact unshifted softmax mixture. -/
theorem kimi_attn_res_exact (s v : ℕ → ℝ) (m : ℝ) (n : ℕ) :
    (∑ j ∈ range n, Real.exp (s j - m) * v j) / (∑ j ∈ range n, Real.exp (s j - m)) =
      (∑ j ∈ range n, Real.exp (s j) * v j) / (∑ j ∈ range n, Real.exp (s j)) := by
  have h1 : ∀ j ∈ range n, Real.exp (s j - m) * v j = Real.exp (-m) * (Real.exp (s j) * v j) := by
    intro j _
    rw [← mul_assoc, ← Real.exp_add]
    ring_nf
  have h2 : ∀ j ∈ range n, Real.exp (s j - m) = Real.exp (-m) * Real.exp (s j) := by
    intro j _
    rw [← Real.exp_add]
    ring_nf
  rw [sum_congr rfl h1, sum_congr rfl h2, ← mul_sum, ← mul_sum,
    mul_div_mul_left _ _ (Real.exp_pos _).ne']

end PallasLean

end
