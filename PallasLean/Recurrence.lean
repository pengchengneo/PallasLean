import Mathlib

/-!
# Semiring linear recurrences

Kernel loops that tile a reduction and update an accumulator obey
`f (k+1) = a k * f k + b k`. Unrolling only needs distributivity, so the
closed form holds in any commutative semiring: `(+, *)` for sums and
rescaled accumulators, `(max, +)` (tropical) for running maxima.
-/

open Finset

namespace PallasLean

/-- `f (k+1) = a k * f k + b k`, `f 0 = f0`. -/
def linRec {S : Type*} [CommSemiring S] (a b : ℕ → S) (f0 : S) : ℕ → S
  | 0 => f0
  | k + 1 => a k * linRec a b f0 k + b k

/-- Closed form: every `b j` is multiplied by every later coefficient. This
closed form is the loop invariant a verifier would otherwise ask for by hand. -/
theorem linRec_closed {S : Type*} [CommSemiring S] (a b : ℕ → S) (f0 : S) (n : ℕ) :
    linRec a b f0 n =
      (∏ i ∈ range n, a i) * f0 + ∑ j ∈ range n, (∏ i ∈ Ico (j + 1) n, a i) * b j := by
  induction n with
  | zero => simp [linRec]
  | succ n ih =>
    have h : ∀ j ∈ range n,
        (∏ i ∈ Ico (j + 1) (n + 1), a i) * b j = a n * ((∏ i ∈ Ico (j + 1) n, a i) * b j) := by
      intro j hj
      rw [mem_range] at hj
      rw [prod_Ico_succ_top (by omega)]
      ring
    rw [linRec, ih, prod_range_succ, sum_range_succ, sum_congr rfl h, ← mul_sum]
    simp
    ring

/-- Products of consecutive ratios telescope: `prod exp (mu i - mu (i+1)) = exp (mu j - mu n)`. -/
theorem prod_exp_telescope (mu : ℕ → ℝ) {j n : ℕ} (h : j ≤ n) :
    ∏ i ∈ Ico j n, Real.exp (mu i - mu (i + 1)) = Real.exp (mu j - mu n) := by
  induction n, h using Nat.le_induction with
  | base => simp
  | succ n hjn ih =>
    rw [prod_Ico_succ_top hjn, ih, ← Real.exp_add]
    ring_nf

/-- Running maximum, the `(max, +)` recurrence `m (k+1) = 0 + m k ⊔ x k`. -/
def runMax (x : ℕ → ℝ) (m0 : ℝ) : ℕ → ℝ
  | 0 => m0
  | k + 1 => max (runMax x m0 k) (x k)

theorem runMax_mono (x : ℕ → ℝ) (m0 : ℝ) {j n : ℕ} (h : j ≤ n) :
    runMax x m0 j ≤ runMax x m0 n := by
  induction n, h using Nat.le_induction with
  | base => exact le_rfl
  | succ n _ ih => exact ih.trans (le_max_left _ _)

/-- Every score seen so far is at most the running max, so `exp (x j - m) ≤ 1`:
this is what keeps the accumulators bounded in the rounding analysis. -/
theorem le_runMax (x : ℕ → ℝ) (m0 : ℝ) {j n : ℕ} (h : j < n) : x j ≤ runMax x m0 n :=
  (le_max_right _ _).trans (runMax_mono x m0 (Nat.succ_le_of_lt h))

end PallasLean
