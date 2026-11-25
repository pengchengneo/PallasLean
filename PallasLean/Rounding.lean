import PallasLean.OnlineSoftmax

/-!
# Rounding error, propagated along the computation tree

The rounding model is mixed relative/absolute, `|fl x - x| <= u |x| + eta`.
A purely relative model is unsound on TPU: subnormal results are flushed to
zero (relative error 1), which `eta` (the smallest normal) absorbs.

Reductions are proven for an arbitrary binary summation tree, so the hardware
accumulation order (MXU, cross-lane `reduce_sum`) is not a trust assumption:
only "some tree of correctly rounded fp32 adds" is.
-/

noncomputable section

namespace PallasLean

structure RoundModel where
  u : ℝ
  eta : ℝ
  fl : ℝ → ℝ
  hu : 0 ≤ u
  heta : 0 ≤ eta
  err : ∀ x, |fl x - x| ≤ u * |x| + eta

/-- A summation order: any binary tree over leaf indices. -/
inductive STree
  | leaf (i : ℕ)
  | node (l r : STree)

namespace STree

def size : STree → ℕ
  | leaf _ => 1
  | node l r => l.size + r.size

/-- Number of additions on the longest possible chain is at most `size - 1`. -/
def d (T : STree) : ℕ := T.size - 1

def real (x : ℕ → ℝ) : STree → ℝ
  | leaf i => x i
  | node l r => l.real x + r.real x

/-- Floating evaluation: leaves are already-computed (possibly inexact) inputs `y`. -/
def fl (R : RoundModel) (y : ℕ → ℝ) : STree → ℝ
  | leaf i => y i
  | node l r => R.fl (l.fl R y + r.fl R y)

def asum (x : ℕ → ℝ) : STree → ℝ
  | leaf i => |x i|
  | node l r => l.asum x + r.asum x

def esum (e : ℕ → ℝ) : STree → ℝ
  | leaf i => e i
  | node l r => l.esum e + r.esum e

theorem size_pos : ∀ T : STree, 1 ≤ T.size
  | leaf _ => le_rfl
  | node l _ => le_add_right (size_pos l)

theorem d_node (l r : STree) : (node l r).d = l.d + r.d + 1 := by
  have := size_pos l; have := size_pos r
  simp only [d, size]; omega

theorem abs_real_le (x : ℕ → ℝ) : ∀ T : STree, |T.real x| ≤ T.asum x
  | leaf _ => le_rfl
  | node l r => (abs_add_le _ _).trans (add_le_add (abs_real_le x l) (abs_real_le x r))

theorem asum_nonneg (x : ℕ → ℝ) : ∀ T : STree, 0 ≤ T.asum x
  | leaf _ => abs_nonneg _
  | node l r => add_nonneg (asum_nonneg x l) (asum_nonneg x r)

theorem esum_nonneg (e : ℕ → ℝ) (he : ∀ i, 0 ≤ e i) : ∀ T : STree, 0 ≤ T.esum e
  | leaf i => he i
  | node l r => add_nonneg (esum_nonneg e he l) (esum_nonneg e he r)

/-- **Order-independent summation error.** With input errors `|y i - x i| <= e i`
and `g = (1+u)^(n-1)` for `n` leaves:
`|fl T - T| <= g * sum e + (g - 1) * sum |x| + (n-1) * g * eta`. -/
theorem err_bound (R : RoundModel) (x y e : ℕ → ℝ) (he : ∀ i, |y i - x i| ≤ e i) :
    ∀ T : STree, |T.fl R y - T.real x| ≤
      (1 + R.u) ^ T.d * T.esum e + ((1 + R.u) ^ T.d - 1) * T.asum x +
        T.d * (1 + R.u) ^ T.d * R.eta
  | leaf i => by simpa [fl, real, esum, asum, d, size] using he i
  | node l r => by
    have he0 : ∀ i, 0 ≤ e i := fun i => (abs_nonneg _).trans (he i)
    have hl := err_bound R x y e he l
    have hr := err_bound R x y e he r
    have hEL := esum_nonneg e he0 l; have hER := esum_nonneg e he0 r
    have hAL := asum_nonneg x l; have hAR := asum_nonneg x r
    have hLa := abs_real_le x l; have hRa := abs_real_le x r
    have h1u : 1 ≤ 1 + R.u := by linarith [R.hu]
    set gl := (1 + R.u) ^ l.d
    set gr := (1 + R.u) ^ r.d
    set g := (1 + R.u) ^ (l.d + r.d + 1)
    have hgl : gl * (1 + R.u) ≤ g := by
      rw [← pow_succ]; exact pow_le_pow_right₀ h1u (by omega)
    have hgr : gr * (1 + R.u) ≤ g := by
      rw [← pow_succ]; exact pow_le_pow_right₀ h1u (by omega)
    have hg1 : 1 ≤ g := one_le_pow₀ h1u
    have hgl1 : 1 ≤ gl := one_le_pow₀ h1u
    have hgr1 : 1 ≤ gr := one_le_pow₀ h1u
    set εl := |l.fl R y - l.real x|
    set εr := |r.fl R y - r.real x|
    have hz : |l.fl R y + r.fl R y| ≤ l.asum x + r.asum x + εl + εr := by
      have : l.fl R y + r.fl R y = l.real x + r.real x + (l.fl R y - l.real x) +
          (r.fl R y - r.real x) := by ring
      rw [this]
      calc _ ≤ |l.real x + r.real x + (l.fl R y - l.real x)| + εr := abs_add_le _ _
        _ ≤ |l.real x + r.real x| + εl + εr := by gcongr; exact abs_add_le _ _
        _ ≤ _ := by gcongr; exact (abs_add_le _ _).trans (add_le_add hLa hRa)
    have hround := R.err (l.fl R y + r.fl R y)
    have htot : |(node l r).fl R y - (node l r).real x| ≤
        R.u * (l.asum x + r.asum x) + R.eta + (1 + R.u) * (εl + εr) := by
      simp only [fl, real]
      have : R.fl (l.fl R y + r.fl R y) - (l.real x + r.real x) =
          (R.fl (l.fl R y + r.fl R y) - (l.fl R y + r.fl R y)) +
            ((l.fl R y - l.real x) + (r.fl R y - r.real x)) := by ring
      rw [this]
      calc _ ≤ |R.fl (l.fl R y + r.fl R y) - (l.fl R y + r.fl R y)| + (εl + εr) :=
            (abs_add_le _ _).trans (by gcongr; exact abs_add_le _ _)
        _ ≤ R.u * (l.asum x + r.asum x + εl + εr) + R.eta + (εl + εr) := by
            gcongr; exact hround.trans (by gcongr; exact R.hu)
        _ = _ := by ring
    rw [d_node]
    simp only [esum, asum]
    have hdl : (0 : ℝ) ≤ l.d := Nat.cast_nonneg _
    have hdr : (0 : ℝ) ≤ r.d := Nat.cast_nonneg _
    have f1 := mul_le_mul_of_nonneg_right hgl hEL
    have f2 := mul_le_mul_of_nonneg_right hgl hAL
    have f3 := mul_le_mul_of_nonneg_right hgl (mul_nonneg hdl R.heta)
    have f4 := mul_le_mul_of_nonneg_right hgr hER
    have f5 := mul_le_mul_of_nonneg_right hgr hAR
    have f6 := mul_le_mul_of_nonneg_right hgr (mul_nonneg hdr R.heta)
    have f7 : R.eta ≤ g * R.eta := le_mul_of_one_le_left R.heta hg1
    have hl' := mul_le_mul_of_nonneg_left hl (by linarith [R.hu] : (0:ℝ) ≤ 1 + R.u)
    have hr' := mul_le_mul_of_nonneg_left hr (by linarith [R.hu] : (0:ℝ) ≤ 1 + R.u)
    push_cast
    nlinarith [htot, hl', hr', f1, f2, f3, f4, f5, f6, f7]

end STree

/-- Products: `|x y - X Y| <= |X| ey + |Y| ex + ex ey`. -/
theorem mul_err {X Y x y ex ey : ℝ} (hx : |x - X| ≤ ex) (hy : |y - Y| ≤ ey) :
    |x * y - X * Y| ≤ |X| * ey + |Y| * ex + ex * ey := by
  have : x * y - X * Y = X * (y - Y) + Y * (x - X) + (x - X) * (y - Y) := by ring
  rw [this]
  calc _ ≤ |X * (y - Y)| + |Y * (x - X)| + |(x - X) * (y - Y)| :=
        (abs_add_le _ _).trans (by gcongr; exact abs_add_le _ _)
    _ = |X| * |y - Y| + |Y| * |x - X| + |x - X| * |y - Y| := by simp [abs_mul]
    _ ≤ _ := by gcongr; exact (abs_nonneg _).trans hx

/-- `exp` is `e^M`-Lipschitz below `M`: arguments that are both `<= M` (after
the running-max shift, `M = 0`) move `exp` by at most `e^M |a - b|`. -/
theorem exp_lipschitz {a b M : ℝ} (ha : a ≤ M) (hb : b ≤ M) :
    |Real.exp a - Real.exp b| ≤ Real.exp M * |a - b| := by
  wlog hab : b ≤ a generalizing a b
  · rw [abs_sub_comm, abs_sub_comm a]; exact this hb ha (le_of_lt (not_le.mp hab))
  have h1 : 1 - Real.exp (b - a) ≤ a - b := by linarith [Real.add_one_le_exp (b - a)]
  have : Real.exp a - Real.exp b = Real.exp a * (1 - Real.exp (b - a)) := by
    rw [mul_sub, mul_one, ← Real.exp_add]; ring_nf
  rw [this, abs_of_nonneg (by
      have := Real.exp_le_exp.mpr (show b - a ≤ 0 by linarith)
      rw [Real.exp_zero] at this; exact mul_nonneg (Real.exp_pos a).le (by linarith)),
    abs_of_nonneg (by linarith)]
  exact mul_le_mul (Real.exp_le_exp.mpr ha) h1 (by
    have := Real.exp_le_exp.mpr (show b - a ≤ 0 by linarith)
    rw [Real.exp_zero] at this; linarith) (Real.exp_pos M).le

/-- Hardware `exp` on non-positive arguments (EUP on TPU). `ue`, `etae` are
measured, not proven: this is the one transcendental trust assumption. -/
structure ExpModel where
  fexp : ℝ → ℝ
  ue : ℝ
  etae : ℝ
  err : ∀ x ≤ 0, |fexp x - Real.exp x| ≤ ue * Real.exp x + etae

/-- **The triangle.** `kernel_Float` vs spec splits into the rounding edge
(`kernel_Float` vs `kernel_Real`), the finite-mask edge (`kernel_Real` vs the
exclusion spec, `mask_approx`) and the algebraic edge (`= 0`,
`online_softmax_exact`). -/
theorem triangle {kf kr km spec εr εm : ℝ} (hr : |kf - kr| ≤ εr) (hm : |kr - km| ≤ εm)
    (ha : km = spec) : |kf - spec| ≤ εr + εm := by
  subst ha
  calc |kf - km| = |(kf - kr) + (kr - km)| := by ring_nf
    _ ≤ |kf - kr| + |kr - km| := abs_add_le _ _
    _ ≤ εr + εm := add_le_add hr hm

end PallasLean

end
