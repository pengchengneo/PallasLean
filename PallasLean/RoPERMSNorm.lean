import Mathlib

/-!
# Fused QK RMSNorm + NeoX RoPE Formal Theory (`sglang.qk_norm_rope`)

Formalizes the algebraic properties of the fused QK RMSNorm + NeoX rotary
position embedding kernel (`sglang_tpu/qk_norm_rope.py`) on TPU v7x:

1. **RoPE 2D Pair Isometry (`ropePair_norm_sq`)**: each rotated channel pair
   `(x₁ cos θ - x₂ sin θ, x₂ cos θ + x₁ sin θ)` preserves Euclidean norm under
   `cos² θ + sin² θ = 1`.
2. **RoPE Relative-Position Dot-Product Invariance (`ropePair_dot_rel`)**:
   rotating `q` by `(cos α, sin α)` and `k` by `(cos β, sin β)` yields an inner
   product depending only on the relative angle `(cos(β - α), sin(β - α))`.
3. **RMSNorm Positive Scale Invariance (`rmsNormElem_scale_pos`)**: scaling an
   input vector by `c > 0` leaves its unregularized RMSNorm unchanged.
4. **QKV `V`-Slice Disjointness (`qkv_v_slice_disjoint`)**: the `BlockSpec` window
   `[0, q_size + kv_size)` is strictly disjoint from the `V` columns
   `[q_size + kv_size, q_size + 2 * kv_size)`.
-/

noncomputable section

namespace PallasLean.RoPERMSNorm

/-- NeoX RoPE rotation on one channel pair `(x₁, x₂)` with `(c, s) = (cos θ, sin θ)`:
`y₁ = x₁ c - x₂ s`, `y₂ = x₂ c + x₁ s`. -/
def rope1 (x1 x2 c s : ℝ) : ℝ := x1 * c - x2 * s
def rope2 (x1 x2 c s : ℝ) : ℝ := x2 * c + x1 * s

/-- **RoPE norm preservation (isometry).** On the unit circle `c² + s² = 1`,
`(rope1 x₁ x₂ c s)² + (rope2 x₁ x₂ c s)² = x₁² + x₂²`. -/
theorem ropePair_norm_sq (x1 x2 c s : ℝ) (hcs : c ^ 2 + s ^ 2 = 1) :
    (rope1 x1 x2 c s) ^ 2 + (rope2 x1 x2 c s) ^ 2 = x1 ^ 2 + x2 ^ 2 := by
  unfold rope1 rope2
  calc (x1 * c - x2 * s) ^ 2 + (x2 * c + x1 * s) ^ 2
      = (x1 ^ 2 + x2 ^ 2) * (c ^ 2 + s ^ 2) := by ring
    _ = (x1 ^ 2 + x2 ^ 2) * 1 := by rw [hcs]
    _ = x1 ^ 2 + x2 ^ 2 := by ring

/-- **RoPE relative-position dot-product invariance.**
The inner product of `(rope1 q₁ q₂ cα sα, rope2 q₁ q₂ cα sα)` and
`(rope1 k₁ k₂ cβ sβ, rope2 k₁ k₂ cβ sβ)` equals the inner product of `(q₁, q₂)` with
`k` rotated by the relative angle `(cα cβ + sα sβ, cα sβ - sα cβ)`. -/
theorem ropePair_dot_rel (q1 q2 k1 k2 ca sa cb sb : ℝ) :
    rope1 q1 q2 ca sa * rope1 k1 k2 cb sb + rope2 q1 q2 ca sa * rope2 k1 k2 cb sb =
      q1 * rope1 k1 k2 (ca * cb + sa * sb) (ca * sb - sa * cb) +
        q2 * rope2 k1 k2 (ca * cb + sa * sb) (ca * sb - sa * cb) := by
  unfold rope1 rope2
  ring

/-- One element of RMSNorm: `x_i * (1 / sqrt(ms + eps)) * w_i`. -/
def rmsNormElem (xi ms eps wi : ℝ) : ℝ :=
  xi * (1 / Real.sqrt (ms + eps)) * wi

/-- **RMSNorm positive scale invariance** (when `eps = 0`): scaling `x` by `c > 0`
scales the mean square `ms` by `c²`, leaving `rmsNormElem` identical. -/
theorem rmsNormElem_scale_pos (c xi ms wi : ℝ) (hc : 0 < c) (hms : 0 < ms) :
    rmsNormElem (c * xi) (c ^ 2 * ms) 0 wi = rmsNormElem xi ms 0 wi := by
  unfold rmsNormElem
  simp only [add_zero]
  rw [Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq hc.le]
  have h1 : Real.sqrt ms ≠ 0 := (Real.sqrt_pos.mpr hms).ne'
  have h2 : c ≠ 0 := hc.ne'
  field_simp

/-- **QKV `V`-slice isolation.** In `sglang.qk_norm_rope`, the input `BlockSpec`
loads columns `col < q_size + kv_size`, so no column in the `V` range
`[q_size + kv_size, q_size + 2 * kv_size)` is ever accessed by the DMA window. -/
theorem qkv_v_slice_disjoint {q_size kv_size col : ℕ}
    (h_in_v : q_size + kv_size ≤ col) :
    ¬ (col < q_size + kv_size) :=
  not_lt.mpr h_in_v

end PallasLean.RoPERMSNorm

end
