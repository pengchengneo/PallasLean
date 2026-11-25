import Mathlib

/-!
# Pallas TPU pipeline semantics for outputs

The grid runs sequentially on one TensorCore (`arbitrary` dims; `parallel`
dims are split across cores, handled by the frontend's disjointness check).
Each output has one VMEM buffer per block. Pallas does **not** load an output
block from HBM when the block index changes; it only writes the buffer back
when the index changes or the grid ends.

`sim` is that rule, for the scalar contents of one output's blocks.
-/

namespace PallasLean.Pipeline

variable {α : Type*}

/-- Pipeline state after some steps: the open block, its buffer, and HBM. -/
structure St (α : Type*) where
  cur : ℕ
  buf : α
  hbm : ℕ → α

/-- One grid step `t` with output block `ob t`. On a block change the old
buffer is written back and the new one starts from `garbage` (it is not read
from HBM). `body t first buf` returns the new buffer contents. -/
def step (ob : ℕ → ℕ) (garbage : α) (body : ℕ → Bool → α → α) (s : St α) (t : ℕ) : St α :=
  if ob t = s.cur then { s with buf := body t false s.buf }
  else { cur := ob t, buf := body t true garbage, hbm := Function.update s.hbm s.cur s.buf }

/-- Run steps `1..n` after step `0` opened block `ob 0`. -/
def run (ob : ℕ → ℕ) (garbage : α) (body : ℕ → Bool → α → α) (hbm0 : ℕ → α) : ℕ → St α
  | 0 => { cur := ob 0, buf := body 0 true garbage, hbm := hbm0 }
  | n + 1 => step ob garbage body (run ob garbage body hbm0 n) (n + 1)

/-- HBM after the final write-back. -/
def final (ob : ℕ → ℕ) (garbage : α) (body : ℕ → Bool → α → α) (hbm0 : ℕ → α) (n : ℕ) : ℕ → α :=
  let s := run ob garbage body hbm0 n
  Function.update s.hbm s.cur s.buf

/-- Scatter spec: block `b` holds the value of the last step `t <= n` with
`ob t = b`, or keeps its initial value (needs `input_output_aliases`). -/
def lastWrite (ob : ℕ → ℕ) (val : ℕ → α) (hbm0 : ℕ → α) : ℕ → ℕ → α
  | 0, b => if ob 0 = b then val 0 else hbm0 b
  | n + 1, b => if ob (n + 1) = b then val (n + 1) else lastWrite ob val hbm0 n b

theorem run_cur (ob : ℕ → ℕ) (g : α) (body : ℕ → Bool → α → α) (h0 : ℕ → α) :
    ∀ n, (run ob g body h0 n).cur = ob n
  | 0 => rfl
  | n + 1 => by
    simp only [run, step]; split_ifs with h
    · exact h.symm
    · rfl

theorem lastWrite_last (ob : ℕ → ℕ) (val : ℕ → α) (h0 : ℕ → α) (n : ℕ) :
    lastWrite ob val h0 n (ob n) = val n := by cases n <;> simp [lastWrite]

/-- **Full-overwrite kernels are scatters.** If the body ignores the buffer
(`kv_page_write`: `out_ref[...] = pages_ref[...]`), the final HBM is
last-write-wins for any index sequence, duplicates and revisits included. -/
theorem overwrite_is_scatter (ob : ℕ → ℕ) (g : α) (val : ℕ → α) (h0 : ℕ → α) (n : ℕ) :
    final ob g (fun t _ _ => val t) h0 n = lastWrite ob val h0 n := by
  -- invariant: buffer holds `val n`; HBM matches the spec off the open block.
  have inv : ∀ n, (run ob g (fun t _ _ => val t) h0 n).buf = val n ∧
      ∀ b, b ≠ ob n → (run ob g (fun t _ _ => val t) h0 n).hbm b = lastWrite ob val h0 n b := by
    intro n
    induction n with
    | zero => exact ⟨rfl, fun b hb => by simp [run, lastWrite, Ne.symm hb]⟩
    | succ n ih =>
      obtain ⟨hbuf, hhbm⟩ := ih
      have hc := run_cur ob g (fun t _ _ => val t) h0 n
      simp only [run, step, hc]
      split_ifs with h
      · refine ⟨rfl, fun b hb => ?_⟩
        simp only [lastWrite, Ne.symm hb, ↓reduceIte]
        exact hhbm b (by rw [← h]; exact hb)
      · refine ⟨rfl, fun b hb => ?_⟩
        simp only [lastWrite, Ne.symm hb, ↓reduceIte]
        by_cases hbn : b = ob n
        · subst hbn; rw [Function.update_self, hbuf, lastWrite_last]
        · rw [Function.update_of_ne hbn]; exact hhbm b hbn
  funext b
  obtain ⟨hbuf, hhbm⟩ := inv n
  have hc := run_cur ob g (fun t _ _ => val t) h0 n
  simp only [final, hc]
  by_cases hb : b = ob n
  · subst hb; rw [Function.update_self, hbuf, lastWrite_last]
  · rw [Function.update_of_ne hb]; exact hhbm b hb

/-- **Accumulators need contiguous visits.** Counter-model: output block `0`
is visited at steps `0` and `2` with block `1` in between. Each visit adds one;
the spec is `2`, the pipeline writes `1` because the second visit restarts from
the (here zero) garbage buffer instead of reloading HBM. -/
theorem noncontiguous_accumulator_loses :
    let ob : ℕ → ℕ := fun t => if t = 1 then 1 else 0
    let body : ℕ → Bool → ℕ → ℕ := fun _ first buf => (if first then 0 else buf) + 1
    final ob 0 body (fun _ => 0) 2 0 = 1 := by
  decide

/-- **Contiguous single-block accumulation is sound.** When every step `t`
targets the same output block `b`, the VMEM buffer never flushes mid-run or
restarts from `garbage`, and each step `n + 1` updates the existing VMEM buffer
in place via `body (n + 1) false`. -/
theorem same_block_step (b : ℕ) (g : α) (body : ℕ → Bool → α → α) (h0 : ℕ → α) (n : ℕ) :
    (run (fun _ => b) g body h0 (n + 1)).buf =
      body (n + 1) false (run (fun _ => b) g body h0 n).buf := by
  have hc := run_cur (fun _ => b) g body h0 n
  simp [run, step, hc]

end PallasLean.Pipeline
