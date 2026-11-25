import Mathlib

/-!
# TensorLean-Style 5-Tuple Kernel Specification & TPU v7x Architecture Model

Following TensorLean (Qiang et al., 2026), a TPU kernel specification decouples
mathematical tensor computation over `ℝ` from machine memory layouts and
floating-point rounding via five components:

* `spec_math`   : mathematical computation `(State → Input → Output)` over `ℝ`
* `spec_mut`    : in-place HBM state update `(State → Input → State)`
* `spec_layout` : mapping from logical tensor coordinates to HBM block / VMEM tile offsets
* `spec_tol`    : allowed floating-point deviation `ε ≥ 0`
* `requires`    : precondition on `State` and `Input` (shapes, bounds, SMEM prefetch invariants)

We also formalize `assert_cut` (the cut rule for `pallaslean_assert` proof hints)
and the hardware constants of **TPU v7x** (`ChipVersion.TPU_7X`).
-/

noncomputable section

namespace PallasLean

/-- **TPU v7x (`TPU7x`) per-TensorCore hardware architecture parameters.** -/
structure TPUv7xArch where
  numSublanes : ℕ := 8
  numLanes : ℕ := 128
  mxuColumnSize : ℕ := 256
  numMXUs : ℕ := 2
  numAccumulators : ℕ := 128
  vmemCapacityBytes : ℕ := 67108864   -- 64 MiB per TensorCore
  smemCapacityBytes : ℕ := 1048576    -- 1 MiB per TensorCore
  hbmCapacityBytes : ℕ := 103000000000

/-- Canonical TPU v7x TensorCore configuration. -/
def v7x : TPUv7xArch := {}

/-- Check whether a kernel's VMEM and SMEM footprint fits on one TPU v7x TensorCore. -/
def fitsV7xMemory (vmemBytes smemBytes : ℕ) : Prop :=
  vmemBytes ≤ v7x.vmemCapacityBytes ∧ smemBytes ≤ v7x.smemCapacityBytes

instance (v s : ℕ) : Decidable (fitsV7xMemory v s) := by
  unfold fitsV7xMemory; infer_instance

/-- **TensorLean 5-tuple specification** for a TPU kernel with read-only `Input`,
in-place mutable `State` (e.g. KV cache), and freshly written `Output` (indexed by `ι`). -/
structure KernelSpec (State Input : Type*) (ι : Type*) where
  /-- Pure mathematical computation over `ℝ` for each output index `i : ι`. -/
  spec_math : State → Input → ι → ℝ
  /-- In-place HBM state update (`State → Input → State`). -/
  spec_mut : State → Input → State
  /-- Logical-to-physical memory layout map `(output index → HBM flat address)`. -/
  spec_layout : ι → ℕ
  /-- Maximum allowed floating-point deviation `ε` per output element. -/
  spec_tol : ℝ
  /-- Precondition on `State` and `Input` (includes SMEM scalar-prefetch invariants). -/
  requires : State → Input → Prop

/-- Extensional correctness of a floating-point kernel execution `(execOut, execMut)`
against a `KernelSpec`: under `K.requires s inp`, every output element is within
`K.spec_tol` of `K.spec_math s inp i` and the mutated state equals `K.spec_mut s inp`. -/
def SatisfiesSpec {State Input ι : Type*} (K : KernelSpec State Input ι)
    (execOut : State → Input → ι → ℝ) (execMut : State → Input → State) : Prop :=
  ∀ s inp, K.requires s inp →
    (∀ i : ι, |execOut s inp i - K.spec_math s inp i| ≤ K.spec_tol) ∧
    execMut s inp = K.spec_mut s inp

/-- **Bridge theorem (Math-to-Machine Triangle for `KernelSpec`).**
If for every valid `(s, inp)` and index `i`:
1. `|execOut s inp i - realOut s inp i| ≤ εr` (Level D: round-off error bound),
2. `|realOut s inp i - maskOut s inp i| ≤ εm` (Level C/D: finite-mask error bound),
3. `maskOut s inp i = K.spec_math s inp i` (Level B/C: algebraic exactness over `ℝ`),
4. `εr + εm ≤ K.spec_tol`, and
5. `execMut s inp = K.spec_mut s inp` (Level A/B: pipeline scatter / state update),
then the kernel satisfies `K`. -/
theorem satisfiesSpec_of_triangle {State Input ι : Type*}
    (K : KernelSpec State Input ι)
    (execOut realOut maskOut : State → Input → ι → ℝ)
    (execMut : State → Input → State)
    (εr εm : ℝ)
    (hround : ∀ s inp, K.requires s inp → ∀ i, |execOut s inp i - realOut s inp i| ≤ εr)
    (hmask : ∀ s inp, K.requires s inp → ∀ i, |realOut s inp i - maskOut s inp i| ≤ εm)
    (halg : ∀ s inp, K.requires s inp → ∀ i, maskOut s inp i = K.spec_math s inp i)
    (htol : εr + εm ≤ K.spec_tol)
    (hmut : ∀ s inp, K.requires s inp → execMut s inp = K.spec_mut s inp) :
    SatisfiesSpec K execOut execMut := by
  intro s inp hreq
  refine ⟨fun i => ?_, hmut s inp hreq⟩
  have hr := hround s inp hreq i
  have hm := hmask s inp hreq i
  have ha := halg s inp hreq i
  rw [← ha]
  calc |execOut s inp i - maskOut s inp i|
      = |(execOut s inp i - realOut s inp i) + (realOut s inp i - maskOut s inp i)| := by ring_nf
    _ ≤ |execOut s inp i - realOut s inp i| + |realOut s inp i - maskOut s inp i| := abs_add_le _ _
    _ ≤ εr + εm := add_le_add hr hm
    _ ≤ K.spec_tol := htol

/-- **`pallaslean_assert` Cut Rule.**
Proving an intermediate assertion `P_assert` once from the kernel context `Γ`
allows every downstream verification condition `P_goal` to assume `P_assert`
instead of expanding the upstream computation tree. -/
theorem assert_cut {Γ P_assert P_goal : Prop}
    (h_assert : Γ → P_assert) (h_goal : Γ → P_assert → P_goal) :
    Γ → P_goal :=
  fun hγ => h_goal hγ (h_assert hγ)

end PallasLean

end
