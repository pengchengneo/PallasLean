"""Level D: the rounding edge of the triangle, |kernel_Float - kernel_Real| <= eps.

Every rule is a Lean lemma in `PallasLean/Rounding.lean`, evaluated with outward
rounding in exact rationals:

* sums and dot products: `STree.err_bound` (any summation tree, so the MXU /
  cross-lane reduction order is not trusted)
* products: `mul_err`
* exp: `exp_lipschitz` (arguments shifted by the running max are <= 0) plus
  the measured hardware term `ExpModel` (ue, etae): the one transcendental
  assumption, calibrated on TPU by `validate/tpu_validate.py`
* casts and every f32 op: `RoundModel` with u = 2^-24 (f32), 2^-8 (bf16) and
  eta = smallest normal (TPU flushes subnormals to zero)

The dtype chain (where the kernel casts to bf16) is read from the traced
jaxpr, not assumed.
"""
from __future__ import annotations

from fractions import Fraction as Fr
from typing import NamedTuple

U = {"float32": Fr(1, 2**24), "bfloat16": Fr(1, 2**8)}
ETA = {"float32": Fr(2) ** -126, "bfloat16": Fr(2) ** -126}


def gamma(u: Fr, n: int) -> Fr:
    """(1+u)^n - 1, the factor in `STree.err_bound`."""
    return (1 + u) ** n - 1


def tree_sum(n: int, asum: Fr, esum: Fr, dt="float32"):
    """`STree.err_bound` for n leaves with sum|x| <= asum, sum e <= esum."""
    u, eta = U[dt], ETA[dt]
    g = (1 + u) ** (n - 1)
    return g * esum + (g - 1) * asum + (n - 1) * g * eta


def rnd(x_abs: Fr, dt="float32") -> Fr:
    """One rounding of a value with |x| <= x_abs (`RoundModel.err`)."""
    return U[dt] * x_abs + ETA[dt]


class ExpModel(NamedTuple):
    ue: Fr  # relative error of hardware exp on (-inf, 0]
    etae: Fr  # absolute error


def exp_rel(arg_err: Fr, em: ExpModel) -> Fr:
    """Relative error of fexp(a~) vs exp(a), |a~ - a| <= arg_err, a <= 0:
    `exp_lipschitz` with M = max(a, a~) gives |exp a~ - exp a| <= exp a * (e^d - 1),
    e^d - 1 <= d (1 + d) for d <= 1; plus the hardware term `ExpModel`."""
    return arg_err * (1 + arg_err) + em.ue * (1 + arg_err * (1 + arg_err))


class AttnShape(NamedTuple):
    n_kv: int  # keys per softmax row
    head_dim: int
    q_abs: Fr  # |q| bound (requires)
    k_abs: Fr
    v_abs: Fr
    scale: Fr
    p_cast: str  # dtype p is cast to before the PV matmul ("float32" = no cast)
    out_dtype: str


def _score_err(sh: AttnShape) -> tuple[Fr, Fr]:
    """Absolute error and magnitude of s = scale * q.k (bf16 x bf16 products are
    exact in f32, so only the D-leaf f32 sum rounds: `STree.err_bound`)."""
    qk = sh.head_dim * sh.q_abs * sh.k_abs
    e = tree_sum(sh.head_dim, qk, Fr(0))
    s_abs = sh.scale * (qk + e)
    return sh.scale * e + rnd(s_abs), s_abs


def _rel_sum(n: int, rho: Fr) -> Fr:
    """Relative error of a sum of n positive leaves each with relative error rho:
    `STree.err_bound` with asum = S, esum = rho S, per unit S (eta terms use S >= 1)."""
    g = (1 + U["float32"]) ** (n - 1)
    return g * rho + (g - 1) + (n - 1) * g * ETA["float32"]


def softmax_then_pv(sh: AttnShape, em: ExpModel) -> dict:
    """sglang single-step path: p = exp(s - m); l = sum p; p /= l;
    o = p.astype(bf16) @ v; o.astype(out). Weights w = p / l sum to 1."""
    e_s, s_abs = _score_err(sh)
    e_arg = 2 * e_s + rnd(2 * s_abs)  # s - m (m is an exact max of computed s)
    rho_p = exp_rel(e_arg, em)
    rho_l = _rel_sum(sh.n_kv, rho_p)
    u32 = U["float32"]
    # w~ = fl(p~ / l~), then the cast: relative errors compose multiplicatively
    rho_w = (1 + rho_p) * (1 + u32) / (1 - rho_l) - 1
    if sh.p_cast != "float32":
        rho_w = (1 + rho_w) * (1 + U[sh.p_cast]) - 1
    # o = sum w~ v: bf16 x bf16 leaves are exact in f32; sum w |v| <= V
    leaf_rel = rho_w if sh.p_cast != "float32" else (1 + rho_w) * (1 + u32) - 1
    e_o = _rel_sum(sh.n_kv, leaf_rel) * (1 + rho_w) * sh.v_abs
    e_out = e_o + (rnd(sh.v_abs + e_o, sh.out_dtype) if sh.out_dtype != "float32" else 0)
    return {"eps_abs": e_out, "eps_rel_V": e_out / sh.v_abs,
            "terms": {"score_abs": e_s, "rho_p": rho_p, "rho_l": rho_l, "rho_w": rho_w,
                      "o_before_cast": e_o}}


def online_normalized(sh: AttnShape, em: ExpModel, n_blocks: int, block: int) -> dict:
    """maxtext ragged: per block s = exp(qk - m_c), l_c = sum s, oc = s @ v (f32),
    m' = max, alpha = exp(m - m'), beta = exp(m_c - m'), l' = alpha l + beta l_c,
    o' = (l alpha o + beta oc) / l'. Every accumulated weight is a positive term,
    so track one relative perturbation per weight: (1 + c)^steps."""
    e_s, s_abs = _score_err(sh)
    u32 = U["float32"]
    rho_p = exp_rel(2 * e_s + rnd(2 * s_abs), em)
    # alpha / beta: exp of a difference of two exact maxima, rounded once
    rho_ab = exp_rel(rnd(2 * s_abs), em)
    # a weight entering at block k: exp, (x v leaf product), x beta, then per later
    # step x alpha, x l (o*l), x o-update add, / l': 5 f32 roundings + alpha
    per_step = (1 + rho_ab) * (1 + u32) ** 5 - 1
    first = (1 + rho_p) * (1 + rho_ab) * (1 + u32) ** 2 - 1
    rho_wt = (1 + first) * (1 + per_step) ** (n_blocks - 1) - 1
    rho_U = _rel_sum(block, rho_wt) + n_blocks * u32  # block sums, then n_blocks adds
    rho_L = _rel_sum(block, rho_wt) + n_blocks * u32
    e_o = (rho_U + rho_L) * sh.v_abs / (1 - rho_L)
    e_out = e_o + (rnd(sh.v_abs + e_o, sh.out_dtype) if sh.out_dtype != "float32" else 0)
    return {"eps_abs": e_out, "eps_rel_V": e_out / sh.v_abs,
            "terms": {"score_abs": e_s, "rho_p": rho_p, "rho_ab": rho_ab, "rho_wt": rho_wt}}


def dtype_chain(model) -> dict:
    """Read from the kernel body whether p is cast to bf16 before PV and the
    output dtype: the frontend never assumes the precision plan."""
    from jax._src import core as jc
    casts = []

    def walk(j):
        for e in j.eqns:
            if e.primitive.name == "convert_element_type":
                casts.append((str(e.invars[0].aval.dtype), str(e.params["new_dtype"]),
                              tuple(getattr(e.invars[0].aval, "shape", ()))))
            for v in e.params.values():
                for x in (v if isinstance(v, (tuple, list)) else (v,)):
                    if isinstance(x, jc.Jaxpr):
                        walk(x)
    walk(model.body)
    return {"f32_to_bf16": [c for c in casts if c[0] == "float32" and c[1] == "bfloat16"],
            "all": casts}


def gmm_roundoff(k_dim: int, tiles_k: int, lhs_abs: Fr, rhs_abs: Fr, out_dtype: str = "float32") -> dict:
    """Level D rounding bound for Grouped Matrix Multiply (`maxtext.megablox_gmm`):
    each `tk = k_dim // tiles_k` tile multiplies bf16*bf16 exactly in f32 and sums
    via `STree.err_bound`, then accumulates `tiles_k` partial sums in f32."""
    tk = k_dim // tiles_k
    tile_asum = tk * lhs_abs * rhs_abs
    e_tile = tree_sum(tk, tile_asum, Fr(0), "float32")
    e_acc = tree_sum(tiles_k, tiles_k * tile_asum, tiles_k * e_tile, "float32")
    full_abs = tiles_k * tile_asum + e_acc
    e_out = e_acc + (rnd(full_abs, out_dtype) if out_dtype != "float32" else Fr(0))
    return {"eps_abs": e_out, "terms": {"tile_err": e_tile, "acc_err": e_acc}}


def qk_norm_rope_roundoff(hd: int, x_abs: Fr, w_abs: Fr, min_rms: Fr, out_dtype: str = "bfloat16") -> dict:
    """Level D rounding bound for `sglang.qk_norm_rope` (RMSNorm + NeoX RoPE):
    `x / rms <= x_abs / min_rms`, scaled by `w_abs`, rotated by `(cos, sin) in [-1, 1]`."""
    u32 = U["float32"]
    # x^2 is exact for bf16->f32; mean of hd squares has relative error gamma(u32, hd)
    rho_ms = (1 + u32) ** hd - 1
    # rsqrt + multiply by x and weight
    rho_norm = (1 + rho_ms) * (1 + u32) ** 4 - 1
    norm_abs = (x_abs / min_rms) * w_abs
    e_norm = rho_norm * norm_abs + 4 * ETA["float32"]
    # RoPE: x1*c - x2*s (2 products + 1 add in f32, |c|,|s| <= 1)
    rope_abs = 2 * norm_abs
    e_rope = 2 * e_norm + 2 * rnd(norm_abs, "float32") + rnd(rope_abs, "float32")
    e_out = e_rope + (rnd(rope_abs + e_rope, out_dtype) if out_dtype != "float32" else Fr(0))
    return {"eps_abs": e_out, "terms": {"e_norm": e_norm, "e_rope": e_rope}}

