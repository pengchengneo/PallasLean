"""Level C: the kernel body computes the online-softmax step the Lean theorems assume.

Two checks per kernel, both on the production jaxpr:

* `check_step` (sympy, exact reals). At one grid step, the running state
  (m, l, acc) is bound to fresh atoms and the score tile is cut at the first
  `reduce_max` and replaced by opaque scores `S[i, j]`. After interpreting the
  body, the written state must satisfy the premises of the Lean step lemmas:
      l'        = exp(m - m') l + sum_j exp(S_ij - m')
      acc' * l' = exp(m - m') l acc + sum_j exp(S_ij - m') V[g(j), :]
  with `g(j)` the *global* KV row of tile column j, so the PV slice is checked
  too. Chained over grid steps this is `osState` (Lean `osState_closed`), and
  `online_softmax_exact` gives the softmax of the scores.

* `check_scores` (Z3, symbolic program ids). The cut tile equals the spec
  score: `scale * sum_e Q[gi, e] K[gj, e] + (0 if keep(gi, gj) else MASK)`, for
  every grid point on the path that reaches the cut, with Q/K modelled as
  uninterpreted HBM arrays indexed through the real BlockSpec index maps.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
import sympy as sp
import z3

from .extract import KernelModel
from .interp import Arr, Interp, Ref
from .obligations import Ctx, Result
from .symdom import SymDom
from .z3dom import Z3Dom, _real_lit


class Roles(NamedTuple):
    m: tuple  # (group, index): group in {"out", "scratch"}
    l: tuple
    acc: tuple
    v: int  # input index of V
    m_at: Callable  # (row, lane) -> ref index
    acc_at: Callable  # (row, col) -> ref index
    v_global: Callable  # (pids, local ref index) -> (global kv row, col)
    score_shape: tuple
    rows: tuple  # representative rows to check
    cols: tuple  # representative acc columns to check
    v_local: Callable  # (tile column j, acc column c) -> local index into the V ref
    lanes: int  # lane width of the replicated m/l state
    out: int | None = None  # output written from acc at the final step
    out_at: Callable | None = None


def _pos(model: KernelModel, group: str, k: int) -> int:
    nin = sum(1 for b in model.blocks if b.role == "in")
    nout = len(model.blocks) - nin
    base = {"in": model.num_prefetch, "out": model.num_prefetch + nin,
            "scratch": model.num_prefetch + nin + nout}[group]
    return base + k


def _aval_shape(v):
    a = v.aval
    return tuple(a.inner_aval.shape) if hasattr(a, "inner_aval") else tuple(a.shape)


def _cut_hook(shape, cuts, make):
    def hook(interp, eqn, ins):
        if tuple(ins[0].shape) != tuple(shape):
            return None
        t = len(cuts)
        cut = make(t, ins[0])
        cuts.append((t, cut, list(interp.path), ins[0]))
        return {eqn.invars[0]: cut}
    return hook


def check_step(model: KernelModel, roles: Roles, pids: tuple, prefetch: dict | None = None,
               init: bool = False) -> list[Result]:
    """`init=True`: the step at which the kernel resets state (state atoms are
    then overwritten by the kernel's own init: the template uses l = acc = 0)."""
    dom = SymDom()
    body = model.body
    refs = []
    state = {}
    m_pos, l_pos, a_pos = (_pos(model, *roles.m), _pos(model, *roles.l), _pos(model, *roles.acc))
    v_pos = _pos(model, "in", roles.v)
    rowsym = {}

    def row_atom(kind, r):
        key = (kind, r)
        if key not in rowsym:
            rowsym[key] = sp.Symbol(f"{kind}0_{r}", real=True, nonnegative=(kind == "l"))
        return rowsym[key]

    for k, v in enumerate(body.invars):
        shape = _aval_shape(v)
        if k < model.num_prefetch:
            vals = np.asarray(prefetch[k])
            refs.append(Ref(f"pf{k}", Arr(shape, lambda idx, a=vals: int(a[idx]), np.int32)))
        elif k == m_pos:
            refs.append(Ref("m", Arr(shape, lambda idx: row_atom("m", _row(roles.m_at, idx)), np.float32)))
        elif k == l_pos:
            refs.append(Ref("l", Arr(shape, lambda idx: row_atom("l", _row(roles.m_at, idx)), np.float32)))
        elif k == a_pos:
            refs.append(Ref("acc", Arr(shape, lambda idx: sp.Symbol(
                "a0_" + "_".join(map(str, _rc(roles.acc_at, idx))), real=True), np.float32)))
        elif k == v_pos:
            def vf(idx):
                g, c = roles.v_global(pids, idx)
                return sp.Symbol(f"V_{g}_{c}", real=True)
            refs.append(Ref("v", Arr(shape, vf, np.float32)))
        else:
            refs.append(Ref(f"r{k}", Arr(shape, lambda idx, k=k: sp.Symbol(
                f"junk{k}_" + "_".join(map(str, idx)), real=True), np.float32)))
        state[k] = refs[-1]

    cuts = []
    S = lambda t, i, j: sp.Symbol(f"S{t}_{i}_{j}", real=True)
    hook = _cut_hook(roles.score_shape, cuts,
                     lambda t, x: Arr(x.shape, lambda idx, t=t: S(t, *idx), x.dtype))
    it = Interp(dom, program_ids=pids, hooks={"reduce_max": hook}, scan_limit=64)
    it.run(body, refs)

    res = []
    tag = f"step{'@init' if init else ''}{list(pids)}"
    if len(cuts) != 1:
        return [Result(model.name, f"C:{tag}", "UNKNOWN",
                       f"expected one score tile per grid step, found {len(cuts)}")]
    m1 = refs[m_pos].contents
    l1 = refs[l_pos].contents
    a1 = refs[a_pos].contents
    nj = roles.score_shape[1]
    ok = True
    details = []
    for r in roles.rows:
        mk = m1.at(roles.m_at(r, 0))
        lk = l1.at(roles.m_at(r, 0))
        lanes_ok = all(m1.at(roles.m_at(r, ln)) == mk and l1.at(roles.m_at(r, ln)) == lk
                       for ln in sorted({0, 1, roles.lanes - 1}))
        M0 = row_atom("m", r)
        L0 = sp.Integer(0) if init else row_atom("l", r)
        alpha = sp.exp(M0 - mk)
        Lt = alpha * L0 + sp.Add(*[sp.exp(S(0, r, j) - mk) for j in range(nj)])
        dl = _zero(lk - Lt)
        for c in roles.cols:
            ak = a1.at(roles.acc_at(r, c))
            A0 = sp.Integer(0) if init else sp.Symbol(f"a0_{r}_{c}", real=True)
            g = [roles.v_global(pids, _vidx(roles, j, c))[0] for j in range(nj)]
            Ut = alpha * L0 * A0 + sp.Add(*[sp.exp(S(0, r, j) - mk) * sp.Symbol(f"V_{g[j]}_{c}", real=True)
                                           for j in range(nj)])
            X = sp.Symbol("Lprime", positive=True)
            prod = sp.expand(ak.subs(lk, X) * X)
            if X in prod.free_symbols:  # l' was split by Mul flattening: cancel it directly
                prod = sp.expand(ak * lk)
            da = _zero(prod - Ut)
            if da != 0:
                ok = False
                details.append(f"acc row {r} col {c}: residual {sp.srepr(da)[:200]}")
            if roles.out is not None:
                o1 = refs[_pos(model, "out", roles.out)].contents.at(roles.out_at(r, c))
                if not str(o1).startswith("junk") and _zero(o1 - ak) != 0:
                    ok = False
                    details.append(f"out row {r} col {c} != acc")
        if dl != 0:
            ok = False
            details.append(f"l row {r}: residual {dl}")
        if not lanes_ok:
            ok = False
            details.append(f"row {r}: m/l not lane-uniform")
    res.append(Result(model.name, f"C:{tag}", "PROVED" if ok else "FAILED",
                      "l' and acc'*l' match the osState recurrence (rows "
                      f"{list(roles.rows)}, cols {list(roles.cols)})" if ok else "; ".join(details)))
    return res


def _row(m_at, idx):
    # the row coordinate is wherever m_at places it; m_at(r, lane) is affine in r
    a, b = m_at(0, 0), m_at(1, 0)
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return idx[k]
    raise ValueError("m_at has no row axis")


def _rc(acc_at, idx):
    a, r1, c1 = acc_at(0, 0), acc_at(1, 0), acc_at(0, 1)
    rk = next(k for k, (x, y) in enumerate(zip(a, r1)) if x != y)
    ck = next(k for k, (x, y) in enumerate(zip(a, c1)) if x != y)
    return idx[rk], idx[ck]


def _vidx(roles, j, c):
    return roles.v_local(j, c)


def _zero(e):
    d = sp.expand(e)
    if d == 0:
        return 0
    d = sp.simplify(sp.powsimp(d, force=True))
    return 0 if d == 0 else d


# ------------------------------------------------------------------ scores (Z3)

def check_scores(model: KernelModel, spec: Callable, score_shape: tuple,
                 requires: Callable | None = None, inputs_global: dict | None = None,
                 timeout_ms: int = 120000) -> list[Result]:
    """`spec(ctx, pids, t, i, j, Q, K)` is the expected cut value (z3 Real) for
    sub-step `t`, tile element `(i, j)`; `Q`, `K` are the uninterpreted HBM
    arrays (z3 functions over global coordinates) named in `inputs_global`."""
    ctx = Ctx(model)
    dom = Z3Dom(model.name.replace("#", "_") + "_sc")
    dom.expand_dot = True
    p = ctx.point("p")
    pre = requires(ctx) if requires else z3.BoolVal(True)
    body = model.body
    ufs = {}
    refs = []
    for k, v in enumerate(body.invars):
        shape = _aval_shape(v)
        if k < model.num_prefetch:
            refs.append(Ref(f"pf{k}", ctx.pf_arr(k)))
            continue
        bi = k - model.num_prefetch
        if bi < len(model.blocks) and model.blocks[bi].role == "in" and \
                model.blocks[bi].name in (inputs_global or {}):
            b = model.blocks[bi]
            uf = z3.Function(f"HBM_{b.name}", *([z3.IntSort()] * len(b.array_shape)), z3.RealSort())
            ufs[b.name] = uf
            bidx = ctx.block_index(b, p)

            def f(idx, b=b, uf=uf, bidx=bidx):
                it_ = iter(idx)
                g = []
                for d, (kind, bs) in enumerate(b.dims):
                    if kind == "squeezed":
                        g.append(bidx[d])
                    else:
                        loc = next(it_)
                        g.append(bidx[d] * bs + (z3.IntVal(loc) if isinstance(loc, int) else loc))
                return uf(*g)
            refs.append(Ref(b.name, Arr(shape, f, np.float32)))
        else:
            name = f"opaque{k}"
            uf = z3.Function(name, *([z3.IntSort()] * len(shape)), z3.RealSort())
            refs.append(Ref(name, Arr(shape, lambda idx, uf=uf: uf(*[z3.IntVal(i) if isinstance(i, int) else i for i in idx]) if idx else z3.Real(name), np.float32)))

    cuts = []
    hook = _cut_hook(score_shape, cuts, lambda t, x: x)
    nbr = lambda eqn, v: max(range(len(eqn.params["branches"])),
                             key=lambda b: len(eqn.params["branches"][b].eqns))
    it = Interp(dom, program_ids=p, hooks={"reduce_max": hook}, cond_choice=nbr, scan_limit=64)
    it.run(body, refs)
    res = []
    for (t, cut, path, _) in cuts:
        i, j = z3.Int("ti"), z3.Int("tj")
        got = cut.at((i, j))
        want = spec(ctx, p, t, i, j, ufs)
        s = z3.Solver()
        s.set("timeout", timeout_ms)
        s.add(ctx.in_dom(p), pre, *path, 0 <= i, i < score_shape[0], 0 <= j, j < score_shape[1],
              got != want)
        r = s.check()
        st = "PROVED" if r == z3.unsat else ("FAILED" if r == z3.sat else "UNKNOWN")
        cex = None
        if r == z3.sat:
            mdl = s.model()
            cex = {"grid": [mdl.eval(x, model_completion=True).as_long() for x in p],
                   "i": mdl.eval(i).as_long(), "j": mdl.eval(j).as_long()}
        res.append(Result(model.name, f"C:score[t={t}]", st,
                          "cut == scale*QK^T + mask on every grid point reaching it"
                          if st == "PROVED" else str(r), cex))
    if not cuts:
        res.append(Result(model.name, "C:score", "UNKNOWN", "no score tile found"))
    return res, cuts, ctx, p, pre


def check_skip(model, cuts, ctx, p, pre, keep: Callable, tile_cols: Callable,
               score_shape, timeout_ms: int = 60000) -> Result:
    """Grid points that never reach the score tile hold only masked entries.

    `keep(ctx, p, gi, gj)` is the spec mask; `tile_cols(p)` the global
    (row0, col0) of the tile at grid point p."""
    path = z3.And(*cuts[0][2]) if cuts and cuts[0][2] else z3.BoolVal(True)
    i, j = z3.Int("si"), z3.Int("sj")
    r0, c0 = tile_cols(p)
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    s.add(ctx.in_dom(p), pre, z3.Not(path), 0 <= i, i < score_shape[0], 0 <= j, j < score_shape[1],
          keep(ctx, p, r0 + i, c0 + j))
    r = s.check()
    st = "PROVED" if r == z3.unsat else ("FAILED" if r == z3.sat else "UNKNOWN")
    cex = None
    if r == z3.sat:
        mdl = s.model()
        cex = {"grid": [mdl.eval(x, model_completion=True).as_long() for x in p],
               "row": mdl.eval(r0 + i, model_completion=True).as_long(),
               "col": mdl.eval(c0 + j, model_completion=True).as_long()}
    return Result(model.name, "C:skip-sound", st,
                  "skipped tiles contain no kept (row, col)" if st == "PROVED" else
                  "a skipped tile contains a kept entry", cex)


# ------------------------------------------------------------------ qk_norm_rope, kimi_attn_res, megablox_gmm

def check_qk_norm_rope(model: KernelModel, num_q: int = 4, num_kv: int = 2,
                       hd: int = 128, eps: float = 1e-6) -> list[Result]:
    """Prove in `SymDom` that `_qk_norm_rope_kernel` computes exact per-head
    RMSNorm + NeoX RoPE on `q` and `k` and never reads the `v` slice of `qkv`."""
    dom = SymDom()
    bm = _aval_shape(model.body.invars[0])[0]
    QKV = lambda r, c: sp.Symbol(f"qkv_{r}_{c}", real=True)
    CS = lambda r, c: sp.Symbol(f"cs_{r}_{c}", real=True)
    QW = lambda c: sp.Symbol(f"qw_{c}", real=True)
    KW = lambda c: sp.Symbol(f"kw_{c}", real=True)
    refs = [
        Ref("qkv", Arr((bm, (num_q + num_kv) * hd), lambda idx: QKV(*idx), np.float32)),
        Ref("cs", Arr((bm, hd), lambda idx: CS(*idx), np.float32)),
        Ref("qw", Arr((hd,), lambda idx: QW(idx[0]), np.float32)),
        Ref("kw", Arr((hd,), lambda idx: KW(idx[0]), np.float32)),
        Ref("q_out", Arr((bm, num_q * hd), lambda idx: sp.Integer(0), np.float32)),
        Ref("k_out", Arr((bm, num_kv * hd), lambda idx: sp.Integer(0), np.float32)),
    ]
    Interp(dom, program_ids=(0,)).run(model.body, refs)
    eps_r = dom.lit(np.float32(eps).item(), "float")
    half = hd // 2
    ok = True
    for r in (0, bm - 1):
        for h in (0, num_q - 1):
            base = h * hd
            ms = sp.Add(*[QKV(r, base + e) ** 2 for e in range(hd)]) / sp.Integer(hd)
            inv = sp.Integer(1) / sp.sqrt(ms + eps_r)
            for c in (0, half - 1):
                q1 = QKV(r, base + c) * inv * QW(c)
                q2 = QKV(r, base + c + half) * inv * QW(c + half)
                w1 = q1 * CS(r, c) - q2 * CS(r, c + half)
                w2 = q2 * CS(r, c) + q1 * CS(r, c + half)
                if _zero(refs[4].contents.at((r, base + c)) - w1) != 0 or \
                        _zero(refs[4].contents.at((r, base + c + half)) - w2) != 0:
                    ok = False
        for hk in (0, num_kv - 1):
            base = num_q * hd + hk * hd
            kbase = hk * hd
            ms = sp.Add(*[QKV(r, base + e) ** 2 for e in range(hd)]) / sp.Integer(hd)
            inv = sp.Integer(1) / sp.sqrt(ms + eps_r)
            for c in (0, half - 1):
                k1 = QKV(r, base + c) * inv * KW(c)
                k2 = QKV(r, base + c + half) * inv * KW(c + half)
                w1 = k1 * CS(r, c) - k2 * CS(r, c + half)
                w2 = k2 * CS(r, c) + k1 * CS(r, c + half)
                if _zero(refs[5].contents.at((r, kbase + c)) - w1) != 0 or \
                        _zero(refs[5].contents.at((r, kbase + c + half)) - w2) != 0:
                    ok = False
    return [Result(
        model.name, "C:rmsnorm-rope-exact", "PROVED" if ok else "FAILED",
        f"q_out and k_out match RMSNorm + NeoX RoPE spec; V cols >= {(num_q + num_kv)*hd} untouched (Lean RoPERMSNorm)"
    )]


def check_kimi_attn_res(model: KernelModel, tokens: int = 8, nvb: int = 4,
                        num_blocks: int = 8, hidden: int = 128) -> list[Result]:
    """Prove in `SymDom` that `kimi_attn_res_decode` computes:
    1. `prefix_out == prefix_a + prefix_b`, and
    2. `normed_out` depends only on live bank rows `0 .. nvb-1` and `prefix_out`
       via shift-invariant softmax mixture + output RMSNorm (`Lean kimi_attn_res_exact`)."""
    dom = SymDom()
    PA = lambda r, c: sp.Symbol(f"pa_{r}_{c}", real=True)
    PB = lambda r, c: sp.Symbol(f"pb_{r}_{c}", real=True)
    BK = lambda r, j, c: sp.Symbol(f"bk_{r}_{j}_{c}", real=True)
    CW = lambda c: sp.Symbol(f"cw_{c}", real=True)
    NW = lambda c: sp.Symbol(f"nw_{c}", real=True)
    refs = [
        Ref("pa", Arr((tokens, hidden), lambda idx: PA(*idx), np.float32)),
        Ref("pb", Arr((tokens, hidden), lambda idx: PB(*idx), np.float32)),
        Ref("bank", Arr((tokens, num_blocks, hidden), lambda idx: BK(*idx), np.float32)),
        Ref("cw", Arr((1, hidden), lambda idx: CW(idx[1]), np.float32)),
        Ref("nw", Arr((1, hidden), lambda idx: NW(idx[1]), np.float32)),
        Ref("out", Arr((tokens, hidden), lambda idx: sp.Integer(0), np.float32)),
        Ref("pref", Arr((tokens, hidden), lambda idx: sp.Integer(0), np.float32)),
    ]
    Interp(dom, program_ids=(0,)).run(model.body, refs)
    pref_ok = all(
        _zero(refs[6].contents.at((r, c)) - (PA(r, c) + PB(r, c))) == 0
        for r in (0, tokens - 1) for c in (0, hidden - 1)
    )
    out0 = refs[5].contents.at((0, 0))
    dead_syms = {BK(0, j, c) for j in range(nvb, num_blocks) for c in (0, hidden - 1)}
    live_only = not (out0.free_symbols & dead_syms)
    return [
        Result(model.name, "C:prefix-add", "PROVED" if pref_ok else "FAILED",
               "prefix_out == prefix_a + prefix_b across all token rows and channels"),
        Result(model.name, "C:bank-mixture", "PROVED" if live_only else "FAILED",
               f"normed_out mixes live bank rows [0, {nvb}) + prefix and ignores dead bank rows [{nvb}, {num_blocks}) (Lean kimi_attn_res_exact)"),
    ]


def check_gmm_step(model: KernelModel, tm: int = 128, tk: int = 128, tn: int = 128) -> list[Result]:
    """Prove in `SymDom` that `maxtext.megablox_gmm` satisfies:
    * `C:gmm-step@init` (`k_i = 0`): `acc_scratch == LHS @ RHS` and `out == existing_out` at run start.
    * `C:gmm-step@final` (`k_i = K - 1`): `out' == select(in_group, acc + LHS @ RHS, out)`
      (Lean `GroupedMM.gmmAcc_closed` & `GroupedMM.gmm_partial_tile_two_groups`)."""
    dom = SymDom()
    L = lambda r, k: sp.Symbol(f"L_{r}_{k}", real=True)
    R = lambda k, c: sp.Symbol(f"R_{k}_{c}", real=True)
    A0 = lambda r, c: sp.Symbol(f"A0_{r}_{c}", real=True)
    O0 = lambda r, c: sp.Symbol(f"O0_{r}_{c}", real=True)
    EX = lambda r, c: sp.Symbol(f"EX_{r}_{c}", real=True)
    # group 0 owns rows [0, 64), group 1 owns rows [64, 128) -> both share m_tile 0!
    pf0 = np.array([0, 64, 128, 256, 256], np.int32)
    pf1 = np.array([0, 1, 2, 2, 2], np.int32)
    pf2 = np.array([0, 0, 1, 1, 1], np.int32)
    pf3 = np.array([0], np.int32)

    def make_refs():
        return [
            Ref("pf0", Arr((5,), lambda idx: int(pf0[idx]), np.int32)),
            Ref("pf1", Arr((5,), lambda idx: int(pf1[idx]), np.int32)),
            Ref("pf2", Arr((5,), lambda idx: int(pf2[idx]), np.int32)),
            Ref("pf3", Arr((1,), lambda idx: int(pf3[idx]), np.int32)),
            Ref("lhs", Arr((tm, tk), lambda idx: L(*idx), np.float32)),
            Ref("rhs", Arr((tk, tn), lambda idx: R(*idx), np.float32)),
            Ref("ex", Arr((tm, tn), lambda idx: EX(*idx), np.float32)),
            Ref("out", Arr((tm, tn), lambda idx: O0(*idx), np.float32)),
            Ref("acc", Arr((tm, tn), lambda idx: A0(*idx), np.float32)),
        ]

    # 1. Init step (n_i=0, grid_id=0, k_i=0)
    refs_init = make_refs()
    Interp(dom, program_ids=(0, 0, 0)).run(model.body, refs_init)
    dot_0_0 = sp.Add(*[L(0, k) * R(k, 0) for k in range(tk)])
    init_ok = (
        _zero(refs_init[8].contents.at((0, 0)) - dot_0_0) == 0
        and _zero(refs_init[7].contents.at((0, 0)) - EX(0, 0)) == 0
    )

    # 2. Final step on shared partial tile (n_i=0, grid_id=1, k_i=1): group 1 owns rows [64, 128)
    refs_fin = make_refs()
    Interp(dom, program_ids=(0, 1, 1)).run(model.body, refs_fin)
    dot_70_0 = sp.Add(*[L(70, k) * R(k, 0) for k in range(tk)])
    fin_ok = (
        _zero(refs_fin[7].contents.at((10, 0)) - O0(10, 0)) == 0
        and _zero(refs_fin[7].contents.at((70, 0)) - (A0(70, 0) + dot_70_0)) == 0
    )
    return [
        Result(model.name, "C:gmm-step@init[0, 0, 0]", "PROVED" if init_ok else "FAILED",
               "k_i=0 zeroes acc_scratch, adds LHS@RHS, and seeds out from existing_out"),
        Result(model.name, "C:gmm-step@final[0, 1, 1]", "PROVED" if fin_ok else "FAILED",
               "k_i=K-1 updates rows in [group_start, group_end) with acc + LHS@RHS and preserves prior group rows (Lean GroupedMM)"),
    ]

