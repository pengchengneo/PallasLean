"""Index-level, pipeline, termination, and TPU v7x Mosaic obligations for one `KernelModel`.

These are fully automatic and run on the production jaxpr and TPU v7x Mosaic MLIR
at production shapes. Each obligation is the TPU v7x analogue of a TensorLean check:

* in-bounds        every block index any grid step can produce lies inside the
                   HBM array (TensorLean: memory access errors)
* prefetch-bounds  scalar-prefetch reads stay inside the TPU v7x SMEM array
* parallel-race    blocks written from different `parallel` grid coordinates
                   are disjoint across TPU v7x TensorCores (TensorLean: data races)
* contiguity       an output block's visits form one run of consecutive grid
                   steps (Pallas never reloads an output from HBM; Lean
                   `Pipeline.noncontiguous_accumulator_loses`)
* init-at-start    an accumulator (read before written) is initialised exactly
                   at the first step of each run (TensorLean: uninitialised
                   memory) and never reset mid-run (`no-midrun-reset`)
* write-at-end     a conditionally written output is written at the last step
                   of each run
* coverage         every output block is produced, unless the output aliases
                   an input (`input_output_aliases`)
* termination      every grid dimension and loop (`scan`/`fori_loop`) has a
                   well-founded bound and DMA semaphores are balanced
* v7x-mosaic       lowers cleanly to TPU v7x Mosaic MLIR (`#tpu.core_type<tc>`)
* v7x-vmem-budget  pipelined + scratch VMEM <= 64 MiB and SMEM <= 1 MiB on TPU v7x
                   (Lean `PallasLean.fitsV7xMemory`)
* unused-input     a pipelined input whose block is never read (wasted DMA)
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple

import numpy as np
import z3
from jax._src import core as jc

from .extract import KernelModel
from .interp import Arr, Interp, Unresolved
from .mosaic import V7X_SMEM_CAPACITY_BYTES, V7X_VMEM_CAPACITY_BYTES, analyze_v7x_mosaic, export
from .z3dom import Z3Dom


class Result(NamedTuple):
    kernel: str
    name: str
    status: str  # PROVED | FAILED | UNKNOWN | WARN | N/A
    detail: str
    cex: dict | None = None


# ---------------------------------------------------------------- effects

class Eff:
    __slots__ = ("reads", "writes_some", "writes_all", "rbw")

    def __init__(self):
        self.reads = False
        self.writes_some = False
        self.writes_all = False
        self.rbw = False  # a read that may precede any definite write

    def seq(self, e: "Eff"):
        if e.reads and not self.writes_all:
            self.rbw = True
        self.reads |= e.reads
        self.writes_some |= e.writes_some
        self.writes_all |= e.writes_all
        return self


def _isv(v, pos) -> bool:
    return isinstance(v, jc.Var) and v in pos


def _used_vars(jaxpr) -> set:
    used = set()
    for e in jaxpr.eqns:
        for v in e.invars:
            if isinstance(v, jc.Var):
                used.add(v)
    for v in jaxpr.outvars:
        if isinstance(v, jc.Var):
            used.add(v)
    return used


def _covers_full(ref_aval, idx_vals, tree) -> bool:
    from jax import tree_util
    from jax._src.state import indexing as stix
    transforms = tree_util.tree_unflatten(tree, list(idx_vals))
    shape = ref_aval.inner_aval.shape
    for t in transforms or ():
        if isinstance(t, stix.NDIndexer):
            for d, ix in enumerate(t.indices):
                size = shape[d]
                if isinstance(ix, stix.Slice):
                    if not (isinstance(ix.start, (int, np.integer)) and int(ix.start) == 0
                            and int(ix.size) == size and ix.stride == 1):
                        return False
                elif size != 1:
                    return False
    return True


def _dma_refs(eqn):
    """Return (src_var, dst_var) for a Pallas `dma_start` equation."""
    from jax import tree_util
    try:
        items = tree_util.tree_unflatten(eqn.params["tree"], list(eqn.invars))
        src = getattr(items[0], "ref", items[0])
        dst = getattr(items[1], "ref", items[1])
        return src, dst
    except Exception:
        return (eqn.invars[0] if eqn.invars else None, None)


def effects(jaxpr) -> dict:
    """Per invar position: `Eff` of the whole jaxpr on that ref, in program order."""
    if hasattr(jaxpr, "jaxpr"):
        jaxpr = jaxpr.jaxpr
    pos = {v: i for i, v in enumerate(jaxpr.invars)}
    used = _used_vars(jaxpr)
    out = {}

    def acc(i, e):
        out.setdefault(i, Eff()).seq(e)

    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        if name == "dma_start":
            src, dst = _dma_refs(eqn)
            if _isv(src, pos):
                e = Eff()
                e.reads = True
                acc(pos[src], e)
            if _isv(dst, pos):
                e = Eff()
                e.writes_some = True
                acc(pos[dst], e)
            continue
        if name in ("get", "swap") and _isv(eqn.invars[0], pos):
            e = Eff()
            if name == "get":
                e.reads = True
            else:
                e.reads = isinstance(eqn.outvars[0], jc.Var) and eqn.outvars[0] in used
                e.writes_some = True
                e.writes_all = _covers_full(eqn.invars[0].aval, eqn.invars[2:], eqn.params["tree"])
            acc(pos[eqn.invars[0]], e)
            continue
        if name == "cond":
            ops = eqn.invars[1:]
            per = [effects(b) for b in eqn.params["branches"]]
            for k, v in enumerate(ops):
                if not _isv(v, pos):
                    continue
                e = Eff()
                bs = [p.get(k, Eff()) for p in per]
                e.reads = any(b.reads for b in bs)
                e.rbw = any(b.rbw for b in bs)
                e.writes_some = any(b.writes_some for b in bs)
                e.writes_all = all(b.writes_all for b in bs)
                if e.rbw:
                    e.reads = True
                acc(pos[v], _rbw_first(e))
            continue
        sub = None
        if name == "scan":
            sub = eqn.params["jaxpr"]
        elif name in ("jit", "pjit", "closed_call", "run_scoped"):
            sub = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
        elif name == "while":
            sub = eqn.params["body_jaxpr"]
        if sub is not None:
            se = effects(sub)
            for k, v in enumerate(eqn.invars):
                if _isv(v, pos) and k in se:
                    acc(pos[v], _rbw_first(se[k]))
    return out


def _rbw_first(e: Eff) -> Eff:
    """Collapse a compound effect into one op for sequencing: if its first
    access may be a read, the read precedes its write."""
    x = Eff()
    x.reads = e.rbw or (e.reads and not e.writes_all)
    x.writes_some = e.writes_some
    x.writes_all = e.writes_all
    if e.reads and not e.rbw and e.writes_all:
        x.reads = False  # reads only after its own definite write
    return x


# ---------------------------------------------------------------- context

class Ctx:
    """Symbolic grid point(s) and scalar-prefetch arrays for one kernel."""

    def __init__(self, model: KernelModel):
        self.m = model
        self.G = [
            int(g) if isinstance(g, int) else z3.Int(f"dyn_G{d}")
            for d, g in enumerate(model.grid)
        ]
        self.dyn_bounds = [
            z3.And(g >= 1, g <= 65536) for g, raw in zip(self.G, model.grid) if not isinstance(raw, int)
        ]
        self.dom = Z3Dom(model.name.replace("#", "_"))
        self.pf = [z3.Array(f"pf{k}", z3.IntSort(), z3.IntSort())
                   for k in range(model.num_prefetch)]
        self.pf_reads = []  # (k, flat index term, size)
        self._n = 0

    def point(self, tag):
        return [z3.Int(f"{tag}_g{d}") for d in range(len(self.G))]

    def in_dom(self, p):
        return z3.And(*self.dyn_bounds, *[z3.And(0 <= x, x < g) for x, g in zip(p, self.G)])

    def pf_arr(self, k):
        aval = self.m.prefetch_avals[k]
        shape = tuple(aval.inner_aval.shape) if hasattr(aval, "inner_aval") else tuple(aval.shape)
        size = int(np.prod(shape)) if shape else 1
        A = self.pf[k]

        def f(idx, A=A, shape=shape, size=size, k=k):
            flat = 0
            stride = 1
            for i, s in zip(reversed(idx), reversed(shape)):
                flat = flat + (z3.IntVal(i) if isinstance(i, int) else i) * stride
                stride *= s
            flat = z3.IntVal(flat) if isinstance(flat, int) else flat
            self.pf_reads.append((k, flat, size))
            return A[flat]
        return Arr(shape, f, np.int32)

    def pf_values(self, k, pred: Callable):
        """`pred(i, value)` for every element of prefetch operand `k`."""
        aval = self.m.prefetch_avals[k]
        shape = tuple(aval.inner_aval.shape) if hasattr(aval, "inner_aval") else tuple(aval.shape)
        size = int(np.prod(shape)) if shape else 1
        return z3.And(*[pred(i, self.pf[k][i]) for i in range(size)])

    def block_index(self, b, p):
        it = Interp(self.dom)
        args = [Arr((), lambda idx, x=x: x, np.int32) for x in p]
        args += [self.pf_arr(k) for k in range(self.m.num_prefetch)]
        outs = it.run(b.index_map, args)
        return [o.scalar() for o in outs]

    def out_key(self, p, outs=None):
        outs = outs if outs is not None else [b for b in self.m.blocks if b.role == "out"]
        key = []
        for b in outs:
            key.extend(self.block_index(b, p))
        return key


def _nblocks(b):
    out = []
    for (kind, bs), n in zip(b.dims, b.array_shape):
        out.append(n if kind == "squeezed" else -(-n // bs))
    return out


def lex_lt(p, q):
    terms = []
    for k in range(len(p)):
        terms.append(z3.And(*([p[j] == q[j] for j in range(k)] + [p[k] < q[k]])))
    return z3.Or(*terms)


def succ(G, p, q):
    """`q` is the step right after `p` in grid (row-major) order."""
    n = len(G)
    terms = []
    for k in range(n):
        same = [p[j] == q[j] for j in range(k)]
        carry = [z3.And(p[j] == G[j] - 1, q[j] == 0) for j in range(k + 1, n)]
        terms.append(z3.And(*(same + [q[k] == p[k] + 1] + carry)))
    return z3.Or(*terms)


def _keys_eq(a, b):
    return z3.And(*[x == y for x, y in zip(a, b)]) if a else z3.BoolVal(True)


# ---------------------------------------------------------------- body predicates

class BodyPreds(NamedTuple):
    reads: Any  # z3 Bool per body invar position, or None if never
    writes: Any  # definite full writes
    inits: Any  # definite full writes, before any prior read on that path
    effects: dict


def body_preds(ctx: Ctx, p) -> BodyPreds:
    """Recursively extract path-sensitive read/write/init predicates over root body invars."""
    m = ctx.m
    body = m.body
    it = Interp(ctx.dom, program_ids=p)
    root_env = {}
    root_pos = {}
    for k, v in enumerate(body.invars):
        root_pos[v] = k
        if k < m.num_prefetch:
            root_env[v] = ctx.pf_arr(k)
        else:
            root_env[v] = None  # data refs: not evaluated at top level

    reads, writes, inits = {}, {}, {}
    written_boxes: dict[int, list] = {}
    used_cache = {}
    T = z3.BoolVal(True)
    F = z3.BoolVal(False)

    def add(d, i, c):
        d[i] = c if i not in d else z3.Or(d[i], c)

    def walk(j, env, pos_map, path_cond):
        if hasattr(j, "jaxpr"):
            j = j.jaxpr
        for cv, c in zip(j.constvars, getattr(j, "consts", ()) or ()):
            arr = np.asarray(c)
            if arr.shape == ():
                env[cv] = Arr((), lambda idx, a=arr: ctx.dom.lit(a.item(), "int" if np.issubdtype(a.dtype, np.integer) else "float"), arr.dtype)
        if id(j) not in used_cache:
            used_cache[id(j)] = _used_vars(j)
        used = used_cache[id(j)]

        for eqn in j.eqns:
            name = eqn.primitive.name
            if name == "cond":
                try:
                    idx = it.read(env, eqn.invars[0]).scalar()
                except Exception:
                    idx = ctx.dom.fresh("cidx", z3.IntSort())
                nb = len(eqn.params["branches"])
                clamped = z3.If(idx < 0, 0, z3.If(idx > nb - 1, nb - 1, idx))
                ops = eqn.invars[1:]
                for bi, br in enumerate(eqn.params["branches"]):
                    b_jaxpr = br.jaxpr if hasattr(br, "jaxpr") else br
                    taken = z3.And(path_cond, clamped == bi)
                    b_env = {}
                    b_pos = {}
                    for bv, ov in zip(b_jaxpr.invars, ops):
                        if isinstance(ov, jc.Var):
                            if ov in pos_map:
                                b_pos[bv] = pos_map[ov]
                            if ov in env:
                                b_env[bv] = env[ov]
                        elif isinstance(ov, jc.Literal):
                            b_env[bv] = it.read(env, ov)
                    walk(b_jaxpr, b_env, b_pos, taken)
                continue

            if name == "dma_start":
                src, dst = _dma_refs(eqn)
                if _isv(src, pos_map) and env.get(src) is None:
                    add(reads, pos_map[src], path_cond)
                if _isv(dst, pos_map) and env.get(dst) is None:
                    di = pos_map[dst]
                    prior_r = reads.get(di, F)
                    add(inits, di, z3.And(path_cond, z3.Not(prior_r)))
                    add(writes, di, path_cond)
                continue

            if name in ("get", "swap") and _isv(eqn.invars[0], pos_map) and env.get(eqn.invars[0]) is None:
                i = pos_map[eqn.invars[0]]
                box = _static_slice_box(eqn.invars[0].aval, eqn.invars[2:] if name == "swap" else eqn.invars[1:], eqn.params["tree"])
                if name == "get":
                    if not (z3.is_true(path_cond) and _box_covered(box, written_boxes.get(i, []))):
                        add(reads, i, path_cond)
                else:
                    swap_reads = isinstance(eqn.outvars[0], jc.Var) and eqn.outvars[0] in used
                    if swap_reads and not (z3.is_true(path_cond) and _box_covered(box, written_boxes.get(i, []))):
                        add(reads, i, path_cond)
                    if z3.is_true(path_cond) and box is not None:
                        written_boxes.setdefault(i, []).append(box)
                    full_now = _covers_full(eqn.invars[0].aval, eqn.invars[2:], eqn.params["tree"])
                    if not full_now and z3.is_true(path_cond) and hasattr(eqn.invars[0].aval, "inner_aval"):
                        full_box = tuple((0, int(s)) for s in eqn.invars[0].aval.inner_aval.shape)
                        full_now = _box_covered(full_box, written_boxes.get(i, []))
                    if full_now:
                        prior_r = reads.get(i, F)
                        if not swap_reads:
                            add(inits, i, z3.And(path_cond, z3.Not(prior_r)))
                        add(writes, i, path_cond)
                continue

            if name in ("scan", "while", "jit", "pjit", "closed_call", "run_scoped") and any(_isv(v, pos_map) for v in eqn.invars):
                sub_j = (
                    eqn.params["body_jaxpr"]
                    if name == "while"
                    else (eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr"))
                )
                sub = effects(sub_j)
                for k, v in enumerate(eqn.invars):
                    if _isv(v, pos_map) and k in sub:
                        i = pos_map[v]
                        prior_r = reads.get(i, F)
                        if sub[k].reads:
                            add(reads, i, path_cond)
                        if sub[k].writes_all:
                            if not sub[k].rbw:
                                add(inits, i, z3.And(path_cond, z3.Not(prior_r)))
                            add(writes, i, path_cond)
                continue

            if all(getattr(o.aval, "shape", None) == () for o in eqn.outvars):
                try:
                    ins = [it.read(env, v) for v in eqn.invars]
                    if any(x is None for x in ins):
                        continue
                    outs = it.eval_eqn(eqn, ins)
                    for v, o in zip(eqn.outvars, outs):
                        env[v] = o
                except (Unresolved, NotImplementedError, KeyError, TypeError, AttributeError):
                    pass

    walk(body, root_env, root_pos, T)
    return BodyPreds(reads, writes, inits, effects(body))


# ---------------------------------------------------------------- termination

def _static_slice_box(ref_aval, idx_vals, tree):
    """Return a tuple of (lo, hi) integer bounds per dim if all indices are static constants, else None."""
    from jax import tree_util
    from jax._src.state import indexing as stix
    try:
        transforms = tree_util.tree_unflatten(tree, list(idx_vals))
        shape = ref_aval.inner_aval.shape
        box = [(0, int(s)) for s in shape]
        for t in transforms or ():
            if not isinstance(t, stix.NDIndexer):
                return None
            new_box = []
            for d, ix in enumerate(t.indices):
                lo, hi = box[d]
                if isinstance(ix, stix.Slice):
                    if isinstance(ix.start, jc.Literal) and isinstance(ix.size, (int, np.integer)) and ix.stride == 1:
                        st = int(ix.start.val)
                        new_box.append((lo + st, lo + st + int(ix.size)))
                    elif isinstance(ix.start, (int, np.integer)) and isinstance(ix.size, (int, np.integer)) and ix.stride == 1:
                        st = int(ix.start)
                        new_box.append((lo + st, lo + st + int(ix.size)))
                    else:
                        return None
                elif isinstance(ix, jc.Literal):
                    st = int(ix.val)
                    new_box.append((lo + st, lo + st + 1))
                elif isinstance(ix, (int, np.integer)):
                    st = int(ix)
                    new_box.append((lo + st, lo + st + 1))
                else:
                    return None
            box = new_box
        return tuple(box)
    except Exception:
        return None


def _box_covered(box, written_boxes) -> bool:
    """Return True if static slice `box` is covered by the union of `written_boxes` along a single sliced axis."""
    if not box or not written_boxes:
        return False
    # Exact containment in any single written box
    for wb in written_boxes:
        if len(wb) == len(box) and all(wlo <= blo and bhi <= whi for (blo, bhi), (wlo, whi) in zip(box, wb)):
            return True
    # Union of contiguous intervals along one axis with identical other axes
    ndim = len(box)
    for ax in range(ndim):
        matching = [
            wb[ax] for wb in written_boxes
            if len(wb) == ndim and all(d == ax or (wb[d][0] <= box[d][0] and box[d][1] <= wb[d][1]) for d in range(ndim))
        ]
        if not matching:
            continue
        matching.sort()
        cur_lo, cur_hi = matching[0]
        if cur_lo > box[ax][0]:
            continue
        for mlo, mhi in matching[1:]:
            if mlo <= cur_hi:
                cur_hi = max(cur_hi, mhi)
        if cur_hi >= box[ax][1]:
            return True
    return False


def _check_dma_bounds(model: KernelModel) -> Result | None:
    """Check static slice bounds on every explicit `dma_start` (src and dst) inside the kernel body."""
    from jax import tree_util
    from jax._src.state import indexing as stix

    checked = 0
    violations = []

    def check_tref(tref, role):
        nonlocal checked
        ref = getattr(tref, "ref", tref)
        aval = getattr(ref, "aval", None)
        if aval is None or not hasattr(aval, "inner_aval"):
            return
        shape = tuple(int(s) for s in aval.inner_aval.shape)
        transforms = getattr(tref, "transforms", ()) or ()
        for t in transforms:
            if not isinstance(t, stix.NDIndexer):
                continue
            for d, ix in enumerate(t.indices):
                if d >= len(shape):
                    continue
                dim = shape[d]
                if isinstance(ix, stix.Slice):
                    st = ix.start.val if isinstance(ix.start, jc.Literal) else (ix.start if isinstance(ix.start, (int, np.integer)) else None)
                    sz = ix.size.val if isinstance(ix.size, jc.Literal) else (ix.size if isinstance(ix.size, (int, np.integer)) else None)
                    if st is not None and sz is not None:
                        checked += 1
                        if int(st) < 0 or int(sz) < 0 or int(st) + int(sz) > dim:
                            violations.append(f"{role} dim {d}: [{st}:{int(st)+int(sz)}] out of [0:{dim}]")
                else:
                    val = ix.val if isinstance(ix, jc.Literal) else (ix if isinstance(ix, (int, np.integer)) else None)
                    if val is not None:
                        checked += 1
                        if int(val) < 0 or int(val) >= dim:
                            violations.append(f"{role} dim {d}: idx={val} out of [0:{dim})")

    def walk(j):
        if hasattr(j, "jaxpr"):
            j = j.jaxpr
        for e in j.eqns:
            if e.primitive.name == "dma_start":
                try:
                    items = tree_util.tree_unflatten(e.params["tree"], list(e.invars))
                    check_tref(items[0], "dma_src")
                    check_tref(items[1], "dma_dst")
                except Exception:
                    pass
            for v in e.params.values():
                for x in (v if isinstance(v, (tuple, list)) else (v,)):
                    if isinstance(x, jc.Jaxpr) or hasattr(x, "jaxpr"):
                        walk(x)

    walk(model.body)
    if violations:
        return Result(model.name, "dma-bounds", "FAILED", "; ".join(violations[:3]))
    if checked > 0:
        return Result(model.name, "dma-bounds", "PROVED", f"{checked} static DMA slice bounds verified")
    return None


def _check_termination(model: KernelModel) -> Result:
    """Check that all grid dimensions, scan/while loops, and DMA semaphores terminate."""
    scans = []
    unbounded_whiles = []
    dma_starts = 0
    dma_waits = 0

    def walk(j):
        nonlocal dma_starts, dma_waits
        if hasattr(j, "jaxpr"):
            j = j.jaxpr
        for e in j.eqns:
            nm = e.primitive.name
            if nm == "scan":
                scans.append(int(e.params.get("length", 0)))
            elif nm == "while":
                unbounded_whiles.append(nm)
            elif nm == "dma_start":
                dma_starts += 1
            elif nm == "dma_wait":
                dma_waits += 1
            for v in e.params.values():
                for x in (v if isinstance(v, (tuple, list)) else (v,)):
                    if isinstance(x, jc.Jaxpr) or hasattr(x, "jaxpr"):
                        walk(x)

    walk(model.body)
    if unbounded_whiles:
        return Result(model.name, "termination", "WARN",
                      f"contains data-dependent while loop ({len(unbounded_whiles)})")
    if dma_starts != dma_waits:
        return Result(model.name, "termination", "FAILED",
                      f"unbalanced DMA semaphores: {dma_starts} starts vs {dma_waits} waits")
    g_desc = [str(g) for g in model.grid]
    s_desc = f", scan_lengths={scans}" if scans else ""
    d_desc = f", dma_pairs={dma_starts}" if dma_starts else ""
    return Result(model.name, "termination", "PROVED",
                  f"bounded grid [{', '.join(g_desc)}]{s_desc}{d_desc}")


# ---------------------------------------------------------------- driver

def _status(r) -> str:
    if r == z3.unsat:
        return "PROVED"
    if r == z3.sat:
        return "FAILED"
    return "UNKNOWN"


def _check(s: z3.Solver, timeout_ms=60000):
    s.set("timeout", timeout_ms)
    r = s.check()
    return r


def _cex(ctx, model, pts: dict):
    out = {}
    for tag, p in pts.items():
        out[tag] = [model.eval(x, model_completion=True).as_long() for x in p]
    for k, A in enumerate(ctx.pf):
        vals = {}
        for (kk, flat, size) in ctx.pf_reads:
            if kk != k:
                continue
            try:
                fi = model.eval(flat, model_completion=True).as_long()
            except Exception:
                continue
            if 0 <= fi < size or True:
                vals[fi] = model.eval(A[fi], model_completion=True).as_long()
        if vals:
            out[f"prefetch{k}"] = dict(sorted(vals.items()))
    return out


def check_kernel(model: KernelModel, requires: Callable | None = None,
                 timeout_ms: int = 60000,
                 fn_args: tuple[Callable, tuple] | None = None) -> list[Result]:
    ctx = Ctx(model)
    pre = requires(ctx) if requires else z3.BoolVal(True)
    res = []
    name = model.name
    p = ctx.point("p")
    base = [ctx.in_dom(p), pre]

    # in-bounds
    for b in model.blocks:
        ix = ctx.block_index(b, p)
        nbl = _nblocks(b)
        bad = z3.Or(*[z3.Or(x < 0, x >= n) for x, n in zip(ix, nbl)])
        s = z3.Solver()
        s.add(*base, bad)
        r = _check(s, timeout_ms)
        if r == z3.unsat:
            res.append(Result(name, f"in-bounds[{b.name}]", "PROVED", f"blocks {nbl}"))
        elif r == z3.sat:
            mdl = s.model()
            cx = _cex(ctx, mdl, {"grid": p})
            cx["block_index"] = [mdl.eval(x, model_completion=True).as_long() for x in ix]
            cx["num_blocks"] = nbl
            res.append(Result(name, f"in-bounds[{b.name}]", "FAILED",
                              "block index out of range", cx))
        else:
            res.append(Result(name, f"in-bounds[{b.name}]", "UNKNOWN", str(r)))

    # prefetch reads (index maps + body), collected above and below
    bp = body_preds(ctx, p)
    for k in range(model.num_prefetch):
        reads = [(flat, size) for (kk, flat, size) in ctx.pf_reads if kk == k]
        if not reads:
            continue
        bad = z3.Or(*[z3.Or(f < 0, f >= size) for f, size in reads])
        s = z3.Solver()
        s.add(*base, bad)
        r = _check(s, timeout_ms)
        st = _status(r)
        res.append(Result(name, f"prefetch-bounds[{k}]", st, f"{len(reads)} reads",
                          _cex(ctx, s.model(), {"grid": p}) if r == z3.sat else None))

    outs = [b for b in model.blocks if b.role == "out"]
    nin = sum(1 for b in model.blocks if b.role == "in")
    out_pos = {b.pos: model.num_prefetch + nin + b.pos for b in outs}
    aliased = {o for (_, o) in model.aliases}

    # parallel race across cores
    par = [d for d, s_ in enumerate(model.dim_sem) if s_ == "parallel"]
    for b in outs:
        if not par:
            res.append(Result(name, f"parallel-race[{b.name}]", "N/A", "no parallel dims"))
            continue
        q = ctx.point("q")
        s = z3.Solver()
        s.add(*base, ctx.in_dom(q), z3.Or(*[p[d] != q[d] for d in par]),
              _keys_eq(ctx.block_index(b, p), ctx.block_index(b, q)))
        r = _check(s, timeout_ms)
        st = _status(r)
        res.append(Result(name, f"parallel-race[{b.name}]", st,
                          f"parallel dims {par}",
                          _cex(ctx, s.model(), {"p": p, "q": q}) if r == z3.sat else None))

    # per-output pipeline obligations
    G = ctx.G
    zero = z3.And(*[x == 0 for x in p])
    last = z3.And(*[x == g - 1 for x, g in zip(p, G)])
    for b in outs:
        i = out_pos[b.pos]
        R = bp.reads.get(i)
        W = bp.writes.get(i)
        I = bp.inits.get(i)
        is_hbm_ref = b.memory_space in ("any", "hbm")
        unconditional_overwrite = (R is None and W is not None
                                   and z3.is_true(z3.simplify(W)))
        if is_hbm_ref:
            res.append(Result(name, f"contiguity[{b.name}]", "N/A",
                              f"explicit HBM ref (memory_space={b.memory_space}): no VMEM output pipeline"))
            alias_in_writes = any(
                bp.effects.get(in_op, Eff()).writes_some
                for in_op, out_op in model.aliases
                if out_op == b.pos
            )
            if bp.effects.get(i, Eff()).writes_some or alias_in_writes:
                res.append(Result(name, f"write-at-end[{b.name}]", "PROVED",
                                  f"explicit HBM DMA write (memory_space={b.memory_space})"))
            else:
                res.append(Result(name, f"write-at-end[{b.name}]", "FAILED",
                                  f"explicit HBM ref (memory_space={b.memory_space}) is never written"))
            if b.pos in aliased:
                res.append(Result(name, f"coverage[{b.name}]", "N/A",
                                  "aliased to an input: unvisited blocks keep their value"))
            else:
                res.append(_coverage(ctx, b, pre, timeout_ms))
            continue
        if unconditional_overwrite:
            res.append(Result(name, f"contiguity[{b.name}]", "N/A",
                              "unconditional full overwrite: scatter semantics "
                              "(Lean Pipeline.overwrite_is_scatter)"))
        else:
            res.extend(_contiguity(ctx, b, base, p, timeout_ms))
        Kp = ctx.block_index(b, p)
        qv = ctx.point("pred")
        Kq = ctx.block_index(b, qv)
        if R is not None:
            # accumulator: init exactly at run start
            first = z3.Or(zero, z3.And(ctx.in_dom(qv), succ(G, qv, p), z3.Not(_keys_eq(Kp, Kq))))
            Ip = I if I is not None else z3.BoolVal(False)
            res.append(_sat_obl(ctx, name, f"init-at-start[{b.name}]",
                                [*base, first, z3.Not(Ip)], {"grid": p, "pred": qv}, timeout_ms,
                                "accumulator read without init at run start"))
            res.append(_sat_obl(ctx, name, f"no-midrun-reset[{b.name}]",
                                [*base, Ip, z3.Not(zero), ctx.in_dom(qv), succ(G, qv, p),
                                 _keys_eq(Kp, Kq)], {"grid": p, "pred": qv}, timeout_ms,
                                "accumulator re-initialised mid-run"))
        elif not unconditional_overwrite:
            nx = ctx.point("next")
            Kn = ctx.block_index(b, nx)
            lastrun = z3.Or(last, z3.And(ctx.in_dom(nx), succ(G, p, nx), z3.Not(_keys_eq(Kp, Kn))))
            Wp = W if W is not None else z3.BoolVal(False)
            res.append(_sat_obl(ctx, name, f"write-at-end[{b.name}]",
                                [*base, lastrun, z3.Not(Wp)], {"grid": p, "next": nx}, timeout_ms,
                                "run ends without writing the output block"))
        # coverage
        if b.pos in aliased:
            res.append(Result(name, f"coverage[{b.name}]", "N/A",
                              "aliased to an input: unvisited blocks keep their value"))
        else:
            res.append(_coverage(ctx, b, pre, timeout_ms))

    # scratch accumulators share the partition of all outputs (or inner reduction axis)
    nscr = len(model.scratch_avals)
    scr0 = model.num_prefetch + len(model.blocks)
    for j in range(nscr):
        i = scr0 + j
        R = bp.reads.get(i)
        if R is None:
            continue
        I = bp.inits.get(i, z3.BoolVal(False))
        if z3.is_true(z3.simplify(I)):
            res.append(Result(name, f"init-at-start[scratch{j}]", "PROVED",
                              "per-step staging buffer: written before read on every grid step"))
            res.append(Result(name, f"no-midrun-reset[scratch{j}]", "N/A",
                              "per-step staging buffer (not a cross-step accumulator)"))
            continue
        qv = ctx.point("pred")
        # If outputs use prefetch-dependent grouping (e.g. megablox_gmm's m_tile_ids)
        # while the scratch accumulator resets every inner reduction loop (p[-1] == 0),
        # the scratch run partition is p[:-1].
        if len(p) > 1 and model.num_prefetch > 0:
            Kp, Kq = p[:-1], qv[:-1]
        else:
            Kp, Kq = ctx.out_key(p), ctx.out_key(qv)
        first = z3.Or(zero, z3.And(ctx.in_dom(qv), succ(G, qv, p), z3.Not(_keys_eq(Kp, Kq))))
        res.append(_sat_obl(ctx, name, f"init-at-start[scratch{j}]",
                            [*base, first, z3.Not(I)], {"grid": p, "pred": qv}, timeout_ms,
                            "scratch read without init at output-run start"))
        res.append(_sat_obl(ctx, name, f"no-midrun-reset[scratch{j}]",
                            [*base, I, z3.Not(zero), ctx.in_dom(qv), succ(G, qv, p),
                             _keys_eq(Kp, Kq)], {"grid": p, "pred": qv}, timeout_ms,
                            "scratch re-initialised mid-run"))

    # termination & explicit DMA bounds obligations
    res.append(_check_termination(model))
    dma_res = _check_dma_bounds(model)
    if dma_res is not None:
        res.append(dma_res)

    # TPU v7x Mosaic MLIR lowering + VMEM/SMEM capacity verification
    if fn_args is not None:
        fn, args = fn_args
        try:
            lowered = export(fn, *args)
            k_name, mlir_txt = lowered[0]
            ms = analyze_v7x_mosaic(k_name, mlir_txt, len(model.blocks), len(model.scratch_avals))
            ops_str = ", ".join(f"{k}={v}" for k, v in ms.op_counts.items())
            res.append(Result(name, "v7x-mosaic-lower", "PROVED",
                              f"TPU7x TC module @{ms.kernel_name} ({ops_str})"))
            st_mem = "PROVED" if (ms.fits_v7x_vmem and ms.fits_v7x_smem) else "FAILED"
            res.append(Result(
                name, "v7x-vmem-budget", st_mem,
                f"VMEM {ms.vmem_bytes / 1024:.1f} KiB / {V7X_VMEM_CAPACITY_BYTES // (1024*1024)} MiB, "
                f"SMEM {ms.smem_bytes} B / {V7X_SMEM_CAPACITY_BYTES // 1024} KiB (Lean PallasLean.fitsV7xMemory)"
            ))
        except Exception as e:
            res.append(Result(name, "v7x-mosaic-lower", "FAILED", str(e)[:200]))

    # unused pipelined inputs
    for b in model.blocks:
        if b.role != "in":
            continue
        if b.memory_space in ("any", "hbm"):
            continue
        i = model.num_prefetch + b.pos
        if not bp.effects.get(i, Eff()).reads:
            res.append(Result(name, f"unused-input[{b.name}]", "WARN",
                              "block is DMA'd HBM->VMEM every step but never read; "
                              "pass it with memory_space=pl.ANY"))
    return res


def _sat_obl(ctx, kname, oname, constraints, pts, timeout_ms, msg):
    s = z3.Solver()
    s.add(*constraints)
    r = _check(s, timeout_ms)
    if r == z3.unsat:
        return Result(kname, oname, "PROVED", "")
    if r == z3.sat:
        return Result(kname, oname, "FAILED", msg, _cex(ctx, s.model(), pts))
    return Result(kname, oname, "UNKNOWN", str(r))


def _contiguity(ctx, b, base, p, timeout_ms):
    q, r_ = ctx.point("q"), ctx.point("r")
    Kp, Kq, Kr = (ctx.block_index(b, x) for x in (p, q, r_))
    s = z3.Solver()
    s.add(*base, ctx.in_dom(q), ctx.in_dom(r_), lex_lt(p, r_), lex_lt(r_, q),
          _keys_eq(Kp, Kq), z3.Not(_keys_eq(Kp, Kr)))
    r = _check(s, timeout_ms)
    st = _status(r)
    return [Result(ctx.m.name, f"contiguity[{b.name}]", st,
                   "" if r != z3.sat else "block revisited after another block",
                   _cex(ctx, s.model(), {"first": p, "between": r_, "again": q}) if r == z3.sat else None)]


def _coverage(ctx, b, pre, timeout_ms):
    nbl = _nblocks(b)
    tgt = [z3.Int(f"cov{d}") for d in range(len(nbl))]
    p = ctx.point("cv")
    K = ctx.block_index(b, p)
    s = z3.Solver()
    if p:
        s.add(*ctx.dyn_bounds, pre, *[z3.And(0 <= t, t < n) for t, n in zip(tgt, nbl)],
              z3.ForAll(p, z3.Implies(ctx.in_dom(p), z3.Not(_keys_eq(K, tgt)))))
    else:
        s.add(*ctx.dyn_bounds, pre, *[z3.And(0 <= t, t < n) for t, n in zip(tgt, nbl)],
              z3.Not(_keys_eq(K, tgt)))
    r = _check(s, timeout_ms)
    if r == z3.unsat:
        return Result(ctx.m.name, f"coverage[{b.name}]", "PROVED", "")
    if r == z3.sat:
        mdl = s.model()
        return Result(ctx.m.name, f"coverage[{b.name}]", "FAILED", "output block never written",
                      {"block": [mdl.eval(t).as_long() for t in tgt]})
    return Result(ctx.m.name, f"coverage[{b.name}]", "UNKNOWN", str(r))
