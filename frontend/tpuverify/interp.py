"""A lazy, element-wise interpreter for Pallas kernel bodies and index maps.

Arrays are functions from an index tuple to a domain element, evaluated only
at the indices an obligation asks for. One interpreter, three domains:

* `Z3Dom`    integers/bools exactly, floats as reals with uninterpreted `exp`
             and `dot`: index maps, masks, predicates, score definitions.
* `SymDom`   sympy reals: the online-softmax step algebra (Level C).
* `RangeDom` real interval + absolute rounding error per value, uniform over
             a tile (Level D, the rounding edge of the triangle).

The domain decides what an op means; the interpreter only knows shapes,
indexing, refs and control flow.
"""
from __future__ import annotations

import itertools
from typing import Any, Callable

import numpy as np
from jax._src import core as jc
from jax import tree_util
from jax._src.state import indexing as stix


class Arr:
    """A (possibly 0-d) array given by an element function."""

    __slots__ = ("shape", "fn", "dtype", "_cache")

    def __init__(self, shape, fn, dtype):
        self.shape = tuple(int(s) for s in shape)
        self.fn = fn
        self.dtype = dtype
        self._cache = {}

    def at(self, idx):
        if all(isinstance(i, (int, np.integer)) for i in idx):
            key = tuple(int(i) for i in idx)
            hit = self._cache.get(key)
            if hit is None:
                hit = self.fn(key)
                self._cache[key] = hit
            return hit
        return self.fn(tuple(idx))

    def scalar(self):
        return self.at(())


class Ref:
    """A mutable ref cell; its identity survives renaming into cond/scan bodies."""

    __slots__ = ("name", "contents")

    def __init__(self, name, contents):
        self.name = name
        self.contents = contents


def const_arr(shape, value, dtype):
    return Arr(shape, lambda idx: value, dtype)


class Unresolved(Exception):
    """Control flow or a predicate the domain cannot decide."""


def _kind(dtype) -> str:
    d = np.dtype(dtype)
    if d == np.bool_:
        return "bool"
    if np.issubdtype(d, np.integer):
        return "int"
    return "float"


class Interp:
    """Evaluate a jaxpr over a domain.

    `refs` maps ref vars to their current `Arr` contents; `get`/`swap` read and
    update it functionally. `hooks` maps primitive name to a callback
    `(interp, eqn, in_vals) -> None` used to record values at cut points.
    `cond_choice(eqn, idx_val) -> int` resolves branches the domain cannot.
    """

    def __init__(self, dom, *, program_ids=(), hooks=None, cond_choice=None,
                 scan_limit: int = 64):
        self.dom = dom
        self.pids = tuple(program_ids)
        self.hooks = hooks or {}
        self.cond_choice = cond_choice
        self.scan_limit = scan_limit
        self.path = []  # path conditions taken (domain predicates)

    # ---- env ----
    def read(self, env, v):
        if isinstance(v, jc.Literal):
            val = v.val
            aval = v.aval
            shape = getattr(aval, "shape", ())
            dt = getattr(aval, "dtype", np.asarray(val).dtype)
            if np.ndim(val) == 0:
                e = self.dom.lit(np.asarray(val).item(), _kind(dt))
                return const_arr(shape, e, dt)
            arr = np.asarray(val)
            return Arr(arr.shape, lambda idx, a=arr: self.dom.lit(a[idx].item(), _kind(dt)), dt)
        return env[v]

    def run(self, jaxpr: jc.Jaxpr, args: list):
        env = {}
        for cv, c in zip(jaxpr.constvars, getattr(jaxpr, "consts", ()) or ()):
            arr = np.asarray(c)
            env[cv] = Arr(arr.shape, lambda idx, a=arr: self.dom.lit(a[idx].item(), _kind(a.dtype)), arr.dtype)
        for v, a in zip(jaxpr.invars, args):
            env[v] = a
        for eqn in jaxpr.eqns:
            ins = [self.read(env, v) for v in eqn.invars]
            hook = self.hooks.get(eqn.primitive.name)
            if hook is not None:
                rebind = hook(self, eqn, ins)
                if rebind:
                    env.update(rebind)
                    ins = [self.read(env, v) for v in eqn.invars]
            outs = self.eval_eqn(eqn, ins)
            for v, o in zip(eqn.outvars, outs):
                if not isinstance(v, jc.DropVar):
                    env[v] = o
        return [self.read(env, v) for v in jaxpr.outvars]

    # ---- primitives ----
    def eval_eqn(self, eqn, ins):
        name = eqn.primitive.name
        p = eqn.params
        outs_avals = [v.aval for v in eqn.outvars]
        dom = self.dom

        if name in ("jit", "pjit", "closed_call", "core_call", "custom_jvp_call", "custom_vjp_call"):
            sub = p.get("jaxpr") or p.get("call_jaxpr") or p.get("fun_jaxpr")
            return self.run(sub, ins)
        if name == "program_id":
            return [const_arr((), self.pids[p["axis"]], np.int32)]
        if name == "get":
            return [self._get(ins[0], ins[1:], p["tree"])]
        if name == "swap":
            old = self._get(ins[0], ins[2:], p["tree"])
            self._swap(ins[0], ins[1], ins[2:], p["tree"])
            return [old]
        if name == "cond":
            return self._cond(eqn, ins)
        if name == "scan":
            return self._scan(eqn, ins)
        if name == "iota":
            dim = p["dimension"]
            return [Arr(p["shape"], lambda idx, d=dim: dom.index(idx[d]), p["dtype"])]
        if name == "broadcast_in_dim":
            x = ins[0]
            bd = p["broadcast_dimensions"]
            xs = x.shape

            def f(idx, x=x, bd=bd, xs=xs):
                return x.at(tuple(idx[d] if xs[k] != 1 else 0 for k, d in enumerate(bd)))
            return [Arr(p["shape"], f, x.dtype)]
        if name == "tile":
            x = ins[0]
            reps = p["reps"]
            xs = x.shape
            rank = len(reps)
            out_shape = tuple(int(r) * s for r, s in zip(reps, (1,) * (rank - len(xs)) + xs))
            pad = rank - len(xs)
            return [Arr(out_shape, lambda idx, x=x: x.at(tuple(dom.imod(i, s) for i, s in zip(idx[pad:], xs))), x.dtype)]
        if name == "squeeze":
            x = ins[0]
            dims = set(d % len(x.shape) for d in p["dimensions"])

            def f(idx, x=x, dims=dims):
                it = iter(idx)
                return x.at(tuple(0 if k in dims else next(it) for k in range(len(x.shape))))
            return [Arr(outs_avals[0].shape, f, x.dtype)]
        if name == "transpose":
            x = ins[0]
            perm = p["permutation"]
            inv = np.argsort(perm)
            return [Arr(outs_avals[0].shape, lambda idx, x=x: x.at(tuple(idx[i] for i in inv)), x.dtype)]
        if name == "reshape":
            x = ins[0]
            out_shape = outs_avals[0].shape

            def f(idx, x=x, out_shape=out_shape):
                flat = int(np.ravel_multi_index(tuple(int(i) for i in idx), out_shape)) if out_shape else 0
                return x.at(tuple(int(i) for i in np.unravel_index(flat, x.shape)) if x.shape else ())
            return [Arr(out_shape, f, x.dtype)]
        if name == "slice":
            x = ins[0]
            starts = tuple(int(s) for s in p["start_indices"])
            strides = tuple(int(s) for s in (p.get("strides") or (1,) * len(starts)))
            out_shape = outs_avals[0].shape

            def f(idx, x=x, starts=starts, strides=strides):
                return x.at(tuple(dom.iadd(st, dom.imul(i, sr)) for i, st, sr in zip(idx, starts, strides)))
            return [Arr(out_shape, f, x.dtype)]
        if name == "concatenate":
            dim = int(p["dimension"]) % len(outs_avals[0].shape)
            out_shape = outs_avals[0].shape
            offsets = []
            acc_off = 0
            for a in ins:
                offsets.append((acc_off, acc_off + a.shape[dim], a))
                acc_off += a.shape[dim]

            def f(idx, offsets=offsets, dim=dim):
                coord = int(idx[dim])
                for lo, hi, arr in offsets:
                    if lo <= coord < hi:
                        sub_idx = list(idx)
                        sub_idx[dim] = coord - lo
                        return arr.at(tuple(sub_idx))
                raise IndexError(f"concatenate index {coord} out of range")
            return [Arr(out_shape, f, outs_avals[0].dtype)]
        if name == "convert_element_type":
            x = ins[0]
            src, dst = _kind(x.dtype), _kind(p["new_dtype"])
            return [Arr(x.shape, lambda idx, x=x: dom.convert(x.at(idx), src, dst, p["new_dtype"]), p["new_dtype"])]
        if name == "select_n":
            pred, cases = ins[0], ins[1:]
            shape = outs_avals[0].shape
            pk = _kind(pred.dtype)
            bshape = pred.shape

            def f(idx, pred=pred, cases=cases):
                pv = pred.at(idx if bshape else ())
                return dom.select(pv, [lambda c=c: c.at(idx) for c in cases], pk)
            return [Arr(shape, f, outs_avals[0].dtype)]
        if name == "dot_general":
            return [self._dot(eqn, ins[0], ins[1])]
        if name in ("reduce_sum", "reduce_max", "reduce_min"):
            x = ins[0]
            axes = tuple(a % max(len(x.shape), 1) for a in p["axes"])
            out_shape = outs_avals[0].shape
            kept = [d for d in range(len(x.shape)) if d not in axes]
            ranges = [range(x.shape[a]) for a in axes]
            n = int(np.prod([x.shape[a] for a in axes])) if axes else 1
            kind = name.split("_")[1]

            def f(idx, x=x):
                def full(red):
                    full_idx = [0] * len(x.shape)
                    for d, i in zip(kept, idx):
                        full_idx[d] = i
                    for a, r in zip(axes, red):
                        full_idx[a] = r
                    return x.at(tuple(full_idx))
                return dom.reduce(kind, full, itertools.product(*ranges), n, eqn)
            return [Arr(out_shape, f, x.dtype)]
        if name in _ELEMENTWISE:
            shape = outs_avals[0].shape
            k = _kind(ins[0].dtype)

            def f(idx, ins=ins, shape=shape):
                args = [a.at(idx if a.shape else ()) if a.shape == shape or not a.shape else
                        a.at(tuple(i if s != 1 else 0 for i, s in zip(idx, a.shape))) for a in ins]
                return dom.op(name, args, k, eqn)
            return [Arr(shape, f, outs_avals[0].dtype)]
        if name == "integer_pow":
            x = ins[0]
            y = p["y"]
            return [Arr(x.shape, lambda idx, x=x: dom.op("integer_pow", [x.at(idx), y], _kind(x.dtype), eqn), x.dtype)]
        if name == "multiple_of":
            return [ins[0]]
        raise NotImplementedError(f"primitive {name}")

    # ---- refs ----
    def _indexers(self, idx_args, tree):
        transforms = tree_util.tree_unflatten(tree, list(idx_args))
        return [t for t in transforms if isinstance(t, stix.NDIndexer)] if transforms else []

    def _view(self, shape, indexers):
        """Map an output index of a ref view to the underlying ref index."""
        mappers = []
        cur_shape = tuple(shape)
        for ix in indexers:
            spec = []
            out_shape = []
            for dim_ix in ix.indices:
                if isinstance(dim_ix, stix.Slice):
                    start = self._as_index(dim_ix.start)
                    spec.append(("slice", start, int(dim_ix.stride)))
                    out_shape.append(int(dim_ix.size))
                else:
                    spec.append(("int", self._as_index(dim_ix)))
            mappers.append(spec)
            cur_shape = tuple(out_shape)

        def to_ref(idx):
            for spec in reversed(mappers):
                it = iter(idx)
                full = []
                for s in spec:
                    if s[0] == "slice":
                        full.append(self.dom.iadd(s[1], self.dom.imul(next(it), s[2])))
                    else:
                        full.append(s[1])
                full.extend(it)
                idx = tuple(full)
            return idx

        return to_ref, cur_shape if indexers else tuple(shape)

    def _as_index(self, v):
        if isinstance(v, Arr):
            return v.scalar()
        if isinstance(v, (int, np.integer)):
            return int(v)
        return v

    def _get(self, ref, idx_args, tree):
        contents = ref.contents if isinstance(ref, Ref) else ref
        indexers = self._indexers(idx_args, tree)
        if not indexers:
            return contents
        to_ref, _ = self._view(contents.shape, indexers)
        out_shape = self._out_shape(contents.shape, indexers)
        return Arr(out_shape, lambda idx: contents.at(to_ref(idx)), contents.dtype)

    def _out_shape(self, shape, indexers):
        shp = list(shape)
        for ix in indexers:
            new = []
            for k, dim_ix in enumerate(ix.indices):
                if isinstance(dim_ix, stix.Slice):
                    new.append(int(dim_ix.size))
            new.extend(shp[len(ix.indices):])
            shp = new
        return tuple(shp)

    def _swap(self, ref, val, idx_args, tree):
        if not isinstance(ref, Ref):
            raise Unresolved("write to a read-only ref")
        contents = ref.contents
        indexers = self._indexers(idx_args, tree)
        if not indexers:
            ref.contents = Arr(contents.shape, lambda idx, v=val: v.at(idx) if v.shape else v.scalar(), contents.dtype)
            return
        to_ref, _ = self._view(contents.shape, indexers)
        out_shape = self._out_shape(contents.shape, indexers)
        # Only static slices / concrete ints are supported for writes.
        region = {}
        for out_idx in itertools.product(*[range(s) for s in out_shape]):
            region[tuple(int(i) for i in self.dom.concrete(to_ref(out_idx)))] = out_idx
        old = contents

        def f(idx, old=old, region=region, val=val):
            hit = region.get(tuple(idx))
            if hit is None:
                return old.at(idx)
            return val.at(hit) if val.shape else val.scalar()
        ref.contents = Arr(contents.shape, f, contents.dtype)

    # ---- control flow ----
    def _cond(self, eqn, ins):
        idx = self.dom.cond_index(ins[0].scalar(), len(eqn.params["branches"]))
        if idx is None and self.cond_choice is not None:
            idx = self.cond_choice(eqn, ins[0].scalar())
        if idx is None:
            raise Unresolved(f"cond index {ins[0].scalar()}")
        branch = eqn.params["branches"][idx]
        pushed = False
        if self.dom.cond_index(ins[0].scalar(), len(eqn.params["branches"])) is None:
            self.path.append(self.dom.eq_index(ins[0].scalar(), idx))
            pushed = True
        try:
            return self.run(branch, ins[1:])
        finally:
            if pushed:
                self.path.pop()

    def _scan(self, eqn, ins):
        p = eqn.params
        n = int(p["length"])
        if n > self.scan_limit:
            raise Unresolved(f"scan length {n} > {self.scan_limit}")
        if "num_consts" in p:
            nc, ncar = p["num_consts"], p["num_carry"]
        else:  # jax >= 0.12: ft_in = (consts, carry, xs) flat-tree groups
            groups = p["ft_in"].elts
            nc, ncar = (len(list(g.vals)) for g in groups[:2])
        consts, carry, xs = ins[:nc], ins[nc:nc + ncar], ins[nc + ncar:]
        body = p["jaxpr"]
        ys_all = []
        order = range(n - 1, -1, -1) if p.get("reverse") else range(n)
        for t in order:
            x_t = [Arr(x.shape[1:], lambda idx, x=x, t=t: x.at((t, *idx)), x.dtype) for x in xs]
            outs = self.run(body, list(consts) + list(carry) + x_t)
            carry, ys = outs[:ncar], outs[ncar:]
            ys_all.append(ys)
        ys_out = []
        for k in range(len(eqn.outvars) - ncar):
            col = [ys[k] for ys in ys_all]
            ys_out.append(Arr((n, *col[0].shape), lambda idx, col=col: col[idx[0]].at(idx[1:]), col[0].dtype))
        return list(carry) + ys_out

    def _dot(self, eqn, a, b):
        (ca, cb), (ba, bb) = eqn.params["dimension_numbers"]
        out_shape = eqn.outvars[0].aval.shape
        fa = [d for d in range(len(a.shape)) if d not in ca and d not in ba]
        fb = [d for d in range(len(b.shape)) if d not in cb and d not in bb]
        ranges = [range(a.shape[d]) for d in ca]
        n = int(np.prod([a.shape[d] for d in ca])) if ca else 1

        def f(idx):
            nb = len(ba)
            bidx, ia, ib = idx[:nb], idx[nb:nb + len(fa)], idx[nb + len(fa):]

            def term(red):
                xa = [0] * len(a.shape)
                xb = [0] * len(b.shape)
                for d, i in zip(ba, bidx):
                    xa[d] = i
                for d, i in zip(bb, bidx):
                    xb[d] = i
                for d, i in zip(fa, ia):
                    xa[d] = i
                for d, i in zip(fb, ib):
                    xb[d] = i
                for da, db, r in zip(ca, cb, red):
                    xa[da] = r
                    xb[db] = r
                return a.at(tuple(xa)), b.at(tuple(xb))
            return self.dom.dot(term, itertools.product(*ranges), n, eqn, idx)
        return Arr(out_shape, f, eqn.outvars[0].aval.dtype)


_ELEMENTWISE = {
    "add", "sub", "mul", "div", "rem", "max", "min", "neg", "exp", "log", "sqrt",
    "rsqrt", "abs", "tanh", "logistic", "pow", "eq", "ne", "lt", "le", "gt", "ge",
    "and", "or", "not", "xor", "sign", "floor", "ceil", "exp2", "square",
    "shift_left", "shift_right_logical", "shift_right_arithmetic",
}
