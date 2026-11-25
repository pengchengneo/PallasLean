"""Sympy domain: exact real semantics for the step algebra (Level C).

Floats are reals (casts are identities), `exp` is `sympy.exp`, maxima are
fresh opaque symbols: online softmax is shift-invariant, so the step identity
must hold for *any* running shift (Lean `online_softmax_exact`).
"""
from __future__ import annotations

from fractions import Fraction
import math

import sympy as sp

from .interp import Unresolved


def _sign(e):
    if e.is_zero:
        return 0
    if e.is_positive:
        return 1
    if e.is_negative:
        return -1
    return None


class SymDom:
    def __init__(self):
        self._max = {}
        self._n = 0

    def fresh(self, name):
        self._n += 1
        return sp.Symbol(f"{name}{self._n}", real=True)

    def lit(self, v, kind):
        if kind == "bool":
            return bool(v)
        if kind == "int":
            return int(v)
        v = float(v)
        if math.isinf(v):
            return sp.oo if v > 0 else -sp.oo
        return sp.Rational(Fraction(v))

    def index(self, i):
        return i

    def imod(self, i, s):
        return i % s

    def iadd(self, a, b):
        return a + b

    def imul(self, a, b):
        return a * b

    def concrete(self, idx):
        return tuple(int(i) for i in idx)

    def convert(self, x, src, dst, dtype):
        if src == dst or (src == "float" and dst == "float"):
            return x
        if src == "bool":
            return (1 if x else 0) if dst == "int" else sp.Integer(1 if x else 0)
        if src == "int" and dst == "float":
            return sp.Integer(x)
        if src == "float" and dst == "int" and isinstance(x, sp.Integer):
            return int(x)
        raise Unresolved(f"convert {src}->{dst} of {x}")

    def _truth(self, pv):
        if isinstance(pv, bool):
            return pv
        if pv is sp.true or pv is sp.false:
            return bool(pv)
        if isinstance(pv, sp.Eq):
            d = pv.lhs - pv.rhs
            s = _sign(d)
            if s is None:
                s = _sign(sp.simplify(d))
            if s is None:
                raise Unresolved(f"undecided {pv}")
            return s == 0
        if isinstance(pv, (sp.StrictLessThan, sp.LessThan, sp.StrictGreaterThan, sp.GreaterThan)):
            s = _sign(sp.simplify(pv.lhs - pv.rhs))
            if s is None:
                raise Unresolved(f"undecided {pv}")
            return {sp.StrictLessThan: s < 0, sp.LessThan: s <= 0,
                    sp.StrictGreaterThan: s > 0, sp.GreaterThan: s >= 0}[type(pv)]
        raise Unresolved(f"predicate {pv!r}")

    def select(self, pred, thunks, pk):
        if pk == "bool":
            return thunks[1]() if self._truth(pred) else thunks[0]()
        return thunks[int(pred)]()

    def _cmp(self, name, a, b):
        if not isinstance(a, sp.Basic) and not isinstance(b, sp.Basic):
            return {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b,
                    "gt": a > b, "ge": a >= b}[name]
        a, b = sp.sympify(a), sp.sympify(b)
        if name == "eq":
            return sp.Eq(a, b, evaluate=False) if not (a - b).is_number else bool(a == b)
        if name == "ne":
            return not self._truth(sp.Eq(a, b, evaluate=False))
        r = {"lt": sp.Lt, "le": sp.Le, "gt": sp.Gt, "ge": sp.Ge}[name](a, b)
        return r if not isinstance(r, (bool, type(sp.true), type(sp.false))) else bool(r)

    def op(self, name, args, kind, eqn):
        a = args
        if name == "add": return a[0] + a[1]
        if name == "sub": return a[0] - a[1]
        if name == "mul": return a[0] * a[1]
        if name == "neg": return -a[0]
        if name == "div":
            if kind == "int":
                q = abs(a[0]) // abs(a[1])
                return q if (a[0] >= 0) == (a[1] > 0) else -q
            return a[0] / a[1]
        if name == "rem": return int(math.fmod(a[0], a[1]))
        if name in ("max", "min"):
            if all(isinstance(x, (int, float)) or (isinstance(x, sp.Basic) and x.is_number) for x in a):
                return (max if name == "max" else min)(a[0], a[1])
            key = (name, a[0], a[1])
            if key not in self._max:
                self._max[key] = self.fresh("M")
            return self._max[key]
        if name in ("eq", "ne", "lt", "le", "gt", "ge"):
            return self._cmp(name, a[0], a[1])
        if name == "exp": return sp.exp(a[0])
        if name == "sqrt": return sp.sqrt(a[0])
        if name == "rsqrt": return sp.Integer(1) / sp.sqrt(a[0])
        if name == "logistic": return sp.Integer(1) / (sp.Integer(1) + sp.exp(-a[0]))
        if name == "and": return a[0] and a[1] if kind == "bool" else a[0] & a[1]
        if name == "or": return a[0] or a[1] if kind == "bool" else a[0] | a[1]
        if name == "not": return (not a[0]) if kind == "bool" else ~a[0]
        if name == "abs": return abs(a[0])
        if name == "integer_pow": return a[0] ** a[1]
        raise Unresolved(f"sym op {name}")

    def reduce(self, kind, full, it, n, eqn):
        terms = [full(r) for r in it]
        if kind == "sum":
            return sp.Add(*terms)
        key = (kind, tuple(terms))
        if key not in self._max:
            self._max[key] = self.fresh("M")
        return self._max[key]

    def dot(self, term, it, n, eqn, idx):
        out = []
        for r in it:
            x, y = term(r)
            out.append(x * y)
        return sp.Add(*out)

    def cond_index(self, v, n):
        return max(0, min(n - 1, int(v)))

    def eq_index(self, v, k):
        return v == k
