"""Z3 domain: exact integers and booleans, floats as reals with opaque exp/dot."""
from __future__ import annotations

from fractions import Fraction
import math

import z3

from .interp import Unresolved


def _real_lit(v):
    if isinstance(v, float) and math.isinf(v):
        return z3.Real("NEG_INF" if v < 0 else "POS_INF")
    return z3.RealVal(Fraction(v).limit_denominator(10**30) if isinstance(v, float) else v)


def tdiv(a, b):
    """Truncating integer division, as `lax.div` on int32."""
    q = a / b
    return z3.If(z3.Or(a >= 0, a % b == 0), q, z3.If(b > 0, q + 1, q - 1))


def trem(a, b):
    return a - b * tdiv(a, b)


class Z3Dom:
    def __init__(self, tag: str = "k"):
        self.tag = tag
        self._fresh = 0
        self.exp = z3.Function(f"exp_{tag}", z3.RealSort(), z3.RealSort())
        self._fns = {}

    def fresh(self, name, sort=None):
        self._fresh += 1
        return z3.Const(f"{name}_{self._fresh}", sort or z3.RealSort())

    def fn(self, name, arity, ret=None):
        key = (name, arity)
        if key not in self._fns:
            self._fns[key] = z3.Function(name, *([z3.IntSort()] * arity), ret or z3.RealSort())
        return self._fns[key]

    # literals and indices
    def lit(self, v, kind):
        if kind == "bool":
            return z3.BoolVal(bool(v))
        if kind == "int":
            return z3.IntVal(int(v))
        return _real_lit(float(v))

    def index(self, i):
        return z3.IntVal(i) if isinstance(i, int) else i

    def _i(self, x):
        return z3.IntVal(x) if isinstance(x, int) else x

    def imod(self, i, s):
        return i % s if isinstance(i, int) else i % s

    def iadd(self, a, b):
        return a + b if isinstance(a, int) and isinstance(b, int) else self._i(a) + self._i(b)

    def imul(self, a, b):
        return a * b if isinstance(a, int) and isinstance(b, int) else self._i(a) * self._i(b)

    def concrete(self, idx):
        out = []
        for i in idx:
            if isinstance(i, int):
                out.append(i)
                continue
            s = z3.simplify(i)
            if z3.is_int_value(s):
                out.append(s.as_long())
            else:
                raise Unresolved(f"symbolic write index {i}")
        return tuple(out)

    def convert(self, x, src, dst, dtype):
        if src == dst:
            return x
        if src == "bool":
            return z3.If(x, self.lit(1, dst), self.lit(0, dst))
        if src == "int" and dst == "float":
            return z3.ToReal(x)
        if src == "float" and dst == "int":
            return z3.ToInt(x)
        if dst == "bool":
            return x != self.lit(0, src)
        raise Unresolved(f"convert {src}->{dst}")

    def select(self, pred, thunks, pk):
        if pk == "bool":
            return z3.If(pred, thunks[1](), thunks[0]())
        out = thunks[-1]()
        for k in range(len(thunks) - 2, -1, -1):
            out = z3.If(pred == k, thunks[k](), out)
        return out

    def op(self, name, args, kind, eqn):
        a = args
        if name == "add": return a[0] + a[1]
        if name == "sub": return a[0] - a[1]
        if name == "mul": return a[0] * a[1]
        if name == "neg": return -a[0]
        if name == "div": return tdiv(a[0], a[1]) if kind == "int" else a[0] / a[1]
        if name == "rem": return trem(a[0], a[1])
        if name == "max": return z3.If(a[0] >= a[1], a[0], a[1])
        if name == "min": return z3.If(a[0] <= a[1], a[0], a[1])
        if name == "eq": return a[0] == a[1]
        if name == "ne": return a[0] != a[1]
        if name == "lt": return a[0] < a[1]
        if name == "le": return a[0] <= a[1]
        if name == "gt": return a[0] > a[1]
        if name == "ge": return a[0] >= a[1]
        if name == "and": return z3.And(a[0], a[1]) if kind == "bool" else self.fn("band", 2, z3.IntSort())(a[0], a[1])
        if name == "or": return z3.Or(a[0], a[1]) if kind == "bool" else self.fn("bor", 2, z3.IntSort())(a[0], a[1])
        if name == "not": return z3.Not(a[0]) if kind == "bool" else -a[0] - 1
        if name == "exp": return self.exp(a[0])
        if name == "abs": return z3.If(a[0] >= 0, a[0], -a[0])
        if name == "integer_pow":
            out = 1
            for _ in range(a[1]):
                out = out * a[0]
            return out
        f = z3.Function(f"{name}_{self.tag}", *([z3.RealSort()] * len(a)), z3.RealSort())
        return f(*a)

    def reduce(self, kind, full, it, n, eqn):
        if kind == "sum":
            return z3.Sum([full(r) for r in it])
        return self.fresh(f"red{kind}")

    expand_dot = False

    def dot(self, term, it, n, eqn, idx):
        if self.expand_dot:
            return z3.Sum([x * y for x, y in (term(r) for r in it)])
        return self.fn(f"dot{id(eqn) % 100000}", len(idx))(*[self._i(i) for i in idx])

    def cond_index(self, v, n):
        if isinstance(v, int):
            return max(0, min(n - 1, v))
        s = z3.simplify(v)
        if z3.is_int_value(s):
            return max(0, min(n - 1, s.as_long()))
        return None

    def eq_index(self, v, k):
        return v == k
