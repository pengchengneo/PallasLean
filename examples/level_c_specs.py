"""Level C role maps and score specs for the attention kernels.

A role map names which body refs hold the running (m, l, acc) state and where
the V tile comes from; it is the analogue of TensorLean's `spec_layout`.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import z3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "frontend"))
from tpuverify.level_c import Roles  # noqa: E402
from tpuverify.z3dom import _real_lit  # noqa: E402

F32_MASK = -0.7 * float(np.finfo(np.float32).max)


def ragged_roles(heads=8, block=256, dh=128):
    # o/m/l are outputs [heads, 128] (block of [B, heads, dh]); V tile [block, dh].
    return Roles(
        m=("out", 1), l=("out", 2), acc=("out", 0), v=2,
        m_at=lambda r, ln: (r, ln), acc_at=lambda r, c: (r, c),
        v_global=lambda pids, idx: (f"b{pids[0]}r{pids[1] * block + idx[0]}", idx[1]),
        v_local=lambda j, c: (j, c),
        score_shape=(heads, block), rows=(0, heads - 1), cols=(0, dh - 1), lanes=128,
    )


def flash_roles(block_q=128, block_k=128, dh=128):
    # scratch m/l [1,1,bq,128], acc [1,1,bq,dh]; V tile [1,1,bk,dh]; out [1,1,bq,dh].
    return Roles(
        m=("scratch", 0), l=("scratch", 1), acc=("scratch", 2), v=2,
        m_at=lambda r, ln: (0, 0, r, ln), acc_at=lambda r, c: (0, 0, r, c),
        v_global=lambda pids, idx: (f"b{pids[0]}h{pids[1]}r{pids[3] * block_k + idx[2]}", idx[3]),
        v_local=lambda j, c: (0, 0, j, c),
        score_shape=(block_q, block_k), rows=(0, block_q - 1), cols=(0, dh - 1), lanes=128,
        out=0, out_at=lambda r, c: (0, 0, r, c),
    )


# ---- score specs: cut == scale * sum_e Q K + (0 if keep else MASK)

def ragged_keep(ctx, p, gi, gj):
    return gj < ctx.pf[0][p[0]]


def ragged_score(block=256, dh=128):
    def spec(ctx, p, t, i, j, ufs):
        Q, K = ufs["q"], ufs["k"]
        gj = p[1] * block + j
        dot = z3.Sum([Q(p[0], i, e) * K(p[0], gj, e) for e in range(dh)])
        return dot + z3.If(ragged_keep(ctx, p, i, gj), z3.RealVal(0), _real_lit(F32_MASK))
    return spec


def flash_keep(block_q, block_k):
    def keep(ctx, p, gi, gj):
        return gj <= gi
    return keep


def flash_score(block_q=128, block_k=128, dh=128, scale=0.125, causal=True):
    sc = _real_lit(scale)

    def spec(ctx, p, t, i, j, ufs):
        Q, K = ufs["q"], ufs["k"]
        gi = p[2] * block_q + i
        gj = p[3] * block_k + j
        dot = z3.Sum([Q(p[0], p[1], gi, e) * K(p[0], p[1], gj, e) for e in range(dh)])
        s = dot * sc
        if not causal:
            return s
        return s + z3.If(gj <= gi, z3.RealVal(0), _real_lit(F32_MASK))
    return spec
