"""Production kernels under verification, traced from their checkouts unmodified.

Each entry defines a TensorLean-style 5-tuple specification:
* traceable production function `fn` and abstract `args` at production shapes
* operand `names` (`spec_layout` bindings)
* `requires` precondition (TensorLean's `requires`)
* `spec_tol` floating-point tolerance (TensorLean's `spec_tol`)
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction as Fr

import jax
import jax.numpy as jnp
import z3
from jax.experimental import pallas as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "frontend"))
from tpuverify.loader import load_file, stub_module  # noqa: E402

SGL = os.environ.get("SGLANG_TPU_ROOT", "../sglang-torchtpu")
SGL_JAX = os.environ.get("SGLANG_JAX_ROOT", "../sglang-jax")
MXT = os.environ.get("MAXTEXT_ROOT", "../maxtext")
F32_MASK = -0.7 * float(jnp.finfo(jnp.float32).max)


def S(shape, dt):
    return jax.ShapeDtypeStruct(shape, dt)


def _ensure_stubs():
    stub_module("torch", dtype=object, Tensor=object, bfloat16=object, empty_like=lambda x: x)
    stub_module(
        "sglang_tpu.pallas_jax_op",
        cpu_export_jax_op=lambda name, fn: type(
            "Op", (), {"register_fake": lambda self, f: None, "__call__": staticmethod(fn)}
        )(),
    )
    stub_module("maxtext.common.common_types", DEFAULT_MASK_VALUE=F32_MASK)

    class DummyQArray:
        pass

    stub_module("qwix")
    stub_module(
        "qwix.pallas",
        QArray=DummyQArray,
        dot_general=jax.lax.dot_general,
        pallas_call=pl.pallas_call,
    )
    stub_module("sgl_jax.srt.utils", cdiv=lambda a, b: (a + b - 1) // b)
    stub_module("sgl_jax.srt.utils.jax_utils", get_device_kind=lambda: "TPU7x")
    from jax.experimental.pallas import tpu as pltpu
    if not hasattr(pltpu.MemorySpace, "ANY"):
        pltpu.MemorySpace.ANY = pl.ANY


def kv_page_write(n_pages=16, num_blocks=4096, page=(16, 8, 128)):
    _ensure_stubs()
    m = load_file("sglang_kv_page_write", f"{SGL}/sglang_tpu/kv_page_write.py")
    args = (S((n_pages,), jnp.int32), S((n_pages, *page), jnp.bfloat16),
            S((num_blocks, *page), jnp.bfloat16))

    def requires(ctx):  # every block id is a valid page of the KV buffer
        return ctx.pf_values(0, lambda i, v: z3.And(0 <= v, v < num_blocks))
    return dict(fn=m.write_pages, args=args, names=("pages", "kv", "kv_out"),
                requires=requires, spec_tol=Fr(0), module=m)


def qk_norm_rope(tokens=512, num_q=4, num_kv=2, hd=128, eps=1e-6):
    _ensure_stubs()
    m = load_file("sglang_qk_norm_rope", f"{SGL}/sglang_tpu/qk_norm_rope.py")
    fn = lambda qkv, cs, qw, kw: m._qk_norm_rope_jax(qkv, cs, qw, kw, eps, num_q, num_kv, hd)
    args = (
        S((tokens, (num_q + 2 * num_kv) * hd), jnp.bfloat16),
        S((tokens, hd), jnp.bfloat16),
        S((hd,), jnp.bfloat16),
        S((hd,), jnp.bfloat16),
    )
    return dict(
        fn=fn,
        args=args,
        names=("qkv", "cos_sin", "q_w", "k_w", "q_out", "k_out"),
        requires=None,
        spec_tol=Fr(1, 50),  # 0.02 bf16 tolerance
        module=m,
    )


def kimi_attn_res_decode(tokens=8, num_blocks=8, nvb=4, hidden=128, eps=1e-6):
    _ensure_stubs()
    m = load_file("sglang_kimi_attn_res", f"{SGL}/sglang_tpu/kimi_attn_residual.py")
    fn = lambda a, b, bank, cw, w: m.attention_residual_decode_jax(
        a, b, bank, cw, w, nvb=nvb, eps=eps, out_eps=eps
    )
    args = (
        S((tokens, hidden), jnp.bfloat16),
        S((tokens, hidden), jnp.bfloat16),
        S((tokens, num_blocks, hidden), jnp.bfloat16),
        S((hidden,), jnp.float32),
        S((hidden,), jnp.float32),
    )
    return dict(
        fn=fn,
        args=args,
        names=("prefix_a", "prefix_b", "bank", "cw", "w", "normed_out", "prefix_out"),
        requires=None,
        spec_tol=Fr(1, 50),
        module=m,
    )


def ragged_mqa(batch=8, heads=8, seq=4096, dh=128, block=256, min_len=0):
    _ensure_stubs()
    m = load_file("maxtext_ragged_attention", f"{MXT}/src/MaxText/kernels/attention/ragged_attention.py")
    fn = lambda q, k, v, lengths: m.ragged_mqa(q, k, v, lengths, block_size=block)
    args = (S((batch, heads, dh), jnp.float32), S((batch, seq, dh), jnp.bfloat16),
            S((batch, seq, dh), jnp.bfloat16), S((batch,), jnp.int32))

    def requires(ctx):  # min_len <= lengths[b] <= seq_len
        return ctx.pf_values(0, lambda i, v: z3.And(min_len <= v, v <= seq))
    return dict(fn=fn, args=args, names=("q", "k", "v", "o", "m", "l"),
                requires=requires, spec_tol=Fr(1, 1000), module=m)


def megablox_gmm(m_rows=256, k_cols=256, n_cols=256, num_groups=4, tiling=(128, 128, 128), with_existing_out=False):
    _ensure_stubs()
    m = load_file("maxtext_megablox_backend", f"{MXT}/src/MaxText/kernels/megablox/backend.py")
    lhs = S((m_rows, k_cols), jnp.bfloat16)
    rhs = S((num_groups, k_cols, n_cols), jnp.bfloat16)
    gs = S((num_groups,), jnp.int32)
    tm, tk, tn = tiling
    m_tiles = m_rows // tm

    def requires(ctx):
        # pf0: group_offsets[num_groups+1] in 0..m_rows
        # pf1: group_ids[max_tiles] in 0..num_groups-1
        # pf2: m_tile_ids[max_tiles] sorted in 0..m_tiles-1 and covering 0..m_tiles-1
        # pf3: group_offset[1] == 0
        max_tiles = m_tiles + num_groups - 1
        c0 = ctx.pf_values(0, lambda i, v: z3.And(0 <= v, v <= m_rows))
        c1 = ctx.pf_values(1, lambda i, v: z3.And(0 <= v, v < num_groups))
        c2 = ctx.pf_values(2, lambda i, v: z3.And(0 <= v, v < m_tiles))
        mono = z3.And(*[ctx.pf[2][i] <= ctx.pf[2][i + 1] for i in range(max_tiles - 1)])
        cov = z3.And(ctx.pf[2][0] == 0, ctx.pf[2][1] == 1, ctx.G[1] == m_tiles)
        c3 = ctx.pf_values(3, lambda i, v: v == 0)
        return z3.And(c0, c1, c2, mono, cov, c3)

    if with_existing_out:
        ex = S((m_rows, n_cols), jnp.float32)
        fn = lambda l, r, g, e: m.gmm(l, r, g, existing_out=e, tiling=tiling)
        args = (lhs, rhs, gs, ex)
        names = ("lhs", "rhs", "existing_out", "out")
    else:
        fn = lambda l, r, g: m.gmm(l, r, g, tiling=tiling)
        args = (lhs, rhs, gs)
        names = ("lhs", "rhs", "out")
    return dict(fn=fn, args=args, names=names, requires=requires, spec_tol=Fr(1, 5000), module=m)


def sglang_flash(causal=True, b=1, h=2, q_len=256, kv_len=512, dh=128,
                 block_q=128, block_k_major=256, block_k=128, ab=False, sm_scale=0.125):
    _ensure_stubs()
    m = load_file("sglang_flash_attention", f"{SGL}/sglang_tpu/kernels/flash_attention.py")
    bs = m.BlockSizes(block_q=block_q, block_k_major=block_k_major, block_k=block_k, block_b=1)
    if ab:
        fn = lambda q, k, v, ab_: m.flash_attention(q, k, v, ab_, causal=causal, sm_scale=sm_scale,
                                                    block_sizes=bs, vmem_limit_bytes=None)
        args = (S((b, h, q_len, dh), jnp.bfloat16), S((b, h, kv_len, dh), jnp.bfloat16),
                S((b, h, kv_len, dh), jnp.bfloat16), S((b, h, q_len, kv_len), jnp.float32))
        names = ("q", "k", "v", "ab", "o")
    else:
        fn = lambda q, k, v: m.flash_attention(q, k, v, causal=causal, sm_scale=sm_scale,
                                               block_sizes=bs, vmem_limit_bytes=None)
        args = (S((b, h, q_len, dh), jnp.bfloat16), S((b, h, kv_len, dh), jnp.bfloat16),
                S((b, h, kv_len, dh), jnp.bfloat16))
        names = ("q", "k", "v", "o")
    return dict(fn=fn, args=args, names=names, requires=None, spec_tol=Fr(1, 100), module=m)


def sglang_jax_gmm(m_rows=256, k_cols=512, n_cols=256, num_groups=4, num_quant_blocks=2,
                   tiling=(128, 128, 128)):
    _ensure_stubs()
    gmm_dir = f"{SGL_JAX}/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel"
    common_mod = load_file("sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.common", f"{gmm_dir}/common.py")
    tbs_path = f"{gmm_dir}/tuned_block_sizes.py"
    tbs_mod = load_file("sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.tuned_block_sizes", tbs_path) if os.path.exists(tbs_path) else None
    stub_module("sgl_jax.srt.kernels.gmm.megablox_gmm_kernel", common=common_mod, tuned_block_sizes=tbs_mod)
    m = load_file("sglang_jax_gmm", f"{gmm_dir}/gmm.py")
    lhs = S((m_rows, k_cols), jnp.bfloat16)
    rhs = S((num_groups, k_cols, n_cols), jnp.bfloat16)
    gs = S((num_groups,), jnp.int32)
    rs = S((num_groups, num_quant_blocks, 1, n_cols), jnp.float32)
    ex = S((m_rows, n_cols), jnp.float32)
    tm, tk, tn = tiling
    m_tiles = m_rows // tm

    def requires(ctx):
        max_tiles = m_tiles + num_groups - 1
        c0 = ctx.pf_values(0, lambda i, v: z3.And(0 <= v, v <= m_rows))
        c1 = ctx.pf_values(1, lambda i, v: z3.And(0 <= v, v < num_groups))
        c2 = ctx.pf_values(2, lambda i, v: z3.And(0 <= v, v < m_tiles))
        mono = z3.And(*[ctx.pf[2][i] <= ctx.pf[2][i + 1] for i in range(max_tiles - 1)])
        cov = z3.And(ctx.pf[2][0] == 0, ctx.pf[2][1] == 1, ctx.G[1] == m_tiles)
        c3 = ctx.pf_values(3, lambda i, v: v == 0)
        return z3.And(c0, c1, c2, mono, cov, c3)

    fn = lambda l, r, g, s, e: m.gmm(
        l, r, g, rhs_scale=s, existing_out=e, tiling=tiling
    )
    args = (lhs, rhs, gs, rs, ex)
    names = ("lhs", "rhs", "rhs_scale", "existing_out", "out")
    return dict(fn=fn, args=args, names=names, requires=requires, spec_tol=Fr(1, 5000), module=m)


def sglang_jax_update_kv_cache(new_kv_tokens=32, total_kv_tokens=1024, num_heads=8, head_dim=128,
                               page_size=16, num_slices_per_block=8):
    _ensure_stubs()
    m = load_file("sglang_jax_update_kv_cache",
                  f"{SGL_JAX}/python/sgl_jax/srt/kernels/update_kv_cache/update_kv_cache.py")
    fn = lambda new_kv, slices, kv_cache, n_upd: m.kv_cache_update(
        new_kv, slices, kv_cache, n_upd,
        page_size=page_size, num_slices_per_block=num_slices_per_block,
    )
    args = (
        S((new_kv_tokens, num_heads, head_dim), jnp.bfloat16),
        S((3, num_slices_per_block), jnp.int32),
        S((total_kv_tokens, num_heads, head_dim), jnp.bfloat16),
        S((1,), jnp.int32),
    )
    return dict(
        fn=fn,
        args=args,
        names=("new_kv", "kv_cache", "kv_out"),
        requires=None,
        spec_tol=Fr(0),
        module=m,
        needs_mesh=True,
    )
