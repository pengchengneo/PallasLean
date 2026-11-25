#!/usr/bin/env python3
"""Run each PallasLean-discovered bug reproduction in an isolated subprocess on TPU v7x
so that a hardware BoundsCheck core halt in one bug cannot poison subsequent tests."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stub(name: str, **attrs):
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        sys.modules.setdefault(".".join(parts[:i]), types.ModuleType(".".join(parts[:i])))
    for k, v in attrs.items():
        setattr(sys.modules[name], k, v)


def run_one(which: str, src: str):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    F32_MASK = -0.7 * float(jnp.finfo(jnp.float32).max)
    stub("torch", dtype=object, Tensor=object, bfloat16=object, empty_like=lambda x: x)
    stub(
        "sglang_tpu.pallas_jax_op",
        cpu_export_jax_op=lambda name, fn: type(
            "Op", (), {"register_fake": lambda self, f: None, "__call__": staticmethod(fn)}
        )(),
    )
    stub("maxtext.common.common_types", DEFAULT_MASK_VALUE=F32_MASK)

    class DummyQArray:
        pass

    stub("qwix")
    stub("qwix.pallas", QArray=DummyQArray, dot_general=jax.lax.dot_general, pallas_call=pl.pallas_call)
    stub("sgl_jax.srt.utils", cdiv=lambda a, b: (a + b - 1) // b)
    rng = np.random.default_rng(0)

    if which == "bug2_maxtext_gmm_uninit_vmem_leak":
        mb = load("mb", f"{src}/src/MaxText/kernels/megablox/backend.py")
        M, K, N, G = 256, 256, 256, 4
        lhs = jnp.ones((M, K), jnp.bfloat16)
        rhs = jnp.zeros((G, K, N), jnp.bfloat16).at[0].set(1.0).at[1].set(2.0)
        gs = jnp.array([128, 32, 0, 0], jnp.int32)
        out_no_ex = np.asarray(mb.gmm(lhs, rhs, gs, tiling=(128, 128, 128)), np.float32)
        out_ex = np.asarray(
            mb.gmm(lhs, rhs, gs, existing_out=jnp.zeros((M, N), jnp.float32), tiling=(128, 128, 128)),
            np.float32,
        )
        return {
            "tile0_mean": float(out_no_ex[:128].mean()),
            "tile1_active_mean": float(out_no_ex[128:160].mean()),
            "tile1_unmasked_tail_mean": float(out_no_ex[160:].mean()),
            "tile1_unmasked_tail_max": float(np.abs(out_no_ex[160:]).max()),
            "with_existing_out_tail_max": float(np.abs(out_ex[160:]).max()),
            "bug_reproduced": bool(np.abs(out_no_ex[160:]).max() > 1.0 and np.abs(out_ex[160:]).max() == 0.0),
        }

    if which == "bug4_sglang_flash_odd_block_q":
        fa = load("fa", f"{src}/sglang_tpu/kernels/flash_attention.py")
        B, H, Q, KV, D = 1, 1, 129, 256, 128
        q = jnp.asarray(rng.uniform(-1, 1, (B, H, Q, D)), jnp.bfloat16)
        k = jnp.asarray(rng.uniform(-1, 1, (B, H, KV, D)), jnp.bfloat16)
        v = jnp.asarray(rng.uniform(-1, 1, (B, H, KV, D)), jnp.bfloat16)
        bs = fa.BlockSizes(block_q=129, block_k_major=128, block_k=128, block_b=1)
        o = np.asarray(
            fa.flash_attention(q, k, v, causal=True, sm_scale=0.125, block_sizes=bs, vmem_limit_bytes=None).astype(
                jnp.float32
            ),
            np.float64,
        )
        qf, kf, vf = (np.asarray(t.astype(jnp.float32), np.float64) for t in (q, k, v))

        def ref_attn(qq, kk, vv, causal):
            s = np.einsum("bhqd,bhkd->bhqk", qq, kk) * 0.125
            if causal:
                s = np.where(np.arange(s.shape[3])[None, :] <= np.arange(s.shape[2])[:, None], s, -np.inf)
            s = s - s.max(-1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(-1, keepdims=True)
            return np.einsum("bhqk,bhkd->bhqd", p, vv)

        ref = ref_attn(qf, kf, vf, True)
        ref_drop = ref_attn(qf[:, :, 128:129], kf[:, :, :128], vf[:, :, :128], False)
        err = np.abs(o - ref)
        return {
            "row128_err_vs_spec": float(err[0, 0, 128].max()),
            "row128_err_vs_dropped_col128": float(np.abs(o[0, 0, 128] - ref_drop[0, 0, 0]).max()),
            "rows0_127_max_err": float(err[0, 0, :128].max()),
            "bug_reproduced": bool(
                err[0, 0, 128].max() > 0.05 and np.abs(o[0, 0, 128] - ref_drop[0, 0, 0]).max() < 0.01
            ),
        }

    if which == "bug5_sglang_jax_gmm_quant_midrun_reset":
        c = load("sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.common", f"{src}/sgl_jax/common.py")
        stub("sgl_jax.srt.kernels.gmm.megablox_gmm_kernel", common=c)
        sj = load("sj", f"{src}/sgl_jax/gmm.py")
        M, K, N, G = 256, 512, 256, 4
        lhs_np = np.zeros((M, K), np.float32)
        lhs_np[:, :256] = 1.0  # b_i = 0 contributes 256.0; b_i = 1 contributes 0.0
        out = np.asarray(
            sj.gmm(
                jnp.asarray(lhs_np, jnp.bfloat16),
                jnp.ones((G, N, K), jnp.bfloat16),
                jnp.array([64, 64, 64, 64], jnp.int32),
                rhs_scale=jnp.ones((G, 2, 1, N), jnp.float32),
                existing_out=jnp.full((M, N), 10.0, jnp.float32),
                tiling=(128, 128, 128),
                transpose_rhs=True,
            ),
            np.float32,
        )
        return {
            "expected": 266.0,
            "got_mean": float(out.mean()),
            "got_min": float(out.min()),
            "got_max": float(out.max()),
            "b0_wiped_bug_reproduced": bool(abs(out.mean() - 10.0) < 1e-3),
        }

    if which == "bug6_sglang_jax_gmm_alias_crash":
        c = load("sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.common", f"{src}/sgl_jax/common.py")
        stub("sgl_jax.srt.kernels.gmm.megablox_gmm_kernel", common=c)
        sj = load("sj", f"{src}/sgl_jax/gmm.py")
        M, K, N, G = 256, 256, 256, 4
        try:
            sj.gmm(
                jnp.ones((M, K), jnp.bfloat16),
                jnp.ones((G, N, K), jnp.bfloat16),
                jnp.array([64, 64, 64, 64], jnp.int32),
                rhs_scale=None,
                existing_out=jnp.zeros((M, N), jnp.float32),
                tiling=(128, 128, 128),
                transpose_rhs=True,
            )
            return {"crashed": False}
        except Exception as e:
            return {"crashed": True, "error": f"{type(e).__name__}: {e}"[:500]}

    if which == "kv_page_write_duplicates":
        kvw = load("kvw", f"{src}/sglang_tpu/kv_page_write.py")
        ids = jnp.asarray([5, 7, 5, 9, 7], jnp.int32)
        pages = jnp.asarray(np.arange(5)[:, None, None, None] * np.ones((5, 16, 8, 128)) + 1, jnp.bfloat16)
        kv = jnp.zeros((16, 16, 8, 128), jnp.bfloat16)
        out = np.asarray(kvw.write_pages(ids, pages, kv).astype(jnp.float32))
        got = {b: float(out[b].mean()) for b in (5, 7, 9, 0)}
        return {"block_means": got, "matches_last_write_wins": got == {5: 3.0, 7: 5.0, 9: 4.0, 0: 0.0}}

    if which == "bug3_sglang_flash_ab_oob":
        fa = load("fa", f"{src}/sglang_tpu/kernels/flash_attention.py")
        B, H, Q, KV, D = 1, 2, 200, 512, 128
        q = jnp.asarray(rng.uniform(-1, 1, (B, H, Q, D)), jnp.bfloat16)
        k = jnp.asarray(rng.uniform(-1, 1, (B, H, KV, D)), jnp.bfloat16)
        v = jnp.asarray(rng.uniform(-1, 1, (B, H, KV, D)), jnp.bfloat16)
        ab = jnp.zeros((B, H, Q, KV), jnp.float32)
        bs = fa.BlockSizes(block_q=128, block_k_major=128, block_k=128, block_b=1)
        try:
            o = np.asarray(
                fa.flash_attention(q, k, v, ab, causal=True, sm_scale=0.125, block_sizes=bs, vmem_limit_bytes=None).astype(
                    jnp.float32
                )
            )
            return {"ran": True, "shape": list(o.shape)}
        except Exception as e:
            return {"ran": False, "error": f"{type(e).__name__}: {e}"[:800]}

    if which == "bug1_maxtext_ragged_negative_block":
        ra = load("ra", f"{src}/src/MaxText/kernels/attention/ragged_attention.py")
        T, D = 4096, 128
        q = rng.uniform(-1 / 8, 1 / 8, (1, 8, D)).astype(np.float32)
        k = jnp.asarray(rng.uniform(-1, 1, (1, T, D)), jnp.bfloat16)
        v = jnp.asarray(rng.choice([-1.0, 1.0], (1, T, D)), jnp.bfloat16)
        try:
            o, _, _ = ra.ragged_mqa(jnp.asarray(q), k, v, jnp.asarray([100], jnp.int32), block_size=256)
            return {"ran": True, "shape": list(o.shape)}
        except Exception as e:
            return {"ran": False, "hardware_bounds_check_halt": "BoundsCheck" in str(e), "error": f"{type(e).__name__}: {e}"[:800]}

    raise ValueError(which)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        print(json.dumps(run_one(sys.argv[2], "/tmp/tpu_logs/tl/src")))
        return

    tests = [
        "bug2_maxtext_gmm_uninit_vmem_leak",
        "bug4_sglang_flash_odd_block_q",
        "bug5_sglang_jax_gmm_quant_midrun_reset",
        "bug6_sglang_jax_gmm_alias_crash",
        "kv_page_write_duplicates",
        "bug3_sglang_flash_ab_oob",
        "bug1_maxtext_ragged_negative_block",
    ]
    out = {}
    for t in tests:
        p = subprocess.run([sys.executable, __file__, "--one", t], capture_output=True, text=True)
        try:
            out[t] = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            out[t] = {"rc": p.returncode, "stdout": p.stdout[-400:], "stderr": p.stderr[-600:]}
        print(t, json.dumps(out[t]), flush=True)
    Path("/tmp/tpu_logs/tl/isolated_repro_results.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
