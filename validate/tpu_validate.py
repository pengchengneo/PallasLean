"""TPU-side validation of PallasLean's claims and trust assumptions.

Runs on a v7x host. Each test checks one statement from the report:

* exp_calibration / div_calibration: the `ExpModel` (ue, etae) and the f32
  division model used by Level D, measured on the EUP/VPU through Pallas.
* sglang_flash_bound / maxtext_ragged_bound: measured |kernel - fp64 ref| stays
  below the Level D eps (the rounding edge of the triangle).
* ragged_negative_block / flash_ab_oob: what the hardware does at the
  out-of-range block index Level A found.
* flash_odd_block_q: the causal-skip counterexample (Lean
  `CausalSkip.skip_unsound_odd_block_q`) on real hardware.
* kv_page_write_duplicates: last-write-wins (Lean `overwrite_is_scatter`).

Usage: python tpu_validate.py --src <dir with the kernel files> --out results.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
import types

import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stub(name, **attrs):
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        sys.modules.setdefault(".".join(parts[:i]), types.ModuleType(".".join(parts[:i])))
    for k, v in attrs.items():
        setattr(sys.modules[name], k, v)


def run(tests, name, fn):
    try:
        tests[name] = {"ok": True, **fn()}
    except Exception as e:  # record, keep going
        tests[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:2000],
                       "trace": traceback.format_exc()[-3000:]}
    print(name, json.dumps(tests[name], default=str)[:600], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    tests = {"env": {"jax": jax.__version__, "devices": [str(d) for d in jax.devices()][:2],
                     "device_kind": jax.devices()[0].device_kind}}
    F32_MASK = -0.7 * float(jnp.finfo(jnp.float32).max)
    stub("torch", dtype=object, Tensor=object, bfloat16=object, empty_like=lambda x: x)
    stub("sglang_tpu.pallas_jax_op", cpu_export_jax_op=lambda name, fn: type(
        "Op", (), {"register_fake": lambda self, f: None, "__call__": staticmethod(fn)})())
    stub("maxtext.common.common_types", DEFAULT_MASK_VALUE=F32_MASK)

    class DummyQArray:
        pass

    stub("qwix")
    stub("qwix.pallas", QArray=DummyQArray, dot_general=jax.lax.dot_general, pallas_call=pl.pallas_call)
    stub("sgl_jax.srt.utils", cdiv=lambda a, b: (a + b - 1) // b)
    fa = load("fa", f"{a.src}/sglang_tpu/kernels/flash_attention.py")
    kvw = load("kvw", f"{a.src}/sglang_tpu/kv_page_write.py")
    qkr = load("qkr", f"{a.src}/sglang_tpu/qk_norm_rope.py")
    ra = load("ra", f"{a.src}/src/MaxText/kernels/attention/ragged_attention.py")
    mb = load("mb", f"{a.src}/src/MaxText/kernels/megablox/backend.py")
    sj_common = load("sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.common", f"{a.src}/sgl_jax/common.py")
    stub("sgl_jax.srt.kernels.gmm.megablox_gmm_kernel", common=sj_common)
    sj_gmm = load("sj_gmm", f"{a.src}/sgl_jax/gmm.py")
    rng = np.random.default_rng(0)

    def ew_kernel(f):
        def k(x_ref, o_ref):
            o_ref[...] = f(x_ref[...])
        return k

    def pallas_ew(f, x):
        rows = x.shape[0]
        blk = 512
        return pl.pallas_call(ew_kernel(f), out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                              grid=(rows // blk,),
                              in_specs=[pl.BlockSpec((blk, 128), lambda i: (i, 0))],
                              out_specs=pl.BlockSpec((blk, 128), lambda i: (i, 0)))(x)

    def exp_cal():
        x = np.linspace(-104.0, 0.0, 4096 * 128, dtype=np.float32).reshape(-1, 128)
        y = np.asarray(pallas_ew(jnp.exp, jnp.asarray(x))).astype(np.float64)
        ref = np.exp(x.astype(np.float64))
        normal = ref >= 2.0 ** -126
        rel = np.abs(y - ref)[normal] / ref[normal]
        absr = np.abs(y - ref)[~normal]
        return {"ue_max_rel": float(rel.max()), "ue_in_ulps_2^-24": float(rel.max() * 2 ** 24),
                "etae_max_abs_below_normal": float(absr.max()) if absr.size else 0.0,
                "ftz": bool(np.all(y[~normal] == 0)) if absr.size else None,
                "exp0_exact": bool(y[-1, -1] == 1.0)}

    def div_cal():
        x = rng.uniform(1.0, 2.0, (4096, 128)).astype(np.float32)
        y = np.asarray(pallas_ew(lambda v: 1.0 / v, jnp.asarray(x))).astype(np.float64)
        rel = np.abs(y - 1.0 / x.astype(np.float64)) * x
        z = np.asarray(pallas_ew(lambda v: v / 3.0, jnp.asarray(x))).astype(np.float64)
        rel2 = np.abs(z - x.astype(np.float64) / 3.0) / (x / 3.0)
        return {"recip_max_rel": float(rel.max()), "div_max_rel": float(rel2.max()),
                "u_f32": 2.0 ** -24}

    def ref_attn(q, k, v, causal, scale, keep=None):
        q, k, v = (np.asarray(t, np.float64) for t in (q, k, v))
        s = np.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if causal:
            qi = np.arange(s.shape[2])[:, None]
            ki = np.arange(s.shape[3])[None, :]
            s = np.where(ki <= qi, s, -np.inf)
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        return np.einsum("bhqk,bhkd->bhqd", p, v)

    def flash_bound():
        out = {}
        for case in ("uniform", "peaked"):
            B, H, Q, KV, D = 1, 8, 1024, 1024, 128
            if case == "uniform":
                q = rng.uniform(-1, 1, (B, H, Q, D))
                k = rng.uniform(-1, 1, (B, H, KV, D))
            else:  # align q with a few keys so softmax is near one-hot
                k = rng.uniform(-1, 1, (B, H, KV, D))
                q = np.sign(k[:, :, :Q]) * 1.0
            v = rng.choice([-1.0, 1.0], (B, H, KV, D))
            qb, kb, vb = (jnp.asarray(t, jnp.bfloat16) for t in (q, k, v))
            o = fa.flash_attention(qb, kb, vb, causal=False, sm_scale=0.125, vmem_limit_bytes=None)
            ref = ref_attn(*(np.asarray(t.astype(jnp.float32)) for t in (qb, kb, vb)), False, 0.125)
            err = float(np.abs(np.asarray(o.astype(jnp.float32), np.float64) - ref).max())
            out[case] = {"max_abs_err": err}
        out["eps_bound"] = 0.008468251550567895
        out["within_bound"] = all(out[c]["max_abs_err"] <= out["eps_bound"] for c in ("uniform", "peaked"))
        return out

    def ragged_ref(q, k, v, lengths):
        q, k, v = (np.asarray(t, np.float64) for t in (q, k, v))
        s = np.einsum("bhd,btd->bht", q, k)
        mask = np.arange(k.shape[1])[None] < np.asarray(lengths)[:, None]
        s = np.where(mask[:, None, :], s, -np.inf)
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        return np.einsum("bht,btd->bhd", p, v)

    def ragged_bound():
        B, H, T, D = 8, 8, 4096, 128
        q = rng.uniform(-1 / 8, 1 / 8, (B, H, D)).astype(np.float32)
        k = jnp.asarray(rng.uniform(-1, 1, (B, T, D)), jnp.bfloat16)
        v = jnp.asarray(rng.choice([-1.0, 1.0], (B, T, D)), jnp.bfloat16)
        lengths = np.array([4096, 1, 255, 256, 257, 3000, 4095, 2048], np.int32)
        o, _, _ = ra.ragged_mqa(jnp.asarray(q), k, v, jnp.asarray(lengths), block_size=256)
        ref = ragged_ref(q, np.asarray(k.astype(jnp.float32)), np.asarray(v.astype(jnp.float32)), lengths)
        err = float(np.abs(np.asarray(o, np.float64) - ref).max())
        return {"max_abs_err": err, "eps_bound": 0.0006111707445914636,
                "within_bound": err <= 0.0006111707445914636}

    def ragged_negative_block():
        out = {}
        for B, lens in (("b8_last100", [4096] * 7 + [100]), ("b1_len100", [100]), ("b8_last0", [4096] * 7 + [0])):
            nb = len(lens)
            T, D = 4096, 128
            q = rng.uniform(-1 / 8, 1 / 8, (nb, 8, D)).astype(np.float32)
            k = jnp.asarray(rng.uniform(-1, 1, (nb, T, D)), jnp.bfloat16)
            v = jnp.asarray(rng.choice([-1.0, 1.0], (nb, T, D)), jnp.bfloat16)
            try:
                o, _, _ = ra.ragged_mqa(jnp.asarray(q), k, v, jnp.asarray(lens, jnp.int32), block_size=256)
                o = np.asarray(o, np.float64)
                valid = [i for i, L in enumerate(lens) if L > 0]
                ref = ragged_ref(q[valid], np.asarray(k.astype(jnp.float32))[valid],
                                 np.asarray(v.astype(jnp.float32))[valid], np.asarray(lens)[valid])
                out[B] = {"ran": True, "max_abs_err_valid_rows": float(np.abs(o[valid] - ref).max()),
                          "len0_row_output_absmax": float(np.abs(o[[i for i, L in enumerate(lens) if L == 0]]).max())
                          if 0 in lens else None}
            except Exception as e:
                out[B] = {"ran": False, "error": f"{type(e).__name__}: {e}"[:800]}
        return out

    def flash_ab_oob():
        B, H, Q, KV, D = 1, 2, 200, 512, 128
        q = jnp.asarray(rng.uniform(-1, 1, (B, H, Q, D)), jnp.bfloat16)
        k = jnp.asarray(rng.uniform(-1, 1, (B, H, KV, D)), jnp.bfloat16)
        v = jnp.asarray(rng.uniform(-1, 1, (B, H, KV, D)), jnp.bfloat16)
        ab = jnp.zeros((B, H, Q, KV), jnp.float32)
        bs = fa.BlockSizes(block_q=128, block_k_major=128, block_k=128, block_b=1)
        try:
            o = fa.flash_attention(q, k, v, ab, causal=True, sm_scale=0.125, block_sizes=bs, vmem_limit_bytes=None)
            ref = ref_attn(*(np.asarray(t.astype(jnp.float32)) for t in (q, k, v)), True, 0.125)
            return {"ran": True, "max_abs_err": float(np.abs(np.asarray(o.astype(jnp.float32), np.float64) - ref).max())}
        except Exception as e:
            return {"ran": False, "error": f"{type(e).__name__}: {e}"[:800]}

    def flash_odd_block_q():
        B, H, Q, KV, D = 1, 1, 129, 256, 128
        q = jnp.asarray(rng.uniform(-1, 1, (B, H, Q, D)), jnp.bfloat16)
        k = jnp.asarray(rng.uniform(-1, 1, (B, H, KV, D)), jnp.bfloat16)
        v = jnp.asarray(rng.uniform(-1, 1, (B, H, KV, D)), jnp.bfloat16)
        bs = fa.BlockSizes(block_q=129, block_k_major=128, block_k=128, block_b=1)
        try:
            o = np.asarray(fa.flash_attention(q, k, v, causal=True, sm_scale=0.125, block_sizes=bs,
                                              vmem_limit_bytes=None).astype(jnp.float32), np.float64)
        except Exception as e:
            return {"ran": False, "error": f"{type(e).__name__}: {e}"[:800]}
        qf, kf, vf = (np.asarray(t.astype(jnp.float32)) for t in (q, k, v))
        ref = ref_attn(qf, kf, vf, True, 0.125)
        # what the kernel computes if column 128 is dropped for row 128
        kd, vd = kf[:, :, :128], vf[:, :, :128]
        ref_drop = ref_attn(qf[:, :, 128:129], kd, vd, False, 0.125)
        err = np.abs(o - ref)
        return {"ran": True, "row128_err_vs_spec": float(err[0, 0, 128].max()),
                "row128_err_vs_dropped_col128": float(np.abs(o[0, 0, 128] - ref_drop[0, 0, 0]).max()),
                "rows0_127_max_err": float(err[0, 0, :128].max())}

    def kv_dup():
        ids = jnp.asarray([5, 7, 5, 9, 7], jnp.int32)
        pages = jnp.asarray(np.arange(5)[:, None, None, None] * np.ones((5, 16, 8, 128)) + 1, jnp.bfloat16)
        kv = jnp.zeros((16, 16, 8, 128), jnp.bfloat16)
        out = np.asarray(kvw.write_pages(ids, pages, kv).astype(jnp.float32))
        got = {b: float(out[b].mean()) for b in (5, 7, 9, 0)}
        want = {5: 3.0, 7: 5.0, 9: 4.0, 0: 0.0}
        return {"block_means": got, "expected_last_write_wins": want, "matches": got == want}

    def qk_norm_rope_bound():
        T, HQ, HKV, D = 512, 4, 2, 128
        qkv = rng.uniform(-1, 1, (T, (HQ + 2 * HKV) * D)).astype(np.float32)
        angles = rng.uniform(-np.pi, np.pi, (T, D // 2)).astype(np.float32)
        cs = np.concatenate([np.cos(angles), np.sin(angles)], axis=-1)
        qw = rng.uniform(0.5, 1.0, (D,)).astype(np.float32)
        kw = rng.uniform(0.5, 1.0, (D,)).astype(np.float32)
        qo, ko = qkr._qk_norm_rope_jax(
            jnp.asarray(qkv, jnp.bfloat16), jnp.asarray(cs, jnp.bfloat16),
            jnp.asarray(qw, jnp.bfloat16), jnp.asarray(kw, jnp.bfloat16),
            1e-6, HQ, HKV, D
        )
        qf = np.asarray(jnp.asarray(qkv, jnp.bfloat16).astype(jnp.float32), np.float64)[:, :HQ * D].reshape(T, HQ, D)
        csf = np.asarray(jnp.asarray(cs, jnp.bfloat16).astype(jnp.float32), np.float64)
        qwf = np.asarray(jnp.asarray(qw, jnp.bfloat16).astype(jnp.float32), np.float64)
        rms = np.sqrt(np.mean(qf * qf, axis=-1, keepdims=True) + 1e-6)
        qn = (qf / rms) * qwf
        c, s = csf[:, None, :D // 2], csf[:, None, D // 2:]
        q_ref = np.concatenate([qn[..., :D // 2] * c - qn[..., D // 2:] * s,
                                qn[..., D // 2:] * c + qn[..., :D // 2] * s], axis=-1).reshape(T, HQ * D)
        err = float(np.abs(np.asarray(qo.astype(jnp.float32), np.float64) - q_ref).max())
        return {"max_abs_err": err, "eps_bound": 0.0157, "within_bound": err <= 0.0157}

    def maxtext_gmm_uninit_vmem_leak():
        # Bug 2: existing_out=None leaves out[...] uninitialized; when sum(group_sizes) < m,
        # unmasked tail rows in m_tile=1 leak stale VMEM values from m_tile=0!
        M, K, N, G = 256, 256, 256, 4
        lhs = jnp.ones((M, K), jnp.bfloat16)
        rhs = jnp.zeros((G, K, N), jnp.bfloat16)
        # Group 0 fills m_tile 0 (rows 0..127) with 256.0; group 1 fills rows 128..159 with 512.0;
        # rows 160..255 belong to no group (padded tail).
        rhs = rhs.at[0].set(1.0).at[1].set(2.0)
        gs = jnp.array([128, 32, 0, 0], jnp.int32)
        out_no_ex = np.asarray(mb.gmm(lhs, rhs, gs, tiling=(128, 128, 128)), np.float32)
        out_with_ex = np.asarray(
            mb.gmm(lhs, rhs, gs, existing_out=jnp.zeros((M, N), jnp.float32), tiling=(128, 128, 128)),
            np.float32,
        )
        return {
            "no_existing_out_tile0_mean": float(out_no_ex[:128].mean()),
            "no_existing_out_tile1_active_mean": float(out_no_ex[128:160].mean()),
            "no_existing_out_tile1_unmasked_tail_mean": float(out_no_ex[160:].mean()),
            "no_existing_out_tile1_unmasked_tail_max": float(np.abs(out_no_ex[160:]).max()),
            "with_existing_out_tile1_unmasked_tail_max": float(np.abs(out_with_ex[160:]).max()),
            "bug_reproduced": bool(np.abs(out_no_ex[160:]).max() > 1.0 and np.abs(out_with_ex[160:]).max() == 0.0),
        }

    def sglang_jax_gmm_quant_midrun_reset():
        # Bug 5: num_quant_blocks=2 with existing_out resets out[...] = existing_out[...] at b_i=1, k_i=0,
        # wiping out quantization block b_i=0's accumulated result!
        M, K, N, G = 256, 512, 256, 4
        lhs_np = np.zeros((M, K), np.float32)
        lhs_np[:, :256] = 1.0  # b_i = 0 contributes 256.0
        lhs_np[:, 256:] = 0.0  # b_i = 1 contributes 0.0
        lhs = jnp.asarray(lhs_np, jnp.bfloat16)
        rhs = jnp.ones((G, N, K), jnp.bfloat16)  # transpose_rhs=True
        gs = jnp.array([64, 64, 64, 64], jnp.int32)
        rs = jnp.ones((G, 2, 1, N), jnp.float32)
        ex = jnp.full((M, N), 10.0, jnp.float32)
        out = np.asarray(
            sj_gmm.gmm(lhs, rhs, gs, rhs_scale=rs, existing_out=ex, tiling=(128, 128, 128), transpose_rhs=True),
            np.float32,
        )
        expected = 10.0 + 256.0  # 266.0
        got_mean = float(out.mean())
        return {
            "expected": expected,
            "got_mean": got_mean,
            "got_min": float(out.min()),
            "got_max": float(out.max()),
            "b0_wiped_bug_reproduced": bool(abs(got_mean - 10.0) < 1e-3),
        }

    def sglang_jax_gmm_alias_crash():
        # Bug 6: hardcoded input_output_aliases = {7: 0} crashes when rhs_scale=None with existing_out
        M, K, N, G = 256, 256, 256, 4
        lhs = jnp.ones((M, K), jnp.bfloat16)
        rhs = jnp.ones((G, N, K), jnp.bfloat16)
        gs = jnp.array([64, 64, 64, 64], jnp.int32)
        ex = jnp.zeros((M, N), jnp.float32)
        try:
            sj_gmm.gmm(lhs, rhs, gs, rhs_scale=None, existing_out=ex, tiling=(128, 128, 128), transpose_rhs=True)
            return {"crashed": False}
        except Exception as e:
            return {"crashed": True, "error": f"{type(e).__name__}: {e}"[:500]}

    run(tests, "exp_calibration", exp_cal)
    run(tests, "div_calibration", div_cal)
    run(tests, "sglang_flash_bound", flash_bound)
    run(tests, "maxtext_ragged_bound", ragged_bound)
    run(tests, "sglang_qk_norm_rope_bound", qk_norm_rope_bound)
    run(tests, "bug1_maxtext_ragged_negative_block", ragged_negative_block)
    run(tests, "bug2_maxtext_gmm_uninit_vmem_leak", maxtext_gmm_uninit_vmem_leak)
    run(tests, "bug3_sglang_flash_ab_oob", flash_ab_oob)
    run(tests, "bug4_sglang_flash_odd_block_q", flash_odd_block_q)
    run(tests, "bug5_sglang_jax_gmm_quant_midrun_reset", sglang_jax_gmm_quant_midrun_reset)
    run(tests, "bug6_sglang_jax_gmm_alias_crash", sglang_jax_gmm_alias_crash)
    run(tests, "kv_page_write_duplicates", kv_dup)
    with open(a.out, "w") as f:
        json.dump(tests, f, indent=2, default=str)


if __name__ == "__main__":
    main()
