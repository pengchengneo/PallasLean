#!/usr/bin/env python3
"""Broad 5-repo PallasLean formal verification sweep across:
- sglang-torchtpu (1f79f6bfbb)
- sglang-jax      (b3e3e16)
- maxtext         (55ed4cffb)
- tpu-inference   (c3b3d4fe4)
- vllm-torchtpu   (11f86257)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import z3
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "frontend"))

from tpuverify.extract import extract  # noqa: E402
from tpuverify.loader import load_file, stub_module  # noqa: E402
from tpuverify.obligations import check_kernel  # noqa: E402


class DummyLogger:
    def __getattr__(self, name):
        return lambda *a, **k: None


def setup_stubs():
    if not hasattr(pltpu.MemorySpace, "ANY"):
        pltpu.MemorySpace.ANY = pl.ANY
    F32_MASK = -0.7 * float(jnp.finfo(jnp.float32).max)
    stub_module("requests")
    stub_module("torch", dtype=object, Tensor=object, bfloat16=object, empty_like=lambda x: x)
    stub_module("torch_tpu")
    stub_module("vllm.logger", _VllmLogger=DummyLogger)
    SGL_ROOT = os.environ.get("SGLANG_TPU_ROOT", "../sglang-torchtpu")
    SJ_ROOT = os.environ.get("SGLANG_JAX_ROOT", "../sglang-jax")
    TI_ROOT = os.environ.get("TPU_INFERENCE_ROOT", "../tpu-inference")
    VT_ROOT = os.environ.get("VLLM_TORCHTPU_ROOT", "../vllm-torchtpu")
    stub_module("sglang_tpu", __path__=[f"{SGL_ROOT}/sglang_tpu"])
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
    stub_module("qwix.pallas", QArray=DummyQArray, dot_general=jax.lax.dot_general, pallas_call=pl.pallas_call)
    stub_module("sgl_jax", __path__=[f"{SJ_ROOT}/python/sgl_jax"])
    stub_module("sgl_jax.srt", __path__=[f"{SJ_ROOT}/python/sgl_jax/srt"])
    stub_module("sgl_jax.srt.utils", cdiv=lambda a, b: (a + b - 1) // b)
    stub_module("sgl_jax.srt.utils.jax_utils", get_device_kind=lambda: "TPU7x")
    stub_module("tpu_inference", __path__=[f"{TI_ROOT}/tpu_inference"])
    stub_module("tpu_inference.logger", init_logger=lambda n: DummyLogger())
    stub_module("vllm_torchtpu", __path__=[f"{VT_ROOT}/src/vllm_torchtpu"])
    stub_module("vllm_torchtpu.logger", init_logger=lambda n: DummyLogger())


S = jax.ShapeDtypeStruct


def run_one(report: list, repo: str, label: str, fn, args, req=None, arg_names=()):
    t0 = time.monotonic()
    try:
        mdls = extract(fn, *args, name=label, arg_names=arg_names)
        entry = {"repo": repo, "kernel": label, "ok": True, "subkernels": [], "seconds": 0.0}
        for idx, mdl in enumerate(mdls):
            res = check_kernel(mdl, req, fn_args=(fn, args) if idx == 0 else None)
            counts = {}
            failed = []
            warns = []
            vmem = ""
            for r in res:
                counts[r.status] = counts.get(r.status, 0) + 1
                if r.status == "FAILED":
                    failed.append({"name": r.name, "detail": r.detail, "cex": r.cex})
                elif r.status == "WARN":
                    warns.append({"name": r.name, "detail": r.detail})
                elif r.name == "v7x-vmem-budget":
                    vmem = r.detail
            entry["subkernels"].append({
                "name": mdl.name,
                "grid": list(mdl.grid),
                "counts": counts,
                "vmem": vmem,
                "failed": failed,
                "warns": warns,
            })
        entry["seconds"] = round(time.monotonic() - t0, 2)
        report.append(entry)
        sk = entry["subkernels"][0]
        tag = f"FAILED({len(sk['failed'])})" if sk["failed"] else ("WARN" if sk["warns"] else "PROVED")
        print(f"[{repo}] {label}: {tag} {sk['counts']} ({entry['seconds']}s)", flush=True)
        for f in sk["failed"]:
            print(f"    FAILED {f['name']}: {f['detail']} | cex={f['cex']}", flush=True)
        for w in sk["warns"]:
            print(f"    WARN   {w['name']}: {w['detail']}", flush=True)
    except Exception as e:
        entry = {"repo": repo, "kernel": label, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        report.append(entry)
        print(f"[{repo}] {label}: TRACE_ERR {entry['error']}", flush=True)


def main():
    setup_stubs()
    report = []
    SGL = f'{os.environ.get("SGLANG_TPU_ROOT", "../sglang-torchtpu")}/sglang_tpu'
    SJ = f'{os.environ.get("SGLANG_JAX_ROOT", "../sglang-jax")}/python/sgl_jax/srt/kernels'
    MXT = f'{os.environ.get("MAXTEXT_ROOT", "../maxtext")}/src/MaxText/kernels'
    TI = f'{os.environ.get("TPU_INFERENCE_ROOT", "../tpu-inference")}/tpu_inference/kernels'
    VT = f'{os.environ.get("VLLM_TORCHTPU_ROOT", "../vllm-torchtpu")}/src/vllm_torchtpu/kernels'

    # ---------------- 1. sglang-torchtpu extra kernels ----------------
    try:
        m = load_file("sgl_kv_row", f"{SGL}/kv_row_write.py")
        run_one(report, "sglang-torchtpu", "sglang.kv_row_write",
                lambda ns, seg, rows, kv: m.write_rows(ns, seg, rows, kv),
                (S((1,), jnp.int32), S((16, 4), jnp.int32), S((16, 8, 128), jnp.bfloat16), S((256, 16, 8, 128), jnp.bfloat16)))
    except Exception as e:
        print("skip kv_row_write:", e)

    try:
        m = load_file("sgl_band_fa", f"{SGL}/kernels/band_skip_flash_attention.py")
        bs = m.BlockSizes(block_q=128, block_k_major=128, block_k=128, block_b=1)
        run_one(report, "sglang-torchtpu", "sglang.band_skip_flash_attention",
                lambda q, k, v: m.flash_attention(q, k, v, causal=True, sm_scale=0.125, block_sizes=bs, vmem_limit_bytes=None),
                (S((1, 2, 512, 128), jnp.bfloat16), S((1, 2, 512, 128), jnp.bfloat16), S((1, 2, 512, 128), jnp.bfloat16)))
    except Exception as e:
        print("skip band_skip_fa:", e)

    # ---------------- 2. sglang-jax extra kernels ----------------
    try:
        m = load_file("sj_state_init", f"{SJ}/dsv4/state_init.py")
        run_one(report, "sglang-jax", "sglang_jax.dsv4_state_init",
                lambda st, slots, tmpl: m.init_state_slots(st, slots, tmpl),
                (S((64, 8, 128), jnp.bfloat16), S((16,), jnp.int32), S((8, 128), jnp.bfloat16)),
                req=lambda ctx: ctx.pf_values(0, lambda i, v: z3.And(0 <= v, v < 64)))
    except Exception as e:
        print("skip sj_state_init:", e)

    try:
        m = load_file("sj_paged_row", f"{SJ}/dsv4/paged_row_write.py")
        run_one(report, "sglang-jax", "sglang_jax.dsv4_paged_row_write",
                lambda cache, vals, loc, valid: m.paged_row_write(cache, vals, loc, valid),
                (S((256, 16, 8, 128), jnp.bfloat16), S((16, 8, 128), jnp.bfloat16), S((16,), jnp.int32), S((16,), jnp.bool_)))
    except Exception as e:
        print("skip sj_paged_row:", e)

    try:
        from sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels import kernel as sj_qmm
        run_one(report, "sglang-jax", "sglang_jax.quantized_matmul",
                lambda x, w, s: sj_qmm.quantized_matmul_kernel(x, w, s),
                (S((256, 256), jnp.bfloat16), S((256, 256), jnp.int8), S((256,), jnp.float32)))
    except Exception as e:
        print("skip sj_qmm:", e)

    # ---------------- 3. maxtext extra kernels ----------------
    try:
        m = load_file("mxt_ragged_gather", f"{MXT}/ragged/ragged_gather_tc.py")
        run_one(report, "maxtext", "maxtext.ragged_gather_tc",
                lambda x, idx: m.ragged_gather_tc(x, idx, start=0, num_tokens=64, max_out_tokens=128, block_size=128),
                (S((256, 128), jnp.bfloat16), S((128,), jnp.int32)))
    except Exception as e:
        print("skip mxt_ragged_gather:", e)

    # ---------------- 4. tpu-inference kernels ----------------
    from tpu_inference.kernels.megablox import gmm as ti_gmm
    M, K, N, G = 256, 256, 256, 4

    def gmm_req(ctx):
        m_tiles = 2
        max_tiles = m_tiles + G - 1
        return z3.And(
            ctx.pf_values(0, lambda i, v: z3.And(0 <= v, v <= M)),
            ctx.pf_values(1, lambda i, v: z3.And(0 <= v, v < G)),
            ctx.pf_values(2, lambda i, v: z3.And(0 <= v, v < m_tiles)),
            z3.And(*[ctx.pf[2][i] <= ctx.pf[2][i + 1] for i in range(max_tiles - 1)]),
            ctx.pf[2][0] == 0, ctx.pf[2][1] == 1, ctx.G[1] == m_tiles,
            ctx.pf_values(3, lambda i, v: v == 0),
        )

    run_one(report, "tpu-inference", "tpu_inference.megablox_gmm[no_existing_out]",
            lambda l, r, g: ti_gmm.gmm(l, r, g, tiling=(128, 128, 128)),
            (S((M, K), jnp.bfloat16), S((G, K, N), jnp.bfloat16), S((G,), jnp.int32)),
            req=gmm_req, arg_names=("lhs", "rhs", "out"))

    run_one(report, "tpu-inference", "tpu_inference.megablox_gmm[with_existing_out_alias_bug]",
            lambda l, r, g, e: ti_gmm.gmm(l, r, g, existing_out=e, tiling=(128, 128, 128)),
            (S((M, K), jnp.bfloat16), S((G, K, N), jnp.bfloat16), S((G,), jnp.int32), S((M, N), jnp.float32)),
            req=gmm_req, arg_names=("lhs", "rhs", "existing_out", "out"))

    from tpu_inference.kernels.quantized_matmul import kernel as ti_qmm
    run_one(report, "tpu-inference", "tpu_inference.quantized_matmul",
            lambda x, w, s: ti_qmm.quantized_matmul_kernel(x, w, s),
            (S((256, 256), jnp.bfloat16), S((256, 256), jnp.int8), S((256,), jnp.float32)))

    from tpu_inference.kernels.quantized_matmul import blockwise_kernel as ti_bqmm
    run_one(report, "tpu-inference", "tpu_inference.blockwise_quantized_matmul",
            lambda x, w, s: ti_bqmm.quantized_matmul_kernel(x, w, s, block_size=128),
            (S((256, 256), jnp.bfloat16), S((256, 256), jnp.int8), S((2, 1, 256), jnp.float32)))

    from tpu_inference.kernels.gdn import triangle_solver as ti_tri
    run_one(report, "tpu-inference", "tpu_inference.gdn.newton_schulz_inverse",
            lambda a: ti_tri.newton_schulz_inverse_pallas(a),
            (S((4, 64, 64), jnp.float32),))
    run_one(report, "tpu-inference", "tpu_inference.gdn.decompose_tri_inv",
            lambda a: ti_tri.decompose_triangular_matrix_inverse_pallas(a),
            (S((4, 64, 64), jnp.float32),))

    from tpu_inference.kernels.mla.v2 import transpose as ti_xpose
    run_one(report, "tpu-inference", "tpu_inference.mla_v2.xpose_full",
            lambda x: ti_xpose.xpose_full(x, transpose_axes=(1, 0)),
            (S((256, 256), jnp.bfloat16),))

    # ---------------- 5. vllm-torchtpu kernels ----------------
    from vllm_torchtpu.kernels.quantized_matmul import kernel as vt_qmm
    run_one(report, "vllm-torchtpu", "vllm_torchtpu.quantized_matmul",
            lambda x, w, s: vt_qmm.quantized_matmul_kernel(x, w, s),
            (S((256, 256), jnp.bfloat16), S((256, 256), jnp.int8), S((256,), jnp.float32)))

    from vllm_torchtpu.kernels.mla.v2 import transpose as vt_xpose
    run_one(report, "vllm-torchtpu", "vllm_torchtpu.mla_v2.xpose_full",
            lambda x: vt_xpose.xpose_full(x, transpose_axes=(1, 0)),
            (S((256, 256), jnp.bfloat16),))

    from vllm_torchtpu.kernels.megablox import gmm_v2 as vt_gmm
    ti_cfg = vt_gmm.TileSizes(tile_m=128, tile_k=128, tile_n=128, bucket_base=128)

    def gmm_v2_req(ctx):
        return z3.And(ctx.pf_values(0, lambda i, v: z3.And(0 <= v, v <= M)), ctx.pf_values(1, lambda i, v: v == 0))

    run_one(report, "vllm-torchtpu", "vllm_torchtpu.megablox_gmm_v2[zero_init=True]",
            lambda l, r, g: vt_gmm.gmm_v2(l, r, g, tile_info=ti_cfg, zero_initialize=True),
            (S((M, K), jnp.bfloat16), S((G, K, N), jnp.bfloat16), S((G,), jnp.int32)),
            req=gmm_v2_req, arg_names=("lhs", "rhs", "out"))

    run_one(report, "vllm-torchtpu", "vllm_torchtpu.megablox_gmm_v2[zero_init=False]",
            lambda l, r, g: vt_gmm.gmm_v2(l, r, g, tile_info=ti_cfg, zero_initialize=False),
            (S((M, K), jnp.bfloat16), S((G, K, N), jnp.bfloat16), S((G,), jnp.int32)),
            req=gmm_v2_req, arg_names=("lhs", "rhs", "out"))

    out_path = ROOT / "examples" / "scan_5repos_results.json"
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\nSaved {len(report)} kernel sweep results to {out_path}")


if __name__ == "__main__":
    main()
