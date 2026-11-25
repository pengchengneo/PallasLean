#!/usr/bin/env python3
"""Offline (CPU-hosted) TPU v7x `libtpu` AOT compilation & Final LLO verification.

Runs on Linux x86-64 with `jax`, `jaxlib`, and `libtpu` installed (no physical TPU
required). Uses `get_topology_desc('tpu7x:2x2x1', platform='tpu')`
to invoke `libtpu.so`'s real TPU v7x Mosaic + LLO compiler on every production kernel
from `sglang-torchtpu` and `maxtext`, dumps `*final_bundles.txt` and `mosaic/*.mlir`,
and verifies the emitted TPU v7x VLIW instruction bundles.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "frontend"))


def dump_flags(out_dir: Path) -> str:
    return (
        f"--xla_jf_dump_to={out_dir}/llo "
        f"--xla_mosaic_dump_to={out_dir}/mosaic "
        "--xla_jf_dump_llo_text=true "
        "--xla_jf_dump_llo_pass_label_regex=.*final.* "
        "--xla_jf_dump_category_filter=passes "
        "--xla_jf_dump_llo_html=false "
        "--xla_mosaic_enable_llo_source_annotations=true "
        "--xla_mosaic_enable_dump_debug_info=true "
        "--xla_jf_emit_annotations=true "
        "--xla_jf_collect_llo_stack_trace=true"
    )


def load_mod(name: str, path: str):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="Root containing sglang_tpu/ and src/MaxText/")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for LLO dumps and report")
    ap.add_argument("--target", default="tpu7x:2x2x1", help="TPU v7x topology string")
    a = ap.parse_args()

    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for child in out.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    (out / "llo").mkdir(parents=True, exist_ok=True)
    (out / "mosaic").mkdir(parents=True, exist_ok=True)

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["LIBTPU_INIT_ARGS"] = dump_flags(out)

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.topologies import get_topology_desc
    import numpy as np
    from tpuverify.llo_verify import parse_final_bundles

    S = jax.ShapeDtypeStruct
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
    from jax.experimental.pallas import tpu as pltpu
    if not hasattr(pltpu.MemorySpace, "ANY"):
        pltpu.MemorySpace.ANY = pl.ANY

    fa = load_mod("fa", f"{a.src}/sglang_tpu/kernels/flash_attention.py")
    kvw = load_mod("kvw", f"{a.src}/sglang_tpu/kv_page_write.py")
    qkr = load_mod("qkr", f"{a.src}/sglang_tpu/qk_norm_rope.py")
    kimi = load_mod("kimi", f"{a.src}/sglang_tpu/kimi_attn_residual.py")
    ra = load_mod("ra", f"{a.src}/src/MaxText/kernels/attention/ragged_attention.py")
    mb = load_mod("mb", f"{a.src}/src/MaxText/kernels/megablox/backend.py")
    sj_common = load_mod("sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.common", f"{a.src}/sgl_jax/common.py")
    stub("sgl_jax.srt.kernels.gmm.megablox_gmm_kernel", common=sj_common)
    sj_gmm = load_mod("sj_gmm", f"{a.src}/sgl_jax/gmm.py")
    sj_ukv = load_mod("sj_ukv", f"{a.src}/sgl_jax/update_kv_cache.py")

    device = get_topology_desc(a.target, platform="tpu").devices[0]
    sharding = jax.sharding.SingleDeviceSharding(device)
    mesh = jax.sharding.Mesh(np.array([device]), ("tensor",))

    bs_vis = fa.BlockSizes(block_q=128, block_k_major=1024, block_k=1024, block_b=1)
    bs_cau = fa.BlockSizes(block_q=128, block_k_major=128, block_k=128, block_b=1)

    workloads = [
        (
            "sglang.kv_page_write",
            kvw.write_pages,
            (S((16,), jnp.int32), S((16, 16, 8, 128), jnp.bfloat16), S((4096, 16, 8, 128), jnp.bfloat16)),
        ),
        (
            "sglang.qk_norm_rope",
            lambda qkv, cs, qw, kw: qkr._qk_norm_rope_jax(qkv, cs, qw, kw, 1e-6, 4, 2, 128),
            (S((512, 1024), jnp.bfloat16), S((512, 128), jnp.bfloat16), S((128,), jnp.bfloat16), S((128,), jnp.bfloat16)),
        ),
        (
            "sglang.kimi_attn_res_decode",
            lambda pa, pb, bank, cw, w: kimi.attention_residual_decode_jax(pa, pb, bank, cw, w, nvb=4, eps=1e-6, out_eps=1e-6),
            (S((8, 128), jnp.bfloat16), S((8, 128), jnp.bfloat16), S((8, 8, 128), jnp.bfloat16), S((128,), jnp.float32), S((128,), jnp.float32)),
        ),
        (
            "sglang.flash_vision_single_step",
            lambda q, k, v: fa.flash_attention(q, k, v, causal=False, sm_scale=0.125, block_sizes=bs_vis, vmem_limit_bytes=None),
            (S((1, 2, 1024, 128), jnp.bfloat16), S((1, 2, 1024, 128), jnp.bfloat16), S((1, 2, 1024, 128), jnp.bfloat16)),
        ),
        (
            "sglang.flash_causal_multi_block",
            lambda q, k, v: fa.flash_attention(q, k, v, causal=True, sm_scale=0.125, block_sizes=bs_cau, vmem_limit_bytes=None),
            (S((1, 2, 512, 128), jnp.bfloat16), S((1, 2, 512, 128), jnp.bfloat16), S((1, 2, 512, 128), jnp.bfloat16)),
        ),
        (
            "maxtext.ragged_mqa",
            lambda q, k, v, l: ra.ragged_mqa(q, k, v, l, block_size=256),
            (S((8, 8, 128), jnp.float32), S((8, 4096, 128), jnp.bfloat16), S((8, 4096, 128), jnp.bfloat16), S((8,), jnp.int32)),
        ),
        (
            "maxtext.megablox_gmm",
            lambda l, r, g, e: mb.gmm(l, r, g, existing_out=e, tiling=(128, 128, 128)),
            (S((256, 256), jnp.bfloat16), S((4, 256, 256), jnp.bfloat16), S((4,), jnp.int32), S((256, 256), jnp.float32)),
        ),
        (
            "sglang_jax.megablox_gmm",
            lambda l, r, g, s, e: sj_gmm.gmm(l, r, g, rhs_scale=s, existing_out=e, tiling=(128, 128, 128), transpose_rhs=True),
            (
                S((256, 256), jnp.bfloat16),
                S((4, 256, 256), jnp.bfloat16),
                S((4,), jnp.int32),
                S((4, 1, 1, 256), jnp.float32),
                S((256, 256), jnp.float32),
            ),
        ),
        (
            "sglang_jax.update_kv_cache",
            lambda new_kv, slices, kv_cache, n_upd: sj_ukv.kv_cache_update(
                new_kv, slices, kv_cache, n_upd, page_size=16, num_slices_per_block=8
            ),
            (
                S((32, 8, 128), jnp.bfloat16),
                S((3, 8), jnp.int32),
                S((1024, 8, 128), jnp.bfloat16),
                S((1,), jnp.int32),
            ),
        ),
    ]

    versions = {}
    for pkg in ("jax", "jaxlib", "libtpu"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "?"

    results = {"target": a.target, "versions": versions, "kernels": {}}
    seen_llo: set[Path] = set()

    with jax.set_mesh(mesh):
        for name, fn, inputs in workloads:
            t0 = time.monotonic()
            try:
                lowered = jax.jit(fn, in_shardings=(sharding,) * len(inputs)).lower(*inputs)
                compiled = lowered.compile()
                dt = round(time.monotonic() - t0, 3)
                new_finals = sorted(
                    f for f in (out / "llo").rglob("*final*.txt")
                    if "schedule-analysis" not in f.name and f not in seen_llo
                )
                seen_llo.update(new_finals)
                summaries = [parse_final_bundles(name, f)._asdict() for f in new_finals]
                results["kernels"][name] = {
                    "ok": True,
                    "compile_seconds": dt,
                    "memory_analysis": str(compiled.memory_analysis()),
                    "llo_bundles": summaries,
                }
            except Exception as e:
                results["kernels"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            print(name, json.dumps(results["kernels"][name], indent=2), flush=True)

    (out / "aot_v7x_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {out / 'aot_v7x_results.json'}")


if __name__ == "__main__":
    main()
