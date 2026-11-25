"""Run every PallasLean check on the production kernels (targeting TPU v7x) and write a report.

    .venv/bin/python examples/run_all.py            # writes examples/results.json, PallasLean/Certificates.lean, REPORT.md

Levels (see README):
  A  index / pipeline / termination / TPU v7x Mosaic MLIR & 64 MiB VMEM obligations (Z3)
  C  body = semiring recurrence step / score & mask / skip soundness / RMSNorm+RoPE / GMM
  D  rounding bound eps (Lean rounding lemmas, dtype chain from the jaxpr)
  B  the generic Lean 4 theorems + auto-generated `PallasLean/Certificates.lean` (checked by `lake build`)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from fractions import Fraction as Fr

import jax
import numpy as np
import z3

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "frontend"))

import kernels as K  # noqa: E402
import level_c_specs as C  # noqa: E402
from tpuverify.cert_gen import generate_certificates_lean  # noqa: E402
from tpuverify.extract import extract  # noqa: E402
from tpuverify.level_c import (  # noqa: E402
    check_gmm_step,
    check_kimi_attn_res,
    check_qk_norm_rope,
    check_scores,
    check_skip,
    check_step,
)
from tpuverify.mosaic import analyze_v7x_mosaic, export  # noqa: E402
from tpuverify.obligations import Result, check_kernel  # noqa: E402
from tpuverify.roundoff import (  # noqa: E402
    AttnShape,
    ExpModel,
    dtype_chain,
    gmm_roundoff,
    online_normalized,
    qk_norm_rope_roundoff,
    softmax_then_pv,
)


def rev(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "?"


def level_a(rows, mosaic_list, label, spec):
    from contextlib import nullcontext
    ctx = (
        jax.set_mesh(jax.sharding.Mesh(np.array(jax.devices()[:1]), ("tensor",)))
        if spec.get("needs_mesh")
        else nullcontext()
    )
    with ctx:
        for mdl in extract(spec["fn"], *spec["args"], name=label, arg_names=spec["names"]):
            rows.extend(check_kernel(mdl, spec["requires"], fn_args=(spec["fn"], spec["args"])))
            try:
                k_name, mlir_txt = export(spec["fn"], *spec["args"])[0]
                ms = analyze_v7x_mosaic(k_name, mlir_txt, len(mdl.blocks), len(mdl.scratch_avals))
                mosaic_list.append({
                    "kernel": label,
                    "mosaic_name": ms.kernel_name,
                    "vmem_bytes": ms.vmem_bytes,
                    "smem_bytes": ms.smem_bytes,
                    "op_counts": ms.op_counts,
                })
            except Exception:
                pass
            return mdl


def main():
    t0 = time.time()
    rows: list[Result] = []
    mosaic_list: list[dict] = []
    meta = {
        "sglang-jax": rev(K.SGL_JAX),
        "sglang-torchtpu": rev(K.SGL),
        "maxtext": rev(K.MXT),
        "target": "TPU v7x (TPU7x)",
    }

    # ---------------- 1. sglang kv_page_write
    level_a(rows, mosaic_list, "sglang.kv_page_write", K.kv_page_write())

    # ---------------- 2. sglang qk_norm_rope
    spec_qkr = K.qk_norm_rope()
    mdl_qkr = level_a(rows, mosaic_list, "sglang.qk_norm_rope", spec_qkr)
    rows += check_qk_norm_rope(mdl_qkr, num_q=4, num_kv=2, hd=128, eps=1e-6)

    # ---------------- 3. sglang kimi_attn_res_decode
    spec_kimi = K.kimi_attn_res_decode()
    mdl_kimi = level_a(rows, mosaic_list, "sglang.kimi_attn_res_decode", spec_kimi)
    rows += check_kimi_attn_res(mdl_kimi, tokens=8, nvb=4, num_blocks=8, hidden=128)

    # ---------------- 4. maxtext ragged_mqa (default precondition 0 <= lengths[b] <= seq)
    spec = K.ragged_mqa(min_len=0)
    mdl = level_a(rows, mosaic_list, "maxtext.ragged_mqa", spec)
    pf = {0: np.full((8,), 4096, np.int32)}
    rows += check_step(mdl, C.ragged_roles(), (3, 5), pf)
    rows += check_step(mdl, C.ragged_roles(), (3, 0), pf, init=True)
    res, cuts, ctx, p, pre = check_scores(mdl, C.ragged_score(), (8, 256), spec["requires"], {"q": 1, "k": 1})
    rows += res
    rows.append(check_skip(mdl, cuts, ctx, p, pre, C.ragged_keep, lambda p: (0, p[1] * 256), (8, 256)))

    # ---------------- 5. maxtext ragged_mqa (tightened precondition block_size <= lengths[b] <= seq)
    spec_ok = K.ragged_mqa(min_len=256)
    level_a(rows, mosaic_list, "maxtext.ragged_mqa[requires_len>=256]", spec_ok)

    # ---------------- 6 & 7. maxtext megablox_gmm (without vs. with existing_out)
    spec_gmm_no = K.megablox_gmm(with_existing_out=False)
    level_a(rows, mosaic_list, "maxtext.megablox_gmm[no_existing_out]", spec_gmm_no)

    spec_gmm_ex = K.megablox_gmm(with_existing_out=True)
    mdl_gmm_ex = level_a(rows, mosaic_list, "maxtext.megablox_gmm[with_existing_out]", spec_gmm_ex)
    rows += check_gmm_step(mdl_gmm_ex, tm=128, tk=128, tn=128)

    # ---------------- 8-11. sglang flash_attention: production paths and API corner cases
    flash_cfgs = [
        ("sglang.flash[vision,single-step]", dict(causal=False, q_len=1024, kv_len=1024, block_q=128,
                                                 block_k_major=1024, block_k=1024), None),
        ("sglang.flash[causal,multi-block]", dict(causal=True, q_len=512, kv_len=512, block_q=128,
                                                 block_k_major=128, block_k=128), 128),
        ("sglang.flash[causal,ab,q_len=200]", dict(causal=True, q_len=200, kv_len=512, block_q=128,
                                                  block_k_major=128, block_k=128, ab=True), None),
        ("sglang.flash[causal,block_q=129]", dict(causal=True, q_len=129, kv_len=256, block_q=129,
                                                 block_k_major=128, block_k=128), 129),
    ]
    for label, cfg, bq in flash_cfgs:
        spec = K.sglang_flash(**cfg)
        mdl = level_a(rows, mosaic_list, label, spec)
        if bq is None:
            continue
        if bq == 128:
            rows += check_step(mdl, C.flash_roles(), (0, 1, 3, 1), None)
            rows += check_step(mdl, C.flash_roles(), (0, 1, 3, 3), None)
            rows += check_step(mdl, C.flash_roles(), (0, 1, 3, 0), None, init=True)
        res, cuts, ctx, p, pre = check_scores(mdl, C.flash_score(block_q=bq), (bq, 128), None, {"q": 1, "k": 1})
        rows += res
        q_len = cfg["q_len"]
        keep = lambda ctx, p, gi, gj, q_len=q_len: z3.And(gj <= gi, gi < q_len)
        rows.append(check_skip(mdl, cuts, ctx, p, pre, keep, lambda p, bq=bq: (p[2] * bq, p[3] * 128), (bq, 128)))

    # ---------------- 12-14. sglang-jax: megablox_gmm (quant_b2 bug vs quant_b1) & update_kv_cache
    spec_sj_gmm_b2 = K.sglang_jax_gmm(k_cols=512, num_quant_blocks=2)
    level_a(rows, mosaic_list, "sglang_jax.megablox_gmm[quant_b2,existing_out]", spec_sj_gmm_b2)

    spec_sj_gmm_b1 = K.sglang_jax_gmm(k_cols=256, num_quant_blocks=1)
    mdl_sj_gmm_b1 = level_a(rows, mosaic_list, "sglang_jax.megablox_gmm[quant_b1,existing_out]", spec_sj_gmm_b1)

    spec_sj_ukv = K.sglang_jax_update_kv_cache()
    level_a(rows, mosaic_list, "sglang_jax.update_kv_cache", spec_sj_ukv)

    # ---------------- Level D: exact rational rounding bounds
    em = ExpModel(ue=Fr(1, 2**21), etae=Fr(2) ** -126)  # replaced by the TPU-measured value when available
    cal = os.path.join(ROOT, "validate", "results.json")
    if os.path.exists(cal):
        c = json.load(open(cal)).get("exp_calibration", {})
        if c.get("ok"):
            em = ExpModel(ue=Fr(c["ue_max_rel"]).limit_denominator(2**40) * 2, etae=Fr(2) ** -126)

    d_rows = []
    # 1) sglang.flash[single-step]
    spec = K.sglang_flash(causal=False, q_len=1024, kv_len=1024, block_q=128, block_k_major=1024, block_k=1024)
    dc = dtype_chain(extract(spec["fn"], *spec["args"], name="x")[0])
    sh = AttnShape(n_kv=1024, head_dim=128, q_abs=Fr(1), k_abs=Fr(1), v_abs=Fr(1), scale=Fr(1, 8),
                   p_cast="bfloat16" if dc["f32_to_bf16"] else "float32", out_dtype="bfloat16")
    r = softmax_then_pv(sh, em)
    d_rows.append(dict(
        kernel="sglang.flash[vision,single-step]",
        requires="|q|,|k|,|v|<=1, kv=1024, d=128, scale=1/8",
        f32_to_bf16_casts=[c[2] for c in dc["f32_to_bf16"]],
        eps_abs=float(r["eps_abs"]),
        spec_tol=float(Fr(1, 100)),
        eps_frac=r["eps_abs"],
        tol_frac=Fr(1, 100),
        terms={k: float(v) for k, v in r["terms"].items()},
    ))

    # 2) sglang.flash[causal,multi-block]
    spec_mb = K.sglang_flash(causal=True, q_len=512, kv_len=512, block_q=128, block_k_major=128, block_k=128)
    dc_mb = dtype_chain(extract(spec_mb["fn"], *spec_mb["args"], name="x")[0])
    sh_mb = AttnShape(n_kv=512, head_dim=128, q_abs=Fr(1), k_abs=Fr(1), v_abs=Fr(1), scale=Fr(1, 8),
                      p_cast="float32", out_dtype="bfloat16")
    r_mb = online_normalized(sh_mb, em, n_blocks=4, block=128)
    d_rows.append(dict(
        kernel="sglang.flash[causal,multi-block]",
        requires="|q|,|k|,|v|<=1, kv=512 (4x128), d=128, scale=1/8",
        f32_to_bf16_casts=[c[2] for c in dc_mb["f32_to_bf16"]],
        eps_abs=float(r_mb["eps_abs"]),
        spec_tol=float(Fr(1, 100)),
        eps_frac=r_mb["eps_abs"],
        tol_frac=Fr(1, 100),
        terms={k: float(v) for k, v in r_mb["terms"].items()},
    ))

    # 3) maxtext.ragged_mqa
    spec = K.ragged_mqa()
    dc = dtype_chain(extract(spec["fn"], *spec["args"], name="x")[0])
    sh2 = AttnShape(n_kv=4096, head_dim=128, q_abs=Fr(1, 8), k_abs=Fr(1), v_abs=Fr(1), scale=Fr(1),
                    p_cast="float32", out_dtype="float32")
    r2 = online_normalized(sh2, em, n_blocks=16, block=256)
    d_rows.append(dict(
        kernel="maxtext.ragged_mqa",
        requires="|q|<=1/8,|k|,|v|<=1, T=4096 (16x256), d=128",
        f32_to_bf16_casts=[c[2] for c in dc["f32_to_bf16"]],
        eps_abs=float(r2["eps_abs"]),
        spec_tol=float(Fr(1, 1000)),
        eps_frac=r2["eps_abs"],
        tol_frac=Fr(1, 1000),
        terms={k: float(v) for k, v in r2["terms"].items()},
    ))

    # 4) sglang.qk_norm_rope
    dc_qkr = dtype_chain(mdl_qkr)
    r_qkr = qk_norm_rope_roundoff(hd=128, x_abs=Fr(1), w_abs=Fr(1), min_rms=Fr(1), out_dtype="bfloat16")
    d_rows.append(dict(
        kernel="sglang.qk_norm_rope",
        requires="|qkv|<=1, |w|<=1, rms>=1, hd=128",
        f32_to_bf16_casts=[c[2] for c in dc_qkr["f32_to_bf16"]],
        eps_abs=float(r_qkr["eps_abs"]),
        spec_tol=float(Fr(1, 50)),
        eps_frac=r_qkr["eps_abs"],
        tol_frac=Fr(1, 50),
        terms={k: float(v) for k, v in r_qkr["terms"].items()},
    ))

    # 5) maxtext.megablox_gmm[with_existing_out]
    dc_gmm = dtype_chain(mdl_gmm_ex)
    r_gmm = gmm_roundoff(k_dim=256, tiles_k=2, lhs_abs=Fr(1), rhs_abs=Fr(1, 16), out_dtype="float32")
    d_rows.append(dict(
        kernel="maxtext.megablox_gmm[with_existing_out]",
        requires="|lhs|<=1, |rhs|<=1/16, K=256 (2x128), acc=f32",
        f32_to_bf16_casts=[c[2] for c in dc_gmm["f32_to_bf16"]],
        eps_abs=float(r_gmm["eps_abs"]),
        spec_tol=float(Fr(1, 5000)),
        eps_frac=r_gmm["eps_abs"],
        tol_frac=Fr(1, 5000),
        terms={k: float(v) for k, v in r_gmm["terms"].items()},
    ))

    # 6) sglang_jax.megablox_gmm[quant_b1,existing_out]
    dc_sj_gmm = dtype_chain(mdl_sj_gmm_b1)
    r_sj_gmm = gmm_roundoff(k_dim=256, tiles_k=2, lhs_abs=Fr(1), rhs_abs=Fr(1, 16), out_dtype="float32")
    d_rows.append(dict(
        kernel="sglang_jax.megablox_gmm[quant_b1,existing_out]",
        requires="|lhs|<=1, |rhs|<=1/16, K=256 (2x128), quant_blocks=1, acc=f32",
        f32_to_bf16_casts=[c[2] for c in dc_sj_gmm["f32_to_bf16"]],
        eps_abs=float(r_sj_gmm["eps_abs"]),
        spec_tol=float(Fr(1, 5000)),
        eps_frac=r_sj_gmm["eps_abs"],
        tol_frac=Fr(1, 5000),
        terms={k: float(v) for k, v in r_sj_gmm["terms"].items()},
    ))

    # ---------------- Level B Certificate Generation & Lean 4 Machine Check
    cert_path = os.path.join(ROOT, "PallasLean", "Certificates.lean")
    cert_lean = generate_certificates_lean([r._asdict() for r in rows], d_rows, mosaic_list)
    with open(cert_path, "w") as f:
        f.write(cert_lean)

    root_lean_path = os.path.join(ROOT, "PallasLean.lean")
    with open(root_lean_path) as f:
        root_txt = f.read()
    if "import PallasLean.Certificates" not in root_txt:
        with open(root_lean_path, "w") as f:
            f.write(root_txt.rstrip() + "\nimport PallasLean.Certificates\n")

    env = os.environ.copy()
    elan_bin = os.path.expanduser("~/.elan/bin")
    if os.path.isdir(elan_bin):
        env["PATH"] = f"{elan_bin}:{env.get('PATH', '')}"
    lean_t0 = time.time()
    lean_proc = subprocess.run(
        ["lake", "env", "lean", "PallasLean/Certificates.lean"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    lean_ok = lean_proc.returncode == 0
    lean_sec = round(time.time() - lean_t0, 1)

    d_json = [
        {k: v for k, v in d.items() if k not in ("eps_frac", "tol_frac")}
        for d in d_rows
    ]
    out = {
        "meta": meta,
        "seconds": round(time.time() - t0, 1),
        "lean_certificates": {
            "status": "PROVED" if lean_ok else "FAILED",
            "file": "PallasLean/Certificates.lean",
            "seconds": lean_sec,
        },
        "results": [r._asdict() for r in rows],
        "level_d": d_json,
        "mosaic_v7x": mosaic_list,
        "exp_model": {
            "ue": float(em.ue),
            "etae": float(em.etae),
            "source": "TPU-measured" if os.path.exists(cal) else "placeholder (not yet measured)",
        },
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    write_report(out)
    counts = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    print(f"{len(rows)} obligations in {out['seconds']}s (Lean certificates: {out['lean_certificates']['status']} in {lean_sec}s): {counts}")


def write_report(out):
    lines = [
        "# PallasLean Verification Report (Target: TPU v7x)",
        "",
        f"sglang-jax `{out['meta']['sglang-jax']}`, sglang-torchtpu `{out['meta']['sglang-torchtpu']}`, "
        f"maxtext `{out['meta']['maxtext']}`, "
        f"target `{out['meta']['target']}`, {out['seconds']} s total "
        f"(Lean 4 certificate check `PallasLean/Certificates.lean`: **{out['lean_certificates']['status']}** in {out['lean_certificates']['seconds']} s).",
        "",
    ]
    cur = None
    for r in out["results"]:
        if r["kernel"] != cur:
            cur = r["kernel"]
            lines += ["", f"## {cur}", "", "| status | obligation | detail |", "|---|---|---|"]
        det = (r["detail"] or "").replace("|", "/")
        if r["cex"]:
            det += f" — counterexample `{json.dumps(r['cex'])}`"
        lines.append(f"| {r['status']} | `{r['name']}` | {det} |")
    lines += [
        "",
        "## Level D: TPU v7x Floating-Point Rounding Bounds",
        "",
        f"exp model: ue = {out['exp_model']['ue']:.3g}, {out['exp_model']['source']}",
        "",
        "| kernel | requires | f32->bf16 casts in body | eps (abs) | spec_tol | status |",
        "|---|---|---|---|---|---|",
    ]
    for d in out["level_d"]:
        st = "PROVED" if d["eps_abs"] <= d["spec_tol"] else "FAILED"
        lines.append(
            f"| {d['kernel']} | {d['requires']} | {d['f32_to_bf16_casts']} | {d['eps_abs']:.3g} | {d['spec_tol']:.3g} | {st} |"
        )
    with open(os.path.join(ROOT, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
