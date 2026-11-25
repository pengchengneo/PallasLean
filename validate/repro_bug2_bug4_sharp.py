#!/usr/bin/env python3
"""Targeted high-contrast hardware repro on TPU v7x for:
- Bug 2 (maxtext.megablox_gmm[no_existing_out]): 3 M-tiles (M=384) to hit Pallas 2-slot VMEM buffer reuse!
- Bug 4 (sglang.flash[causal,block_q=129]): peaked diagonal k[128] == q[128] so dropping col 128 gives ~1.0+ abs error!
"""
import importlib.util, json, sys, types
import numpy as np
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl

def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
def stub(n, **a):
    ps = n.split(".")
    for i in range(1, len(ps)+1): sys.modules.setdefault(".".join(ps[:i]), types.ModuleType(".".join(ps[:i])))
    for k, v in a.items(): setattr(sys.modules[n], k, v)

F32_MASK = -0.7 * float(jnp.finfo(jnp.float32).max)
stub("torch", dtype=object, Tensor=object, bfloat16=object, empty_like=lambda x: x)
stub("sglang_tpu.pallas_jax_op", cpu_export_jax_op=lambda name, fn: type("Op", (), {"register_fake": lambda self, f: None, "__call__": staticmethod(fn)})())
stub("maxtext.common.common_types", DEFAULT_MASK_VALUE=F32_MASK)
class DummyQArray: pass
stub("qwix")
stub("qwix.pallas", QArray=DummyQArray, dot_general=jax.lax.dot_general, pallas_call=pl.pallas_call)

# 1) Bug 2: 3 M-tiles (M=384) so m_tile 2 reuses m_tile 0's VMEM double-buffer slot!
mb = load("mb", "/tmp/tpu_logs/tl/src/src/MaxText/kernels/megablox/backend.py")
M, K, N, G = 384, 256, 256, 4
lhs = jnp.ones((M, K), jnp.bfloat16)
rhs = jnp.zeros((G, K, N), jnp.bfloat16).at[0].set(1.0).at[1].set(2.0).at[2].set(3.0)
gs = jnp.array([128, 128, 32, 0], jnp.int32)  # tile0=256, tile1=512, tile2 active[256:288]=768, tail[288:384]=unmasked!
out_no_ex = np.asarray(mb.gmm(lhs, rhs, gs, tiling=(128, 128, 128)), np.float32)
out_ex = np.asarray(mb.gmm(lhs, rhs, gs, existing_out=jnp.zeros((M, N), jnp.float32), tiling=(128, 128, 128)), np.float32)
b2 = {
    "tile0_mean": float(out_no_ex[:128].mean()),
    "tile1_mean": float(out_no_ex[128:256].mean()),
    "tile2_active_mean": float(out_no_ex[256:288].mean()),
    "tile2_unmasked_tail_mean": float(out_no_ex[288:].mean()),
    "tile2_unmasked_tail_max": float(np.abs(out_no_ex[288:]).max()),
    "with_existing_out_tail_max": float(np.abs(out_ex[288:]).max()),
    "vmem_leak_reproduced": bool(np.abs(out_no_ex[288:]).max() > 1.0 and np.abs(out_ex[288:]).max() == 0.0),
}
print("BUG2_3TILES:", json.dumps(b2), flush=True)

# 2) Bug 4: Peaked diagonal at (row=128, col=128) so dropping col 128 causes massive ~2.0 error!
fa = load("fa", "/tmp/tpu_logs/tl/src/sglang_tpu/kernels/flash_attention.py")
rng = np.random.default_rng(0)
B, H, Q, KV, D = 1, 1, 129, 256, 128
q_np = rng.uniform(-0.2, 0.2, (B, H, Q, D)).astype(np.float32)
k_np = rng.uniform(-0.2, 0.2, (B, H, KV, D)).astype(np.float32)
v_np = np.full((B, H, KV, D), -1.0, np.float32)
q_np[0, 0, 128, :] = 1.0
k_np[0, 0, 128, :] = 1.0   # score(128, 128) = 128 * 0.125 = 16.0 -> softmax weight ~ 1.0!
v_np[0, 0, 128, :] = 1.0   # true output for row 128 is +1.0; dropping col 128 gives -1.0!
q, k, v = (jnp.asarray(x, jnp.bfloat16) for x in (q_np, k_np, v_np))
bs = fa.BlockSizes(block_q=129, block_k_major=128, block_k=128, block_b=1)
o = np.asarray(fa.flash_attention(q, k, v, causal=True, sm_scale=0.125, block_sizes=bs, vmem_limit_bytes=None).astype(jnp.float32), np.float64)
b4 = {
    "row128_true_spec_expected": 1.0,
    "row128_tpu_v7x_got_mean": float(o[0, 0, 128].mean()),
    "row128_abs_err": float(np.abs(o[0, 0, 128] - 1.0).max()),
    "diagonal_dropped_reproduced": bool(o[0, 0, 128].mean() < -0.9),
}
print("BUG4_PEAKED:", json.dumps(b4), flush=True)
