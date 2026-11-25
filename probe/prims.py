import os, sys, importlib.util, jax, jax.numpy as jnp
from jax._src import core as jc
sys.path.insert(0,"frontend"); from tpuverify.loader import load_file; fa = load_file("fa", f'{os.environ.get("SGLANG_TPU_ROOT", "../sglang-torchtpu")}/sglang_tpu/kernels/flash_attention.py')
S = lambda *s: jax.ShapeDtypeStruct(s, jnp.bfloat16)
fn = lambda q,k,v: fa.flash_attention(q,k,v,causal=True,sm_scale=0.125,block_sizes=fa.BlockSizes(128,256,128,1),vmem_limit_bytes=None)
cj = jax.make_jaxpr(fn)(S(1,2,256,128), S(1,2,512,128), S(1,2,512,128))
def walk(j, depth=0, acc=None):
    for e in j.eqns:
        acc.add(e.primitive.name)
        for k,v in e.params.items():
            if isinstance(v, jc.Jaxpr): walk(v.jaxpr, depth+1, acc)
            elif isinstance(v, jc.Jaxpr): walk(v, depth+1, acc)
            elif isinstance(v, (tuple,list)):
                for x in v:
                    if isinstance(x, jc.Jaxpr): walk(x.jaxpr, depth+1, acc)
                    elif isinstance(x, jc.Jaxpr): walk(x, depth+1, acc)
s=set(); walk(cj.jaxpr, 0, s); print(sorted(s))
def find(j):
    for e in j.eqns:
        if e.primitive.name=="pallas_call": return e
        for v in e.params.values():
            if isinstance(v, jc.Jaxpr):
                r=find(v.jaxpr)
                if r: return r
e=find(cj.jaxpr); print(e.params["compiler_params"]); body=e.params["jaxpr"]
for q in body.eqns: print(q.primitive.name, [str(v.aval) for v in q.invars][:4])
