import os, sys, types, importlib.util, jax, jax.numpy as jnp
from jax._src.pallas import core as plcore
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
which = sys.argv[1]
if which == "flash":
    fa = load("fa", f'{os.environ.get("SGLANG_TPU_ROOT", "../sglang-torchtpu")}/sglang_tpu/kernels/flash_attention.py')
    S = lambda *s: jax.ShapeDtypeStruct(s, jnp.bfloat16)
    fn = lambda q,k,v: fa.flash_attention(q,k,v,causal=True,sm_scale=0.125,block_sizes=fa.BlockSizes(128,256,128,1),vmem_limit_bytes=None)
    args = (S(1,2,256,128), S(1,2,512,128), S(1,2,512,128))
else:
    sys.modules["maxtext"]=types.ModuleType("maxtext"); m2=types.ModuleType("maxtext.common"); m3=types.ModuleType("maxtext.common.common_types"); m3.DEFAULT_MASK_VALUE=-0.7*float(jnp.finfo(jnp.float32).max)
    sys.modules["maxtext.common"]=m2; sys.modules["maxtext.common.common_types"]=m3
    ra = load("ra", f'{os.environ.get("MAXTEXT_ROOT", "../maxtext")}/src/MaxText/kernels/attention/ragged_attention.py')
    fn = lambda q,k,v,l: ra.ragged_mqa(q,k,v,l,block_size=256)
    args = (jax.ShapeDtypeStruct((4,8,128),jnp.float32), jax.ShapeDtypeStruct((4,1024,128),jnp.bfloat16), jax.ShapeDtypeStruct((4,1024,128),jnp.bfloat16), jax.ShapeDtypeStruct((4,),jnp.int32))
cj = jax.make_jaxpr(fn)(*args)
def find(j):
    for e in j.eqns:
        if e.primitive.name == "pallas_call": return e
        for sub in jax.core.jaxprs_in_params(e.params) if hasattr(jax.core,'jaxprs_in_params') else []:
            r = find(sub);
            if r: return r
        for v in e.params.values():
            if isinstance(v, jax.extend.core.ClosedJaxpr): r=find(v.jaxpr); 
            elif isinstance(v, jax.extend.core.Jaxpr): r=find(v)
            else: continue
            if r: return r
e = find(cj.jaxpr)
gm = e.params["grid_mapping"]
print("grid", gm.grid, "nsp", gm.num_index_operands, "scratch", gm.num_scratch_operands)
print("compiler_params", e.params["compiler_params"])
for bm in gm.block_mappings:
    print(bm.block_shape, bm.array_aval if hasattr(bm,'array_aval') else bm.array_shape_dtype, bm.index_map_jaxpr)
print(e.params["jaxpr"])
