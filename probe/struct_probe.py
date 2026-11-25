import os, sys, types, importlib.util, jax, jax.numpy as jnp
sys.modules.setdefault("torch", types.ModuleType("torch"))
spec = importlib.util.spec_from_file_location("kvw", f'{os.environ.get("SGLANG_TPU_ROOT", "../sglang-torchtpu")}/sglang_tpu/kv_page_write.py'); kv = importlib.util.module_from_spec(spec); spec.loader.exec_module(kv)
cj = jax.make_jaxpr(kv.write_pages)(jax.ShapeDtypeStruct((4,),jnp.int32), jax.ShapeDtypeStruct((4,16,8,128),jnp.bfloat16), jax.ShapeDtypeStruct((64,16,8,128),jnp.bfloat16))
e = cj.jaxpr.eqns[0]
gm = e.params["grid_mapping"]
print([a for a in dir(gm) if not a.startswith("_")])
bm = gm.block_mappings[1]
print([a for a in dir(bm) if not a.startswith("_")])
print(type(bm.block_shape[0]), type(bm.block_shape[1]), bm.block_shape[1].block_size)
print(bm.array_aval if hasattr(bm,"array_aval") else None)
im = bm.index_map_jaxpr
for q in im.jaxpr.eqns: print(q.primitive.name, q.params.keys(), [str(v.aval) for v in q.invars])
print(gm.num_inputs, gm.num_outputs, gm.num_index_operands, gm.num_scratch_operands, gm.grid)
print(e.params["compiler_params"], e.params["input_output_aliases"])
body = e.params["jaxpr"]
for q in body.eqns: print(q.primitive.name, list(q.params.keys()), [ (type(v).__name__, str(v.aval)) for v in q.invars], [str(v.aval) for v in q.outvars])
