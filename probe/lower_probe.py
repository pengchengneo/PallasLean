import os, sys, types, importlib.util, jax, jax.numpy as jnp
sys.modules.setdefault("torch", types.ModuleType("torch"))
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
kv = load("kvw", f'{os.environ.get("SGLANG_TPU_ROOT", "../sglang-torchtpu")}/sglang_tpu/kv_page_write.py')
ids = jax.ShapeDtypeStruct((4,), jnp.int32)
pages = jax.ShapeDtypeStruct((4, 16, 8, 128), jnp.bfloat16)
cache = jax.ShapeDtypeStruct((64, 16, 8, 128), jnp.bfloat16)
f = jax.jit(kv.write_pages)
print(jax.make_jaxpr(kv.write_pages)(ids, pages, cache))
low = f.trace(ids, pages, cache).lower(lowering_platforms=("tpu",))
txt = low.as_text()
print(len(txt)); print("tpu_custom_call" in txt)
