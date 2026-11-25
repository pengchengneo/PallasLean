"""Extract the Pallas TPU v7x kernels a function launches into `KernelModel`s.

Works on a CPU host: `tpu_v7x_context()` binds `pltpu.get_tpu_info()` to TPU v7x
(`ChipVersion.TPU_7X`) and `jax.make_jaxpr` traces `pallas_call`, so grid
(including `DynamicGridDim`), BlockSpecs, index maps and the kernel body all
come from the production code path unmodified.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import numpy as np
from jax._src import core as jc
from jax._src.pallas import core as plcore

from .mosaic import tpu_v7x_context


class Block(NamedTuple):
    role: str  # "in" | "out"
    pos: int  # operand position within its role
    name: str
    dims: tuple  # per array dim: ("squeezed", 1) | ("blocked", size)
    array_shape: tuple
    dtype: Any
    index_map: jc.Jaxpr  # (grid indices..., prefetch refs...) -> block indices
    memory_space: str = "vmem"


class KernelModel(NamedTuple):
    name: str
    grid: tuple  # int | "dynamic" per grid dim
    dim_sem: tuple  # "parallel" | "arbitrary" per grid dim
    num_prefetch: int
    prefetch_avals: tuple
    blocks: tuple  # inputs then outputs
    aliases: tuple  # (input operand index incl. prefetch, output index)
    scratch_avals: tuple
    body: jc.Jaxpr  # invars: prefetch, inputs, outputs, scratch refs
    eqn: Any


def _sub_jaxprs(eqn):
    for v in eqn.params.values():
        items = v if isinstance(v, (tuple, list)) else (v,)
        for x in items:
            if isinstance(x, jc.Jaxpr):
                yield x
            elif hasattr(x, "jaxpr") and isinstance(x.jaxpr, jc.Jaxpr):
                yield x.jaxpr


def find_pallas_calls(jaxpr: jc.Jaxpr) -> list:
    found = []
    for e in jaxpr.eqns:
        if e.primitive.name == "pallas_call":
            found.append(e)
        else:
            for sub in _sub_jaxprs(e):
                found.extend(find_pallas_calls(sub))
    return found


def _dims(block_shape) -> tuple:
    out = []
    for d in block_shape:
        if isinstance(d, plcore.Squeezed):
            out.append(("squeezed", 1))
        elif isinstance(d, plcore.Blocked):
            out.append(("blocked", int(d.block_size)))
        elif isinstance(d, plcore.Element):
            pad_hi = int(d.padding[1]) if getattr(d, "padding", None) else 0
            out.append(("element", int(d.block_size), pad_hi))
        else:
            raise NotImplementedError(f"block dim kind {type(d).__name__}")
    return tuple(out)


def _grid_dim(g):
    if isinstance(g, (int, np.integer)):
        return int(g)
    return "dynamic"


def model_from_eqn(eqn, name: str, arg_names=()) -> KernelModel:
    gm = eqn.params["grid_mapping"]
    cp = eqn.params["compiler_params"]
    grid = tuple(_grid_dim(g) for g in gm.grid)
    if cp is not None and getattr(cp, "dimension_semantics", None):
        dim_sem = tuple(str(s).lower().split(".")[-1] for s in cp.dimension_semantics)
    else:
        dim_sem = ("arbitrary",) * len(grid)  # Mosaic default
    nin, nout = gm.num_inputs, gm.num_outputs
    blocks = []
    for k, bm in enumerate(gm.block_mappings):
        role, pos = ("in", k) if k < nin else ("out", k - nin)
        label = arg_names[k] if k < len(arg_names) else f"{role}{pos}"
        mem = str(getattr(bm.block_aval, "memory_space", "vmem")).lower().split(".")[-1]
        blocks.append(
            Block(role, pos, label, _dims(bm.block_shape), tuple(bm.array_aval.shape),
                  bm.array_aval.dtype, bm.index_map_jaxpr, mem)
        )
    body = eqn.params["jaxpr"]
    npf = gm.num_index_operands
    return KernelModel(
        name=name,
        grid=grid,
        dim_sem=dim_sem,
        num_prefetch=npf,
        prefetch_avals=tuple(v.aval for v in body.invars[:npf]),
        blocks=tuple(blocks),
        aliases=tuple(eqn.params["input_output_aliases"]),
        scratch_avals=tuple(gm.scratch_avals),
        body=body,
        eqn=eqn,
    )


def extract(fn, *args, name: str, arg_names=()) -> list[KernelModel]:
    with tpu_v7x_context():
        jaxpr = jax.make_jaxpr(fn)(*args)
    eqns = find_pallas_calls(jaxpr)
    if not eqns:
        raise ValueError(f"{name}: no pallas_call reached")
    return [model_from_eqn(e, f"{name}#{i}" if len(eqns) > 1 else name, arg_names)
            for i, e in enumerate(eqns)]
