"""Capture and verify the TPU v7x Mosaic MLIR module of every pallas_call.

Mosaic IR (dialects tpu/vector/arith/math/memref/scf/func) is the TPU v7x
analogue of PTX for TensorLean: the machine-targeted IR right before libtpu's
Mosaic compiler allocates physical VREGs and emits LLO bundles.

Because we target **TPU v7x (`ChipVersion.TPU_7X`)**, `tpu_v7x_context()` binds
JAX's `tpu_info` to TPU v7x hardware parameters (8 sublanes x 128 lanes, dual
256x256 MXUs, 64 MiB VMEM, 1 MiB SMEM) so lowering runs deterministically on any
CPU or TPU host.
"""
from __future__ import annotations

import contextlib
import math
import re
from typing import NamedTuple

import jax
from jax._src import tpu_info as _ti
from jax._src.pallas.mosaic import pallas_call_registration as _reg
from jax._src.pallas.mosaic import tpu_info as _mti
from jax.experimental.pallas import tpu as pltpu

# TPU v7x hardware limits per TensorCore (matches PallasLean/Spec.lean `TPUv7xArch`)
V7X_SUBLANES = 8
V7X_LANES = 128
V7X_MXU_SIZE = 256
V7X_NUM_MXUS = 2
V7X_VMEM_CAPACITY_BYTES = 67_108_864  # 64 MiB
V7X_SMEM_CAPACITY_BYTES = 1_048_576   # 1 MiB

_DTYPE_BYTES = {
    "f32": 4,
    "i32": 4,
    "bf16": 2,
    "f16": 2,
    "i16": 2,
    "i8": 1,
    "f8E4M3FN": 1,
    "f8E5M2": 1,
    "i1": 1,
}


@contextlib.contextmanager
def tpu_v7x_context():
    """Patch JAX / Pallas `tpu_info` to TPU v7x (`TPU7x`, 1 TensorCore per Pallas call)."""
    orig_ti_kind = getattr(_ti, "get_device_kind", None)
    orig_ti_cores = getattr(_ti, "get_num_device_cores", None)
    orig_ti_is_tpu = getattr(_ti, "is_tpu_device", None)
    orig_ti_get = getattr(_ti, "get_tpu_info", None)
    orig_mti_get = getattr(_mti, "get_tpu_info", None)
    orig_pltpu_get = getattr(pltpu, "get_tpu_info", None)

    chip = _ti.chip_version_from_device_kind("TPU7x")
    info = _ti.get_tpu_info_for_chip(chip, 1)

    _ti.get_device_kind = lambda: "TPU7x"
    _ti.get_num_device_cores = lambda: 1
    _ti.is_tpu_device = lambda: True
    _ti.get_tpu_info = lambda: info
    _mti.get_tpu_info = lambda: info
    pltpu.get_tpu_info = lambda: info
    try:
        yield info
    finally:
        if orig_ti_kind is not None:
            _ti.get_device_kind = orig_ti_kind
        if orig_ti_cores is not None:
            _ti.get_num_device_cores = orig_ti_cores
        if orig_ti_is_tpu is not None:
            _ti.is_tpu_device = orig_ti_is_tpu
        if orig_ti_get is not None:
            _ti.get_tpu_info = orig_ti_get
        if orig_mti_get is not None:
            _mti.get_tpu_info = orig_mti_get
        if orig_pltpu_get is not None:
            pltpu.get_tpu_info = orig_pltpu_get


@contextlib.contextmanager
def _capture(sink: list):
    orig = _reg._lower_to_custom_call

    def hook(ctx, *in_nodes, mosaic_module, name, **kw):
        sink.append((name, mosaic_module.operation.get_asm(enable_debug_info=False)))
        return orig(ctx, *in_nodes, mosaic_module=mosaic_module, name=name, **kw)

    _reg._lower_to_custom_call = hook
    try:
        yield
    finally:
        _reg._lower_to_custom_call = orig


_EXPORT_CACHE: dict[tuple, list[tuple[str, str]]] = {}


def export(fn, *args) -> list[tuple[str, str]]:
    """Return `[(kernel_name, TPU v7x Mosaic MLIR text)]` for each `pallas_call` in `fn`."""
    key = (id(fn), tuple((getattr(a, "shape", None), str(getattr(a, "dtype", None))) for a in args))
    if key in _EXPORT_CACHE:
        return _EXPORT_CACHE[key]
    sink: list = []
    with tpu_v7x_context(), _capture(sink):
        jax.jit(lambda *xs: fn(*xs)).trace(*args).lower(lowering_platforms=("tpu",))
    if not sink:
        raise ValueError("no pallas_call reached the TPU v7x Mosaic lowering")
    _EXPORT_CACHE[key] = sink
    return sink


class MosaicV7xSummary(NamedTuple):
    kernel_name: str
    mlir_text: str
    vmem_bytes: int
    smem_bytes: int
    fits_v7x_vmem: bool
    fits_v7x_smem: bool
    op_counts: dict[str, int]
    transform_funcs: int
    f32_to_bf16_truncs: int


_MEMREF_RE = re.compile(
    r"memref<([^>]+),\s*#tpu\.memory_space<(vmem|smem)>>"
)


def _memref_bytes(spec_str: str) -> int:
    parts = spec_str.strip().split("x")
    dt = parts[-1]
    dims = [int(d) for d in parts[:-1] if d.isdigit()]
    elems = math.prod(dims) if dims else 1
    return elems * _DTYPE_BYTES.get(dt, 4)


def analyze_v7x_mosaic(kernel_name: str, mlir_text: str, num_blocks: int, num_scratch: int) -> MosaicV7xSummary:
    """Analyze a lowered TPU v7x Mosaic MLIR module for memory footprint & instruction profile."""
    # Find the main kernel function signature (the func.func with window_params / tpu.core_type)
    main_header = mlir_text
    for line in mlir_text.splitlines():
        if "func.func @" in line and not re.search(r"func\.func @transform_\d+", line):
            main_header = line
            break
    sync_count = main_header.count("#tpu.pipeline_mode<synchronous>")

    vmem_args = []
    smem_args = []
    for m in _MEMREF_RE.finditer(main_header):
        spec, space = m.group(1), m.group(2)
        b = _memref_bytes(spec)
        if space == "vmem":
            vmem_args.append(b)
        else:
            smem_args.append(b)

    # In TPU v7x Mosaic pipeline, pipelined BlockSpec VMEM operands are double-buffered (2x),
    # while synchronous BlockSpecs and scratch VMEM operands are single-buffered (1x).
    pipelined_vmem = vmem_args[: max(0, len(vmem_args) - num_scratch)]
    scratch_vmem = vmem_args[max(0, len(vmem_args) - num_scratch) :]
    # Double-buffer all pipelined VMEM refs except synchronous ones
    vmem_bytes = sum(2 * b for b in pipelined_vmem) + sum(scratch_vmem)
    if sync_count > 0 and pipelined_vmem:
        # Subtract 1x for the smallest `sync_count` pipelined VMEM refs (conservative or exact)
        for b in sorted(pipelined_vmem)[:sync_count]:
            vmem_bytes -= b
    smem_bytes = sum(smem_args)

    ops_of_interest = (
        "tpu.matmul",
        "vector.contract",
        "math.exp",
        "math.rsqrt",
        "arith.extf",
        "arith.truncf",
        "vector.load",
        "vector.store",
        "tpu.enqueue_dma",
        "tpu.wait_dma",
        "scf.for",
        "scf.if",
    )
    op_counts = {op: mlir_text.count(op) for op in ops_of_interest if mlir_text.count(op) > 0}
    transform_funcs = len(re.findall(r"func\.func @transform_\d+", mlir_text))
    truncs = len(re.findall(r"arith\.truncf\b.*?\bf32\s+to\s+vector<[^>]+xbf16>", mlir_text))

    return MosaicV7xSummary(
        kernel_name=kernel_name,
        mlir_text=mlir_text,
        vmem_bytes=vmem_bytes,
        smem_bytes=smem_bytes,
        fits_v7x_vmem=(vmem_bytes <= V7X_VMEM_CAPACITY_BYTES),
        fits_v7x_smem=(smem_bytes <= V7X_SMEM_CAPACITY_BYTES),
        op_counts=op_counts,
        transform_funcs=transform_funcs,
        f32_to_bf16_truncs=truncs,
    )
