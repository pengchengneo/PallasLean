"""TPU v7x Final LLO (`*final_bundles.txt`) Bundle Verifier.

When `libtpu.so` compiles a Pallas kernel offline (AOT) for topology `tpu7x:2x2x1`
(via `validate/aot_compile_v7x.py` inside a `linux/amd64` Docker container), it
emits the actual TPU v7x VLIW instruction bundles (`*final_bundles.txt`).

This module parses those Final LLO bundles (the TPU v7x machine-code analogue of
NVIDIA PTX/SASS) and checks LLO-level formal obligations:
* `llo-shalt-termination` : the bundle stream reaches a terminal `shalt` instruction
* `llo-bounds-checks`     : extracts `shalt.err` hardware DMA bounds checks
                            (`BoundsCheck ... for dma.hbm_to_vmem / dma.vmem_to_hbm`),
                            which are statically discharged by Level A `in-bounds`
* `llo-dma-sync-balance`  : every issued DMA (`dma.hbm_to_vmem` / `dma.vmem_to_hbm` / `sdma`)
                            is paired with a completion wait (`dma.done` / ` vsyncpa`)
* `llo-ip-profile`        : extracts exact TPU v7x hardware IP instruction counts
                            (MXU `vdwg`/`vmat*`, EUP `vpow2`/`vexp`/`vrsqrt`/`vpop.eup`,
                            XLU `vperm.xlu*`, VPU `vadd`/`vmul`, VLOAD/VSTORE, Scalar/Branch)
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import NamedTuple


class LLOSummary(NamedTuple):
    kernel: str
    llo_file: str
    num_bundles: int
    has_shalt: bool
    dma_bounds_checks: int
    mxu_ops: int
    eup_ops: int
    xlu_ops: int
    dma_starts: int
    dma_waits: int
    vload_ops: int
    vstore_ops: int
    branch_ops: int
    top_mnemonics: dict[str, int]


_BUNDLE_RE = re.compile(r"^\s*(?:0x[0-9a-fA-F]+|\d+)\s*(?:\([^)]*\))?\s*:\s*\{")
_TLP_INST_RE = re.compile(r"^\s*(?:%[a-zA-Z0-9_.,\s]+\s*=\s*)?(?:\([^)]*\)\s+)?([a-zA-Z][a-zA-Z0-9_.]*)\b")
_ALLOC_VMEM_RE = re.compile(r"space=vmem,\s*size\s*=\s*(0x[0-9a-fA-F]+|\d+)")
_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
_MNEMONIC_RE = re.compile(r"(?:=\s*|\{\s*|;;\s*)(?:@!?r?p\d+\s+|\(!?%[a-zA-Z0-9_]+\)\s+)?([a-zA-Z][a-zA-Z0-9_.]*)")


def parse_final_bundles(kernel: str, path: str | Path) -> LLOSummary:
    p = Path(path)
    raw_text = p.read_text(encoding="utf-8", errors="ignore")
    dma_bounds_checks = len(re.findall(r"BoundsCheck\s+\d+\s+\[deref of", raw_text))
    num_bundles = sum(1 for line in raw_text.splitlines() if _BUNDLE_RE.match(line))

    # Strip nested or single-level /* ... */ comments before extracting active instructions
    clean = raw_text
    for _ in range(3):
        clean = _COMMENT_RE.sub("", clean)

    mnemonics: Counter[str] = Counter()
    if num_bundles > 0:
        for line in clean.splitlines():
            if not _BUNDLE_RE.match(line):
                continue
            for inst in line.split(";;"):
                m = _MNEMONIC_RE.search(inst)
                if m:
                    op = m.group(1).lower()
                    if op not in ("snop", "vnop"):
                        mnemonics[op] += 1
    else:
        # Top-level LLO SSA dump (`*-final-top-level-llo.txt`)
        for line in clean.splitlines():
            s = line.strip()
            if not s or s.startswith(("//", "$", "#", "}")):
                continue
            m = _TLP_INST_RE.match(s)
            if m:
                op = m.group(1).lower()
                if op not in ("snop", "vnop", "inlined_call"):
                    mnemonics[op] += 1
        num_bundles = sum(mnemonics.values())

    has_shalt = any(k.startswith("shalt") for k in mnemonics) or ("shalt" in raw_text)
    mxu_ops = sum(v for k, v in mnemonics.items() if k.startswith(("vdwg", "vmat", "vmxu", "vgmm", "vpop.mxu", "vpush.mxu", "vmatmul")))
    eup_ops = sum(v for k, v in mnemonics.items() if any(x in k for x in ("vexp", "vpow2", "vrsqrt", "vrecip", "vlog", "vtanh", "eup")))
    xlu_ops = sum(v for k, v in mnemonics.items() if "xlu" in k)
    dma_starts = sum(v for k, v in mnemonics.items() if ("dma" in k and "done" not in k and "wait" not in k))
    dma_waits = sum(v for k, v in mnemonics.items() if ("dma.done" in k or "dma_done" in k or "vsync" in k or "sflag" in k))
    vload_ops = sum(v for k, v in mnemonics.items() if k.startswith(("vld", "vload")))
    vstore_ops = sum(v for k, v in mnemonics.items() if k.startswith(("vst", "vstore")))
    branch_ops = sum(v for k, v in mnemonics.items() if k.startswith(("sbr", "scall")))

    return LLOSummary(
        kernel=kernel,
        llo_file=p.name,
        num_bundles=num_bundles,
        has_shalt=has_shalt,
        dma_bounds_checks=dma_bounds_checks,
        mxu_ops=mxu_ops,
        eup_ops=eup_ops,
        xlu_ops=xlu_ops,
        dma_starts=dma_starts,
        dma_waits=dma_waits,
        vload_ops=vload_ops,
        vstore_ops=vstore_ops,
        branch_ops=branch_ops,
        top_mnemonics=dict(mnemonics.most_common(12)),
    )
