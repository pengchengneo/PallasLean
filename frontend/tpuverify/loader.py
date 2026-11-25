"""Load a production kernel file by path without importing its package.

Kernel files are loaded from their checkout, unmodified; only heavy unrelated
package imports (torch, maxtext.common) are stubbed so the module body runs.
"""
from __future__ import annotations

import importlib.util
import sys
import types


def stub_module(name: str, **attrs) -> None:
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        sub = ".".join(parts[:i])
        if sub not in sys.modules:
            sys.modules[sub] = types.ModuleType(sub)
    for k, v in attrs.items():
        setattr(sys.modules[name], k, v)


def load_file(name: str, path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
