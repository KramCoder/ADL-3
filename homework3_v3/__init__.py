"""Public API for the Homework 3 submission package.

The course grader imports symbols directly from the top-level
`homework3_v3` package (for example ``homework3_v3.BaseLLM`` or
``homework3_v3.data.Dataset``).  The starter code, however, organises the
implementation inside the nested ``homework`` package.  The original
submission therefore exposed no attributes at the expected locations,
causing ``AttributeError`` failures during grading.

To keep the existing module structure intact while matching the public
interface required by the grader, we simply re-export the relevant
classes, loader helpers, and utility submodules from here.
"""

from __future__ import annotations

from importlib import import_module

from .homework.base_llm import BaseLLM

# Re-export frequently used submodules so that the grader can access
# ``homework3_v3.data.Dataset`` and similar lookups.
data = import_module(".homework.data", package=__name__)
cot = import_module(".homework.cot", package=__name__)
sft = import_module(".homework.sft", package=__name__)
rft = import_module(".homework.rft", package=__name__)
datagen = import_module(".homework.datagen", package=__name__)
conversion_utils = import_module(".homework.conversion_utils", package=__name__)

# Convenience loaders required by the grader helpers.  They mirror the
# functions exported from the respective submodules but provide a flat
# namespace at the package root.
load_cot = cot.load
load_sft = sft.load
load_rft = rft.load

__all__ = [
    "BaseLLM",
    "conversion_utils",
    "cot",
    "data",
    "datagen",
    "load_cot",
    "load_rft",
    "load_sft",
    "rft",
    "sft",
]

