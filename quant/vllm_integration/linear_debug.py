"""Optional debug hooks for residue vLLM linear methods."""

from __future__ import annotations

import hashlib
import json
import os

import torch

_RESIDUE_TENSOR_DUMP = os.getenv("RESIDUE_TENSOR_DUMP")
_residue_tap_counter = [0]


def _residue_tap_record(t: torch.Tensor) -> None:
    b = t.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    rec = {
        "idx": _residue_tap_counter[0],
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "sha1": hashlib.sha1(b).hexdigest()[:16],
    }
    _residue_tap_counter[0] += 1
    with open(_RESIDUE_TENSOR_DUMP, "a") as f:
        f.write(json.dumps(rec) + "\n")


if _RESIDUE_TENSOR_DUMP:

    @torch.library.custom_op("residue_dbg::tap", mutates_args=())
    def _residue_tap(x: torch.Tensor) -> torch.Tensor:
        # Skip warmup dummy runs so paired dumps line up on real prompts.
        if x.dim() >= 2 and x.shape[0] <= 400:
            _residue_tap_record(x)
        return x.clone()

    @_residue_tap.register_fake
    def _residue_tap_fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)


def maybe_tap_tensor(x: torch.Tensor) -> torch.Tensor:
    if not _RESIDUE_TENSOR_DUMP:
        return x
    return torch.ops.residue_dbg.tap(x)
