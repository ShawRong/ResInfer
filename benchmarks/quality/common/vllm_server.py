"""Start the upstream vLLM CLI with the selected ResInfer integration."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from benchmarks.quality.common.runtime import initialize_runtime


def main():
    initialize_runtime()
    from vllm.entrypoints.cli.main import main as serve

    runtime = {"pid": os.getpid(), "python": sys.version.split()[0],
               "versions": {name: getattr(sys.modules[name], "__version__", None)
                            for name in ("torch", "vllm", "flashinfer", "transformers")
                            if name in sys.modules}}
    if os.environ.get("RESIDUE_NVFP4") == "1":
        spec = importlib.util.find_spec("nvfp4._C")
        if spec and spec.origin:
            extension = Path(spec.origin).resolve()
            runtime["extension"] = str(extension)
            runtime["extension_sha256"] = hashlib.sha256(extension.read_bytes()).hexdigest()
    print("[quality:runtime] " + json.dumps(runtime, sort_keys=True), flush=True)
    serve()


if __name__ == "__main__":
    main()
