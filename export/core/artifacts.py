# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""
Artifact management for runner system.

Runners communicate through explicit artifacts:
- calibrate -> writes calib_state + metadata
- export -> reads calib artifacts, writes model + metadata
- eval -> reads exported model, writes results

Each artifact includes full provenance for reproducibility.
"""
import os
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


@dataclass
class DatasetProvenance:
    """Dataset provenance for reproducibility."""
    dataset_ref: str                # e.g., "c4"
    repo: str                       # e.g., "allenai/c4"
    config: Optional[str]           # e.g., "en"
    split: str                      # e.g., "train"
    revision: Optional[str]         # commit hash or None
    seed: int                       # sampling seed
    num_samples: int                # number of samples
    seq_len: int                    # sequence length
    indices_hash: Optional[str]     # hash of sample indices for verification


@dataclass
class EnvironmentProvenance:
    """Environment provenance for reproducibility."""
    python_version: str
    torch_version: str
    transformers_version: str
    datasets_version: str
    modelopt_version: Optional[str]
    git_commit: Optional[str]
    git_dirty: bool
    hostname: str
    cuda_version: Optional[str]


@dataclass
class QuantConfig:
    """Quantization configuration."""
    mode: str                       # residue, w4a4, w4a8, fp16
    fmt: str                        # nvfp4, mxfp4, nvfp4_fp8
    ratio: float = 0.25             # residue ratio
    swap: bool = False              # swap mode
    fake_quant: bool = False        # fake quantization
    block_size: int = 16            # quantization block size
    calib_block_size: int = 8       # calibration block size
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationArtifact:
    """Calibration artifact metadata."""
    artifact_id: str                # unique identifier
    artifact_type: str = "calibration"
    model_id: str = ""              # e.g., "llama3_8b"
    model_path: str = ""            # resolved path

    # Provenance
    dataset: Optional[DatasetProvenance] = None
    environment: Optional[EnvironmentProvenance] = None
    quant_config: Optional[QuantConfig] = None

    # Timestamps
    created_at: str = ""
    duration_seconds: float = 0.0

    # Output files (relative to artifact dir)
    files: Dict[str, str] = field(default_factory=dict)
    # e.g., {"calib_state": "calib_state.json", "scales": "scales.safetensors"}


@dataclass
class ExportArtifact:
    """Export artifact metadata."""
    artifact_id: str
    artifact_type: str = "export"
    model_id: str = ""

    # Input artifact reference
    calib_artifact_id: Optional[str] = None
    calib_artifact_path: Optional[str] = None

    # Export config
    export_fmt: str = "safetensors"  # safetensors, trtllm, onnx
    quant_config: Optional[QuantConfig] = None

    # Provenance
    environment: Optional[EnvironmentProvenance] = None

    # Timestamps
    created_at: str = ""
    duration_seconds: float = 0.0

    # Output files
    files: Dict[str, str] = field(default_factory=dict)
    # e.g., {"model": "model.safetensors", "config": "config.json"}


def get_git_info() -> tuple[Optional[str], bool]:
    """Get current git commit and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()

        # Check if dirty
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        dirty = len(status) > 0

        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


def get_environment_provenance() -> EnvironmentProvenance:
    """Collect environment provenance."""
    import sys
    import socket

    # Get versions
    python_version = sys.version.split()[0]

    try:
        import torch
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except ImportError:
        torch_version = "unknown"
        cuda_version = None

    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = "unknown"

    try:
        import datasets
        datasets_version = datasets.__version__
    except ImportError:
        datasets_version = "unknown"

    try:
        import modelopt
        modelopt_version = modelopt.__version__
    except (ImportError, AttributeError):
        modelopt_version = None

    git_commit, git_dirty = get_git_info()

    return EnvironmentProvenance(
        python_version=python_version,
        torch_version=torch_version,
        transformers_version=transformers_version,
        datasets_version=datasets_version,
        modelopt_version=modelopt_version,
        git_commit=git_commit,
        git_dirty=git_dirty,
        hostname=socket.gethostname(),
        cuda_version=cuda_version,
    )


def compute_indices_hash(indices: List[int]) -> str:
    """Compute hash of sample indices for verification."""
    # Sort and hash to get deterministic result
    sorted_indices = sorted(indices)
    indices_str = ",".join(map(str, sorted_indices))
    return hashlib.sha256(indices_str.encode()).hexdigest()[:16]


def generate_artifact_id(
    model_id: str,
    artifact_type: str,
    **kwargs
) -> str:
    """Generate a human-readable artifact ID."""
    parts = [model_id, artifact_type]

    if "dataset_ref" in kwargs:
        parts.append(kwargs["dataset_ref"])
    if "seed" in kwargs:
        parts.append(f"seed{kwargs['seed']}")
    if "num_samples" in kwargs:
        parts.append(f"n{kwargs['num_samples']}")
    if "quant_mode" in kwargs and kwargs["quant_mode"]:
        parts.append(kwargs["quant_mode"])
    if "ratio" in kwargs and kwargs.get("quant_mode") == "residue":
        parts.append(f"r{kwargs['ratio']}")
    if "swap" in kwargs and kwargs.get("swap"):
        parts.append("swap")

    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts.append(timestamp)

    return "_".join(parts)


class ArtifactManager:
    """
    Manages artifact storage and retrieval.

    Directory structure:
        artifacts/
          llama3_8b/
            calibration/
              c4_seed42_n128_20260125_120000/
                meta.json
                calib_state.json
                scales.safetensors
            export/
              nvfp4_residue_r0.125_swap_20260125_130000/
                meta.json
                model.safetensors
                config.json
    """

    def __init__(self, root: str = "artifacts"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get_artifact_dir(
        self,
        model_id: str,
        artifact_type: str,
        artifact_id: str
    ) -> Path:
        """Get directory for an artifact."""
        path = self.root / model_id / artifact_type / artifact_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_calibration_artifact(
        self,
        artifact: CalibrationArtifact,
        files: Dict[str, Path]
    ) -> Path:
        """
        Save calibration artifact.

        Args:
            artifact: CalibrationArtifact metadata
            files: Dict of {file_key: source_path} to copy into artifact dir

        Returns:
            Path to artifact directory
        """
        import shutil

        artifact_dir = self.get_artifact_dir(
            artifact.model_id,
            "calibration",
            artifact.artifact_id
        )

        # Copy files
        artifact.files = {}
        for key, src_path in files.items():
            dst_name = src_path.name
            dst_path = artifact_dir / dst_name
            shutil.copy2(src_path, dst_path)
            artifact.files[key] = dst_name

        # Save metadata
        meta_path = artifact_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(asdict(artifact), f, indent=2, default=str)

        return artifact_dir

    def save_export_artifact(
        self,
        artifact: ExportArtifact,
        files: Dict[str, Path]
    ) -> Path:
        """Save export artifact."""
        import shutil

        artifact_dir = self.get_artifact_dir(
            artifact.model_id,
            "export",
            artifact.artifact_id
        )

        # Copy files
        artifact.files = {}
        for key, src_path in files.items():
            dst_name = src_path.name
            dst_path = artifact_dir / dst_name
            shutil.copy2(src_path, dst_path)
            artifact.files[key] = dst_name

        # Save metadata
        meta_path = artifact_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(asdict(artifact), f, indent=2, default=str)

        return artifact_dir

    def load_artifact_meta(self, artifact_path: Path) -> Dict[str, Any]:
        """Load artifact metadata."""
        meta_path = artifact_path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Artifact metadata not found: {meta_path}")

        with open(meta_path) as f:
            return json.load(f)

    def find_artifact(
        self,
        model_id: str,
        artifact_type: str,
        artifact_id: Optional[str] = None,
        **match_kwargs
    ) -> Optional[Path]:
        """
        Find an artifact by ID or matching criteria.

        Args:
            model_id: Model ID
            artifact_type: "calibration" or "export"
            artifact_id: Exact artifact ID (if known)
            **match_kwargs: Criteria to match in metadata

        Returns:
            Path to artifact directory, or None if not found
        """
        base_dir = self.root / model_id / artifact_type
        if not base_dir.exists():
            return None

        if artifact_id:
            artifact_path = base_dir / artifact_id
            if artifact_path.exists():
                return artifact_path
            return None

        # Search by criteria
        for artifact_dir in sorted(base_dir.iterdir(), reverse=True):
            if not artifact_dir.is_dir():
                continue

            try:
                meta = self.load_artifact_meta(artifact_dir)

                # Check all match criteria
                matches = True
                for key, value in match_kwargs.items():
                    # Support nested keys like "dataset.seed"
                    parts = key.split(".")
                    obj = meta
                    for part in parts:
                        if isinstance(obj, dict) and part in obj:
                            obj = obj[part]
                        else:
                            matches = False
                            break

                    if matches and obj != value:
                        matches = False

                if matches:
                    return artifact_dir

            except (FileNotFoundError, json.JSONDecodeError):
                continue

        return None

    def list_artifacts(
        self,
        model_id: Optional[str] = None,
        artifact_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all artifacts."""
        results = []

        if model_id:
            model_dirs = [self.root / model_id]
        else:
            model_dirs = [d for d in self.root.iterdir() if d.is_dir()]

        for model_dir in model_dirs:
            if not model_dir.exists():
                continue

            if artifact_type:
                type_dirs = [model_dir / artifact_type]
            else:
                type_dirs = [d for d in model_dir.iterdir() if d.is_dir()]

            for type_dir in type_dirs:
                if not type_dir.exists():
                    continue

                for artifact_dir in type_dir.iterdir():
                    if not artifact_dir.is_dir():
                        continue

                    try:
                        meta = self.load_artifact_meta(artifact_dir)
                        meta["_path"] = str(artifact_dir)
                        results.append(meta)
                    except (FileNotFoundError, json.JSONDecodeError):
                        continue

        return results
