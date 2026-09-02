# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""
Base classes for pipeline stages.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging


@dataclass
class StageContext:
    """
    Context passed between pipeline stages.

    This holds all state during pipeline execution:
    - refs: In-memory objects (model, tokenizer, etc.)
    - artifacts: Paths to artifacts (calib, export, logs)
    - metrics: Aggregated metrics for case.json
    - meta: Metadata (quant_cfg, dataset provenance, etc.)
    - axis/variant: For case.json output
    - logs: Log file paths
    """

    # In-memory references (not serialized)
    refs: Dict[str, Any] = field(default_factory=dict)

    # Artifact paths (relative to case_dir or absolute)
    artifacts: Dict[str, Path] = field(default_factory=dict)

    # Metrics for case.json
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Metadata (not in case.json, but useful for stages)
    meta: Dict[str, Any] = field(default_factory=dict)

    # For case.json output
    axis: Dict[str, str] = field(default_factory=dict)
    variant: Dict[str, Any] = field(default_factory=dict)

    # Log paths
    logs: List[Path] = field(default_factory=list)

    # Case directory (for resolving relative paths)
    case_dir: Optional[Path] = None

    def get_ref(self, key: str, default: Any = None) -> Any:
        """Get an in-memory reference."""
        return self.refs.get(key, default)

    def set_ref(self, key: str, value: Any):
        """Set an in-memory reference."""
        self.refs[key] = value

    def get_artifact(self, key: str, default: Optional[Path] = None) -> Optional[Path]:
        """Get an artifact path."""
        return self.artifacts.get(key, default)

    def set_artifact(self, key: str, path: Path):
        """Set an artifact path."""
        self.artifacts[key] = Path(path)

    def get_meta(self, key: str, default: Any = None) -> Any:
        """Get metadata."""
        return self.meta.get(key, default)

    def set_meta(self, key: str, value: Any):
        """Set metadata."""
        self.meta[key] = value

    def add_log(self, log_path: Path):
        """Add a log file path."""
        self.logs.append(Path(log_path))

    def resolve_path(self, path: Path) -> Path:
        """
        Resolve a path relative to case_dir if not absolute.

        Args:
            path: Path to resolve

        Returns:
            Absolute path
        """
        path = Path(path)
        if path.is_absolute():
            return path
        if self.case_dir:
            return self.case_dir / path
        return path


class Stage(ABC):
    """
    Base class for pipeline stages.

    Stages are reusable processing units that:
    1. Take a StageContext and params dict
    2. Perform some operation (load model, calibrate, export, eval, etc.)
    3. Update the context with results
    4. Return the updated context

    Stages should be stateless - all state goes in the context.
    Stages should not manage retries/errors - the runner handles that.
    Stages should not write case.json - the runner handles that.
    """

    def __init__(self, name: str = ""):
        """
        Initialize stage.

        Args:
            name: Stage name (for logging)
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"stage.{self.name}")

    @abstractmethod
    def run(self, ctx: StageContext, params: Dict[str, Any]) -> StageContext:
        """
        Execute the stage.

        Args:
            ctx: Stage context with refs/artifacts/metrics/etc.
            params: Stage-specific parameters from YAML

        Returns:
            Updated context

        Raises:
            Exception: On failure (runner will handle retry/logging)
        """
        raise NotImplementedError

    def validate_inputs(self, ctx: StageContext, params: Dict[str, Any]):
        """
        Validate required inputs before running.

        Override this to check ctx.refs/artifacts/params before running.
        Raise ValueError with clear message if inputs are invalid.

        Args:
            ctx: Stage context
            params: Stage parameters

        Raises:
            ValueError: If required inputs are missing or invalid
        """
        pass

    def __call__(self, ctx: StageContext, params: Dict[str, Any]) -> StageContext:
        """
        Execute the stage with validation.

        Args:
            ctx: Stage context
            params: Stage parameters

        Returns:
            Updated context
        """
        self.logger.info(f"Running stage: {self.name}")
        self.validate_inputs(ctx, params)
        result = self.run(ctx, params)
        self.logger.info(f"Stage {self.name} completed successfully")
        return result


class ConditionalStage(Stage):
    """
    Stage with built-in condition checking.

    Useful for stages that should only run under certain conditions.
    """

    def __init__(self, name: str = "", condition: Optional[str] = None):
        """
        Initialize conditional stage.

        Args:
            name: Stage name
            condition: Condition expression (e.g., "${needs_calib}")
        """
        super().__init__(name)
        self.condition = condition

    def should_run(self, ctx: StageContext, params: Dict[str, Any]) -> bool:
        """
        Check if stage should run.

        Args:
            ctx: Stage context
            params: Stage parameters

        Returns:
            True if stage should run, False to skip
        """
        if not self.condition:
            return True

        # Simple condition evaluation
        # TODO: Implement full expression parser for complex conditions
        condition = self.condition.strip()

        # Handle ${param} syntax
        if condition.startswith("${") and condition.endswith("}"):
            param_name = condition[2:-1].strip()
            value = params.get(param_name)

            # Truthy check
            if value is None or value is False or value == "":
                return False
            return True

        # Handle ${param} != null syntax
        if "!=" in condition and "null" in condition:
            param_name = condition.split("!=")[0].strip()
            if param_name.startswith("${") and param_name.endswith("}"):
                param_name = param_name[2:-1].strip()
            value = params.get(param_name)
            return value is not None

        # Default: always run
        return True

    def __call__(self, ctx: StageContext, params: Dict[str, Any]) -> StageContext:
        """
        Execute stage if condition is met.

        Args:
            ctx: Stage context
            params: Stage parameters

        Returns:
            Updated context (or unchanged if skipped)
        """
        if not self.should_run(ctx, params):
            self.logger.info(f"Skipping stage {self.name} (condition not met: {self.condition})")
            return ctx

        return super().__call__(ctx, params)
