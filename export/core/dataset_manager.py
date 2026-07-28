# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""
DatasetManager - 统一数据集管理基础设施

设计原则：
1. Suite只声明语义需求（dataset_ref）
2. DatasetManager统一管理缓存、版本、抽样
3. Runner只消费稳定数据

三锁合一：
- 锁定缓存路径：HF_HOME / HF_DATASETS_CACHE / HF_HUB_CACHE
- 锁定数据集脚本版本：revision=<commit> (锁的是 dataset repo 的 commit，不是 datasets 库版本)
- 锁定抽样结果：seed + n → 确定性 indices

注意：datasets 库版本需要通过 uv.lock / requirements.txt 锁定
"""
import os
import random
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DatasetSpec:
    """数据集规格"""
    ref: str           # 引用名，如 "c4"
    repo: str          # HuggingFace repo，如 "allenai/c4"
    config: str        # 配置名，如 "en"
    split: str         # 分割，如 "train"
    revision: str      # 版本锁定
    description: str   # 描述
    # 新增字段（用于大数据集）
    streaming_default: bool = False  # 是否默认使用 streaming
    text_field: str = "text"         # 文本字段名
    is_large: bool = False           # 是否为大数据集（防呆）


@dataclass
class MaterializeRequest:
    """数据集物化请求"""
    ref: str
    n_samples: int
    seed: int = 42
    max_length: int = 2048
    batch_size: int = 8
    shuffle_buffer: int = 10000      # streaming shuffle buffer; 0 = no shuffle
    tokenizer_name: Optional[str] = None
    mode: str = "tokens"             # "jsonl" or "tokens"


# 全局缓存
_DATASET_REGISTRY: Optional[Dict[str, Any]] = None


def load_dataset_registry(registry_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    加载数据集注册表。

    Args:
        registry_path: 注册表路径，默认 configs/datasets.yaml

    Returns:
        注册表字典
    """
    global _DATASET_REGISTRY

    if _DATASET_REGISTRY is not None:
        return _DATASET_REGISTRY

    if registry_path is None:
        # __file__ is export/core/dataset_manager.py.
        project_root = Path(__file__).parent.parent.parent
        registry_path = project_root / "configs" / "datasets.yaml"

    if not registry_path.exists():
        print(f"Warning: Dataset registry not found at {registry_path}")
        return {"datasets": {}, "cache": {"root": "/tmp/hf_cache"}}

    with open(registry_path) as f:
        registry = yaml.safe_load(f) or {}

    local_path = registry_path.with_name("datasets.local.yaml")
    if local_path.exists():
        with open(local_path) as f:
            local_registry = yaml.safe_load(f) or {}
        registry = {
            **registry,
            **{k: v for k, v in local_registry.items() if k not in ("cache", "datasets")},
            "cache": {
                **(registry.get("cache") or {}),
                **(local_registry.get("cache") or {}),
            },
            "datasets": {
                **(registry.get("datasets") or {}),
                **(local_registry.get("datasets") or {}),
            },
        }

    _DATASET_REGISTRY = registry

    return _DATASET_REGISTRY


class DatasetManager:
    """
    统一数据集管理。

    职责：
    1. 统一设置 HF_HOME / HF_DATASETS_CACHE / HF_HUB_CACHE
    2. 统一 load_dataset 调用（锁定 revision）
    3. 统一可复现抽样（seed + sample indices）

    Example:
        >>> from export.core.dataset_manager import DatasetManager, load_dataset_registry
        >>> cfg = load_dataset_registry()
        >>> dm = DatasetManager(cfg, cache_root="/path/to/datasets")
        >>> ds = dm.get("c4")
        >>> samples = dm.sample(ds, n=128)
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        cache_root: Optional[str] = None,
        seed: int = 42,
    ):
        """
        初始化 DatasetManager。

        Args:
            cfg: 数据集配置（从 datasets.yaml 加载）
            cache_root: 缓存根目录（覆盖配置中的值）
            seed: 随机种子（可复现抽样）
        """
        self.cfg = cfg
        self.seed = seed

        # 确定缓存路径
        if cache_root is None:
            # 优先使用环境变量
            cache_root = os.environ.get(
                "RESIDUE_DATASET_CACHE",
                cfg.get("cache", {}).get("root", "/tmp/hf_cache")
            )

        self.cache_root = Path(cache_root)
        self._setup_cache()
        self._log_versions()

    def _setup_cache(self):
        """设置 HuggingFace 缓存路径。"""
        cache_root = str(self.cache_root)

        # 统一设置所有 HF 缓存路径
        os.environ["HF_HOME"] = cache_root
        os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_root, "datasets")
        os.environ["HF_HUB_CACHE"] = os.path.join(cache_root, "hub")

        # 确保所有目录存在（避免多进程 race condition）
        self.cache_root.mkdir(parents=True, exist_ok=True)
        (self.cache_root / "datasets").mkdir(parents=True, exist_ok=True)
        (self.cache_root / "hub").mkdir(parents=True, exist_ok=True)

    def _log_versions(self):
        """Log datasets library version for reproducibility."""
        try:
            import datasets
            self.datasets_version = datasets.__version__
        except ImportError:
            self.datasets_version = "unknown"

    def get_spec(self, ref: str) -> DatasetSpec:
        """
        获取数据集规格。

        Args:
            ref: 数据集引用名（如 "c4"）

        Returns:
            DatasetSpec 对象

        Raises:
            ValueError: 如果引用不存在
        """
        datasets = self.cfg.get("datasets", {})

        if ref not in datasets:
            raise ValueError(
                f"Dataset ref '{ref}' not found. "
                f"Available: {list(datasets.keys())}"
            )

        d = datasets[ref]
        return DatasetSpec(
            ref=ref,
            repo=d["repo"],
            config=d.get("config"),
            split=d.get("split", "train"),
            revision=d.get("revision"),
            description=d.get("description", ""),
            streaming_default=d.get("streaming_default", False),
            text_field=d.get("text_field", "text"),
            is_large=d.get("is_large", False),
        )

    def get(self, ref: str):
        """
        获取数据集。

        Args:
            ref: 数据集引用名（如 "c4"）

        Returns:
            HuggingFace Dataset 对象

        Note:
            延迟导入 datasets 库，避免启动时加载
        """
        from datasets import load_dataset

        spec = self.get_spec(ref)

        # 构建 load_dataset 参数
        kwargs = {
            "split": spec.split,
            "trust_remote_code": True,
        }

        if spec.config:
            kwargs["name"] = spec.config

        if spec.revision:
            kwargs["revision"] = spec.revision

        return load_dataset(spec.repo, **kwargs)

    def sample(self, ds, n: int, seed: Optional[int] = None):
        """
        可复现抽样。

        同一个 seed + n 永远得到同一组样本。

        Args:
            ds: HuggingFace Dataset 对象
            n: 抽样数量
            seed: 随机种子（默认使用实例 seed）

        Returns:
            抽样后的 Dataset 子集

        Note:
            使用 random.sample(range(total), n) 而不是 list(range(total)) + shuffle，
            因为 C4 等大数据集有几亿行，物化全量 list 会 OOM。
            Python 对 range 的 random.sample 是高效的，不会生成全量列表。
        """
        if seed is None:
            seed = self.seed

        rng = random.Random(seed)
        total = len(ds)

        if n >= total:
            return ds

        # 高效抽样：不物化全量 list，避免 C4 等大数据集 OOM
        indices = rng.sample(range(total), k=n)

        return ds.select(indices)

    def get_sampled(self, ref: str, n: int, seed: Optional[int] = None):
        """
        获取并抽样数据集（便捷方法）。

        Args:
            ref: 数据集引用名
            n: 抽样数量
            seed: 随机种子

        Returns:
            抽样后的 Dataset 子集
        """
        ds = self.get(ref)
        return self.sample(ds, n, seed)

    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存配置信息（用于日志和复现）。"""
        return {
            "HF_HOME": os.environ.get("HF_HOME", ""),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE", ""),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", ""),
            "cache_root": str(self.cache_root),
            "datasets_version": self.datasets_version,
            "seed": self.seed,
        }

    def list_datasets(self) -> list:
        """列出所有可用的数据集引用。"""
        return list(self.cfg.get("datasets", {}).keys())

    # ========== Streaming + Materialize Pattern ==========
    # 避免下载整个大数据集，使用 streaming 抽样后固化到本地文件

    def _dataset_artifact_dir(self, run_dir: Path, ref: str) -> Path:
        """获取数据集工件目录"""
        d = run_dir / "artifacts" / "datasets" / ref
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_manifest(self, path: Path, manifest: Dict[str, Any]) -> None:
        """写入 manifest 文件"""
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def materialize_samples(
        self,
        ref: str,
        n: int,
        *,
        seed: Optional[int] = None,
        run_dir: str | Path,
        shuffle_buffer: int = 10000,
    ) -> Path:
        """
        使用 streaming 模式抽样并固化到 jsonl 文件。

        避免下载整个数据集，只获取需要的样本。

        Args:
            ref: 数据集引用名
            n: 样本数量
            seed: 随机种子
            run_dir: 运行目录
            shuffle_buffer: shuffle buffer 大小（0=不 shuffle）

        Returns:
            jsonl 文件路径
        """
        import json
        from datasets import load_dataset

        if seed is None:
            seed = self.seed
        run_dir = Path(run_dir)
        spec = self.get_spec(ref)

        out_dir = self._dataset_artifact_dir(run_dir, ref)
        out_jsonl = out_dir / f"samples_seed{seed}_n{n}.jsonl"
        manifest_path = out_dir / "manifest.json"

        # 已存在则复用
        if out_jsonl.exists():
            return out_jsonl

        print(f"[DatasetManager] Materializing {n} samples from {ref} (streaming)...")

        # 使用 streaming 模式，避免下载整个数据集
        kwargs = {
            "split": spec.split,
            "streaming": True,
            "trust_remote_code": True,
        }
        if spec.config:
            kwargs["name"] = spec.config
        if spec.revision:
            kwargs["revision"] = spec.revision

        ds = load_dataset(spec.repo, **kwargs)

        # Shuffle（如果需要）
        if shuffle_buffer > 0:
            ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

        text_field = spec.text_field
        it = ds.take(n)

        # 构建 manifest
        meta = {
            "ref": ref,
            "repo": spec.repo,
            "config": spec.config,
            "split": spec.split,
            "revision": spec.revision,
            "datasets_version": self.datasets_version,
            "seed": seed,
            "n": n,
            "streaming": True,
            "shuffle_buffer": shuffle_buffer,
            "text_field": text_field,
            "cache": self.get_cache_info(),
        }

        # 写入 jsonl
        with out_jsonl.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"__meta__": meta}, ensure_ascii=False) + "\n")
            for ex in it:
                f.write(json.dumps({text_field: ex[text_field]}, ensure_ascii=False) + "\n")

        self._write_manifest(manifest_path, meta)
        print(f"[DatasetManager] Materialized to {out_jsonl}")
        return out_jsonl

    def materialize_tokens(
        self,
        req: "MaterializeRequest",
        *,
        run_dir: str | Path,
    ) -> Path:
        """
        将样本 tokenize 并固化到 .pt 文件。

        Args:
            req: MaterializeRequest
            run_dir: 运行目录

        Returns:
            tokens .pt 文件路径
        """
        import json
        import torch

        run_dir = Path(run_dir)
        if req.tokenizer_name is None:
            raise ValueError("tokenizer_name is required for token materialization")

        out_dir = self._dataset_artifact_dir(run_dir, req.ref)
        out_pt = out_dir / f"tokens_seed{req.seed}_n{req.n_samples}_L{req.max_length}.pt"

        # 已存在则复用
        if out_pt.exists():
            return out_pt

        # 确保 samples jsonl 存在
        samples_path = self.materialize_samples(
            req.ref,
            req.n_samples,
            seed=req.seed,
            run_dir=run_dir,
            shuffle_buffer=req.shuffle_buffer,
        )

        print(f"[DatasetManager] Tokenizing {req.n_samples} samples...")

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(req.tokenizer_name, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        # 读取 samples
        texts = []
        with samples_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                if i == 0 and "__meta__" in obj:
                    continue
                # 兼容不同字段名
                if "text" in obj:
                    texts.append(obj["text"])
                else:
                    k = next(iter(obj.keys()))
                    texts.append(obj[k])

        # Tokenize
        enc = tok(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=req.max_length,
        )

        payload = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "meta": {
                "tokenizer_name": req.tokenizer_name,
                "max_length": req.max_length,
                "seed": req.seed,
                "n_samples": req.n_samples,
                "ref": req.ref,
            },
        }
        torch.save(payload, out_pt)
        print(f"[DatasetManager] Tokens saved to {out_pt}")
        return out_pt

    def build_dataloader_from_tokens(
        self,
        tokens_path: str | Path,
        *,
        batch_size: int,
    ):
        """
        从 tokens .pt 文件构建 DataLoader。

        Args:
            tokens_path: tokens 文件路径
            batch_size: batch 大小

        Returns:
            DataLoader
        """
        import torch
        from torch.utils.data import Dataset, DataLoader

        class _TokenTensorDataset(Dataset):
            def __init__(self, data: Dict[str, torch.Tensor]):
                self.input_ids = data["input_ids"]
                self.attention_mask = data["attention_mask"]

            def __len__(self):
                return self.input_ids.size(0)

            def __getitem__(self, idx: int):
                return {
                    "input_ids": self.input_ids[idx],
                    "attention_mask": self.attention_mask[idx],
                }

        payload = torch.load(tokens_path, map_location="cpu", weights_only=True)
        ds = _TokenTensorDataset(payload)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    def get_calib_dataloader(
        self,
        *,
        dataset_ref: str,
        tokenizer_name: str,
        n_samples: int,
        seed: int,
        max_length: int,
        batch_size: int,
        run_dir: str | Path,
        shuffle_buffer: int = 10000,
    ):
        """
        获取 calibration DataLoader（便捷方法）。

        Runner 直接调用此方法，不需要关心底层细节。

        Args:
            dataset_ref: 数据集引用名
            tokenizer_name: tokenizer 名称
            n_samples: 样本数量
            seed: 随机种子
            max_length: 最大序列长度
            batch_size: batch 大小
            run_dir: 运行目录（case_dir）
            shuffle_buffer: streaming shuffle buffer

        Returns:
            DataLoader
        """
        req = MaterializeRequest(
            ref=dataset_ref,
            n_samples=n_samples,
            seed=seed,
            max_length=max_length,
            batch_size=batch_size,
            shuffle_buffer=shuffle_buffer,
            tokenizer_name=tokenizer_name,
            mode="tokens",
        )
        tokens_path = self.materialize_tokens(req, run_dir=run_dir)
        return self.build_dataloader_from_tokens(tokens_path, batch_size=batch_size)


# 便捷函数
def get_dataset_manager(
    cache_root: Optional[str] = None,
    seed: int = 42,
) -> DatasetManager:
    """
    获取 DatasetManager 实例（便捷函数）。

    Args:
        cache_root: 缓存根目录
        seed: 随机种子

    Returns:
        DatasetManager 实例
    """
    cfg = load_dataset_registry()
    return DatasetManager(cfg, cache_root=cache_root, seed=seed)
