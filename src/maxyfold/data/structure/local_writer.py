from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from maxyfold.data.structure.cif_writer import CIFWriter
from maxyfold.data.structure.records import StructureRecord


def _json_safe(value: Any) -> Any:
    """Convert common numeric/container types into JSON-compatible values."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()

        if value.numel() == 1:
            return value.item()

        return value.tolist()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    return value


def _safe_name(value: str) -> str:
    """Make a value safe to use as part of a directory name."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value or "unknown"


class LocalStructureWriter:
    """Write prediction bundles inside a Hydra run directory."""

    def __init__(
        self,
        run_dir: str | Path,
        cif_writer: CIFWriter | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.structure_dir = self.run_dir / "structures"
        self.manifest_path = self.structure_dir / "manifest.jsonl"
        self.cif_writer = cif_writer or CIFWriter()

    def write(
        self,
        *,
        record: StructureRecord,
        metrics: dict[str, Any],
        split: str,
        epoch: int,
        global_step: int,
        batch_idx: int,
        sample_idx: int,
    ) -> dict[str, Path]:
        """Write one prediction CIF, its metrics, and a manifest entry."""
        split = _safe_name(split)
        pdb_id = _safe_name(record.pdb_id.upper())

        if split == "test":
            split_dir = self.structure_dir / split
        else:
            split_dir = self.structure_dir / split / f"epoch_{epoch:04d}"

        sample_name = f"{pdb_id}_b{batch_idx:04d}_s{sample_idx:02d}"
        bundle_dir = split_dir / sample_name
        bundle_dir.mkdir(parents=True, exist_ok=True)

        cif_path = bundle_dir / "pred.cif"
        metrics_path = bundle_dir / "metrics.json"

        self.cif_writer.write(record, cif_path)

        context = {
            "pdb_id": record.pdb_id.upper(),
            "split": split,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "batch_idx": int(batch_idx),
            "sample_idx": int(sample_idx),
        }

        metrics_payload = {
            **context,
            "metrics": _json_safe(metrics),
        }

        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics_payload, file, indent=2, sort_keys=True)
            file.write("\n")

        manifest_entry = {
            **context,
            "cif": cif_path.relative_to(self.run_dir).as_posix(),
            "metrics_file": metrics_path.relative_to(self.run_dir).as_posix(),
            "metrics": _json_safe(metrics),
        }

        self.structure_dir.mkdir(parents=True, exist_ok=True)

        # The future callback must call this only on global rank zero.
        with self.manifest_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(manifest_entry, sort_keys=True))
            file.write("\n")

        return {
            "bundle_dir": bundle_dir,
            "cif_path": cif_path,
            "metrics_path": metrics_path,
            "manifest_path": self.manifest_path,
        }