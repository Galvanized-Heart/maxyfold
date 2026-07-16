from __future__ import annotations

import argparse
import html
import json
import sys
import webbrowser
from collections import Counter
from pathlib import Path
from typing import Any

import biotite
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import torch

from maxyfold.data.constants import LIGAND_IDX, elements, restypes
from maxyfold.data.cropping.croppers import ContiguousCropper, SpatialCropper
from maxyfold.data.datasets.pdb_dataset import PDBDataset
from maxyfold.data.storage.lmdb import LMDBBackend
from maxyfold.data.structure import CIFWriter, StructureRecord

"""
uv run --with py3Dmol python \
  src/maxyfold/data/analysis/debug_lmdb_to_cif.py \
  --lmdb-path data/pdb/processed/pdb_dataset.lmdb \
  --pdb-id 100D \
  --cropper contiguous \
  --crop-size 20 \
  --seed 0
"""


COORDINATE_TOLERANCE = 1e-3


def to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert tensors or arrays into NumPy arrays."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def decode_residue(residue_id: int) -> str:
    """Decode a MaxyFold residue token."""
    residue_id = int(residue_id)

    if 0 <= residue_id < len(restypes):
        return str(restypes[residue_id])

    return f"INVALID_{residue_id}"


def decode_element(element_id: int) -> str:
    """Decode a MaxyFold element token."""
    element_id = int(element_id)

    if 0 <= element_id < len(elements):
        return str(elements[element_id]).upper()

    return "X"


def summarize_data(data: dict[str, Any], label: str) -> dict[str, Any]:
    """Return useful diagnostics for raw or cropped structure data."""
    coords = to_numpy(data["coords"])
    mask = to_numpy(data["mask"]).astype(bool)
    res_type = to_numpy(data["res_type"]).astype(np.int64)
    atom_elements = to_numpy(data["atom_elements"]).astype(np.int64)
    chain_ids = to_numpy(data["chain_ids"]).astype(np.int64)

    valid_tokens = mask.any(axis=-1)
    valid_coords = coords[mask]
    valid_element_ids = atom_elements[mask]
    ligand_tokens = res_type == LIGAND_IDX

    residue_counts = Counter(
        decode_residue(token_id)
        for token_id in res_type[valid_tokens]
    )
    element_counts = Counter(
        decode_element(element_id)
        for element_id in valid_element_ids
    )
    chain_token_counts = Counter(
        int(chain_id)
        for chain_id in chain_ids[valid_tokens]
        if chain_id >= 0
    )

    if valid_coords.size > 0:
        coord_min = valid_coords.min(axis=0)
        coord_max = valid_coords.max(axis=0)
        coord_span = coord_max - coord_min
    else:
        coord_min = np.zeros(3)
        coord_max = np.zeros(3)
        coord_span = np.zeros(3)

    return {
        "label": label,
        "pdb_id": str(data["pdb_id"]),
        "coords_shape": list(coords.shape),
        "mask_shape": list(mask.shape),
        "num_tokens": int(len(res_type)),
        "num_valid_tokens": int(valid_tokens.sum()),
        "num_padded_or_empty_tokens": int((~valid_tokens).sum()),
        "num_valid_atoms": int(mask.sum()),
        "num_chains": int(len(set(chain_ids[valid_tokens].tolist()))),
        "chain_token_counts": dict(sorted(chain_token_counts.items())),
        "num_ligand_tokens": int(ligand_tokens.sum()),
        "num_resolved_ligand_atoms": int(
            (mask & ligand_tokens[:, None]).sum()
        ),
        "num_unknown_valid_elements": int((valid_element_ids == 0).sum()),
        "residue_counts": dict(residue_counts.most_common()),
        "element_counts": dict(element_counts.most_common()),
        "coordinate_min": coord_min.round(3).tolist(),
        "coordinate_max": coord_max.round(3).tolist(),
        "coordinate_span": coord_span.round(3).tolist(),
        "all_valid_coordinates_finite": bool(np.isfinite(valid_coords).all()),
    }


def print_section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2))


def create_record(
    data: dict[str, Any],
    metadata: dict[str, Any],
) -> StructureRecord:
    """Create a structure record from raw arrays or dataset tensors."""
    return StructureRecord(
    pdb_id=str(data["pdb_id"]),
    coords=data["coords"],
    mask=data["mask"],
    res_type=data["res_type"],
    atom_elements=data["atom_elements"],
    chain_ids=data["chain_ids"],
    metadata=metadata,
)


def read_cif(path: Path) -> struc.AtomArray:
    """Read the generated CIF back through Biotite."""
    cif_file = pdbx.CIFFile.read(str(path))
    return pdbx.get_structure(
        cif_file,
        model=1,
        use_author_fields=False,
    )


def secondary_structure_summary(
    atom_array: struc.AtomArray,
) -> dict[str, Any]:
    """Estimate secondary structure per polymer chain with Biotite.

    This is not used as a correctness assertion. It tells us whether the
    reconstructed backbone can be interpreted as protein-like geometry.
    """
    polymer = atom_array[~atom_array.hetero]

    if len(polymer) == 0:
        return {"status": "no polymer atoms"}

    total_counts: Counter[str] = Counter()
    chain_results: dict[str, Any] = {}

    for chain_id in np.unique(polymer.chain_id):
        chain = polymer[polymer.chain_id == chain_id]

        try:
            annotation = struc.annotate_sse(chain)
            counts = Counter(annotation.tolist())

            readable_counts = {
                "helix": int(counts.get("a", 0)),
                "sheet": int(counts.get("b", 0)),
                "coil": int(counts.get("c", 0)),
                "unassigned": int(counts.get("", 0)),
            }
            chain_results[str(chain_id)] = readable_counts
            total_counts.update(annotation.tolist())

        except Exception as error:
            chain_results[str(chain_id)] = {
                "error": f"{type(error).__name__}: {error}"
            }

    return {
        "per_chain": chain_results,
        "total": {
            "helix": int(total_counts.get("a", 0)),
            "sheet": int(total_counts.get("b", 0)),
            "coil": int(total_counts.get("c", 0)),
            "unassigned": int(total_counts.get("", 0)),
        },
    }


def validate_roundtrip(
    source_data: dict[str, Any],
    cif_path: Path,
    label: str,
) -> tuple[dict[str, Any], bool]:
    """Compare CIF contents against the tensor representation."""
    coords = to_numpy(source_data["coords"])
    mask = to_numpy(source_data["mask"]).astype(bool)
    res_type = to_numpy(source_data["res_type"]).astype(np.int64)
    atom_elements = to_numpy(source_data["atom_elements"]).astype(np.int64)
    chain_ids = to_numpy(source_data["chain_ids"]).astype(np.int64)

    # Match CIFWriter behaviour: invalid/padded chains must not be emitted.
    write_mask = mask & (chain_ids[:, None] >= 0)

    expected_coords = coords[write_mask]
    expected_elements = np.asarray(
        [decode_element(index) for index in atom_elements[write_mask]],
        dtype=str,
    )
    expected_hetero_count = int(
        (write_mask & (res_type[:, None] == LIGAND_IDX)).sum()
    )

    atom_array = read_cif(cif_path)
    actual_coords = atom_array.coord
    actual_elements = np.char.upper(atom_array.element.astype(str))

    atom_count_matches = len(atom_array) == len(expected_coords)

    if atom_count_matches and len(actual_coords) > 0:
        coordinate_errors = np.abs(actual_coords - expected_coords)
        max_coordinate_error = float(coordinate_errors.max())
        mean_coordinate_error = float(coordinate_errors.mean())
    elif atom_count_matches:
        max_coordinate_error = 0.0
        mean_coordinate_error = 0.0
    else:
        max_coordinate_error = float("inf")
        mean_coordinate_error = float("inf")

    elements_match = (
        atom_count_matches
        and np.array_equal(actual_elements, expected_elements)
    )
    hetero_count_matches = (
        int(atom_array.hetero.sum()) == expected_hetero_count
    )
    coordinates_match = (
        atom_count_matches
        and max_coordinate_error <= COORDINATE_TOLERANCE
    )

    element_mismatches: list[dict[str, Any]] = []

    if atom_count_matches and not elements_match:
        mismatch_indices = np.flatnonzero(
            actual_elements != expected_elements
        )

        for index in mismatch_indices[:10]:
            element_mismatches.append(
                {
                    "atom_index": int(index),
                    "expected": str(expected_elements[index]),
                    "actual": str(actual_elements[index]),
                }
            )

    diagnostics = {
        "label": label,
        "cif_path": str(cif_path.resolve()),
        "expected_atom_count": int(len(expected_coords)),
        "written_atom_count": int(len(atom_array)),
        "atom_count_matches": atom_count_matches,
        "max_coordinate_roundtrip_error": max_coordinate_error,
        "mean_coordinate_roundtrip_error": mean_coordinate_error,
        "coordinate_tolerance": COORDINATE_TOLERANCE,
        "coordinates_match": coordinates_match,
        "elements_match": elements_match,
        "element_mismatches_first_10": element_mismatches,
        "expected_hetero_atom_count": expected_hetero_count,
        "written_hetero_atom_count": int(atom_array.hetero.sum()),
        "hetero_count_matches": hetero_count_matches,
        "written_chains": sorted(
            np.unique(atom_array.chain_id).astype(str).tolist()
        ),
        "written_residue_names": dict(
            Counter(atom_array.res_name.astype(str).tolist()).most_common()
        ),
        "secondary_structure": secondary_structure_summary(atom_array),
    }

    passed = (
        atom_count_matches
        and coordinates_match
        and elements_match
        and hetero_count_matches
    )

    diagnostics["roundtrip_passed"] = passed

    return diagnostics, passed


def create_py3dmol_html(
    cif_path: Path,
    html_path: Path,
    title: str,
    diagnostics: dict[str, Any],
) -> None:
    """Create a standalone interactive py3Dmol HTML page."""
    try:
        import py3Dmol
    except ImportError as error:
        raise RuntimeError(
            "py3Dmol is not installed. Run this script with "
            "`uv run --with py3Dmol python ...`."
        ) from error

    cif_text = cif_path.read_text(encoding="utf-8")

    viewer = py3Dmol.view(width=1100, height=720)
    viewer.addModel(cif_text, "cif")
    viewer.setBackgroundColor("white")

    # Cartoon gives intuitive secondary structure. A line overlay provides a
    # fallback when secondary-structure inference is incomplete.
    viewer.setStyle(
        {"hetflag": False},
        {
            "cartoon": {"colorscheme": "chain"},
            "line": {},
        },
    )

    # Show ligand atoms even when inferred bonds are incomplete.
    viewer.setStyle(
        {"hetflag": True},
        {
            "stick": {"radius": 0.18},
            "sphere": {"scale": 0.30},
        },
    )

    viewer.zoomTo()
    viewer.render()

    # py3Dmol primarily documents notebook use. _make_html() lets this
    # diagnostic script emit a browser-viewable file without Jupyter.
    viewer_html = viewer._make_html()

    page = f"""<!doctype html>
<html lang="en">
  <div class="viewer">
    {viewer_html}
  </div>
  <h2>Diagnostics</h2>
  <pre>{html.escape(json.dumps(diagnostics, indent=2))}</pre>
</body>
</html>
"""

    html_path.write_text(page, encoding="utf-8")


def get_cropper(name: str, crop_size: int):
    if name == "contiguous":
        return ContiguousCropper(crop_size=crop_size)

    if name == "spatial":
        return SpatialCropper(crop_size=crop_size)

    if name == "none":
        return None

    raise ValueError(f"Unsupported cropper: {name}")


def resolve_sample_index(
    backend: LMDBBackend,
    *,
    index: int | None,
    pdb_id: str | None,
    find_ligand: bool,
    scan_limit: int,
) -> int:
    """Resolve a sample by numeric index, PDB ID, or ligand presence."""
    keys = backend.get_keys()

    if not keys:
        raise RuntimeError("The LMDB contains no keys.")

    if pdb_id is not None:
        target = pdb_id.upper()

        try:
            return keys.index(target)
        except ValueError as error:
            raise KeyError(f"PDB ID {target!r} was not found in the LMDB.") from error

    if find_ligand:
        limit = min(scan_limit, len(keys))

        print(f"Searching the first {limit} structures for a ligand...")

        for candidate_index in range(limit):
            data = backend.get_raw_data(candidate_index)

            if np.any(data["res_type"] == LIGAND_IDX):
                print(
                    f"Found ligand-containing sample at index "
                    f"{candidate_index}: {data['pdb_id']}"
                )
                return candidate_index

        raise RuntimeError(
            f"No ligand-containing entry was found in the first {limit} samples."
        )

    resolved_index = 0 if index is None else index

    if not 0 <= resolved_index < len(keys):
        raise IndexError(
            f"Index {resolved_index} is outside LMDB range [0, {len(keys) - 1}]."
        )

    return resolved_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise MaxyFold's LMDB -> cropper/dataset -> CIF -> "
            "Biotite -> py3Dmol path."
        )
    )

    parser.add_argument(
        "--lmdb-path",
        type=Path,
        default=Path("data/pdb/processed/pdb_dataset.lmdb"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/debug_lmdb_to_cif"),
    )

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--index", type=int)
    selection.add_argument("--pdb-id", type=str)
    selection.add_argument("--find-ligand", action="store_true")

    parser.add_argument("--scan-limit", type=int, default=500)
    parser.add_argument(
        "--cropper",
        choices=("none", "contiguous", "spatial"),
        default="contiguous",
    )
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-html", action="store_true")
    parser.add_argument("--open-browser", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.lmdb_path.exists():
        raise FileNotFoundError(
            f"LMDB file does not exist: {args.lmdb_path.resolve()}"
        )

    print_section("ENVIRONMENT")
    print(f"Biotite version: {biotite.__version__}")
    print(f"LMDB path:       {args.lmdb_path.resolve()}")
    print(f"Output root:     {args.output_dir.resolve()}")
    print(f"Cropper:         {args.cropper}")
    print(f"Crop size:       {args.crop_size}")
    print(f"Random seed:     {args.seed}")

    backend = LMDBBackend(str(args.lmdb_path))

    sample_index = resolve_sample_index(
        backend,
        index=args.index,
        pdb_id=args.pdb_id,
        find_ligand=args.find_ligand,
        scan_limit=args.scan_limit,
    )

    raw_data = backend.get_raw_data(sample_index)
    pdb_id = str(raw_data["pdb_id"]).upper()

    run_dir = args.output_dir / pdb_id
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = CIFWriter()
    all_checks_passed = True
    generated_html: list[Path] = []

    # ------------------------------------------------------------------
    # Experiment 1: direct LMDB arrays -> CIF
    # ------------------------------------------------------------------
    print_section("EXPERIMENT 1: RAW LMDB → CIF")

    raw_summary = summarize_data(raw_data, "raw_lmdb")
    print_summary(raw_summary)

    raw_record = create_record(
        raw_data,
        metadata={
            "source": "lmdb",
            "sample_index": sample_index,
            "cropped": False,
        },
    )

    raw_cif_path = run_dir / f"{pdb_id}_raw.cif"
    writer.write(raw_record, raw_cif_path)

    raw_roundtrip, raw_passed = validate_roundtrip(
        raw_data,
        raw_cif_path,
        "raw_lmdb",
    )
    print_section("RAW CIF ROUND-TRIP")
    print_summary(raw_roundtrip)

    all_checks_passed &= raw_passed

    if not args.skip_html:
        raw_html_path = run_dir / f"{pdb_id}_raw.html"
        create_py3dmol_html(
            raw_cif_path,
            raw_html_path,
            f"{pdb_id}: raw LMDB reconstruction",
            {
                "source_summary": raw_summary,
                "roundtrip": raw_roundtrip,
            },
        )
        generated_html.append(raw_html_path)

    # ------------------------------------------------------------------
    # Experiment 2: LMDB -> cropper -> PDBDataset tensors -> CIF
    # ------------------------------------------------------------------
    cropper = get_cropper(args.cropper, args.crop_size)

    if cropper is not None:
        print_section("EXPERIMENT 2: LMDB → CROPPER → PDBDATASET → CIF")

        np.random.seed(args.seed)

        dataset = PDBDataset(
            backend=backend,
            cropper=cropper,
        )
        cropped_data = dataset[sample_index]

        cropped_summary = summarize_data(
            cropped_data,
            f"{args.cropper}_crop",
        )
        print_summary(cropped_summary)

        cropped_record = create_record(
            cropped_data,
            metadata={
                "source": "pdb_dataset",
                "sample_index": sample_index,
                "cropper": args.cropper,
                "crop_size": args.crop_size,
                "seed": args.seed,
            },
        )

        cropped_cif_path = run_dir / f"{pdb_id}_{args.cropper}_crop.cif"
        writer.write(cropped_record, cropped_cif_path)

        cropped_roundtrip, cropped_passed = validate_roundtrip(
            cropped_data,
            cropped_cif_path,
            f"{args.cropper}_crop",
        )
        print_section("CROPPED CIF ROUND-TRIP")
        print_summary(cropped_roundtrip)

        all_checks_passed &= cropped_passed

        if not args.skip_html:
            cropped_html_path = (
                run_dir / f"{pdb_id}_{args.cropper}_crop.html"
            )
            create_py3dmol_html(
                cropped_cif_path,
                cropped_html_path,
                f"{pdb_id}: {args.cropper} crop",
                {
                    "source_summary": cropped_summary,
                    "roundtrip": cropped_roundtrip,
                },
            )
            generated_html.append(cropped_html_path)

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    print_section("FINAL RESULT")
    print(f"PDB ID:             {pdb_id}")
    print(f"LMDB index:         {sample_index}")
    print(f"Output directory:   {run_dir.resolve()}")
    print(f"All numeric checks: {'PASS' if all_checks_passed else 'FAIL'}")

    for path in generated_html:
        print(f"Viewer:              {path.resolve()}")

    if generated_html and args.open_browser:
        opened = webbrowser.open(generated_html[-1].resolve().as_uri())
        print(f"Browser open request succeeded: {opened}")

    if generated_html:
        print()
        print("For remote HPC viewing, copy the HTML locally or serve it with:")
        print(
            f"  uv run python -m http.server 8000 "
            f"--directory {run_dir.resolve()}"
        )

    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())