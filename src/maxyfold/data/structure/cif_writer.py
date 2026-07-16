# src/maxyfold/data/structure/cif_writer.py

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np

from maxyfold.data.constants import (
    LIGAND_IDX,
    get_atom_name,
    get_element_symbol,
    get_residue_name,
)
from maxyfold.data.structure.records import StructureRecord


def _chain_label(chain_index: int) -> str:
    """Convert 0, 1, ..., 25, 26 into A, B, ..., Z, AA."""
    if chain_index < 0:
        raise ValueError(f"Chain index cannot be negative: {chain_index}")

    label = ""

    while True:
        chain_index, remainder = divmod(chain_index, 26)
        label = chr(ord("A") + remainder) + label

        if chain_index == 0:
            return label

        chain_index -= 1


class CIFWriter:
    """Serialize MaxyFold structure records as visualization-oriented mmCIF."""

    def __init__(self, ligand_res_name: str = "LIG") -> None:
        self.ligand_res_name = ligand_res_name

    def to_atom_array(self, record: StructureRecord) -> struc.AtomArray:
        """Convert one structure record into a Biotite AtomArray."""
        record = record.numpy()
        record.validate()

        coords: list[np.ndarray] = []
        chain_ids: list[str] = []
        res_ids: list[int] = []
        res_names: list[str] = []
        atom_names: list[str] = []
        elements: list[str] = []
        hetero: list[bool] = []

        next_polymer_res_id: dict[int, int] = defaultdict(lambda: 1)
        ligand_atom_counts: dict[tuple[int, str], int] = defaultdict(int)

        for token_index in range(len(record.res_type)):
            chain_index = int(record.chain_ids[token_index])

            if chain_index < 0:
                continue

            valid_atom_indices = np.flatnonzero(record.mask[token_index])

            if len(valid_atom_indices) == 0:
                continue

            residue_name = get_residue_name(record.res_type[token_index])
            is_ligand = int(record.res_type[token_index]) == LIGAND_IDX

            if is_ligand:
                residue_id = 1
                output_residue_name = self.ligand_res_name
            else:
                residue_id = next_polymer_res_id[chain_index]
                next_polymer_res_id[chain_index] += 1
                output_residue_name = residue_name

            for atom_index in valid_atom_indices:
                element = get_element_symbol(
                    record.atom_elements[token_index, atom_index]
                )

                if is_ligand:
                    count_key = (chain_index, element)
                    ligand_atom_counts[count_key] += 1
                    atom_name = f"{element}{ligand_atom_counts[count_key]}"
                else:
                    atom_name = get_atom_name(residue_name, atom_index)
                    atom_name = atom_name or f"{element}{atom_index + 1}"

                coords.append(record.coords[token_index, atom_index])
                chain_ids.append(_chain_label(chain_index))
                res_ids.append(residue_id)
                res_names.append(output_residue_name)
                atom_names.append(atom_name)
                elements.append(element)
                hetero.append(is_ligand)

        if not coords:
            raise ValueError(
                f"No valid atoms available for structure {record.pdb_id}"
            )

        atom_array = struc.AtomArray(len(coords))
        atom_array.coord = np.asarray(coords, dtype=np.float32)
        atom_array.chain_id = np.asarray(chain_ids, dtype="U8")
        atom_array.res_id = np.asarray(res_ids, dtype=np.int32)
        atom_array.res_name = np.asarray(res_names, dtype="U8")
        atom_array.atom_name = np.asarray(atom_names, dtype="U8")
        atom_array.element = np.asarray(elements, dtype="U4")
        atom_array.hetero = np.asarray(hetero, dtype=bool)

        return atom_array

    def write(
        self,
        record: StructureRecord,
        output_path: str | Path,
    ) -> Path:
        """Write one structure to an mmCIF file."""
        record = record.numpy()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cif_file = pdbx.CIFFile()
        pdbx.set_structure(
            cif_file,
            self.to_atom_array(record),
            data_block=record.pdb_id,
        )
        cif_file.write(str(output_path))

        return output_path