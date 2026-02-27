import pytest
import lmdb
import gzip
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

from maxyfold.data.splits.pdb_splitter import PDBDataSplitter
from maxyfold.data.components.tarball_writer import TarballWriter

DUMMY_CIF = "data_1ABC\n_entry.id 1ABC\n"

@pytest.fixture
def mock_pdb_env(tmp_path):
    """Sets up a fake PDB directory structure."""
    raw_dir = tmp_path / "raw" / "assemblies"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    
    # 1. Create Fake LMDB (subdir=False is critical!)
    lmdb_path = processed_dir / "pdb_dataset.lmdb"
    with lmdb.open(str(lmdb_path), map_size=1024*1024, subdir=False) as env:
        with env.begin(write=True) as txn:
            txn.put(b"1ABC", b"dummy_data")
            txn.put(b"2DEF", b"dummy_data")
            txn.put(b"3GHI", b"dummy_data")

    # 2. Create Fake Tarball with CIFs
    tar_path = raw_dir / "assemblies_batch_0.tar.gz"
    
    cif_names = ["1abc-assembly1.cif.gz", "2def-assembly1.cif.gz", "3ghi-assembly1.cif.gz"]
    
    with TarballWriter(tar_path) as writer:
        for name in cif_names:
            p = raw_dir / name
            with gzip.open(p, 'wt') as f:
                f.write(DUMMY_CIF.replace("1ABC", name[:4].upper()))
            writer.add_file(p, delete_original=True)
            
    return {
        "lmdb": lmdb_path,
        "raw": raw_dir,
        "out": processed_dir
    }

def test_splitter_logic_end_to_end(mock_pdb_env):
    """
    Tests the splitting logic with MOCKS for extraction and clustering.
    """
    
    # Configs
    mmseqs_cfg = {"seq_id": 0.3, "coverage": 0.8, "cov_mode": 0, "cluster_mode": 0, "threads": 1}
    split_cfg = {"ratios": [0.6, 0.2, 0.2], "seed": 42} 

    splitter = PDBDataSplitter(
        lmdb_path=mock_pdb_env["lmdb"],
        raw_assemblies_dir=mock_pdb_env["raw"],
        output_dir=mock_pdb_env["out"],
        mmseqs_config=mmseqs_cfg,
        splitting_config=split_cfg,
        limit=0
    )
    
    # Mock 1: Extracted Sequences
    mock_seqs = {
        "1ABC_A": "AAAAA"*5, 
        "2DEF_A": "BBBBB"*5,
        "3GHI_A": "CCCCC"*5
    }
    
    # Mock 2: Clustering Results
    mock_cluster_df = pd.DataFrame([
        {'representative': '1ABC_A', 'member': '1ABC_A'},
        {'representative': '1ABC_A', 'member': '2DEF_A'},
        {'representative': '3GHI_A', 'member': '3GHI_A'},
    ])
        
    # Patch 1: Skip reading tarballs, just return dict
    with patch.object(splitter, '_extract_protein_sequences', return_value=mock_seqs) as mock_extract:
        # Patch 2: Skip running binary, just return DataFrame
        with patch.object(splitter, 'run_mmseqs2_clustering', return_value=mock_cluster_df) as mock_run:
            
            splitter.create()
                        
            # 1. Verify our mocks were used
            assert mock_extract.called
            assert mock_run.called
            
            # 2. Verify Output Files
            train_file = mock_pdb_env["out"] / "train_keys.txt"
            val_file = mock_pdb_env["out"] / "val_keys.txt"
            test_file = mock_pdb_env["out"] / "test_keys.txt"
            
            assert train_file.exists()
            assert val_file.exists()
            assert test_file.exists()
            
            # 3. Verify Split Logic
            with open(train_file) as f: train_keys = [x.strip() for x in f.readlines()]
            with open(val_file) as f: val_keys = [x.strip() for x in f.readlines()]
            with open(test_file) as f: test_keys = [x.strip() for x in f.readlines()]
            
            # Check total count
            assert len(train_keys) + len(val_keys) + len(test_keys) == 3
            
            # Check cluster integrity
            # 1ABC and 2DEF must be in the SAME split because they share a representative
            
            def get_split(pdb):
                if pdb in train_keys: return "train"
                if pdb in val_keys: return "val"
                if pdb in test_keys: return "test"
                return "missing"

            assert get_split("1ABC") == get_split("2DEF"), "Homologs were split across sets!"
            assert get_split("3GHI") != "missing"