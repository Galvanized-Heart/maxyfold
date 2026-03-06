import pytest

def test_imports():
    """
    Smoke test to ensure the package structure is correct and 
    classes are exposed in the top-level init files.
    """
    try:
        from maxyfold.data import PDBDownloader
        from maxyfold.data import PDBProcessor
        from maxyfold.data import PDBDataset
    except ImportError as e:
        pytest.fail(f"Import failed: {e}. Check __init__.py files and package installation.")

def test_dataset_instantiation(tmp_path):
    """
    Verify we can instantiate the dataset with a mocked backend.
    """
    from maxyfold.data import DataBackend
    from maxyfold.data import PDBDataset
    
    # Create dummy backend
    class MockBackend(DataBackend):
        def __len__(self): return 10
        def get_raw_data(self, idx): 
            import numpy as np
            return {
                "coords": np.zeros((5,4,3)), 
                "mask": np.ones((5,)), 
                "sequence": "ACDEF"
            }

    # Test
    backend = MockBackend()
    dataset = PDBDataset(backend)
    assert len(dataset) == 10
    item = dataset[0]
    assert "coords" in item
    assert "mask" in item