import torch

from maxyfold.models.metrics import (
    MaskedCoordMAE,
    MaskedCoordMSE,
    MaskedCoordRMSE,
    PaddingFraction,
    RepTokenDistanceMAE,
    ValidAtomFraction,
    masked_coord_mae,
    masked_coord_mse,
    masked_coord_rmse,
)


def test_functional_masked_coord_metrics_ignore_padding():
    pred = torch.ones(1, 2, 1, 3)
    true = torch.zeros(1, 2, 1, 3)
    mask = torch.tensor([[[1.0], [0.0]]])

    assert torch.isclose(masked_coord_mse(pred, true, mask), torch.tensor(3.0))
    assert torch.isclose(
        masked_coord_rmse(pred, true, mask),
        torch.sqrt(torch.tensor(3.0)),
    )
    assert torch.isclose(masked_coord_mae(pred, true, mask), torch.tensor(1.0))


def test_torchmetrics_masked_coord_metrics_ignore_padding():
    pred = torch.ones(1, 2, 1, 3)
    true = torch.zeros(1, 2, 1, 3)
    mask = torch.tensor([[[1.0], [0.0]]])

    mse = MaskedCoordMSE()
    rmse = MaskedCoordRMSE()
    mae = MaskedCoordMAE()

    mse.update(pred, true, mask)
    rmse.update(pred, true, mask)
    mae.update(pred, true, mask)

    assert torch.isclose(mse.compute(), torch.tensor(3.0))
    assert torch.isclose(rmse.compute(), torch.sqrt(torch.tensor(3.0)))
    assert torch.isclose(mae.compute(), torch.tensor(1.0))


def test_zero_mask_does_not_crash():
    pred = torch.randn(1, 2, 3, 3)
    true = torch.randn(1, 2, 3, 3)
    mask = torch.zeros(1, 2, 3)

    metrics = [
        MaskedCoordMSE(),
        MaskedCoordRMSE(),
        MaskedCoordMAE(),
        RepTokenDistanceMAE(),
        ValidAtomFraction(),
        PaddingFraction(),
    ]

    for metric in metrics:
        metric.update(pred, true, mask)
        assert torch.isfinite(metric.compute())


def test_valid_atom_fraction():
    pred = torch.zeros(1, 2, 2, 3)
    true = torch.zeros(1, 2, 2, 3)
    mask = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])

    metric = ValidAtomFraction()
    metric.update(pred, true, mask)

    assert torch.isclose(metric.compute(), torch.tensor(0.25))


def test_padding_fraction_counts_empty_tokens():
    pred = torch.zeros(1, 2, 2, 3)
    true = torch.zeros(1, 2, 2, 3)
    mask = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])

    metric = PaddingFraction()
    metric.update(pred, true, mask)

    assert torch.isclose(metric.compute(), torch.tensor(0.5))


def test_rep_token_distance_mae_zero_for_identical_structures():
    coords = torch.randn(2, 4, 3, 3)
    mask = torch.ones(2, 4, 3)

    metric = RepTokenDistanceMAE()
    metric.update(coords, coords.clone(), mask)

    assert torch.isclose(metric.compute(), torch.tensor(0.0))