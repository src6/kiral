import pytest
import torch

from equidock_diff.models.egnn import EGNNConfig, EGNNScoreNet
from equidock_diff.utils.geometry import (
    aligned_rmsd,
    apply_rigid_transform,
    random_rotation_matrix,
)


def test_rotation_matrix_shapes() -> None:
    device = torch.device("cpu")
    rot = random_rotation_matrix(3, device=device, dtype=torch.float32)
    pos = torch.randn(4, 3)
    trans = torch.zeros(3)
    out = apply_rigid_transform(pos, rot[0], trans)
    assert rot.shape == (3, 3, 3)
    assert out.shape == pos.shape


def test_egnn_score_is_rotation_equivariant() -> None:
    torch.manual_seed(0)
    model = EGNNScoreNet(EGNNConfig(node_dim=4, hidden_dim=32, num_layers=2))
    node_features = torch.randn(6, 4)
    positions = torch.randn(6, 3)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 1, 2, 3, 4], [1, 2, 3, 4, 5, 0, 0, 1, 2, 3]],
        dtype=torch.long,
    )
    time = torch.tensor(0.3)

    base_score = model(node_features, positions, edge_index, time)
    rotation = random_rotation_matrix(1, device=torch.device("cpu"), dtype=torch.float32)[0]
    translation = torch.randn(3)
    transformed_positions = apply_rigid_transform(positions, rotation, translation)
    transformed_score = model(node_features, transformed_positions, edge_index, time)
    expected = apply_rigid_transform(base_score, rotation, torch.zeros(3))

    assert torch.allclose(transformed_score, expected, atol=1e-4, rtol=1e-4)


def test_aligned_rmsd_is_zero_for_rigidly_transformed_points() -> None:
    reference = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    rotation = random_rotation_matrix(1, device=torch.device("cpu"), dtype=torch.float32)[0]
    translation = torch.tensor([1.5, -0.25, 0.5], dtype=torch.float32)
    transformed = apply_rigid_transform(reference, rotation, translation)

    rmsd = aligned_rmsd(transformed, reference)

    assert rmsd.item() == pytest.approx(0.0, abs=1e-5)


def test_aligned_rmsd_detects_non_rigid_distortion() -> None:
    reference = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    distorted = reference.clone()
    distorted[1, 0] = 1.3

    rmsd = aligned_rmsd(distorted, reference)

    assert rmsd.item() > 0.0
