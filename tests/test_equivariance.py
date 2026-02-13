import torch

from equidock_diff.models.egnn import EGNNConfig, EGNNScoreNet
from equidock_diff.utils.geometry import apply_rigid_transform, random_rotation_matrix


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
