import pytest
import torch

from equidock_diff.models.egnn import EGNNConfig, EGNNScoreNet
from equidock_diff.models.egnn import (
    EDGE_TYPE_LIGAND_LIGAND,
    EDGE_TYPE_LIGAND_PROTEIN,
    EDGE_TYPE_PROTEIN_PROTEIN,
    complete_frame_basis,
    infer_edge_types,
    infer_ligand_mask,
)
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


def test_egnn_score_is_rotation_equivariant_with_hetero_edges() -> None:
    torch.manual_seed(0)
    model = EGNNScoreNet(
        EGNNConfig(
            node_dim=4,
            hidden_dim=32,
            num_layers=2,
            use_hetero_edges=True,
        )
    )
    node_features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
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


def test_egnn_score_is_rotation_equivariant_with_ligand_global_node() -> None:
    torch.manual_seed(0)
    model = EGNNScoreNet(
        EGNNConfig(
            node_dim=4,
            hidden_dim=32,
            num_layers=2,
            use_ligand_global_node=True,
        )
    )
    node_features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
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


def test_egnn_score_is_rotation_equivariant_with_complete_frame() -> None:
    torch.manual_seed(0)
    model = EGNNScoreNet(
        EGNNConfig(
            node_dim=4,
            hidden_dim=32,
            num_layers=2,
            use_complete_frame=True,
        )
    )
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


def test_infer_ligand_mask_uses_partition_columns() -> None:
    node_features = torch.tensor(
        [
            [1.0, 0.0, 0.2, 0.0],
            [1.0, 0.0, 0.7, 0.0],
            [0.0, 1.0, 0.3, 0.0],
            [0.0, 1.0, 0.9, 0.0],
        ],
        dtype=torch.float32,
    )

    mask = infer_ligand_mask(node_features)

    assert mask.tolist() == [True, True, False, False]


def test_infer_edge_types_separates_ligand_and_protein_edges() -> None:
    edge_index = torch.tensor(
        [
            [0, 2, 1, 3, 0, 2],
            [1, 3, 2, 0, 2, 0],
        ],
        dtype=torch.long,
    )
    ligand_mask = torch.tensor([True, True, False, False])

    edge_types = infer_edge_types(edge_index, ligand_mask)

    assert edge_types.tolist() == [
        EDGE_TYPE_LIGAND_LIGAND,
        EDGE_TYPE_PROTEIN_PROTEIN,
        EDGE_TYPE_LIGAND_PROTEIN,
        EDGE_TYPE_LIGAND_PROTEIN,
        EDGE_TYPE_LIGAND_PROTEIN,
        EDGE_TYPE_LIGAND_PROTEIN,
    ]


def test_complete_frame_basis_stays_finite_when_cross_product_degenerates() -> None:
    src_positions = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    dst_positions = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)

    radial, pseudo, orthogonal = complete_frame_basis(src_positions, dst_positions)

    assert torch.isfinite(radial).all()
    assert torch.isfinite(pseudo).all()
    assert torch.isfinite(orthogonal).all()
