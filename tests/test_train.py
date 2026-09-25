from __future__ import annotations

from pathlib import Path

import pytest
import torch

from equidock_diff.diffusion.schedules import cosine_signal_amplitude
from equidock_diff.utils.artifacts import (
    ligand_mask_from_features,
    write_experiment_log,
    write_ligand_artifacts,
)
from equidock_diff.train import (
    build_parser,
    build_sample_schedule,
    build_dataset_examples,
    build_synthetic_graph,
    dataset_example_for_step,
    ligand_bond_length_loss,
    ligand_protein_contact_loss,
    ligand_protein_clash_loss,
    ligand_shape_loss,
    load_checkpoint,
    load_graph_inputs,
    main,
    make_model,
    noised_positions_for_schedule,
    saved_args_to_namespace,
    save_checkpoint,
    sample_positions,
    training_step,
    training_step_with_breakdown,
    validate_resume_compatibility,
)


class _Args:
    hidden_dim = 32
    num_layers = 2
    hetero_edges = False
    ligand_global_node = False
    complete_frame = False
    frame_hetero_backbone = False
    use_edge_attention = False
    use_cross_interface_block = False
    learning_rate = 1e-3
    ligand_bond_weight = 0.0
    ligand_shape_weight = 0.0
    ligand_protein_clash_weight = 0.0
    ligand_protein_contact_weight = 0.0
    beta_min = 0.1
    beta_max = 2.0
    noise_schedule = "linear"
    cosine_offset = 0.008
    cosine_nu = 1.5
    protein_path = None
    ligand_path = None
    crop_cutoff = 10.0
    context_policy = "fixed"
    protein_node_budget = 256
    edge_cutoff = 4.5
    batch_size = 1
    num_nodes = 8
    steps = 5
    sample_steps = 8
    sample_time_power = 1.0
    checkpoint_path = None
    checkpoint_every = 0
    resume_from = None
    dataset_root = None
    dataset_split = None
    dataset_limit = 0
    dataset_cache_dir = Path("data/.cache/equidock_diff_graphs")
    sampler_diagnostics_json = None
    seed = 42


def test_training_step_is_finite() -> None:
    device = torch.device("cpu")
    model = make_model(_Args(), device)
    node_features, positions, edge_index = build_synthetic_graph(8, 1, device)

    loss, beta_t = training_step(
        model,
        node_features,
        positions,
        edge_index,
        beta_min=0.1,
        beta_max=2.0,
    )

    assert torch.isfinite(loss)
    assert beta_t > 0.0


def test_training_step_is_finite_with_hetero_edges() -> None:
    class _HeteroArgs(_Args):
        hetero_edges = True

    device = torch.device("cpu")
    model = make_model(_HeteroArgs(), device)
    node_features, positions, edge_index = build_synthetic_graph(8, 1, device)

    loss, beta_t = training_step(
        model,
        node_features,
        positions,
        edge_index,
        beta_min=0.1,
        beta_max=2.0,
    )

    assert torch.isfinite(loss)
    assert beta_t > 0.0


def test_training_step_is_finite_with_ligand_global_node() -> None:
    class _GlobalArgs(_Args):
        ligand_global_node = True

    device = torch.device("cpu")
    model = make_model(_GlobalArgs(), device)
    node_features, positions, edge_index = build_synthetic_graph(8, 1, device)

    loss, beta_t = training_step(
        model,
        node_features,
        positions,
        edge_index,
        beta_min=0.1,
        beta_max=2.0,
    )

    assert torch.isfinite(loss)
    assert beta_t > 0.0


def test_training_step_is_finite_with_complete_frame() -> None:
    class _CompleteFrameArgs(_Args):
        complete_frame = True

    device = torch.device("cpu")
    model = make_model(_CompleteFrameArgs(), device)
    node_features, positions, edge_index = build_synthetic_graph(8, 1, device)

    loss, beta_t = training_step(
        model,
        node_features,
        positions,
        edge_index,
        beta_min=0.1,
        beta_max=2.0,
    )

    assert torch.isfinite(loss)
    assert beta_t > 0.0


def test_training_step_is_finite_with_frame_hetero_backbone() -> None:
    class _FrameHeteroArgs(_Args):
        frame_hetero_backbone = True

    device = torch.device("cpu")
    model = make_model(_FrameHeteroArgs(), device)
    node_features, positions, edge_index = build_synthetic_graph(8, 1, device)

    loss, beta_t = training_step(
        model,
        node_features,
        positions,
        edge_index,
        beta_min=0.1,
        beta_max=2.0,
    )

    assert torch.isfinite(loss)
    assert beta_t > 0.0


def test_training_step_is_finite_with_frame_hetero_backbone_and_edge_attention() -> None:
    class _FrameAttentionArgs(_Args):
        frame_hetero_backbone = True
        use_edge_attention = True

    device = torch.device("cpu")
    model = make_model(_FrameAttentionArgs(), device)
    node_features, positions, edge_index = build_synthetic_graph(8, 1, device)

    loss, beta_t = training_step(
        model,
        node_features,
        positions,
        edge_index,
        beta_min=0.1,
        beta_max=2.0,
    )

    assert torch.isfinite(loss)
    assert beta_t > 0.0


def test_training_step_is_finite_with_frame_hetero_backbone_and_cross_interface_block() -> None:
    class _FrameCrossArgs(_Args):
        frame_hetero_backbone = True
        use_cross_interface_block = True

    device = torch.device("cpu")
    model = make_model(_FrameCrossArgs(), device)
    node_features, positions, edge_index = build_synthetic_graph(8, 1, device)

    loss, beta_t = training_step(
        model,
        node_features,
        positions,
        edge_index,
        beta_min=0.1,
        beta_max=2.0,
    )

    assert torch.isfinite(loss)
    assert beta_t > 0.0


def test_main_rejects_cross_interface_block_without_frame_backbone() -> None:
    with pytest.raises(ValueError, match="requires --frame-hetero-backbone"):
        main(["--dry-run", "--use-cross-interface-block"])


def test_main_rejects_cross_interface_block_with_edge_attention() -> None:
    with pytest.raises(ValueError, match="cannot be combined with --use-edge-attention"):
        main(["--dry-run", "--frame-hetero-backbone", "--use-cross-interface-block", "--use-edge-attention"])


def test_load_graph_inputs_requires_both_real_paths() -> None:
    class _RealArgs(_Args):
        protein_path = Path("only_protein.pdb")
        ligand_path = None
        crop_cutoff = 10.0
        num_nodes = 8
        batch_size = 1

    with pytest.raises(ValueError, match="Pass both --protein-path and --ligand-path"):
        load_graph_inputs(_RealArgs(), torch.device("cpu"))


def test_training_step_is_finite_for_real_pair_graph(tmp_path: Path) -> None:
    rdkit = pytest.importorskip("rdkit")
    assert rdkit is not None

    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    ligand_path = tmp_path / "ethanol.sdf"
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert mol is not None
    assert AllChem.EmbedMolecule(mol, randomSeed=0) == 0
    writer = Chem.SDWriter(str(ligand_path))
    writer.write(mol)
    writer.close()

    protein_path = tmp_path / "toy_protein.pdb"
    protein_path.write_text(
        "\n".join(
            [
                "ATOM      1  N   GLY A   1       0.000   4.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  GLY A   1       0.000   6.000   0.000  1.00  0.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class _RealArgs(_Args):
        crop_cutoff = 10.0
        num_nodes = 8
        batch_size = 1

    real_args = _RealArgs()
    real_args.protein_path = protein_path
    real_args.ligand_path = ligand_path

    device = torch.device("cpu")
    node_features, positions, edge_index, ligand_bond_index, resolved_crop_cutoff, retained_protein_nodes = load_graph_inputs(real_args, device)
    model = make_model(real_args, device, node_dim=node_features.shape[1])

    loss, beta_t = training_step(
        model,
        node_features,
        positions,
        edge_index,
        beta_min=0.1,
        beta_max=2.0,
        ligand_bond_index=ligand_bond_index,
    )

    assert node_features.shape[1] == 17
    assert ligand_bond_index is not None
    assert resolved_crop_cutoff == pytest.approx(10.0)
    assert retained_protein_nodes is not None
    assert torch.isfinite(loss)
    assert beta_t > 0.0


def test_load_graph_inputs_uses_cache_for_real_pair_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RealArgs(_Args):
        protein_path = Path("toy_protein.pdb")
        ligand_path = Path("toy_ligand.sdf")
        dataset_cache_dir = Path("graph_cache")

    captured: dict[str, object] = {}

    def _fake_cached_loader(
        protein_path: Path,
        ligand_path: Path,
        *,
        cutoff: float,
        edge_cutoff: float,
        cache_dir: Path | None,
        context_policy: str,
        protein_node_budget: int,
    ):
        captured["protein_path"] = protein_path
        captured["ligand_path"] = ligand_path
        captured["cutoff"] = cutoff
        captured["edge_cutoff"] = edge_cutoff
        captured["cache_dir"] = cache_dir
        captured["context_policy"] = context_policy
        captured["protein_node_budget"] = protein_node_budget
        return type(
            "_Batch",
            (),
            {
                "node_features": torch.zeros((2, 17), dtype=torch.float32),
                "positions": torch.zeros((2, 3), dtype=torch.float32),
                "edge_index": torch.zeros((2, 0), dtype=torch.long),
                "ligand_bond_index": None,
                "resolved_crop_cutoff": 10.0,
                "retained_protein_nodes": 1,
            },
        )()

    monkeypatch.setattr("equidock_diff.train.load_protein_ligand_graph_cached", _fake_cached_loader)

    node_features, positions, edge_index, ligand_bond_index, resolved_crop_cutoff, retained_protein_nodes = load_graph_inputs(
        _RealArgs(),
        torch.device("cpu"),
    )

    assert node_features.shape == (2, 17)
    assert positions.shape == (2, 3)
    assert edge_index.shape == (2, 0)
    assert ligand_bond_index is None
    assert resolved_crop_cutoff == pytest.approx(10.0)
    assert retained_protein_nodes == 1
    assert captured["cache_dir"] == Path("graph_cache")
    assert captured["context_policy"] == "fixed"
    assert captured["protein_node_budget"] == 256


def test_training_step_is_finite_with_cosine_schedule() -> None:
    class _CosineArgs(_Args):
        noise_schedule = "cosine"
        cosine_nu = 1.5

    device = torch.device("cpu")
    model = make_model(_CosineArgs(), device)
    node_features, positions, edge_index = build_synthetic_graph(8, 1, device)

    loss, beta_t = training_step(
        model,
        node_features,
        positions,
        edge_index,
        beta_min=0.1,
        beta_max=2.0,
        noise_schedule="cosine",
        cosine_offset=0.008,
        cosine_nu=1.5,
    )

    assert torch.isfinite(loss)
    assert beta_t > 0.0


def test_noised_positions_for_schedule_uses_alpha_bar_coefficients_for_cosine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = torch.tensor([0.5], dtype=torch.float32)
    clean_positions = torch.tensor([[2.0, -4.0, 6.0]], dtype=torch.float32)
    alpha_bar = cosine_signal_amplitude(t, offset=0.008, nu=1.5)
    expected_signal_scale = torch.sqrt(alpha_bar)
    expected_sigma = torch.sqrt(1.0 - alpha_bar)

    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.zeros_like(tensor))
    noised_positions, _ = noised_positions_for_schedule(
        clean_positions,
        t,
        beta_min=0.1,
        beta_max=2.0,
        noise_schedule="cosine",
        cosine_offset=0.008,
        cosine_nu=1.5,
    )
    assert torch.allclose(noised_positions, expected_signal_scale.unsqueeze(-1) * clean_positions)

    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.ones_like(tensor))
    noised_positions, _ = noised_positions_for_schedule(
        torch.zeros_like(clean_positions),
        t,
        beta_min=0.1,
        beta_max=2.0,
        noise_schedule="cosine",
        cosine_offset=0.008,
        cosine_nu=1.5,
    )
    assert torch.allclose(
        noised_positions,
        expected_sigma.unsqueeze(-1) * torch.ones_like(clean_positions),
    )


def test_sample_positions_stays_finite_with_clipping() -> None:
    class _SampleArgs(_Args):
        hidden_dim = 32
        num_layers = 2

    device = torch.device("cpu")
    node_features, positions, edge_index = build_synthetic_graph(8, 1, device)
    model = make_model(_SampleArgs(), device, node_dim=node_features.shape[1])

    sampled_positions, trajectory, diagnostics = sample_positions(
        model,
        node_features,
        edge_index,
        positions.size(0),
        device,
        sample_steps=8,
        beta_min=0.1,
        beta_max=2.0,
        score_clip=10.0,
        position_clip=50.0,
    )

    assert torch.isfinite(sampled_positions).all()
    assert len(trajectory) == 9
    assert torch.isfinite(trajectory[-1]).all()
    assert diagnostics is None


def test_sample_positions_keeps_protein_anchor_with_frame_hetero_backbone() -> None:
    class _FrameHeteroArgs(_Args):
        frame_hetero_backbone = True

    device = torch.device("cpu")
    node_features, positions, edge_index = build_synthetic_graph(8, 1, device)
    model = make_model(_FrameHeteroArgs(), device, node_dim=node_features.shape[1])

    sampled_positions, trajectory, _ = sample_positions(
        model,
        node_features,
        edge_index,
        positions.size(0),
        device,
        sample_steps=4,
        beta_min=0.1,
        beta_max=2.0,
        score_clip=10.0,
        position_clip=50.0,
        reference_positions=positions,
        anchor_protein=True,
    )

    protein_mask = node_features[:, 1] > 0.5
    assert torch.allclose(sampled_positions[protein_mask], positions[protein_mask])
    assert torch.allclose(trajectory[-1][protein_mask], positions[protein_mask].cpu())


def test_checkpoint_round_trip_restores_training_state(tmp_path: Path) -> None:
    device = torch.device("cpu")
    args = _Args()
    model = make_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    node_features, positions, edge_index = build_synthetic_graph(args.num_nodes, 1, device)

    optimizer.zero_grad(set_to_none=True)
    loss, beta_t, _, _, _, _, _ = training_step_with_breakdown(
        model,
        node_features,
        positions,
        edge_index,
        None,
        args.beta_min,
        args.beta_max,
        ligand_bond_weight=args.ligand_bond_weight,
        ligand_shape_weight=args.ligand_shape_weight,
        ligand_protein_clash_weight=args.ligand_protein_clash_weight,
        ligand_protein_contact_weight=args.ligand_protein_contact_weight,
        frame_hetero_backbone=args.frame_hetero_backbone,
    )
    loss.backward()
    optimizer.step()
    loss_rows = [(1, float(loss.item()), beta_t)]

    checkpoint_path = tmp_path / "resume.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        args=args,
        completed_steps=1,
        training_seconds=1.25,
        loss_rows=loss_rows,
        graph_source="synthetic",
        node_feature_dim=node_features.size(-1),
    )

    restored_model = make_model(args, device)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=args.learning_rate)
    checkpoint_state = load_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        device=device,
    )

    validate_resume_compatibility(
        args,
        checkpoint_state,
        graph_source="synthetic",
        node_feature_dim=node_features.size(-1),
    )

    assert checkpoint_state.completed_steps == 1
    assert checkpoint_state.training_seconds == pytest.approx(1.25)
    assert checkpoint_state.loss_rows == loss_rows

    for current_param, restored_param in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.allclose(current_param, restored_param)


def test_validate_resume_compatibility_rejects_mismatched_settings(tmp_path: Path) -> None:
    device = torch.device("cpu")
    args = _Args()
    model = make_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    checkpoint_path = tmp_path / "resume.pt"

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        args=args,
        completed_steps=2,
        training_seconds=0.5,
        loss_rows=[(1, 0.1, 0.2), (2, 0.05, 0.3)],
        graph_source="synthetic",
        node_feature_dim=4,
    )

    checkpoint_state = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        device=device,
    )

    class _MismatchedArgs(_Args):
        hidden_dim = 64

    with pytest.raises(ValueError, match="Checkpoint is not compatible"):
        validate_resume_compatibility(
            _MismatchedArgs(),
            checkpoint_state,
            graph_source="synthetic",
            node_feature_dim=4,
        )


def test_validate_resume_compatibility_rejects_changed_dataset_order(tmp_path: Path) -> None:
    device = torch.device("cpu")
    args = _Args()
    model = make_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    checkpoint_path = tmp_path / "resume.pt"

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        args=args,
        completed_steps=2,
        training_seconds=0.5,
        loss_rows=[(1, 0.1, 0.2), (2, 0.05, 0.3)],
        graph_source="dataset",
        node_feature_dim=17,
        source_ids=["10gs", "11gs"],
    )

    checkpoint_state = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        device=device,
    )

    with pytest.raises(ValueError, match="dataset/source ordering changed"):
        validate_resume_compatibility(
            args,
            checkpoint_state,
            graph_source="dataset",
            node_feature_dim=17,
            source_ids=["11gs", "10gs"],
        )


def test_build_dataset_examples_uses_split_order(tmp_path: Path) -> None:
    dataset_root = tmp_path / "pdbbind_v2020"
    base_dir = dataset_root / "protein_ligand_general_minus_refined" / "1981-2000"
    for complex_id in ("10gs", "11gs", "1a30"):
        complex_dir = base_dir / complex_id
        complex_dir.mkdir(parents=True, exist_ok=True)
        (complex_dir / f"{complex_id}_protein.pdb").write_text("", encoding="utf-8")
        (complex_dir / f"{complex_id}_ligand.sdf").write_text("", encoding="utf-8")

    split_path = tmp_path / "train.txt"
    split_path.write_text("1a30\n10gs\n", encoding="utf-8")

    args = _Args()
    args.dataset_root = dataset_root
    args.dataset_split = split_path
    args.dataset_limit = 0

    examples = build_dataset_examples(args)

    assert [example.complex_id for example in examples] == ["1a30", "10gs"]
    assert dataset_example_for_step(examples, 3).complex_id == "1a30"


def test_ligand_bond_length_loss_is_zero_for_matching_bonds() -> None:
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    bond_index = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.long)

    loss = ligand_bond_length_loss(positions, positions.clone(), bond_index)

    assert loss.item() == pytest.approx(0.0)


def test_training_step_with_bond_breakdown_is_finite() -> None:
    class _SampleArgs(_Args):
        hidden_dim = 32
        num_layers = 2

    device = torch.device("cpu")
    node_features, positions, edge_index = build_synthetic_graph(8, 1, device)
    model = make_model(_SampleArgs(), device, node_dim=node_features.shape[1])
    bond_index = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.long, device=device)

    loss, beta_t, score_loss, bond_loss, shape_loss, clash_loss, contact_loss = training_step_with_breakdown(
        model,
        node_features,
        positions,
        edge_index,
        bond_index,
        beta_min=0.1,
        beta_max=2.0,
        ligand_bond_weight=0.5,
        ligand_shape_weight=0.25,
        ligand_protein_clash_weight=0.0,
        ligand_protein_contact_weight=0.0,
        frame_hetero_backbone=False,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(score_loss)
    assert torch.isfinite(bond_loss)
    assert torch.isfinite(shape_loss)
    assert torch.isfinite(clash_loss)
    assert torch.isfinite(contact_loss)
    assert beta_t > 0.0


def test_ligand_protein_clash_loss_is_zero_when_pairs_are_separated() -> None:
    predicted_positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    node_features = torch.zeros((2, 17), dtype=torch.float32)
    node_features[0, 0] = 1.0
    node_features[1, 0] = 1.0
    ligand_mask = torch.tensor([True, False], dtype=torch.bool)

    loss = ligand_protein_clash_loss(predicted_positions, node_features, ligand_mask)

    assert loss.item() == pytest.approx(0.0)


def test_ligand_protein_clash_loss_is_positive_for_overlapping_pairs() -> None:
    predicted_positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    node_features = torch.zeros((2, 17), dtype=torch.float32)
    node_features[0, 0] = 1.0
    node_features[1, 0] = 1.0
    ligand_mask = torch.tensor([True, False], dtype=torch.bool)

    loss = ligand_protein_clash_loss(predicted_positions, node_features, ligand_mask)

    assert loss.item() > 0.0


def test_ligand_protein_contact_loss_prefers_plausible_contact_window() -> None:
    node_features = torch.zeros((2, 17), dtype=torch.float32)
    node_features[0, 0] = 1.0
    node_features[1, 0] = 1.0
    ligand_mask = torch.tensor([True, False], dtype=torch.bool)

    near_target = torch.tensor([[0.0, 0.0, 0.0], [4.9, 0.0, 0.0]], dtype=torch.float32)
    far_from_target = torch.tensor([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype=torch.float32)

    good_loss = ligand_protein_contact_loss(near_target, node_features, ligand_mask)
    bad_loss = ligand_protein_contact_loss(far_from_target, node_features, ligand_mask)

    assert good_loss.item() < bad_loss.item()
    assert good_loss.item() >= 0.0


def test_training_step_with_clash_breakdown_is_finite() -> None:
    class _FrameArgs(_Args):
        frame_hetero_backbone = True

    device = torch.device("cpu")
    node_features = torch.zeros((4, 17), device=device)
    node_features[:2, 0] = 1.0
    node_features[2:, 0] = 1.0
    node_features[:2, -1] = 1.0
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [2.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [2, 3, 2, 3, 0, 1, 0, 1]],
        dtype=torch.long,
        device=device,
    )
    model = make_model(_FrameArgs(), device, node_dim=node_features.shape[1])
    bond_index = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.long, device=device)

    loss, beta_t, score_loss, bond_loss, shape_loss, clash_loss, contact_loss = training_step_with_breakdown(
        model,
        node_features,
        positions,
        edge_index,
        bond_index,
        beta_min=0.1,
        beta_max=2.0,
        ligand_bond_weight=0.1,
        ligand_shape_weight=0.1,
        ligand_protein_clash_weight=0.1,
        ligand_protein_contact_weight=0.1,
        frame_hetero_backbone=True,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(score_loss)
    assert torch.isfinite(bond_loss)
    assert torch.isfinite(shape_loss)
    assert torch.isfinite(clash_loss)
    assert torch.isfinite(contact_loss)
    assert beta_t > 0.0


def test_build_sample_schedule_default_matches_uniform_grid() -> None:
    schedule = build_sample_schedule(
        4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        time_power=1.0,
    )

    times = [float(t.item()) for t, _ in schedule]
    dts = [float(dt.item()) for _, dt in schedule]
    assert times == pytest.approx([1.0, 0.75, 0.5, 0.25])
    assert dts == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_build_sample_schedule_is_monotonic_with_power_respacing() -> None:
    schedule = build_sample_schedule(
        5,
        device=torch.device("cpu"),
        dtype=torch.float32,
        time_power=2.0,
    )

    times = [float(t.item()) for t, _ in schedule]
    dts = [float(dt.item()) for _, dt in schedule]
    assert times[0] == pytest.approx(1.0)
    assert times[-1] == pytest.approx(0.04)
    assert all(left > right for left, right in zip(times, times[1:]))
    assert sum(dts) == pytest.approx(1.0)


def test_ligand_shape_loss_is_zero_for_matching_geometry() -> None:
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([True, True, True], dtype=torch.bool)

    loss = ligand_shape_loss(positions, positions.clone(), mask)

    assert loss.item() == pytest.approx(0.0)


def test_ligand_shape_loss_is_positive_for_distorted_geometry() -> None:
    clean = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    predicted = clean.clone()
    predicted[1] = torch.tensor([2.0, 0.0, 0.0])
    mask = torch.tensor([True, True, True], dtype=torch.bool)

    loss = ligand_shape_loss(predicted, clean, mask)

    assert loss.item() > 0.0


def test_saved_args_to_namespace_restores_paths() -> None:
    args = saved_args_to_namespace(
        {
            "protein_path": "data/x/10gs/10gs_protein.pdb",
            "sample_steps": 50,
            "sample_time_power": 1.5,
        }
    )

    assert args.protein_path == Path("data/x/10gs/10gs_protein.pdb")
    assert args.sample_steps == 50
    assert args.sample_time_power == pytest.approx(1.5)


def test_write_experiment_log_records_run_metadata(tmp_path: Path) -> None:
    class _LogArgs(_Args):
        seed = 7
        steps = 5
        sample_steps = 10
        crop_cutoff = 8.0
        edge_cutoff = 4.5
        context_policy = "gated"
        protein_node_budget = 256
        use_cross_interface_block = True

    log_path = tmp_path / "experiment.md"
    output_path = tmp_path / "sample.pdb"
    trajectory_path = tmp_path / "trajectory.pdb"
    loss_csv_path = tmp_path / "loss.csv"
    plot_path = tmp_path / "plot.png"

    write_experiment_log(
        log_path,
        command="uv run python -m equidock_diff.train --steps 5",
        device=torch.device("cpu"),
        graph_source="real_pair",
        args=_LogArgs(),
        loss_rows=[(1, 1.5, 0.2), (5, 0.25, 0.9)],
        training_seconds=0.42,
        node_count=147,
        edge_count=2992,
        sample_path=output_path,
        trajectory_path=trajectory_path,
        loss_csv_path=loss_csv_path,
        plot_path=plot_path,
        extra_metrics={"aligned_ligand_rmsd": 1.2345},
        resolved_crop_cutoff=9.25,
        retained_protein_nodes=123,
    )

    contents = log_path.read_text(encoding="utf-8")
    assert "uv run python -m equidock_diff.train --steps 5" in contents
    assert "- Seed: `7`" in contents
    assert "- Graph source: `real_pair`" in contents
    assert "- Node count: `147`" in contents
    assert "- Edge count: `2992`" in contents
    assert "- Final loss: `0.250000` at step `5`" in contents
    assert "- Context policy: `gated`" in contents
    assert "- Use cross interface block: `True`" in contents
    assert "- Protein node budget: `256`" in contents
    assert "- Retained protein nodes: `123`" in contents
    assert "- Resolved crop cutoff: `9.250000`" in contents
    assert "- Aligned Ligand Rmsd: `1.234500`" in contents
    assert str(output_path) in contents


def test_write_experiment_log_omits_optional_artifacts_when_absent(tmp_path: Path) -> None:
    log_path = tmp_path / "experiment.md"

    write_experiment_log(
        log_path,
        command="uv run python -m equidock_diff.train --steps 5",
        device=torch.device("cpu"),
        graph_source="real_pair",
        args=_Args(),
        loss_rows=[(1, 1.5, 0.2), (5, 0.25, 0.9)],
        training_seconds=0.42,
        node_count=147,
        edge_count=2992,
        sample_path=None,
        trajectory_path=None,
        loss_csv_path=None,
        plot_path=None,
    )

    contents = log_path.read_text(encoding="utf-8")
    assert "Sample artifact" not in contents
    assert "Trajectory artifact" not in contents
    assert "Loss CSV" not in contents


def test_ligand_mask_from_features_uses_indicator_column() -> None:
    node_features = torch.tensor(
        [
            [0.1, 0.0, 1.0],
            [0.2, 0.0, 1.0],
            [0.3, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    mask = ligand_mask_from_features(node_features)

    assert mask.tolist() == [True, True, False]


def test_write_ligand_artifacts_filters_to_ligand_nodes(tmp_path: Path) -> None:
    node_features = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    sampled_positions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    trajectory = [
        sampled_positions.clone(),
        sampled_positions + 1.0,
    ]
    ligand_output = tmp_path / "ligand_sample.pdb"
    ligand_traj_output = tmp_path / "ligand_traj.pdb"

    written_sample, written_trajectory = write_ligand_artifacts(
        node_features=node_features,
        sampled_positions=sampled_positions,
        trajectory=trajectory,
        ligand_output=ligand_output,
        ligand_trajectory_output=ligand_traj_output,
    )

    assert written_sample == ligand_output
    assert written_trajectory == ligand_traj_output
    sample_lines = ligand_output.read_text(encoding="utf-8").splitlines()
    trajectory_lines = ligand_traj_output.read_text(encoding="utf-8").splitlines()
    assert len([line for line in sample_lines if line.startswith("ATOM")]) == 2
    assert len([line for line in trajectory_lines if line.startswith("MODEL")]) == 2
    assert "   9.000" not in ligand_output.read_text(encoding="utf-8")


def test_parser_snr_consistent_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["--snr-consistent", "--snr-mode", "full"])
    assert args.snr_consistent is True
    assert args.snr_mode == "full"


def test_noised_positions_snr_consistent_variance() -> None:
    torch.manual_seed(42)
    clean_positions = torch.zeros(50000, 3, dtype=torch.float32)
    t = torch.tensor([0.5], dtype=torch.float32)
    beta_min = 0.1
    beta_max = 2.0

    noised, beta_t = noised_positions_for_schedule(
        clean_positions,
        t,
        beta_min=beta_min,
        beta_max=beta_max,
        noise_schedule="linear",
        cosine_offset=0.008,
        cosine_nu=1.5,
        snr_consistent=True,
    )

    # Under clean_positions = 0, noised variance should equal 1 - alpha_bar(t)
    alpha_bar = torch.exp(-(beta_min * t + 0.5 * (beta_max - beta_min) * t**2)).item()
    expected_std = (1.0 - alpha_bar) ** 0.5
    observed_std = noised.std().item()
    assert observed_std == pytest.approx(expected_std, rel=0.02)


def test_sample_positions_snr_consistent_ancestral_sampler() -> None:
    torch.manual_seed(42)
    num_nodes = 8
    node_features = torch.tensor(
        [
            [6.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # protein
            [6.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # protein
            [6.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # protein
            [6.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # protein
            [6.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ligand
            [7.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ligand
            [8.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ligand
            [6.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ligand
        ],
        dtype=torch.float32,
    )
    reference_positions = torch.randn(num_nodes, 3, dtype=torch.float32)
    edge_index = torch.empty((2, 0), dtype=torch.long)

    class DummyScoreNet(torch.nn.Module):
        def forward(self, h, pos, edge_idx, t):
            # Dummy displacement prediction: pull toward reference
            return reference_positions - pos

    model = DummyScoreNet()
    device = torch.device("cpu")

    sampled, traj, _ = sample_positions(
        model=model,
        node_features=node_features,
        edge_index=edge_index,
        num_nodes=num_nodes,
        device=device,
        sample_steps=5,
        beta_min=0.1,
        beta_max=2.0,
        score_clip=10.0,
        position_clip=50.0,
        noise_schedule="cosine",
        reference_positions=reference_positions,
        anchor_protein=True,
        snr_consistent=True,
    )

    assert torch.isfinite(sampled).all()
    assert len(traj) == 6
    # Protein nodes (first 4) must strictly match reference_positions
    assert torch.allclose(sampled[:4], reference_positions[:4], atol=1e-5)
