"""Equivalence tests for batched training versus single-complex training."""

from __future__ import annotations

import copy
from pathlib import Path
import pytest
import torch

from kiral.train import (
    build_parser,
    build_dataset_examples,
    load_dataset_example,
    stack_dataset_examples,
    make_model_for_node_dim,
    training_step_with_breakdown,
    ligand_bond_length_loss,
    ligand_shape_loss,
    ligand_protein_clash_loss,
    ligand_protein_contact_loss,
    noised_positions_for_schedule,
    infer_ligand_mask,
)


@pytest.fixture(scope="module")
def panel8_graphs():
    data_root = Path.home() / "data" / "pdbbind_v2020"
    if not data_root.exists():
        pytest.skip(f"PDBbind data root not found at {data_root}")

    args = build_parser().parse_args([
        "--dataset-root", str(data_root),
        "--dataset-cache-dir", "data/.cache",
        "--frame-hetero-backbone",
    ])
    device = torch.device("cpu")
    examples = build_dataset_examples(args)[:8]
    if len(examples) < 8:
        pytest.skip("Less than 8 complexes available in dataset")

    graphs = []
    for ex in examples:
        loaded = load_dataset_example(ex, args, device)
        graphs.append({
            "complex_id": ex.complex_id,
            "feat": loaded[0],
            "pos": loaded[1],
            "edge": loaded[2],
            "bond": loaded[3],
        })
    return graphs, args, device


def test_batch_8_loss_and_gradients_match_batch_1_mean(panel8_graphs):
    """The critical acceptance criterion:
    The loss and gradients for a fixed set of 8 complexes at fixed weights,
    computed at batch 8, equals the mean of the 8 batch-1 losses/gradients within tolerance.
    """
    graphs, args, device = panel8_graphs

    # Loss weights for all 4 geometry terms
    args.ligand_bond_weight = 0.1
    args.ligand_shape_weight = 0.05
    args.ligand_protein_clash_weight = 0.05
    args.ligand_protein_contact_weight = 0.05
    args.noise_schedule = "cosine"

    torch.manual_seed(42)
    t_scalars = [torch.tensor(0.1 * (i + 1)) for i in range(8)]
    noised_list = []
    target_list = []

    for i, g in enumerate(graphs):
        clean_pos = g["pos"]
        lig_mask = infer_ligand_mask(g["feat"])
        noised_pos = clean_pos.clone()
        noised_lig, _ = noised_positions_for_schedule(
            clean_pos[lig_mask],
            t_scalars[i],
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            noise_schedule=args.noise_schedule,
            cosine_offset=args.cosine_offset,
            cosine_nu=args.cosine_nu,
            snr_consistent=False,
        )
        noised_pos[lig_mask] = noised_lig
        target_score = torch.zeros_like(clean_pos)
        target_score[lig_mask] = clean_pos[lig_mask] - noised_lig
        noised_list.append(noised_pos)
        target_list.append(target_score)

    torch.manual_seed(1001)
    model_single = make_model_for_node_dim(args, device, node_dim=graphs[0]["feat"].size(-1))
    model_batched = copy.deepcopy(model_single)

    # 1. Compute sequential batch-1 passes with loss accumulation
    losses_single = []
    model_single.zero_grad()
    for i, g in enumerate(graphs):
        clean_pos = g["pos"]
        noised_pos = noised_list[i]
        target_score = target_list[i]
        t = t_scalars[i]
        lig_mask = infer_ligand_mask(g["feat"])

        pred_score = model_single(g["feat"], noised_pos, g["edge"], t)
        pred_score[~lig_mask] = 0.0
        score_loss = torch.mean((pred_score[lig_mask] - target_score[lig_mask]) ** 2)
        pred_pos = noised_pos + pred_score
        bond_loss = ligand_bond_length_loss(pred_pos, clean_pos, g["bond"])
        shape_loss = ligand_shape_loss(pred_pos, clean_pos, lig_mask)
        clash_loss = ligand_protein_clash_loss(pred_pos, g["feat"], lig_mask)
        contact_loss = ligand_protein_contact_loss(pred_pos, g["feat"], lig_mask)

        loss_c = (
            score_loss
            + args.ligand_bond_weight * bond_loss
            + args.ligand_shape_weight * shape_loss
            + args.ligand_protein_clash_weight * clash_loss
            + args.ligand_protein_contact_weight * contact_loss
        )
        losses_single.append(loss_c)
        # Scale backward by 1/8 so gradients accumulate to the mean
        (loss_c / len(graphs)).backward()

    single_mean_loss = torch.stack(losses_single).mean().item()
    single_grads = {
        name: p.grad.clone()
        for name, p in model_single.named_parameters()
        if p.grad is not None
    }

    # 2. Compute batched pass using stack_dataset_examples
    loaded_tuples = [
        (g["feat"], noised_list[i], g["edge"], g["bond"], None, None)
        for i, g in enumerate(graphs)
    ]
    clean_tuples = [
        (g["feat"], g["pos"], g["edge"], g["bond"], None, None)
        for i, g in enumerate(graphs)
    ]
    (
        node_features,
        noised_positions,
        edge_index,
        batch_index,
        complex_slices,
        ligand_bond_indices,
    ) = stack_dataset_examples(loaded_tuples)

    _, clean_positions, _, _, _, _ = stack_dataset_examples(clean_tuples)

    t_tensor = torch.tensor([float(t) for t in t_scalars])

    model_batched.zero_grad()
    # Run training_step_with_breakdown on the batched graph
    batched_loss, beta_t, score_loss, bond_loss, shape_loss, clash_loss, contact_loss = (
        training_step_with_breakdown(
            model_batched,
            node_features,
            clean_positions,
            edge_index,
            None,
            args.beta_min,
            args.beta_max,
            ligand_bond_weight=args.ligand_bond_weight,
            ligand_shape_weight=args.ligand_shape_weight,
            ligand_protein_clash_weight=args.ligand_protein_clash_weight,
            ligand_protein_contact_weight=args.ligand_protein_contact_weight,
            frame_hetero_backbone=args.frame_hetero_backbone,
            noise_schedule=args.noise_schedule,
            cosine_offset=args.cosine_offset,
            cosine_nu=args.cosine_nu,
            snr_consistent=False,
            batch_index=batch_index,
            complex_slices=complex_slices,
            ligand_bond_indices=ligand_bond_indices,
            t=t_tensor,
            noised_positions=noised_positions,
        )
    )
    batched_loss.backward()

    batched_grads = {
        name: p.grad.clone()
        for name, p in model_batched.named_parameters()
        if p.grad is not None
    }

    # Scalar loss equivalence:
    assert abs(single_mean_loss - batched_loss.item()) < 1e-5, (
        f"Loss mismatch: single mean {single_mean_loss} vs batched {batched_loss.item()}"
    )

    # Gradient equivalence across all parameters:
    for name in single_grads:
        assert torch.allclose(
            single_grads[name], batched_grads[name], rtol=1e-3, atol=1e-4
        ), f"Gradient mismatch for {name}: max diff = {(single_grads[name] - batched_grads[name]).abs().max().item()}"
