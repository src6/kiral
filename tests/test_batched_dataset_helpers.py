def test_dataset_examples_for_step_and_stacking():
    from kiral.train import (
        ProteinLigandPaths,
        dataset_examples_for_step,
        stack_dataset_examples,
    )
    from pathlib import Path
    import torch

    examples = [
        ProteinLigandPaths(protein_path=Path(f"p{i}"), ligand_path=Path(f"l{i}"))
        for i in range(10)
    ]
    # Check step 1 with batch_size 4
    batch = dataset_examples_for_step(examples, 1, batch_size=4)
    assert [e.complex_id for e in batch] == ["p0", "p1", "p2", "p3"]

    # Check step 3 with batch_size 4 (start = 8, wraps around: 8, 9, 0, 1)
    batch3 = dataset_examples_for_step(examples, 3, batch_size=4)
    assert [e.complex_id for e in batch3] == ["p8", "p9", "p0", "p1"]

    # Test stacking dummy graphs
    g1 = (
        torch.randn(3, 4),
        torch.randn(3, 3),
        torch.tensor([[0, 1], [1, 2]]),
        torch.tensor([[0], [1]]),
        None,
        None,
    )
    g2 = (
        torch.randn(2, 4),
        torch.randn(2, 3),
        torch.tensor([[0], [1]]),
        None,
        None,
        None,
    )
    feat, pos, edges, batch_idx, slices, bonds = stack_dataset_examples([g1, g2])
    assert feat.size(0) == 5
    assert pos.size(0) == 5
    assert edges.size(1) == 3
    # Second graph edge should be offset by 3: [[0+3], [1+3]] -> [[3], [4]]
    assert edges[:, 2].tolist() == [3, 4]
    assert batch_idx.tolist() == [0, 0, 0, 1, 1]
    assert slices == [(0, 3), (3, 5)]
    assert bonds[0].equal(g1[3])
    assert bonds[1] is None
