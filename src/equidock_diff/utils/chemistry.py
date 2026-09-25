"""Chemistry utilities for ligand graph featurization (SDF-only contract)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

# Keep these immutable to avoid accidental drift.
ATOM_SYMBOLS: tuple[str, ...] = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "OTHER")
CHIRALITY_CLASSES: tuple[str, ...] = (
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_OTHER",
)
BOND_TYPES: tuple[str, ...] = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
BOND_STEREO_CLASSES: tuple[str, ...] = (
    "STEREONONE",
    "STEREOZ",
    "STEREOE",
    "STEREOANY_OR_OTHER",
)
ATOM_SYMBOL_TO_INDEX = {symbol: idx for idx, symbol in enumerate(ATOM_SYMBOLS[:-1])}
CHIRALITY_TO_INDEX = {
    chirality: idx for idx, chirality in enumerate(CHIRALITY_CLASSES[:-1])
}
BOND_TYPE_TO_INDEX = {bond_type: idx for idx, bond_type in enumerate(BOND_TYPES)}
BOND_STEREO_TO_INDEX = {
    stereo: idx for idx, stereo in enumerate(BOND_STEREO_CLASSES[:-1])
}

ATOM_FEATURE_DIM = 16
BOND_FEATURE_DIM = 8

# Approximate van der Waals radii in Angstrom used for soft clash checks.
ATOM_CLASH_RADII = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
    "OTHER": 1.70,
}

SkipReason = Literal[
    "parse_failed",
    "sanitize_failed",
    "no_conformer",
    "empty_after_h_removal",
    "no_bonds",
]
RawSkipReason = SkipReason | Literal["unsupported_extension"]


@dataclass(frozen=True)
class LigandGraph:
    """Torch tensors for one ligand graph."""

    x: torch.Tensor  # [N, 16] float32
    pos: torch.Tensor  # [N, 3] float32
    edge_index: torch.Tensor  # [2, E] int64
    edge_attr: torch.Tensor  # [E, 8] float32


@dataclass(frozen=True)
class FeaturizeOutcome:
    """Success contains graph; skip contains skip_reason."""

    graph: LigandGraph | None = None
    skip_reason: SkipReason | None = None

    @property
    def skipped(self) -> bool:
        return self.graph is None


@dataclass
class LigandSkipCounter:

    parse_failed: int = 0
    sanitize_failed: int = 0
    no_conformer: int = 0
    empty_after_h_removal: int = 0
    no_bonds: int = 0
    extra: dict[str, int] = field(default_factory=dict)

    def add(self, reason: str) -> None:
        if hasattr(self, reason):
            setattr(self, reason, getattr(self, reason) + 1)
            return
        # Helpful while iterating (e.g. unsupported_extension before mapping to parse_failed).
        self.extra[reason] = self.extra.get(reason, 0) + 1

    def report(self) -> dict[str, int]:
        return {
            "parse_failed": self.parse_failed,  # RDKit parsing or file I/O error
            "sanitize_failed": self.sanitize_failed,  # RDKit sanitization or H-removal failed
            "no_conformer": self.no_conformer,  # Missing or invalid 3D coordinates
            "empty_after_h_removal": self.empty_after_h_removal,  # No heavy atoms found after H removal
            "no_bonds": self.no_bonds,  # No bonds found in molecule
            **self.extra,  # Unmapped errors (e.g., unsupported_extension)
        }


def _skip(reason: RawSkipReason, counter: LigandSkipCounter | None) -> FeaturizeOutcome:
    normalized_reason: SkipReason
    if reason == "unsupported_extension":
        normalized_reason = "parse_failed"
    else:
        normalized_reason = reason
    if counter is not None:
        counter.add(normalized_reason)
    return FeaturizeOutcome(skip_reason=normalized_reason)


def _normalize_atom_symbol(symbol: str) -> str:
    cleaned = "".join(ch for ch in symbol.strip() if ch.isalpha())
    if not cleaned:
        return "OTHER"
    if len(cleaned) >= 2:
        candidate = cleaned[0].upper() + cleaned[1].lower()
        if candidate in ATOM_SYMBOL_TO_INDEX:
            return candidate
    candidate = cleaned[0].upper()
    if candidate in ATOM_SYMBOL_TO_INDEX:
        return candidate
    return "OTHER"


def _build_graph_from_raw_mol(
    atom_symbols: list[str],
    coords: list[list[float]],
    bonds: list[tuple[int, int, str]],
    *,
    skip_counter: LigandSkipCounter | None = None,
) -> FeaturizeOutcome:
    keep_indices = [idx for idx, symbol in enumerate(atom_symbols) if _normalize_atom_symbol(symbol) != "H"]
    if not keep_indices:
        return _skip("empty_after_h_removal", skip_counter)

    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
    filtered_symbols = [atom_symbols[idx] for idx in keep_indices]
    filtered_coords = [coords[idx] for idx in keep_indices]

    filtered_bonds = []
    for begin_idx, end_idx, bond_type_name in bonds:
        if begin_idx not in index_map or end_idx not in index_map:
            continue
        filtered_bonds.append(
            (index_map[begin_idx], index_map[end_idx], bond_type_name)
        )
    if not filtered_bonds:
        return _skip("no_bonds", skip_counter)

    num_atoms = len(filtered_symbols)
    x = torch.zeros((num_atoms, ATOM_FEATURE_DIM), dtype=torch.float32)
    other_atom_index = len(ATOM_SYMBOLS) - 1

    for atom_idx, atom_symbol in enumerate(filtered_symbols):
        atom_type_index = ATOM_SYMBOL_TO_INDEX.get(
            _normalize_atom_symbol(atom_symbol),
            other_atom_index,
        )
        x[atom_idx, atom_type_index] = 1.0
        x[atom_idx, 12] = 1.0

    pos = torch.tensor(filtered_coords, dtype=torch.float32)
    if pos.shape != (num_atoms, 3) or not torch.isfinite(pos).all():
        return _skip("no_conformer", skip_counter)

    num_edges = 2 * len(filtered_bonds)
    edge_index = torch.empty((2, num_edges), dtype=torch.int64)
    edge_attr = torch.zeros((num_edges, BOND_FEATURE_DIM), dtype=torch.float32)

    edge_ptr = 0
    for begin_idx, end_idx, bond_type_name in filtered_bonds:
        bond_type_index = BOND_TYPE_TO_INDEX.get(bond_type_name)
        if bond_type_index is None:
            return _skip("parse_failed", skip_counter)

        edge_index[0, edge_ptr] = begin_idx
        edge_index[1, edge_ptr] = end_idx
        edge_attr[edge_ptr, bond_type_index] = 1.0
        edge_attr[edge_ptr, 4] = 1.0
        edge_ptr += 1

        edge_index[0, edge_ptr] = end_idx
        edge_index[1, edge_ptr] = begin_idx
        edge_attr[edge_ptr, bond_type_index] = 1.0
        edge_attr[edge_ptr, 4] = 1.0
        edge_ptr += 1

    return FeaturizeOutcome(
        graph=LigandGraph(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
    )


def _featurize_ligand_without_rdkit(
    path: Path,
    *,
    skip_counter: LigandSkipCounter | None = None,
) -> FeaturizeOutcome:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(lines) < 3:
            return _skip("parse_failed", skip_counter)

        counts_idx = None
        atom_count = 0
        bond_count = 0
        for idx, line in enumerate(lines[:10]):
            try:
                atom_count = int(line[0:3].strip())
                bond_count = int(line[3:6].strip())
            except Exception:
                continue
            counts_idx = idx
            break
        if counts_idx is None:
            return _skip("parse_failed", skip_counter)
        if atom_count <= 0 or bond_count <= 0:
            return _skip("parse_failed", skip_counter)

        atom_start = counts_idx + 1
        atom_lines = lines[atom_start : atom_start + atom_count]
        bond_start = atom_start + atom_count
        bond_lines = lines[bond_start : bond_start + bond_count]
        if len(atom_lines) != atom_count or len(bond_lines) != bond_count:
            return _skip("parse_failed", skip_counter)

        atom_symbols: list[str] = []
        coords: list[list[float]] = []
        for line in atom_lines:
            coords.append(
                [
                    float(line[0:10].strip()),
                    float(line[10:20].strip()),
                    float(line[20:30].strip()),
                ]
            )
            atom_symbols.append(line[31:34].strip())

        bond_type_names = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 4: "AROMATIC"}
        bonds: list[tuple[int, int, str]] = []
        for line in bond_lines:
            begin_idx = int(line[0:3].strip()) - 1
            end_idx = int(line[3:6].strip()) - 1
            bond_type_code = int(line[6:9].strip())
            bond_type_name = bond_type_names.get(bond_type_code)
            if bond_type_name is None:
                return _skip("parse_failed", skip_counter)
            bonds.append((begin_idx, end_idx, bond_type_name))
    except Exception:
        return _skip("parse_failed", skip_counter)

    return _build_graph_from_raw_mol(
        atom_symbols,
        coords,
        bonds,
        skip_counter=skip_counter,
    )


def featurize_ligand(
    sdf_path: str | Path,
    *,
    skip_counter: LigandSkipCounter | None = None,
) -> FeaturizeOutcome:
    """Parse one `.sdf` ligand and return graph tensors.

    Note:
    - `sanitize_failed` is also used for H-removal failures because those happen in the
      same chemistry-normalization stage before tensor construction.
    """
    path = Path(sdf_path)
    if path.suffix.lower() != ".sdf":
        return _skip("unsupported_extension", skip_counter)

    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        return _featurize_ligand_without_rdkit(path, skip_counter=skip_counter)

    # Parse first valid molecule entry from SDF with sanitize disabled.
    try:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    except Exception:
        return _skip("parse_failed", skip_counter)

    if supplier is None:
        return _skip("parse_failed", skip_counter)

    mol = None
    try:
        for candidate in supplier:
            if candidate is not None:
                mol = candidate
                break
    except Exception:
        return _skip("parse_failed", skip_counter)

    if mol is None:
        return _skip("parse_failed", skip_counter)

    # Explicit sanitization phase, separate from parsing to preserve skip reason.
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return _skip("sanitize_failed", skip_counter)

    try:
        mol = Chem.RemoveHs(mol, sanitize=False)
    except Exception:
        return _skip("sanitize_failed", skip_counter)

    if mol.GetNumAtoms() == 0:
        return _skip("empty_after_h_removal", skip_counter)
    if mol.GetNumBonds() == 0:
        return _skip("no_bonds", skip_counter)

    try:
        if mol.GetNumConformers() == 0:
            return _skip("no_conformer", skip_counter)
        conformer = mol.GetConformer()
        # This RDKit build exposes Is3D() but not GetNumDimensions().
        is_3d = bool(conformer.Is3D()) if hasattr(conformer, "Is3D") else True
        if not is_3d:
            return _skip("no_conformer", skip_counter)
        if conformer.GetNumAtoms() != mol.GetNumAtoms():  # indexes not aligned
            return _skip("no_conformer", skip_counter)

        coords: list[list[float]] = []
        for i in range(mol.GetNumAtoms()):
            atom_pos = conformer.GetAtomPosition(i)
            coords.append([float(atom_pos.x), float(atom_pos.y), float(atom_pos.z)])

        pos = torch.tensor(coords, dtype=torch.float32)
        if pos.shape != (mol.GetNumAtoms(), 3):
            return _skip("no_conformer", skip_counter)
        if not torch.isfinite(pos).all():
            return _skip("no_conformer", skip_counter)
    except Exception:
        return _skip("no_conformer", skip_counter)

    num_atoms = mol.GetNumAtoms()
    x = torch.zeros((num_atoms, ATOM_FEATURE_DIM), dtype=torch.float32)
    other_atom_index = len(ATOM_SYMBOLS) - 1
    other_chirality_index = len(CHIRALITY_CLASSES) - 1

    for atom_idx, atom in enumerate(mol.GetAtoms()):
        atom_type_index = ATOM_SYMBOL_TO_INDEX.get(atom.GetSymbol(), other_atom_index)
        x[atom_idx, atom_type_index] = 1.0
        x[atom_idx, 10] = float(atom.GetIsAromatic())
        x[atom_idx, 11] = float(atom.GetFormalCharge())

        chiral_tag = atom.GetChiralTag().name
        chirality_index = CHIRALITY_TO_INDEX.get(chiral_tag, other_chirality_index)
        x[atom_idx, 12 + chirality_index] = 1.0

    num_bonds = mol.GetNumBonds()
    num_edges = 2 * num_bonds
    edge_index = torch.empty((2, num_edges), dtype=torch.int64)
    edge_attr = torch.zeros((num_edges, BOND_FEATURE_DIM), dtype=torch.float32)
    other_stereo_index = len(BOND_STEREO_CLASSES) - 1

    edge_ptr = 0
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        bond_type_name = bond.GetBondType().name
        bond_type_index = BOND_TYPE_TO_INDEX.get(bond_type_name)
        if bond_type_index is None:
            return _skip("parse_failed", skip_counter)

        stereo_name = bond.GetStereo().name
        stereo_index = BOND_STEREO_TO_INDEX.get(stereo_name, other_stereo_index)

        edge_index[0, edge_ptr] = begin_idx
        edge_index[1, edge_ptr] = end_idx
        edge_attr[edge_ptr, bond_type_index] = 1.0
        edge_attr[edge_ptr, 4 + stereo_index] = 1.0
        edge_ptr += 1

        edge_index[0, edge_ptr] = end_idx
        edge_index[1, edge_ptr] = begin_idx
        edge_attr[edge_ptr, bond_type_index] = 1.0
        edge_attr[edge_ptr, 4 + stereo_index] = 1.0
        edge_ptr += 1

    if x.shape != (num_atoms, ATOM_FEATURE_DIM):
        return _skip("parse_failed", skip_counter)
    if pos.shape != (num_atoms, 3):
        return _skip("no_conformer", skip_counter)
    if edge_index.shape[0] != 2:
        return _skip("parse_failed", skip_counter)
    if edge_attr.shape != (edge_index.shape[1], BOND_FEATURE_DIM):
        return _skip("parse_failed", skip_counter)
    if not torch.isfinite(x).all():
        return _skip("parse_failed", skip_counter)
    if not torch.isfinite(pos).all():
        return _skip("no_conformer", skip_counter)
    if not torch.isfinite(edge_attr).all():
        return _skip("parse_failed", skip_counter)

    return FeaturizeOutcome(
        graph=LigandGraph(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
    )


@dataclass(frozen=True)
class ChemicalHealthReport:
    total_atoms: int
    ligand_atoms: int
    protein_atoms: int
    clash_count: int
    clash_fraction: float
    bond_violation_count: int
    max_bond_deviation: float
    is_valid: bool


def evaluate_steric_clashes(
    positions: torch.Tensor,
    node_features: torch.Tensor,
    clash_ratio_threshold: float = 0.75,
    clash_margin: float = 0.2,
) -> tuple[int, float, list[tuple[int, int, float]]]:
    """Evaluate steric clashes between ligand and protein atoms.

    A clash occurs when the inter-atomic distance is smaller than the sum of
    their van der Waals radii scaled by clash_ratio_threshold or within clash_margin.
    """
    if positions.size(0) == 0 or node_features.size(0) == 0:
        return 0, 0.0, []

    is_ligand = node_features[:, -1] > 0.5
    is_protein = ~is_ligand

    if not is_ligand.any() or not is_protein.any():
        return 0, 0.0, []

    ligand_indices = torch.nonzero(is_ligand, as_tuple=True)[0]
    protein_indices = torch.nonzero(is_protein, as_tuple=True)[0]

    # Assign default clash radii based on one-hot symbol if available
    radii = torch.full((positions.size(0),), ATOM_CLASH_RADII["C"], dtype=positions.dtype, device=positions.device)
    for idx, sym in enumerate(ATOM_SYMBOLS[:-1]):
        if idx < node_features.size(1):
            mask = node_features[:, idx] > 0.5
            if mask.any():
                radii[mask] = ATOM_CLASH_RADII.get(sym, ATOM_CLASH_RADII["OTHER"])

    lig_pos = positions[ligand_indices]
    prot_pos = positions[protein_indices]
    lig_radii = radii[ligand_indices]
    prot_radii = radii[protein_indices]

    dist_matrix = torch.cdist(lig_pos, prot_pos)
    sum_radii = lig_radii.unsqueeze(1) + prot_radii.unsqueeze(0)
    clash_thresholds = torch.maximum(sum_radii * clash_ratio_threshold, sum_radii - clash_margin)

    clashes = dist_matrix < clash_thresholds
    clash_pairs: list[tuple[int, int, float]] = []
    if clashes.any():
        viol_indices = torch.nonzero(clashes, as_tuple=False)
        for v in viol_indices:
            l_idx = int(ligand_indices[v[0]].item())
            p_idx = int(protein_indices[v[1]].item())
            d = float(dist_matrix[v[0], v[1]].item())
            clash_pairs.append((l_idx, p_idx, d))

    clash_count = len(clash_pairs)
    unique_clashing_ligands = len(set(p[0] for p in clash_pairs))
    total_ligand_atoms = int(is_ligand.sum().item())
    clash_fraction = unique_clashing_ligands / max(total_ligand_atoms, 1)
    return clash_count, clash_fraction, clash_pairs


def evaluate_bond_lengths(
    positions: torch.Tensor,
    ligand_bond_index: torch.Tensor | None,
    reference_positions: torch.Tensor | None = None,
    min_allowed_len: float = 0.85,
    max_allowed_len: float = 2.30,
    max_allowed_strain_pct: float = 0.35,
) -> tuple[int, float]:
    """Evaluate covalent bond length validity against biophysical limits and reference poses."""
    if ligand_bond_index is None or ligand_bond_index.numel() == 0:
        return 0, 0.0

    src, dst = ligand_bond_index[0], ligand_bond_index[1]
    keep = src < dst
    if not keep.any():
        return 0, 0.0
    src = src[keep]
    dst = dst[keep]

    lengths = torch.linalg.norm(positions[src] - positions[dst], dim=-1)
    violations = (lengths < min_allowed_len) | (lengths > max_allowed_len)

    max_dev = 0.0
    if reference_positions is not None:
        ref_lengths = torch.linalg.norm(reference_positions[src] - reference_positions[dst], dim=-1)
        strain = torch.abs(lengths - ref_lengths) / torch.clamp_min(ref_lengths, 1e-6)
        violations = violations | (strain > max_allowed_strain_pct)
        max_dev = float(torch.max(torch.abs(lengths - ref_lengths)).item()) if ref_lengths.numel() > 0 else 0.0
    else:
        # Deviation from nominal 1.45 Å bond length
        nominal_diff = torch.abs(lengths - 1.45)
        max_dev = float(torch.max(nominal_diff).item()) if lengths.numel() > 0 else 0.0

    return int(violations.sum().item()), max_dev


def evaluate_chemical_validity(
    positions: torch.Tensor,
    node_features: torch.Tensor,
    ligand_bond_index: torch.Tensor | None = None,
    reference_positions: torch.Tensor | None = None,
    clash_ratio_threshold: float = 0.75,
    clash_margin: float = 0.2,
) -> ChemicalHealthReport:
    """Approximate in-loop chemical diagnostics.

    This is *not* a validity verdict: measured 2026-09-25 it reported every crystal pose on
    the dissertation panel as invalid, because hydrogens carry a carbon radius and the clash
    threshold takes the stricter of two bounds. Use ``equidock_diff.pose_validity`` (the
    reference PoseBusters implementation, exposed as ``kiral validate``) for validity, and
    this for cheap per-step signals.
    """
    is_ligand = node_features[:, -1] > 0.5
    ligand_atoms = int(is_ligand.sum().item())
    protein_atoms = int((~is_ligand).sum().item())
    total_atoms = positions.size(0)

    clash_count, clash_fraction, _ = evaluate_steric_clashes(
        positions,
        node_features,
        clash_ratio_threshold=clash_ratio_threshold,
        clash_margin=clash_margin,
    )
    bond_violations, max_bond_dev = evaluate_bond_lengths(
        positions,
        ligand_bond_index,
        reference_positions=reference_positions,
    )
    is_valid = (clash_count == 0) and (bond_violations == 0)

    return ChemicalHealthReport(
        total_atoms=total_atoms,
        ligand_atoms=ligand_atoms,
        protein_atoms=protein_atoms,
        clash_count=clash_count,
        clash_fraction=clash_fraction,
        bond_violation_count=bond_violations,
        max_bond_deviation=max_bond_dev,
        is_valid=is_valid,
    )
