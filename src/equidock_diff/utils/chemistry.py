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
