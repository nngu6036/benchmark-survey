from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np
import re

try:  # pragma: no cover - availability depends on environment.
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None  # type: ignore


_VALID_ATOMIC_NUMBERS = set(range(1, 119))
_QM9_CLASS_TO_ATOMIC_NUMBER = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9}
_COMMON_VALENCE = {1: 1.0, 5: 3.0, 6: 4.0, 7: 3.0, 8: 2.0, 9: 1.0, 15: 5.0, 16: 6.0, 17: 1.0, 35: 1.0, 53: 1.0}


@dataclass
class MolecularEvaluationResult:
    valid_flags: list[bool]
    smiles: list[str | None]
    validity_backend: str
    rdkit_available: bool
    construction_failures: int


def _is_integer_like_value(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return float(value).is_integer()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return False
        return stripped.lstrip("+-").isdigit()
    return False


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        return int(round(float(value)))
    except Exception:
        return None


def _periodic_atomic_number(value: Any) -> int | None:
    if value is None:
        return None
    ivalue = _as_int(value)
    if ivalue in _VALID_ATOMIC_NUMBERS:
        return ivalue
    if Chem is None:
        return None
    try:
        symbol = str(value).strip()
        if not symbol:
            return None
        atom = Chem.Atom(symbol)
        atomic_number = int(atom.GetAtomicNum())
        return atomic_number if atomic_number in _VALID_ATOMIC_NUMBERS else None
    except Exception:
        return None


def _raw_value_from_canonical(label: Any, raw_values: Sequence[str] | None, *, offset: int = 0) -> Any:
    idx = _as_int(label)
    if raw_values and idx is not None:
        j = idx - int(offset)
        if 0 <= j < len(raw_values):
            return raw_values[j]
    return label


def atomic_number_from_graph_label(
    graph: nx.Graph,
    node: Any,
    *,
    node_label_attr: str,
    raw_node_label_values: Sequence[str] | None = None,
    dataset: str | None = None,
) -> int | None:
    """Resolve a graph node label into an RDKit atomic number.

    For ZINC, PyG's ``atom_type`` labels are categorical ids.  The benchmark
    therefore does **not** interpret integer ZINC category ids as atomic
    numbers unless an explicit ``atomic_number``/``z`` attribute is present or
    the raw label vocabulary contains non-numeric atom symbols such as ``C`` or
    ``Cl``.  This avoids false-positive RDKit validity on ZINC.
    """
    data = graph.nodes[node]
    for key in ("atomic_number", "z", "atomic_num"):
        atomic_number = _as_int(data.get(key))
        if atomic_number in _VALID_ATOMIC_NUMBERS:
            return int(atomic_number)

    dataset_key = str(dataset or "").lower()
    label = data.get(node_label_attr)
    label_int = _as_int(label)
    if dataset_key == "qm9" and label_int in _QM9_CLASS_TO_ATOMIC_NUMBER:
        return _QM9_CLASS_TO_ATOMIC_NUMBER[int(label_int)]

    raw = _raw_value_from_canonical(label, raw_node_label_values, offset=0)
    if dataset_key == "zinc":
        # PyG ZINC atom_type values are categorical ids, not guaranteed
        # periodic-table atomic numbers.  Reject bare numeric category values
        # unless the graph carries explicit atomic_number/z attributes above or
        # the caller supplies an explicit mapping value such as "atomic_number=6".
        if raw is None:
            return None
        raw_text = str(raw).strip()
        if not raw_text:
            return None
        lowered = raw_text.lower()
        if lowered.startswith(("atomic_number", "atomic_num", "z=")) or "atomic_number" in lowered or "atomic_num" in lowered:
            numbers = re.findall(r"[-+]?\d+", raw_text)
            for number in numbers:
                atomic_number = _as_int(number)
                if atomic_number in _VALID_ATOMIC_NUMBERS:
                    return int(atomic_number)
        if _is_integer_like_value(raw_text):
            return None
        for token in raw_text.replace(",", ":").replace("|", ":").replace("=", ":").split(":"):
            token = token.strip()
            if token and not _is_integer_like_value(token):
                atomic_number = _periodic_atomic_number(token)
                if atomic_number is not None:
                    return int(atomic_number)
        return None

    atomic_number = _periodic_atomic_number(raw)
    if atomic_number is not None:
        return atomic_number
    atomic_number = _periodic_atomic_number(label)
    if atomic_number is not None:
        return atomic_number
    return None


def bond_type_from_graph_label(edge_value: Any, raw_edge_label_values: Sequence[str] | None = None):
    raw = _raw_value_from_canonical(edge_value, raw_edge_label_values, offset=1)
    val = _as_int(raw)
    if Chem is None:
        if val == 1:
            return 1.0
        if val == 2:
            return 2.0
        if val == 3:
            return 3.0
        if val == 4:
            return 1.5
        return None
    if val == 1:
        return Chem.BondType.SINGLE
    if val == 2:
        return Chem.BondType.DOUBLE
    if val == 3:
        return Chem.BondType.TRIPLE
    if val == 4:
        return Chem.BondType.AROMATIC
    text = str(raw).strip().lower() if raw is not None else ""
    if text in {"single", "s", "bondtype.single"}:
        return Chem.BondType.SINGLE
    if text in {"double", "d", "bondtype.double"}:
        return Chem.BondType.DOUBLE
    if text in {"triple", "t", "bondtype.triple"}:
        return Chem.BondType.TRIPLE
    if text in {"aromatic", "a", "bondtype.aromatic"}:
        return Chem.BondType.AROMATIC
    return None


def graph_to_rdkit_mol(
    graph: nx.Graph,
    *,
    node_label_attr: str = "node_label",
    edge_label_attr: str = "edge_type",
    raw_node_label_values: Sequence[str] | None = None,
    raw_edge_label_values: Sequence[str] | None = None,
    dataset: str | None = None,
    sanitize: bool = True,
):
    if Chem is None:
        return None
    if not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
        return None
    if any(u == v for u, v in graph.edges()):
        return None
    mol = Chem.RWMol()
    node_to_idx: dict[Any, int] = {}
    aromatic_atom_indices: set[int] = set()
    for node in sorted(graph.nodes()):
        atomic_number = atomic_number_from_graph_label(
            graph,
            node,
            node_label_attr=node_label_attr,
            raw_node_label_values=raw_node_label_values,
            dataset=dataset,
        )
        if atomic_number is None:
            return None
        atom = Chem.Atom(int(atomic_number))
        node_to_idx[node] = int(mol.AddAtom(atom))
    for u, v, data in graph.edges(data=True):
        if u not in node_to_idx or v not in node_to_idx:
            return None
        bond_type = bond_type_from_graph_label(data.get(edge_label_attr, 1), raw_edge_label_values)
        if bond_type is None:
            return None
        ui, vi = node_to_idx[u], node_to_idx[v]
        if mol.GetBondBetweenAtoms(ui, vi) is not None:
            continue
        mol.AddBond(ui, vi, bond_type)
        if str(bond_type).upper().endswith("AROMATIC") or str(bond_type) == "AROMATIC":
            aromatic_atom_indices.update([ui, vi])
    for idx in aromatic_atom_indices:
        mol.GetAtomWithIdx(idx).SetIsAromatic(True)
    out = mol.GetMol()
    if sanitize:
        Chem.SanitizeMol(out)
    return out


def canonical_smiles_from_graph(
    graph: nx.Graph,
    *,
    node_label_attr: str = "node_label",
    edge_label_attr: str = "edge_type",
    raw_node_label_values: Sequence[str] | None = None,
    raw_edge_label_values: Sequence[str] | None = None,
    dataset: str | None = None,
) -> str | None:
    if Chem is None:
        return None
    try:
        mol = graph_to_rdkit_mol(
            graph,
            node_label_attr=node_label_attr,
            edge_label_attr=edge_label_attr,
            raw_node_label_values=raw_node_label_values,
            raw_edge_label_values=raw_edge_label_values,
            dataset=dataset,
            sanitize=True,
        )
        if mol is None:
            return None
        return str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))
    except Exception:
        return None


def _fallback_valence_valid(
    graph: nx.Graph,
    *,
    node_label_attr: str,
    edge_label_attr: str,
    raw_node_label_values: Sequence[str] | None,
    raw_edge_label_values: Sequence[str] | None,
    dataset: str | None,
) -> bool:
    if not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
        return False
    if any(u == v for u, v in graph.edges()):
        return False
    valence: dict[Any, float] = {}
    max_valence: dict[Any, float] = {}
    for node in graph.nodes():
        atomic_number = atomic_number_from_graph_label(
            graph,
            node,
            node_label_attr=node_label_attr,
            raw_node_label_values=raw_node_label_values,
            dataset=dataset,
        )
        if atomic_number not in _COMMON_VALENCE:
            return False
        valence[node] = 0.0
        max_valence[node] = float(_COMMON_VALENCE[atomic_number])
    for u, v, data in graph.edges(data=True):
        order = bond_type_from_graph_label(data.get(edge_label_attr, 1), raw_edge_label_values)
        if order is None:
            return False
        if Chem is not None:
            if order == Chem.BondType.SINGLE:
                numeric = 1.0
            elif order == Chem.BondType.DOUBLE:
                numeric = 2.0
            elif order == Chem.BondType.TRIPLE:
                numeric = 3.0
            elif order == Chem.BondType.AROMATIC:
                numeric = 1.5
            else:
                return False
        else:
            numeric = float(order)
        valence[u] += numeric
        valence[v] += numeric
    return all(valence[node] <= max_valence[node] + 1e-9 for node in valence)


def evaluate_molecular_validity(
    graphs: Sequence[nx.Graph],
    *,
    node_label_attr: str = "node_label",
    edge_label_attr: str = "edge_type",
    raw_node_label_values: Sequence[str] | None = None,
    raw_edge_label_values: Sequence[str] | None = None,
    dataset: str | None = None,
    fallback_to_valence: bool = True,
) -> MolecularEvaluationResult:
    valid_flags: list[bool] = []
    smiles: list[str | None] = []
    failures = 0
    used_fallback = False
    for graph in graphs:
        smi = canonical_smiles_from_graph(
            graph,
            node_label_attr=node_label_attr,
            edge_label_attr=edge_label_attr,
            raw_node_label_values=raw_node_label_values,
            raw_edge_label_values=raw_edge_label_values,
            dataset=dataset,
        )
        ok = smi is not None
        if not ok:
            failures += 1
            if fallback_to_valence:
                ok = _fallback_valence_valid(
                    graph,
                    node_label_attr=node_label_attr,
                    edge_label_attr=edge_label_attr,
                    raw_node_label_values=raw_node_label_values,
                    raw_edge_label_values=raw_edge_label_values,
                    dataset=dataset,
                )
                used_fallback = used_fallback or ok
        valid_flags.append(bool(ok))
        smiles.append(smi if smi is not None else None)
    backend = "rdkit" if Chem is not None and not used_fallback else ("rdkit_with_valence_fallback" if Chem is not None else "valence_fallback")
    return MolecularEvaluationResult(valid_flags, smiles, backend, Chem is not None, failures)


def molecular_quality_metrics(
    generated_graphs: Sequence[nx.Graph],
    train_graphs: Sequence[nx.Graph] | None = None,
    *,
    node_label_attr: str = "node_label",
    edge_label_attr: str = "edge_type",
    raw_node_label_values: Sequence[str] | None = None,
    raw_edge_label_values: Sequence[str] | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    generated_graphs = list(generated_graphs)
    train_graphs = list(train_graphs or [])
    gen_eval = evaluate_molecular_validity(
        generated_graphs,
        node_label_attr=node_label_attr,
        edge_label_attr=edge_label_attr,
        raw_node_label_values=raw_node_label_values,
        raw_edge_label_values=raw_edge_label_values,
        dataset=dataset,
    )
    valid_smiles = [s for ok, s in zip(gen_eval.valid_flags, gen_eval.smiles) if ok and s]
    unique_valid_smiles = sorted(set(valid_smiles))
    train_smiles: set[str] = set()
    if train_graphs:
        train_eval = evaluate_molecular_validity(
            train_graphs,
            node_label_attr=node_label_attr,
            edge_label_attr=edge_label_attr,
            raw_node_label_values=raw_node_label_values,
            raw_edge_label_values=raw_edge_label_values,
            dataset=dataset,
        )
        train_smiles = {s for ok, s in zip(train_eval.valid_flags, train_eval.smiles) if ok and s}
    novel_unique = [s for s in unique_valid_smiles if s not in train_smiles]
    n_total = len(generated_graphs)
    n_valid = int(sum(gen_eval.valid_flags))
    n_unique = len(unique_valid_smiles)
    return {
        "validity_rate": float(n_valid / n_total) if n_total else 0.0,
        "dataset_validity_rate": float(n_valid / n_total) if n_total else 0.0,
        "rdkit_validity_rate": float(len(valid_smiles) / n_total) if n_total else 0.0,
        "uniqueness_rate": float(n_unique / max(len(valid_smiles), 1)) if valid_smiles else 0.0,
        "novelty_rate": float(len(novel_unique) / max(n_unique, 1)) if train_graphs and n_unique else (None if not train_graphs else 0.0),
        "valid_unique_novel_rate": float(len(novel_unique) / n_total) if n_total and train_graphs else None,
        "num_valid_molecules": n_valid,
        "num_unique_valid_molecules": n_unique,
        "num_novel_unique_molecules": len(novel_unique) if train_graphs else None,
        "validity_backend": gen_eval.validity_backend,
        "rdkit_available": bool(gen_eval.rdkit_available),
        "rdkit_construction_failures": int(gen_eval.construction_failures),
        "zinc_categorical_label_note": (
            "PyG ZINC atom_type labels are categorical; integer category ids are not treated as atomic numbers. "
            "Provide atomic_number/z node attributes or a symbol-valued raw_node_label_values mapping for RDKit validity."
            if str(dataset or "").lower() == "zinc" else None
        ),
    }
