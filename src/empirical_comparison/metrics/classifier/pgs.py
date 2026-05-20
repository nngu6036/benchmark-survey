from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from empirical_comparison.graphs.attributes import attribute_descriptor_features, fit_attribute_statistics, normalize_schema
from empirical_comparison.metrics.descriptor.descriptors import (
    clustering_histogram,
    count_orca_4node_orbits,
    degree_histogram,
    spectral_histogram,
)
from empirical_comparison.metrics.learned_feature.encoder import RandomGINPlaceholder


@dataclass
class DescriptorConfig:
    degree_bins: int = 100
    clustering_bins: int = 100
    spectral_bins: int = 200
    max_degree: int = 100
    orca_exec: str | None = None
    graph_attributes: dict[str, Any] | None = None
    gin_dim: int = 128
    seed: int = 42


class DescriptorUnavailable(RuntimeError):
    pass


def _safe_stack(rows: Sequence[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.zeros((0, 1), dtype=np.float64)
    width = max(int(np.asarray(r).size) for r in rows)
    out = np.zeros((len(rows), max(width, 1)), dtype=np.float64)
    for i, r in enumerate(rows):
        arr = np.asarray(r, dtype=np.float64).reshape(-1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        out[i, : arr.size] = arr
    return out


def _orca_exec(orca_exec: str | None) -> str:
    exe = orca_exec or os.environ.get("ORCA_EXEC")
    if not exe:
        raise DescriptorUnavailable("ORCA executable is not configured. Set ORCA_EXEC or pass --orca-exec.")
    p = Path(exe)
    if not p.exists():
        raise DescriptorUnavailable(f"ORCA executable not found: {exe}")
    return str(p)


def _simple_graph(graph: nx.Graph) -> tuple[nx.Graph, dict[Any, int]]:
    g = nx.Graph()
    g.add_nodes_from(graph.nodes())
    g.add_edges_from((u, v) for u, v in graph.edges() if u != v)
    nodes = sorted(g.nodes())
    return g, {node: i for i, node in enumerate(nodes)}


def _count_orca_node_orbits(graph: nx.Graph, *, graphlet_size: int, orca_exec: str | None = None) -> np.ndarray:
    exe = _orca_exec(orca_exec)
    g, node_map = _simple_graph(graph)
    if g.number_of_nodes() == 0:
        return np.zeros(1, dtype=np.float64)
    input_path = None
    output_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f_in:
            input_path = f_in.name
            f_in.write(f"{g.number_of_nodes()} {g.number_of_edges()}\n")
            for u, v in g.edges():
                f_in.write(f"{node_map[u]} {node_map[v]}\n")
        with tempfile.NamedTemporaryFile(mode="r", delete=False) as f_out:
            output_path = f_out.name
        subprocess.run([exe, "node", str(graphlet_size), input_path, output_path], check=True, capture_output=True, text=True)
        rows = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append([float(x) for x in line.strip().split()])
        if not rows:
            return np.zeros(1, dtype=np.float64)
        arr = _safe_stack([np.asarray(r, dtype=np.float64) for r in rows])
        counts = np.log1p(arr.sum(axis=0))
        s = counts.sum()
        return counts / s if s > 0 else counts
    except subprocess.CalledProcessError as exc:
        raise DescriptorUnavailable(f"ORCA {graphlet_size}-node orbit execution failed: {exc.stderr or exc.stdout}") from exc
    finally:
        for path in (input_path, output_path):
            if path and os.path.exists(path):
                os.remove(path)


class PGSDescriptor:
    def __init__(self, name: str, config: DescriptorConfig) -> None:
        self.name = name.lower()
        self.config = config
        self._gin = RandomGINPlaceholder(feature_dim=config.gin_dim, seed=config.seed, normalize_output=True)
        self._attr_stats = None
        self._sub_descriptors: list[PGSDescriptor] | None = None

    def fit(self, graphs: Sequence[nx.Graph]) -> "PGSDescriptor":
        if self.name in {"attributes", "attribute", "attrs"}:
            schema = normalize_schema({"graph_attributes": self.config.graph_attributes or {}})
            self._attr_stats = fit_attribute_statistics(graphs, schema)
        elif self.name in {"concat", "all"}:
            self._sub_descriptors = []
            for sub in ["degree", "clustering", "spectral", "gin", "attributes"]:
                try:
                    d = PGSDescriptor(sub, self.config).fit(graphs)
                    self._sub_descriptors.append(d)
                except Exception:
                    pass
        return self

    def transform(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        name = self.name
        if name in {"degree", "deg"}:
            return _safe_stack([degree_histogram(g, bins=self.config.degree_bins, max_degree=self.config.max_degree) for g in graphs])
        if name in {"clustering", "clust"}:
            return _safe_stack([clustering_histogram(g, bins=self.config.clustering_bins) for g in graphs])
        if name in {"spectral", "spectrum", "eig", "eigen"}:
            return _safe_stack([spectral_histogram(g, bins=self.config.spectral_bins) for g in graphs])
        if name in {"gin", "random_gin"}:
            return _safe_stack([self._gin.encode(g) for g in graphs])
        if name in {"orbit4", "orb4", "orbit", "orbits"}:
            return _safe_stack([count_orca_4node_orbits(g, orca_exec=self.config.orca_exec) for g in graphs])
        if name in {"orbit5", "orb5"}:
            return _safe_stack([_count_orca_node_orbits(g, graphlet_size=5, orca_exec=self.config.orca_exec) for g in graphs])
        if name in {"attributes", "attribute", "attrs"}:
            schema = normalize_schema({"graph_attributes": self.config.graph_attributes or {}})
            stats = self._attr_stats or fit_attribute_statistics(graphs, schema)
            return attribute_descriptor_features(
                graphs,
                schema,
                node_label_values=stats.node_label_values,
                edge_label_values=stats.edge_label_values,
                graph_label_values=stats.graph_label_values,
                node_feature_dim=stats.node_feature_dim,
                edge_feature_dim=stats.edge_feature_dim,
                include_continuous=True,
            )
        if name in {"concat", "all"}:
            descs = self._sub_descriptors or [PGSDescriptor(sub, self.config).fit(graphs) for sub in ["degree", "clustering", "spectral", "gin", "attributes"]]
            mats = []
            for d in descs:
                try:
                    mats.append(d.transform(graphs))
                except Exception:
                    pass
            if not mats:
                return np.zeros((len(graphs), 1), dtype=np.float64)
            return np.concatenate(mats, axis=1)
        raise DescriptorUnavailable(f"Unknown PGS descriptor: {self.name}")


class _ProbabilisticClassifier:
    def __init__(self, kind: str = "auto", *, random_state: int = 0, device: str | None = None) -> None:
        self.kind_requested = kind
        self.kind = kind
        self.random_state = random_state
        self.device = device
        self.model = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_ProbabilisticClassifier":
        kind = self.kind_requested.lower()
        if kind in {"lr", "logistic", "logistic_regression"}:
            self.kind = "standardized_logistic_regression"
            self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", random_state=self.random_state))
        else:
            try:
                from tabpfn import TabPFNClassifier  # type: ignore

                self.kind = "tabpfn"
                try:
                    kwargs = {"random_state": self.random_state}
                    if self.device:
                        kwargs["device"] = self.device
                    self.model = TabPFNClassifier(**kwargs)
                except TypeError:
                    self.model = TabPFNClassifier()
            except Exception:
                if kind == "tabpfn":
                    raise DescriptorUnavailable("TabPFN was requested but is not installed/importable.")
                self.kind = "standardized_logistic_regression"
                self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", random_state=self.random_state))
        self.model.fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=int))
        return self

    def predict_proba_generated(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Classifier has not been fitted.")
        proba = self.model.predict_proba(np.asarray(x, dtype=np.float64))
        classes = getattr(self.model, "classes_", None)
        if classes is None and hasattr(self.model, "steps"):
            classes = self.model.steps[-1][1].classes_
        if classes is None:
            return np.asarray(proba[:, -1], dtype=np.float64)
        classes = list(classes)
        idx = classes.index(1) if 1 in classes else -1
        return np.asarray(proba[:, idx], dtype=np.float64)


def _binary_features(ref_x: np.ndarray, gen_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([ref_x, gen_x], axis=0).astype(np.float64)
    y = np.concatenate([np.zeros(len(ref_x), dtype=int), np.ones(len(gen_x), dtype=int)], axis=0)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), y


def _js_score(prob_generated: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Return the classifier-induced PGS-JS lower-bound score.

    The classifier is trained to distinguish reference graphs (class 0) from
    generated graphs (class 1).  If q(x) is its predicted probability of class
    1, the held-out mean log probability of the true domain label gives the
    usual variational lower-bound estimate of Jensen-Shannon divergence in
    base-2 units: JSD >= E[log2 p_true] + 1.  We report sqrt(JSD) as a
    distance-like score in [0, 1].
    """
    p = np.clip(np.asarray(prob_generated, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y = np.asarray(y_true, dtype=int)
    p_true = np.where(y == 1, p, 1.0 - p)
    mean_ll_base2 = float(np.mean(np.log2(p_true)))
    js_div = float(np.clip(mean_ll_base2 + 1.0, 0.0, 1.0))
    score = float(np.sqrt(js_div))
    y_pred = (p >= 0.5).astype(int)
    return {
        "score": score,
        "pgs_js_distance": score,
        "pgs_js_divergence_lower_bound": js_div,
        "pgs_mean_log2_true_class_probability": mean_ll_base2,
        "pgs_mean_true_class_probability": float(np.mean(p_true)),
        "pgs_binary_accuracy_at_0_5": float(np.mean(y_pred == y)),
    }


def _fit_predict_score(x_train, y_train, x_eval, y_eval, *, classifier: str, seed: int, device: str | None = None) -> tuple[dict[str, float], str]:
    clf = _ProbabilisticClassifier(classifier, random_state=seed, device=device).fit(x_train, y_train)
    probs_eval = clf.predict_proba_generated(x_eval)
    return _js_score(probs_eval, y_eval), clf.kind


def _split_ref_gen(ref_graphs: Sequence[nx.Graph], gen_graphs: Sequence[nx.Graph], seed: int) -> tuple[list[nx.Graph], list[nx.Graph], list[nx.Graph], list[nx.Graph]]:
    n = min(len(ref_graphs), len(gen_graphs))
    if n < 4:
        raise ValueError("PGS requires at least 4 reference and 4 generated graphs after subsampling.")
    rng = np.random.default_rng(seed)
    ref_idx = rng.permutation(len(ref_graphs))[:n]
    gen_idx = rng.permutation(len(gen_graphs))[:n]
    ref = [ref_graphs[int(i)] for i in ref_idx]
    gen = [gen_graphs[int(i)] for i in gen_idx]
    half = max(2, n // 2)
    return ref[:half], ref[half:], gen[:half], gen[half:]


def _descriptor_cv_and_test(ref_fit, ref_test, gen_fit, gen_test, *, descriptor_name: str, descriptor_config: DescriptorConfig, classifier: str, cv_folds: int, seed: int, device: str | None = None) -> dict[str, Any]:
    descriptor = PGSDescriptor(descriptor_name, descriptor_config).fit(ref_fit + gen_fit)
    x_fit = descriptor.transform(ref_fit + gen_fit)
    x_fit_bin, y_fit = _binary_features(x_fit[: len(ref_fit)], x_fit[len(ref_fit) :])
    x_test = descriptor.transform(ref_test + gen_test)
    x_test_bin, y_test = _binary_features(x_test[: len(ref_test)], x_test[len(ref_test) :])
    min_class = int(min(np.bincount(y_fit, minlength=2)))
    k = max(2, min(int(cv_folds), min_class))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    cv_scores = []
    cv_payloads = []
    resolved_classifiers = []
    for fold, (tr, va) in enumerate(cv.split(x_fit_bin, y_fit)):
        score_payload, resolved = _fit_predict_score(x_fit_bin[tr], y_fit[tr], x_fit_bin[va], y_fit[va], classifier=classifier, seed=seed + fold, device=device)
        cv_scores.append(score_payload["score"])
        cv_payloads.append(score_payload)
        resolved_classifiers.append(resolved)
    test_payload, resolved = _fit_predict_score(x_fit_bin, y_fit, x_test_bin, y_test, classifier=classifier, seed=seed + 10_000, device=device)
    return {
        "descriptor": descriptor_name,
        "cv_score": float(np.mean(cv_scores)),
        "cv_score_std": float(np.std(cv_scores, ddof=0)),
        "test_score": float(test_payload["score"]),
        "test_metrics": test_payload,
        "cv_metrics": cv_payloads,
        "classifier": resolved,
        "cv_classifiers": sorted(set(resolved_classifiers)),
        "num_fit_graphs_per_class": int(len(ref_fit)),
        "num_test_graphs_per_class": int(len(ref_test)),
        "feature_dim": int(x_fit_bin.shape[1]),
    }


def polygraphscore(ref_graphs: Sequence[nx.Graph], gen_graphs: Sequence[nx.Graph], *, descriptor_names: Sequence[str], descriptor_config: DescriptorConfig | None = None, classifier: str = "auto", cv_folds: int = 4, seed: int = 42, skip_unavailable: bool = True, device: str | None = None) -> dict[str, Any]:
    if descriptor_config is None:
        descriptor_config = DescriptorConfig(seed=seed)
    descriptor_config.seed = seed
    ref_fit, ref_test, gen_fit, gen_test = _split_ref_gen(list(ref_graphs), list(gen_graphs), seed)
    results = []
    skipped: dict[str, str] = {}
    for name in descriptor_names:
        try:
            results.append(_descriptor_cv_and_test(ref_fit, ref_test, gen_fit, gen_test, descriptor_name=str(name), descriptor_config=descriptor_config, classifier=classifier, cv_folds=cv_folds, seed=seed + 101 * (len(results) + 1), device=device))
        except Exception as exc:
            if not skip_unavailable:
                raise
            skipped[str(name)] = str(exc)
    if not results:
        raise RuntimeError(f"No PGS descriptors could be evaluated. Skipped: {skipped}")
    best = max(results, key=lambda d: d["cv_score"])
    out_results: dict[str, Any] = {
        "pgs_js_distance": float(best["test_score"]),
        "pgs_best_descriptor": best["descriptor"],
        "pgs_cv_selected_score": float(best["cv_score"]),
    }
    for k, v in best["test_metrics"].items():
        if k != "score":
            out_results[k] = float(v) if isinstance(v, (int, float, np.number)) else v
    for item in results:
        key = str(item["descriptor"]).lower().replace(" ", "_")
        out_results[f"pgs_{key}_cv"] = float(item["cv_score"])
        out_results[f"pgs_{key}_test"] = float(item["test_score"])
    return {
        "results": out_results,
        "descriptor_results": results,
        "best_descriptor": best["descriptor"],
        "classifier": best.get("classifier"),
        "skipped_descriptors": skipped,
        "mode": "jsd",
    }
