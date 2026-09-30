from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class GenerationComplexitySpec:
    family: str
    representative: str
    step_definition: str
    time_big_o: str
    space_big_o: str
    dominant_state: str
    note: str


# The formulas deliberately separate the cost of one neural-network evaluation
# (C_theta/M_theta) from the sampler's graph-state update. This is more robust
# than hard-coding one upstream architecture because several wrappers can swap
# backbones while retaining the same diffusion/transport family.
_COMPLEXITY_SPECS: dict[str, GenerationComplexitySpec] = {
    "construct": GenerationComplexitySpec(
        family="Discrete categorical diffusion + projection",
        representative="ConStruct",
        step_definition="one reverse categorical denoising/projection step",
        time_big_o=r"O(C_\theta(n)+n^2 K_e+C_{\mathrm{proj}})",
        space_big_o=r"O(M_\theta(n)+n^2 K_e)",
        dominant_state="dense node/edge categorical tensors",
        note=(
            "Projection cost depends on the enforced constraint. In the benchmark, the "
            "GraphTransformer maintains pairwise edge states, so dense working state is quadratic in n."
        ),
    ),
    "digress": GenerationComplexitySpec(
        family="Discrete categorical diffusion",
        representative="DiGress",
        step_definition="one reverse categorical diffusion step",
        time_big_o=r"O(C_\theta(n)+n^2 K_e)",
        space_big_o=r"O(M_\theta(n)+n^2 K_e)",
        dominant_state="dense node/edge categorical tensors",
        note=(
            "For the dense graph-transformer implementation, C_theta and M_theta also contain "
            "pairwise edge-state operations and therefore scale quadratically in n up to hidden dimensions."
        ),
    ),
    "disco": GenerationComplexitySpec(
        family="Continuous-time discrete-state CTMC",
        representative="DisCo",
        step_definition="one tau-leaping / reverse-rate update",
        time_big_o=r"O(C_\theta(n)+n^2 K_e)",
        space_big_o=r"O(M_\theta(n)+n^2 K_e)",
        dominant_state="categorical node/edge states and jump rates",
        note=(
            "Time is continuous, but a numerical tau-leaping update evaluates categorical jump rates "
            "over node/edge variables. The benchmark GraphTransformer uses dense pairwise edge states."
        ),
    ),
    "edp_gnn": GenerationComplexitySpec(
        family="Continuous adjacency score / Langevin",
        representative="EDP-GNN",
        step_definition="one Langevin score update",
        time_big_o=r"O(C_\theta(n)+n^2)",
        space_big_o=r"O(M_\theta(n)+n^2)",
        dominant_state="continuous dense adjacency matrix",
        note=(
            "The sampler stores and updates a continuous adjacency matrix. With the benchmark dense adjacency "
            "representation, the graph-state term is quadratic in n."
        ),
    ),
    "graphguide": GenerationComplexitySpec(
        family="Bernoulli edge diffusion",
        representative="GraphGUIDE",
        step_definition="one Bernoulli reverse edge update",
        time_big_o=r"O(C_\theta(n,m_t)+n^2)",
        space_big_o=r"O(M_\theta(n,m_t)+n^2)",
        dominant_state="edge-vector state over candidate node pairs",
        note=(
            "The GNN evaluation can exploit the active edge set m_t, but the benchmark conversion maintains an "
            "edge-vector over candidate node pairs; worst-case state/update cost is quadratic in n."
        ),
    ),
    "grum": GenerationComplexitySpec(
        family="Bridge / continuous-state diffusion",
        representative="GruM",
        step_definition="one Euler predictor step",
        time_big_o=r"O(C_\theta(n)+n^2 d_e)",
        space_big_o=r"O(M_\theta(n)+n^2 d_e)",
        dominant_state="continuous node features and dense edge/adjacency state",
        note=(
            "d_e is the edge/adjacency channel dimension. For fixed d_e this reduces to a quadratic graph-state "
            "term; the upstream dataset-specific backbone determines C_theta and M_theta."
        ),
    ),
}


def _get_nested(mapping: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def sampling_steps_from_config(model: str, config: Mapping[str, Any]) -> int | None:
    model = str(model).lower()
    candidates: list[Any]
    if model == "construct":
        candidates = [config.get("sampling_steps"), config.get("diffusion_steps")]
    elif model == "digress":
        candidates = [
            _get_nested(config, "model_overrides", "diffusion_steps"),
            _get_nested(config, "sampling", "num_steps"),
            config.get("diffusion_steps"),
        ]
    elif model == "disco":
        candidates = [config.get("sampling_steps")]
    elif model == "edp_gnn":
        candidates = [config.get("step_num"), _get_nested(config, "sampling", "num_steps")]
    elif model == "graphguide":
        candidates = [config.get("t_limit"), _get_nested(config, "sampling", "num_steps")]
    elif model == "grum":
        candidates = [
            config.get("sample_num_scales"),
            config.get("num_scales"),
            config.get("adj_num_scales"),
            config.get("x_num_scales"),
        ]
    else:
        candidates = []
    for value in candidates:
        if value is None:
            continue
        try:
            steps = int(value)
        except (TypeError, ValueError):
            continue
        if steps > 0:
            return steps
    return None


def graph_size_summary(graphs: Sequence[Any] | None) -> dict[str, Any]:
    if not graphs:
        return {}
    ns: list[int] = []
    ms: list[int] = []
    for graph in graphs:
        try:
            ns.append(int(graph.number_of_nodes()))
            ms.append(int(graph.number_of_edges()))
        except Exception:
            continue
    if not ns:
        return {}
    densities = []
    for n, m in zip(ns, ms):
        denom = n * (n - 1) / 2
        densities.append(float(m) / denom if denom > 0 else 0.0)
    return {
        "num_graphs": len(ns),
        "nodes_mean": sum(ns) / len(ns),
        "nodes_max": max(ns),
        "edges_mean": sum(ms) / len(ms),
        "edges_max": max(ms),
        "undirected_density_mean": sum(densities) / len(densities),
    }


def generation_complexity_report(
    model: str,
    config: Mapping[str, Any] | None = None,
    *,
    graphs: Sequence[Any] | None = None,
) -> dict[str, Any]:
    model_key = str(model).lower()
    spec = _COMPLEXITY_SPECS.get(model_key)
    if spec is None:
        return {
            "model": model_key,
            "available": False,
            "note": "No asymptotic generation-complexity specification is registered for this model.",
        }
    cfg = config or {}
    payload = asdict(spec)
    payload.update({
        "model": model_key,
        "available": True,
        "sampling_steps_configured": sampling_steps_from_config(model_key, cfg),
        "observed_graph_size": graph_size_summary(graphs),
        "symbols": {
            "n": "number of nodes in the generated graph",
            "m_t": "number of active edges at the current generation step",
            "K_e": "number of edge categories",
            "d_e": "edge/adjacency channel dimension",
            "C_theta": "time cost of one neural score/rate/denoiser evaluation",
            "M_theta": "working memory of one neural score/rate/denoiser evaluation",
            "C_proj": "constraint-projection cost when projection is used",
        },
        "scope": (
            "Worst-case asymptotic generation cost per sampler step. Fixed batch size, hidden widths, "
            "category counts, and implementation constants are suppressed unless shown explicitly."
        ),
    })
    return payload


def registered_generation_complexities() -> dict[str, dict[str, Any]]:
    return {name: generation_complexity_report(name, {}) for name in sorted(_COMPLEXITY_SPECS)}
