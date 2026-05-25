import importlib.util
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_data.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_summarize_atom_types_logs_counts(monkeypatch):
    module = _load_script_module()
    messages = []

    def capture(message, *args):
        messages.append(message % args if args else message)

    monkeypatch.setattr(module.logger, "info", capture)
    graph = nx.Graph()
    graph.add_node(0, node_label=12, feats=[2.0])
    graph.add_node(1, node_label=0, feats=[0.0])

    module._summarize_atom_types({"train": [graph], "val": [], "test": []}, title="Canonical ZINC")

    text = "\n".join(messages)
    assert "Canonical ZINC atom_type/node_label counts" in text
    assert "train: {0: 1, 12: 1}" in text
    assert "node_label -> example first feature values" in text
