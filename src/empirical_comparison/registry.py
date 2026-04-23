from empirical_comparison.datasets.sbm import SBMDatasetBuilder
from empirical_comparison.datasets.planar import PlanarDatasetBuilder
from empirical_comparison.models.wrappers.graphguide import GraphGUIDEWrapper
from empirical_comparison.models.wrappers.digress import DiGressWrapper
from empirical_comparison.models.wrappers.construct import ConStructWrapper
from empirical_comparison.models.wrappers.edp_gnn import EDPGNNWrapper
from empirical_comparison.models.wrappers.disco import DisCoWrapper
from empirical_comparison.models.wrappers.grum import GruMWrapper

DATASET_REGISTRY = {"sbm": SBMDatasetBuilder, "planar": PlanarDatasetBuilder}
MODEL_REGISTRY = {
    "graphguide": GraphGUIDEWrapper,
    "digress": DiGressWrapper,
    "construct": ConStructWrapper,
    "edp_gnn": EDPGNNWrapper,
    "disco": DisCoWrapper,
    "grum": GruMWrapper,
}
