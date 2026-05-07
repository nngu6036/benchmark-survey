from __future__ import annotations

# Thin wrappers for packaging entry points.  The scripts directory remains the
# primary executable layer when running from the repository root.
from scripts.prepare_data import main as prepare_data
from scripts.train_model import main as train_model
from scripts.generate_samples import main as generate_samples
from scripts.evaluate_descriptor_metrics import main as eval_descriptor
from scripts.evaluate_learned_feature_metrics import main as eval_learned
from scripts.evaluate_classifier_metrics import main as eval_classifier
from scripts.aggregate_results import main as aggregate_results
from scripts.make_latex_tables import main as make_latex_tables
from scripts.make_figures import main as make_figures
from scripts.run_benchmark import main as run_benchmark


def prepare_data_main() -> None:
    prepare_data()


def train_model_main() -> None:
    train_model()


def generate_samples_main() -> None:
    generate_samples()


def evaluate_descriptor_main() -> None:
    eval_descriptor()


def evaluate_learned_main() -> None:
    eval_learned()


def evaluate_classifier_main() -> None:
    eval_classifier()


def aggregate_results_main() -> None:
    aggregate_results()


def make_latex_tables_main() -> None:
    make_latex_tables()


def make_figures_main() -> None:
    make_figures()


def run_benchmark_main() -> None:
    run_benchmark()
