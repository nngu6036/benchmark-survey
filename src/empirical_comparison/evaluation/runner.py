from empirical_comparison.utils.logging import get_logger
logger = get_logger(__name__)

def run_evaluation_step(name: str) -> None:
    logger.info("Running step: %s", name)
