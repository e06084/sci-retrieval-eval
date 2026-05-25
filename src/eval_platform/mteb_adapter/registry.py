"""Registry for explicit per-dataset MTEB normalizers."""

from eval_platform.mteb_adapter.base import MTEBAdapterError, MTEBTaskNormalizer
from eval_platform.mteb_adapter.normalizers import (
    IFIRNFCorpusNormalizer,
    IFIRScifactNormalizer,
    LitSearchRetrievalNormalizer,
    NFCorpusNormalizer,
    SciFactNormalizer,
)

NORMALIZER_REGISTRY: dict[str, MTEBTaskNormalizer] = {
    "LitSearchRetrieval": LitSearchRetrievalNormalizer(),
    "SciFact": SciFactNormalizer(),
    "IFIRScifact": IFIRScifactNormalizer(),
    "IFIRNFCorpus": IFIRNFCorpusNormalizer(),
    "NFCorpus": NFCorpusNormalizer(),
}


def get_mteb_task_normalizer(task_name: str) -> MTEBTaskNormalizer:
    try:
        return NORMALIZER_REGISTRY[task_name]
    except KeyError as exc:
        raise MTEBAdapterError(f"No MTEB normalizer registered for task: {task_name}") from exc
