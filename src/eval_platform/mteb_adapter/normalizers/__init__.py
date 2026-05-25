"""Explicit dataset normalizers for target MTEB retrieval tasks."""

from eval_platform.mteb_adapter.normalizers.ifir_nfcorpus import IFIRNFCorpusNormalizer
from eval_platform.mteb_adapter.normalizers.ifir_scifact import IFIRScifactNormalizer
from eval_platform.mteb_adapter.normalizers.litsearch import LitSearchRetrievalNormalizer
from eval_platform.mteb_adapter.normalizers.nfcorpus import NFCorpusNormalizer
from eval_platform.mteb_adapter.normalizers.scifact import SciFactNormalizer

__all__ = [
    "IFIRNFCorpusNormalizer",
    "IFIRScifactNormalizer",
    "LitSearchRetrievalNormalizer",
    "NFCorpusNormalizer",
    "SciFactNormalizer",
]
