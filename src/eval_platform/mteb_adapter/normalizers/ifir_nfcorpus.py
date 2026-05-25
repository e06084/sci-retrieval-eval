"""Normalizer for the IFIRNFCorpus MTEB retrieval task."""

from eval_platform.mteb_adapter.base import GenericRetrievalTaskNormalizer


class IFIRNFCorpusNormalizer(GenericRetrievalTaskNormalizer):
    task_name = "IFIRNFCorpus"
