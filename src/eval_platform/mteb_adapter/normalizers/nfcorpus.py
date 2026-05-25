"""Normalizer for the NFCorpus MTEB retrieval task."""

from eval_platform.mteb_adapter.base import GenericRetrievalTaskNormalizer


class NFCorpusNormalizer(GenericRetrievalTaskNormalizer):
    task_name = "NFCorpus"
