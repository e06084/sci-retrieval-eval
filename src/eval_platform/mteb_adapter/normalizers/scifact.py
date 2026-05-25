"""Normalizer for the SciFact MTEB retrieval task."""

from eval_platform.mteb_adapter.base import GenericRetrievalTaskNormalizer


class SciFactNormalizer(GenericRetrievalTaskNormalizer):
    task_name = "SciFact"
