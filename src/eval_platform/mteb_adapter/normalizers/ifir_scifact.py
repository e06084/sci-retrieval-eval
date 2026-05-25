"""Normalizer for the IFIRScifact MTEB retrieval task."""

from eval_platform.mteb_adapter.base import GenericRetrievalTaskNormalizer


class IFIRScifactNormalizer(GenericRetrievalTaskNormalizer):
    task_name = "IFIRScifact"
