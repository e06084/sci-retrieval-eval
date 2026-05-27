"""Tests for centralized manifest metadata key constants."""

from __future__ import annotations

from eval_platform.artifacts import (
    DEPENDENCY_METADATA_KEYS_BY_ARTIFACT_TYPE as EXPORTED_DEPENDENCY_KEYS,
)
from eval_platform.artifacts import (
    METADATA_KEY_CHUNKED_CORPUS_ARTIFACT_ID as EXPORTED_CHUNKED_CORPUS_KEY,
)
from eval_platform.artifacts import (
    METADATA_KEY_COLLECTION_NAME as EXPORTED_COLLECTION_NAME_KEY,
)
from eval_platform.artifacts import (
    METADATA_KEY_EMBEDDINGS_ARTIFACT_ID as EXPORTED_EMBEDDINGS_KEY,
)
from eval_platform.artifacts import METADATA_KEY_INDEX_NAME as EXPORTED_INDEX_NAME_KEY
from eval_platform.artifacts import (
    METADATA_KEY_RAW_DATASET_ARTIFACT_ID as EXPORTED_RAW_DATASET_KEY,
)
from eval_platform.artifacts import (
    METADATA_KEY_SOURCE_CHUNKED_CORPUS_ARTIFACT_ID as EXPORTED_SOURCE_CHUNKS_KEY,
)
from eval_platform.artifacts import (
    METADATA_KEY_SOURCE_EMBEDDINGS_ARTIFACT_ID as EXPORTED_SOURCE_EMBEDDINGS_KEY,
)
from eval_platform.artifacts import (
    METADATA_KEY_SOURCE_NORMALIZED_DATASET_ARTIFACT_ID as EXPORTED_SOURCE_NORMALIZED_KEY,
)
from eval_platform.artifacts.metadata_keys import (
    DEPENDENCY_METADATA_KEYS_BY_ARTIFACT_TYPE,
    METADATA_KEY_CHUNKED_CORPUS_ARTIFACT_ID,
    METADATA_KEY_COLLECTION_NAME,
    METADATA_KEY_EMBEDDINGS_ARTIFACT_ID,
    METADATA_KEY_INDEX_NAME,
    METADATA_KEY_RAW_DATASET_ARTIFACT_ID,
    METADATA_KEY_SOURCE_CHUNKED_CORPUS_ARTIFACT_ID,
    METADATA_KEY_SOURCE_EMBEDDINGS_ARTIFACT_ID,
    METADATA_KEY_SOURCE_NORMALIZED_DATASET_ARTIFACT_ID,
)
from eval_platform.artifacts.types import (
    CHUNKED_CORPUS_ARTIFACT_TYPE,
    EMBEDDINGS_ARTIFACT_TYPE,
    NORMALIZED_DATASET_ARTIFACT_TYPE,
    RAW_DATASET_ARTIFACT_TYPE,
)
from eval_platform.corpus_assets.planner import _dependency_id


def test_dependency_metadata_keys_cover_corpus_asset_planner_inputs() -> None:
    assert DEPENDENCY_METADATA_KEYS_BY_ARTIFACT_TYPE == {
        RAW_DATASET_ARTIFACT_TYPE: (METADATA_KEY_RAW_DATASET_ARTIFACT_ID,),
        NORMALIZED_DATASET_ARTIFACT_TYPE: (
            METADATA_KEY_SOURCE_NORMALIZED_DATASET_ARTIFACT_ID,
        ),
        CHUNKED_CORPUS_ARTIFACT_TYPE: (
            METADATA_KEY_SOURCE_CHUNKED_CORPUS_ARTIFACT_ID,
            METADATA_KEY_CHUNKED_CORPUS_ARTIFACT_ID,
        ),
        EMBEDDINGS_ARTIFACT_TYPE: (
            METADATA_KEY_SOURCE_EMBEDDINGS_ARTIFACT_ID,
            METADATA_KEY_EMBEDDINGS_ARTIFACT_ID,
        ),
    }


def test_corpus_asset_planner_reads_registered_dependency_metadata_keys() -> None:
    record = {
        "metadata_summary": {
            METADATA_KEY_RAW_DATASET_ARTIFACT_ID: "raw-1",
            METADATA_KEY_SOURCE_NORMALIZED_DATASET_ARTIFACT_ID: "normalized-1",
            METADATA_KEY_SOURCE_CHUNKED_CORPUS_ARTIFACT_ID: "chunks-1",
            METADATA_KEY_SOURCE_EMBEDDINGS_ARTIFACT_ID: "embeddings-1",
        }
    }

    assert _dependency_id(record, RAW_DATASET_ARTIFACT_TYPE) == "raw-1"
    assert (
        _dependency_id(record, NORMALIZED_DATASET_ARTIFACT_TYPE) == "normalized-1"
    )
    assert _dependency_id(record, CHUNKED_CORPUS_ARTIFACT_TYPE) == "chunks-1"
    assert _dependency_id(record, EMBEDDINGS_ARTIFACT_TYPE) == "embeddings-1"


def test_corpus_asset_planner_reads_index_specific_dependency_metadata_keys() -> None:
    record = {
        "metadata_summary": {
            METADATA_KEY_CHUNKED_CORPUS_ARTIFACT_ID: "chunks-2",
            METADATA_KEY_EMBEDDINGS_ARTIFACT_ID: "embeddings-2",
        }
    }

    assert _dependency_id(record, CHUNKED_CORPUS_ARTIFACT_TYPE) == "chunks-2"
    assert _dependency_id(record, EMBEDDINGS_ARTIFACT_TYPE) == "embeddings-2"


def test_resource_metadata_key_values_are_stable() -> None:
    assert METADATA_KEY_INDEX_NAME == "index_name"
    assert METADATA_KEY_COLLECTION_NAME == "collection_name"


def test_artifacts_package_exports_metadata_key_registry() -> None:
    assert EXPORTED_DEPENDENCY_KEYS == DEPENDENCY_METADATA_KEYS_BY_ARTIFACT_TYPE
    assert EXPORTED_INDEX_NAME_KEY == METADATA_KEY_INDEX_NAME
    assert EXPORTED_COLLECTION_NAME_KEY == METADATA_KEY_COLLECTION_NAME
    assert EXPORTED_RAW_DATASET_KEY == METADATA_KEY_RAW_DATASET_ARTIFACT_ID
    assert EXPORTED_SOURCE_NORMALIZED_KEY == (
        METADATA_KEY_SOURCE_NORMALIZED_DATASET_ARTIFACT_ID
    )
    assert EXPORTED_SOURCE_CHUNKS_KEY == METADATA_KEY_SOURCE_CHUNKED_CORPUS_ARTIFACT_ID
    assert EXPORTED_SOURCE_EMBEDDINGS_KEY == (
        METADATA_KEY_SOURCE_EMBEDDINGS_ARTIFACT_ID
    )
    assert EXPORTED_CHUNKED_CORPUS_KEY == METADATA_KEY_CHUNKED_CORPUS_ARTIFACT_ID
    assert EXPORTED_EMBEDDINGS_KEY == METADATA_KEY_EMBEDDINGS_ARTIFACT_ID
