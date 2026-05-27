"""Dataset registry for corpus asset planning."""

from __future__ import annotations

from dataclasses import dataclass


class CorpusAssetError(Exception):
    """Raised when corpus asset inventory or planning fails."""


@dataclass(frozen=True)
class DatasetSpec:
    """One target dataset and its immutable raw S3 layout."""

    task_name: str
    slug: str
    raw_dir: str
    raw_format: str
    expected_raw_files: tuple[str, ...]
    notes: str


TARGET_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        task_name="IFIRNFCorpus",
        slug="ifir_nfcorpus",
        raw_dir="ifir_nfcorpus",
        raw_format="jsonl_tsv",
        expected_raw_files=(
            "corpus.jsonl",
            "queries.jsonl",
            "instructions.jsonl",
            "qrels/test.tsv",
        ),
        notes="IFIR layout with query instructions.",
    ),
    DatasetSpec(
        task_name="NFCorpus",
        slug="nfcorpus",
        raw_dir="nfcorpus",
        raw_format="jsonl_tsv",
        expected_raw_files=("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"),
        notes="BEIR-style JSONL corpus/queries plus test qrels TSV.",
    ),
    DatasetSpec(
        task_name="IFIRScifact",
        slug="ifir_scifact",
        raw_dir="ifir_scifact",
        raw_format="jsonl_tsv",
        expected_raw_files=(
            "corpus.jsonl",
            "queries.jsonl",
            "instructions.jsonl",
            "qrels/test.tsv",
        ),
        notes="IFIR layout with query instructions.",
    ),
    DatasetSpec(
        task_name="SciFact",
        slug="scifact",
        raw_dir="scifact",
        raw_format="jsonl_tsv",
        expected_raw_files=("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"),
        notes="BEIR-style JSONL corpus/queries plus test qrels TSV.",
    ),
    DatasetSpec(
        task_name="LitSearchRetrieval",
        slug="litsearch",
        raw_dir="litsearch",
        raw_format="parquet_dir_shards",
        expected_raw_files=(
            "corpus/test-00000-of-00001.parquet",
            "queries/test-00000-of-00001.parquet",
            "qrels/test-00000-of-00001.parquet",
        ),
        notes="MTEB parquet shard layout.",
    ),
)

DATASETS_BY_NAME = {spec.task_name: spec for spec in TARGET_DATASETS}
DATASETS_BY_SLUG = {spec.slug: spec for spec in TARGET_DATASETS}


def dataset_specs_for_selection(selection: str) -> list[DatasetSpec]:
    """Return target dataset specs for one CLI selection."""

    if selection == "all":
        return list(TARGET_DATASETS)
    if selection in DATASETS_BY_NAME:
        return [DATASETS_BY_NAME[selection]]
    if selection in DATASETS_BY_SLUG:
        return [DATASETS_BY_SLUG[selection]]
    valid = sorted([spec.task_name for spec in TARGET_DATASETS] + ["all"])
    raise CorpusAssetError(f"Unknown dataset {selection!r}; expected one of {valid}")
