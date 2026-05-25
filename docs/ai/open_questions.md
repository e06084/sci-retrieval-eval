# Open Questions

## Evaluation Semantics

1. Should MTEB-standard evaluation be doc-level only?
2. Should internal evidence evaluation use chunk-level relevance derived from doc-level qrels?
3. How should chunk results be aggregated back to document IDs for MTEB scoring?
   - max chunk score
   - first chunk score
   - RRF over chunks grouped by doc
   - other method

## Artifact Format

4. Should embeddings be stored as JSONL, Parquet, NumPy shards, or another format?
5. Should retrieval traces be stored as JSONL only, or also summarized into report JSON?

## Indexing

6. Should ES and Milvus index names be generated from artifact fingerprints?
7. Should index artifacts store only metadata, or also sampled validation results?

## Reranking

8. Should reranker score replace RRF score for final ranking, or should both scores be preserved?
9. How many candidates should be reranked by default?

## Chunking

12. Should chunked corpus artifacts always record external chunker provenance, even for local-only experiments?
13. Should dirty external chunker repositories be rejected at runner time only, or also flagged in manifest metadata?

## Frontend

10. Should the first frontend be read-only only?
11. Should evaluation runs be started from CLI first, before adding frontend controls?
