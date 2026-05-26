"""Information retrieval metric formulas."""

from __future__ import annotations

import math

from eval_platform.metrics.schema import RankedDoc

METRIC_PREFIXES = (
    "ndcg",
    "map",
    "recall",
    "precision",
    "mrr",
    "hit_rate",
)


def compute_query_metrics(
    ranked_docs: list[RankedDoc],
    relevant_docs: dict[str, float],
    k_values: list[int],
) -> dict[str, float]:
    """Compute MTEB/pytrec_eval-style metrics for one query."""

    positives = {doc_id: rel for doc_id, rel in relevant_docs.items() if rel > 0}
    metrics: dict[str, float] = {}
    if not positives:
        for k in k_values:
            metrics.update(_zero_metrics(k))
        return metrics

    for k in k_values:
        top_docs = ranked_docs[:k]
        positive_hits = 0
        ap_sum = 0.0
        first_positive_rank: int | None = None
        dcg = 0.0
        for rank, doc in enumerate(top_docs, start=1):
            relevance = positives.get(doc.doc_id, 0.0)
            if relevance > 0:
                positive_hits += 1
                ap_sum += positive_hits / rank
                if first_positive_rank is None:
                    first_positive_rank = rank
            dcg += relevance / math.log2(rank + 1)

        ideal_relevances = sorted(positives.values(), reverse=True)[:k]
        idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal_relevances, start=1))

        metrics[f"precision_at_{k}"] = positive_hits / k
        metrics[f"recall_at_{k}"] = positive_hits / len(positives)
        metrics[f"hit_rate_at_{k}"] = 1.0 if positive_hits > 0 else 0.0
        metrics[f"mrr_at_{k}"] = 0.0 if first_positive_rank is None else 1.0 / first_positive_rank
        metrics[f"map_at_{k}"] = ap_sum / len(positives)
        metrics[f"ndcg_at_{k}"] = 0.0 if idcg == 0 else dcg / idcg

    return metrics


def aggregate_query_metrics(
    query_metrics: list[dict[str, float]],
    k_values: list[int],
) -> dict[str, float]:
    """Average per-query metrics arithmetically."""

    metric_names = [f"{prefix}_at_{k}" for k in k_values for prefix in METRIC_PREFIXES]
    if not query_metrics:
        return dict.fromkeys(metric_names, 0.0)
    aggregate: dict[str, float] = {}
    for name in metric_names:
        aggregate[name] = sum(metrics.get(name, 0.0) for metrics in query_metrics) / len(
            query_metrics
        )
    return aggregate


def _zero_metrics(k: int) -> dict[str, float]:
    return {f"{prefix}_at_{k}": 0.0 for prefix in METRIC_PREFIXES}
