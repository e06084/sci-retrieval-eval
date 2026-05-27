#!/usr/bin/env python3
"""Plan real five-dataset corpus/index artifact builds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval_platform.corpus_assets import (  # noqa: E402
    CorpusAssetError,
    add_common_args,
    build_plan_for_datasets,
    dataset_specs_for_selection,
    inventory_corpus_assets,
    load_config_and_client,
    make_s3_artifact_store,
    output_payload,
    raw_prefix_exists,
    raw_prefix_key,
    redacted_config_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print the build plan without writing artifacts. This is the default.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Reserved for a later implementation; current script refuses real writes.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.execute:
        raise CorpusAssetError(
            "--execute is intentionally not implemented in this planning script; "
            "use the dry-run plan to drive the existing corpus_build runner with explicit clients"
        )

    config, client = load_config_and_client(args.config)
    if not config.s3.bucket:
        raise CorpusAssetError("config.s3.bucket is required")

    datasets = dataset_specs_for_selection(args.dataset)
    store = make_s3_artifact_store(
        config=config,
        s3_prefix=args.s3_prefix,
        client=client,
    )
    inventory = inventory_corpus_assets(
        store=store,
        raw_client=client,
        bucket=config.s3.bucket,
        raw_prefix=args.raw_prefix,
        datasets=datasets,
    )
    raw_exists_by_slug = {
        spec.slug: raw_prefix_exists(
            client,
            bucket=config.s3.bucket,
            prefix=raw_prefix_key(args.raw_prefix, spec),
        )
        for spec in datasets
    }
    plan = build_plan_for_datasets(
        datasets=datasets,
        run_id=args.run_id,
        bucket=config.s3.bucket,
        raw_prefix=args.raw_prefix,
        s3_prefix=args.s3_prefix,
        raw_exists_by_slug=raw_exists_by_slug,
        reuse_existing=args.reuse_existing,
        inventory=inventory,
    )
    payload = {
        "kind": "five_dataset_corpus_asset_build_plan",
        "execute": False,
        "config": redacted_config_summary(config),
        "inventory": inventory,
        **plan,
    }
    output_payload(payload, args.output)
    return payload


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
