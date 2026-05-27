#!/usr/bin/env python3
"""Inventory real five-dataset corpus/index artifacts in S3."""

from __future__ import annotations

import argparse
from typing import Any

from corpus_asset_common import (
    add_common_args,
    inventory_corpus_assets,
    load_config_and_client,
    make_s3_artifact_store,
    output_payload,
    redacted_config_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, client = load_config_and_client(args.config)
    if not config.s3.bucket:
        raise SystemExit("config.s3.bucket is required")

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
    )
    payload = {
        "kind": "five_dataset_corpus_asset_inventory",
        "s3_bucket": config.s3.bucket,
        "artifact_prefix": args.s3_prefix,
        "raw_prefix": args.raw_prefix,
        "config": redacted_config_summary(config),
        **inventory,
    }
    output_payload(payload, args.output)
    return payload


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
