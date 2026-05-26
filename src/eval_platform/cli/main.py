"""Official CLI entry point."""

import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_package_version
from pathlib import Path

import typer

from eval_platform.config import dump_redacted_config, load_platform_config

app = typer.Typer(help="Evaluation platform CLI")
CONFIG_OPTION = typer.Option(None, "--config", help="Path to config YAML.")
S3_PREFIX_OPTION = typer.Option(None, "--s3-prefix", help="Override s3.prefix.")
EMBEDDING_BATCH_SIZE_OPTION = typer.Option(
    None,
    "--embedding-batch-size",
    help="Override embedding.batch_size.",
)


@app.callback()
def cli() -> None:
    """Evaluation platform CLI."""


def _package_version() -> str:
    try:
        return get_package_version("sci-retrieval-eval")
    except PackageNotFoundError:
        return "0.0.0+unknown"


@app.command()
def version() -> None:
    """Print the current version."""
    typer.echo(f"sci-retrieval-eval {_package_version()}")


@app.command("config-show")
def config_show(
    config: Path | None = CONFIG_OPTION,
    s3_prefix: str | None = S3_PREFIX_OPTION,
    embedding_batch_size: int | None = EMBEDDING_BATCH_SIZE_OPTION,
) -> None:
    """Print redacted merged config as JSON."""
    cli_overrides: dict[str, object] = {}
    if s3_prefix is not None:
        cli_overrides.setdefault("s3", {})
        cli_overrides["s3"] = {"prefix": s3_prefix}
    if embedding_batch_size is not None:
        cli_overrides.setdefault("embedding", {})
        cli_overrides["embedding"] = {"batch_size": embedding_batch_size}

    loaded = load_platform_config(config, cli_overrides=cli_overrides or None)
    typer.echo(json.dumps(dump_redacted_config(loaded), indent=2, sort_keys=True))
