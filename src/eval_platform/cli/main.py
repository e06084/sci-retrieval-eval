"""Official CLI entry point."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_package_version

import typer

app = typer.Typer(help="Evaluation platform CLI")


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
