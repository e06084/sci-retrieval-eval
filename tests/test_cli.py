"""Smoke tests for the CLI entry point."""

from eval_platform.cli.main import app


def test_cli_app_importable() -> None:
    assert app is not None
