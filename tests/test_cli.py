"""Smoke tests for the CLI entry point."""

from pathlib import Path

from typer.testing import CliRunner

from eval_platform.cli.main import app

runner = CliRunner()


def test_cli_app_importable() -> None:
    assert app is not None


def test_config_show_outputs_redacted_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "s3:\n"
        "  bucket: test-bucket\n"
        "  access_key_id: REAL-AK\n"
        "embedding:\n"
        "  batch_size: 8\n"
        "  endpoints:\n"
        "    - url: http://embed\n"
        "      api_key: SECRET\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["config-show", "--config", str(config_path)])

    assert result.exit_code == 0
    assert '"bucket": "test-bucket"' in result.stdout
    assert '"access_key_id": "***"' in result.stdout
    assert '"api_key": "***"' in result.stdout


def test_config_show_cli_override_wins(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "s3:\n"
        "  prefix: from-yaml\n"
        "embedding:\n"
        "  batch_size: 8\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "config-show",
            "--config",
            str(config_path),
            "--s3-prefix",
            "from-cli",
            "--embedding-batch-size",
            "16",
        ],
    )

    assert result.exit_code == 0
    assert '"prefix": "from-cli"' in result.stdout
    assert '"batch_size": 16' in result.stdout
