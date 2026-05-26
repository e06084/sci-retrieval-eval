"""Tests for config redaction."""

from eval_platform.config import PlatformConfig, dump_redacted_config


def test_redaction_masks_nested_sensitive_values() -> None:
    config = PlatformConfig.model_validate(
        {
            "s3": {
                "access_key_id": "AK",
                "secret_access_key": "SECRET",
            },
            "embedding": {
                "endpoints": [{"url": "http://embed", "api_key": "KEY"}],
            },
            "elasticsearch": {
                "url": "http://es",
                "password": "PW",
            },
            "search_runtime": {
                "rewrite": {
                    "api_key": "RK",
                    "base_url": "http://rewrite",
                }
            },
        }
    )

    redacted = dump_redacted_config(config)

    assert redacted["s3"]["access_key_id"] == "***"
    assert redacted["s3"]["secret_access_key"] == "***"
    assert redacted["embedding"]["endpoints"][0]["api_key"] == "***"
    assert redacted["elasticsearch"]["password"] == "***"
    assert redacted["search_runtime"]["rewrite"]["api_key"] == "***"
    assert redacted["embedding"]["endpoints"][0]["url"] == "http://embed"


def test_redaction_masks_none_sensitive_values_as_stars() -> None:
    config = PlatformConfig.model_validate(
        {
            "s3": {
                "access_key_id": None,
                "secret_access_key": None,
            },
            "embedding": {
                "endpoints": [{"url": "http://embed", "api_key": None}],
            },
        }
    )

    redacted = dump_redacted_config(config)

    assert redacted["s3"]["access_key_id"] == "***"
    assert redacted["s3"]["secret_access_key"] == "***"
    assert redacted["embedding"]["endpoints"][0]["api_key"] == "***"
