"""Planetary Computer SAS helpers. Offline; no network."""

from datetime import datetime, timezone

from monarc.data.pc_sas import (
    apply_sas_token,
    is_azure_blob_href,
    sas_token_url,
    token_expired,
    unsigned_href,
)


def test_azure_blob_detection():
    assert is_azure_blob_href(
        "https://naipeuwest.blob.core.windows.net/naip/v002/co/x.tif"
    )
    assert not is_azure_blob_href("s3://naip-visualization/co/x.tif")
    assert not is_azure_blob_href("https://colorado-public-imagery.s3.amazonaws.com/x.tif")


def test_token_url_and_expiry():
    assert sas_token_url("naip").endswith("/token/naip")
    assert token_expired("1970-01-01T00:00:00Z", now=datetime(2026, 8, 31, tzinfo=timezone.utc))
    assert not token_expired(
        "2099-01-01T00:00:00Z", now=datetime(2026, 8, 31, tzinfo=timezone.utc)
    )


def test_unsigned_roundtrip():
    href = "https://naipeuwest.blob.core.windows.net/naip/x.tif"
    signed = apply_sas_token(href, "se=2099&sig=z")
    assert unsigned_href(signed) == href
