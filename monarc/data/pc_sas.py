"""Anonymous Planetary Computer SAS tokens. No Azure or AWS account required."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from urllib.parse import quote, urlparse

JsonObj = dict[str, Any]
HttpGet = Callable[[str], JsonObj]

PC_SAS_TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"
PC_SAS_SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
PC_BLOB_HOST_SUFFIX = ".blob.core.windows.net"
DEFAULT_SAS_SKEW = timedelta(minutes=5)


class SasError(RuntimeError):
    """Raised when a SAS token cannot be issued or applied."""


def sas_token_url(collection: str) -> str:
    return PC_SAS_TOKEN_URL.format(collection=str(collection))


def sas_sign_url(href: str) -> str:
    return f"{PC_SAS_SIGN_URL}?href={quote(str(href), safe='')}"


def is_azure_blob_href(href: str) -> bool:
    host = (urlparse(str(href)).hostname or "").lower()
    return host.endswith(PC_BLOB_HOST_SUFFIX)


def parse_expiry(text: str | None) -> datetime | None:
    if not text:
        return None
    raw = str(text).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise SasError(f"unparseable SAS expiry {text!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def token_expired(expiry: str | None, *, now: datetime | None = None, skew: timedelta = DEFAULT_SAS_SKEW) -> bool:
    parsed = parse_expiry(expiry)
    if parsed is None:
        return True
    clock = now or datetime.now(timezone.utc)
    return clock + skew >= parsed


def apply_sas_token(href: str, token: str) -> str:
    """Attach a collection SAS token to an unsigned Azure blob HREF."""
    base = str(href).strip()
    tok = str(token).strip().lstrip("?")
    if not base:
        raise SasError("empty href")
    if not tok:
        raise SasError("empty SAS token")
    if "sig=" in base and "se=" in base:
        return base
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}{tok}"


def unsigned_href(href: str) -> str:
    """Strip a query string (SAS) so the catalog HREF can be re-signed later."""
    raw = str(href).strip()
    if "?" not in raw:
        return raw
    return raw.split("?", 1)[0]


def fetch_sas_token(
    collection: str,
    *,
    http_get: HttpGet,
) -> tuple[str, str]:
    """Return ``(token, msft:expiry)`` for a STAC collection. Anonymous GET."""
    url = sas_token_url(collection)
    data = http_get(url)
    token = str(data.get("token") or "")
    expiry = str(data.get("msft:expiry") or "")
    if not token or not expiry:
        raise SasError(f"SAS response for {collection!r} missing token or expiry")
    return token, expiry


def sign_href(
    href: str,
    *,
    collection: str = "naip",
    token: str | None = None,
    expiry: str | None = None,
    http_get: HttpGet | None = None,
) -> tuple[str, str]:
    """Return ``(signed_href, expiry)``. Fetches a collection token if needed."""
    raw = unsigned_href(href)
    if token is None:
        if http_get is None:
            raise SasError("http_get required to issue a SAS token")
        token, expiry = fetch_sas_token(collection, http_get=http_get)
    if expiry is None:
        raise SasError("SAS expiry is required")
    return apply_sas_token(raw, token), expiry


def sas_refresh_block(collection: str) -> dict[str, str]:
    return {
        "collection": collection,
        "token_url": sas_token_url(collection),
        "sign_url": PC_SAS_SIGN_URL,
        "method": "sas_token",
    }
