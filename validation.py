"""
validation.py - shared input validation for the LiveAlert API
=============================================================
One place for the rules every router uses, so each file doesn't invent its own.

  * clean_text()            - the single text sanitiser (control chars, bidi tricks, <>, length)
  * Annotated field types   - PersonName, EmailAddress, Badge, DocIdStr, Latitude, ...
  * Path/Query helpers      - DocIdPath, require_choice(), optional_doc_id()
  * StrictModel/PatchModel  - base classes for request bodies
  * install_validation_handlers(app)  - 422s that never echo the submitted input back
  * BodySizeLimitMiddleware           - rejects oversized request bodies before parsing

WIRING (main.py, once):
    from validation import install_validation_handlers, BodySizeLimitMiddleware
    install_validation_handlers(app)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=256 * 1024)

Environment switches:
    VALIDATION_FORBID_EXTRA=1   reject request bodies that contain unknown fields
                                (default: unknown fields are ignored, as before)
"""

import os
import re
import unicodedata
from typing import Annotated, Any, ClassVar

from fastapi import HTTPException, Path
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

FORBID_EXTRA = os.getenv("VALIDATION_FORBID_EXTRA", "0").lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
# Invisible / direction-changing characters. They let someone make "Admin" look
# like another name, hide text in logs, or reorder what an operator reads.
_INVISIBLE = frozenset(
    "\u200b\u200e\u200f\u2028\u2029\u202a\u202b\u202c\u202d\u202e"
    "\u2060\u2066\u2067\u2068\u2069\ufeff"
)


def clean_text(
    value: Any,
    *,
    label: str = "Value",
    min_len: int = 0,
    max_len: int = 200,
    multiline: bool = False,
    allow_markup: bool = False,
) -> str:
    """Normalise and validate one piece of user text. Raises ValueError (-> 422)."""
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text.")
    if len(value) > max_len * 2 + 64:  # cheap guard before any per-character work
        raise ValueError(f"{label} is too long (max {max_len} characters).")

    text = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip() if multiline else " ".join(text.split())

    for ch in text:
        if ch in _INVISIBLE:
            raise ValueError(f"{label} contains invisible or direction-control characters.")
        if unicodedata.category(ch) in ("Cc", "Cs", "Co") and not (multiline and ch in "\n\t"):
            raise ValueError(f"{label} contains control characters.")
    if not allow_markup and ("<" in text or ">" in text):
        raise ValueError(f"{label} cannot contain < or >.")
    if len(text) > max_len:
        raise ValueError(f"{label} is too long (max {max_len} characters).")
    if len(text) < min_len:
        raise ValueError(f"{label} is required." if min_len == 1 else f"{label} is too short.")
    return text


def _text_type(label: str, *, min_len: int = 0, max_len: int = 200, multiline: bool = False):
    def _check(v: Any) -> str:
        return clean_text(v, label=label, min_len=min_len, max_len=max_len, multiline=multiline)

    return Annotated[str, Field(max_length=max_len * 2 + 64), AfterValidator(_check)]


def _person_name(v: Any) -> str:
    text = clean_text(v, label="Name", min_len=1, max_len=60)
    for ch in text:
        if not (ch.isalpha() or ch in " '\u2019.-" or unicodedata.category(ch).startswith("M")):
            raise ValueError("Name may only contain letters, spaces, apostrophes, hyphens and periods.")
    if not any(ch.isalpha() for ch in text):
        raise ValueError("Name must contain at least one letter.")
    return text


# ---------------------------------------------------------------------------
# Email, badge, passwords, tokens
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(
    r"^(?P<local>[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*)"
    r"@(?P<domain>(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24})$"
)


def clean_email(value: Any) -> str:
    text = clean_text(value, label="Email", min_len=1, max_len=254)
    m = _EMAIL_RE.match(text)
    if not m or len(m.group("local")) > 64:
        raise ValueError("Enter a valid email address.")
    return text.lower()


_BADGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,31}$")
BADGE_MESSAGE = "Badge number must be 1-32 characters: letters, numbers or hyphens only."


def is_valid_badge(raw: Any) -> bool:
    return isinstance(raw, str) and bool(_BADGE_RE.match(raw.strip()))


def clean_badge(raw: Any) -> str:
    """Trim, validate and UPPER-CASE a badge so PNP-123 and pnp-123 are one badge."""
    if not isinstance(raw, str) or not _BADGE_RE.match(raw.strip()):
        raise ValueError(BADGE_MESSAGE)
    return raw.strip().upper()


def _login_identifier(v: Any) -> str:
    # Bounds + no control characters only. Shape (badge vs email) is checked in
    # the login handler so a malformed identifier fails exactly like a wrong
    # password and counts toward the brute-force limiter.
    return clean_text(v, label="Login", min_len=1, max_len=254, allow_markup=True)


def _login_password(v: Any) -> str:
    if not isinstance(v, str) or not (1 <= len(v) <= 1024) or "\x00" in v:
        raise ValueError("Password must be 1-1024 characters.")
    return v  # never stripped or altered


def _new_password(v: Any) -> str:
    if not isinstance(v, str) or "\x00" in v:
        raise ValueError("Password contains invalid characters.")
    if len(v) > 128:
        raise ValueError("Password must be at most 128 characters.")
    if v != v.strip() or any(unicodedata.category(c) == "Cc" for c in v):
        raise ValueError("Password cannot start or end with spaces or contain control characters.")
    return v


_JWT_RE = re.compile(r"^[A-Za-z0-9_-]{1,2048}\.[A-Za-z0-9_-]{1,2048}\.[A-Za-z0-9_-]{0,2048}$")
_REFRESH_RE = re.compile(r"^[A-Za-z0-9._~+/=-]{16,4096}$")


def _jwt(v: Any) -> str:
    if not isinstance(v, str) or len(v) > 6200 or not _JWT_RE.match(v.strip()):
        raise ValueError("Invalid token.")
    return v.strip()


def _refresh_token(v: Any) -> str:
    if not isinstance(v, str) or not _REFRESH_RE.match(v.strip()):
        raise ValueError("Invalid token.")
    return v.strip()


# ---------------------------------------------------------------------------
# Firestore document ids
# ---------------------------------------------------------------------------
# Must start with a letter/digit, which rules out ".", ".." and the reserved
# "__name__" form Firestore refuses. No "/" is possible. If a real document id
# in your data starts with "_" or "." you will need to loosen this.
DOC_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
_DOC_ID_RE = re.compile(DOC_ID_PATTERN)

# Stream channel names end up inside HTML and JavaScript, so they are stricter.
CHANNEL_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"


def clean_doc_id(v: Any, label: str = "id") -> str:
    if not isinstance(v, str) or not _DOC_ID_RE.match(v.strip()):
        raise ValueError(f"Invalid {label}.")
    return v.strip()


def _doc_id(v: Any) -> str:
    return clean_doc_id(v)


def _doc_id_or_blank(v: Any) -> str:
    return "" if v is None or (isinstance(v, str) and not v.strip()) else clean_doc_id(v)


def _doc_id_or_none(v: Any) -> str | None:
    return None if v is None or (isinstance(v, str) and not v.strip()) else clean_doc_id(v)


DocIdPath = Annotated[str, Path(min_length=1, max_length=128, pattern=DOC_ID_PATTERN)]
ChannelIdPath = Annotated[str, Path(min_length=1, max_length=64, pattern=CHANNEL_ID_PATTERN)]


def optional_doc_id(value: str | None, label: str = "id") -> str | None:
    """For query-string ids: '' / missing -> None, anything else must be a valid id (else 422)."""
    if value is None or not value.strip():
        return None
    try:
        return clean_doc_id(value, label)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ---------------------------------------------------------------------------
# Allowed value sets (all lower-case)
# ---------------------------------------------------------------------------
ROLE_VALUES = frozenset({"responder", "responder_admin", "super_admin"})
ACCOUNT_STATUS_VALUES = frozenset({"on", "off", "inactive", "active", "available", "deployed", "standby"})
DUTY_VALUES = frozenset({"on_duty", "off_duty", "onduty", "offduty", "on", "off", "duty", "deployed",
                         "true", "false", "1", "0"})
UNIT_STATUS_VALUES = frozenset({"on", "off", "inactive"})


def _choice(allowed: frozenset[str], label: str, *, keep_case: bool = False):
    def _check(v: Any) -> str:
        if not isinstance(v, str) or v.strip().lower() not in allowed:
            raise ValueError(f"{label} must be one of: {', '.join(sorted(allowed))}.")
        return v.strip() if keep_case else v.strip().lower()

    return _check


def require_choice(value: str | None, allowed: frozenset[str], label: str) -> str | None:
    """For query-string filters: None/'' -> None, unknown value -> 422."""
    if value is None or not value.strip():
        return None
    if value.strip().lower() not in allowed:
        raise HTTPException(status_code=422, detail=f"{label} must be one of: {', '.join(sorted(allowed))}.")
    return value.strip().lower()


# Free-form but format-checked (extend to an enum once you list the real values).
_REPORT_STATUS_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_ -]{0,31}$")
_REPORT_LEVEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,31}$")


def _report_status(v: Any) -> str:
    if not isinstance(v, str) or not _REPORT_STATUS_RE.match(v.strip()):
        raise ValueError("Invalid status.")
    return v.strip()


def _report_level(v: Any) -> str:
    if not isinstance(v, str) or not _REPORT_LEVEL_RE.match(v.strip()):
        raise ValueError("Invalid level.")
    return v.strip()


# ---------------------------------------------------------------------------
# Annotated types for request models
# ---------------------------------------------------------------------------
PersonName = Annotated[str, Field(max_length=200), AfterValidator(_person_name)]
EmailAddress = Annotated[str, Field(max_length=400), AfterValidator(clean_email)]
Badge = Annotated[str, Field(max_length=100), AfterValidator(clean_badge)]

StationName = _text_type("Station name", min_len=1, max_len=100)
CommanderName = _text_type("Commander name", max_len=100)
UnitName = _text_type("Unit name", min_len=1, max_len=60)
Jurisdiction = _text_type("Jurisdiction", min_len=1, max_len=120)
ShortLabel = _text_type("Value", max_len=120)
IncidentType = _text_type("Incident type", min_len=1, max_len=100)
Description = _text_type("Description", max_len=2000, multiline=True)

DocIdStr = Annotated[str, Field(max_length=128), AfterValidator(_doc_id)]
DocIdOrBlank = Annotated[str, Field(max_length=128), AfterValidator(_doc_id_or_blank)]
OptionalDocId = Annotated[str | None, AfterValidator(_doc_id_or_none)]

RoleValue = Annotated[str, Field(max_length=40), AfterValidator(_choice(ROLE_VALUES, "Role", keep_case=True))]
AccountStatusValue = Annotated[str, Field(max_length=20), AfterValidator(_choice(ACCOUNT_STATUS_VALUES, "Status"))]
DutyValue = Annotated[str, Field(max_length=20), AfterValidator(_choice(DUTY_VALUES, "Duty"))]
UnitStatusValue = Annotated[str, Field(max_length=20), AfterValidator(_choice(UNIT_STATUS_VALUES, "Status"))]
ReportStatusValue = Annotated[str, Field(max_length=40), AfterValidator(_report_status)]
ReportLevelValue = Annotated[str, Field(max_length=40), AfterValidator(_report_level)]

LoginIdentifier = Annotated[str, Field(max_length=800), AfterValidator(_login_identifier)]
LoginPassword = Annotated[str, Field(max_length=1100), AfterValidator(_login_password)]
NewPassword = Annotated[str, Field(max_length=200), AfterValidator(_new_password)]
IdTokenStr = Annotated[str, Field(max_length=6300), AfterValidator(_jwt)]
RefreshTokenStr = Annotated[str, Field(max_length=4200), AfterValidator(_refresh_token)]

Latitude = Annotated[float, Field(ge=-90, le=90, allow_inf_nan=False)]
Longitude = Annotated[float, Field(ge=-180, le=180, allow_inf_nan=False)]


# ---------------------------------------------------------------------------
# Base models
# ---------------------------------------------------------------------------
class StrictModel(BaseModel):
    """Request body base: unknown fields ignored (or rejected with VALIDATION_FORBID_EXTRA=1)."""

    model_config = ConfigDict(extra="forbid" if FORBID_EXTRA else "ignore")


class PatchModel(StrictModel):
    """PATCH body base. An explicit `null` counts as 'not sent' unless the field is listed in
    `nullable_fields`. Before this, {"status": null} on an account quietly became "on"."""

    nullable_fields: ClassVar[frozenset[str]] = frozenset()

    @model_validator(mode="before")
    @classmethod
    def _drop_null_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None or k in cls.nullable_fields}
        return data


# ---------------------------------------------------------------------------
# App-level helpers
# ---------------------------------------------------------------------------
def install_validation_handlers(app) -> None:
    """422 responses with a readable `detail` string and NO echo of the submitted value.
    FastAPI's default 422 includes the offending input - including passwords on /auth/login."""

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request, exc: RequestValidationError):
        errors = []
        for err in exc.errors():
            loc = [str(p) for p in err.get("loc", ()) if p not in ("body", "query", "path")]
            msg = str(err.get("msg", "Invalid value")).removeprefix("Value error, ")
            errors.append({"field": ".".join(loc), "message": msg})
        first = errors[0] if errors else {"field": "", "message": "Invalid request."}
        detail = f"{first['field']}: {first['message']}" if first["field"] else first["message"]
        return JSONResponse(status_code=422, content={"detail": detail, "errors": errors})


class _TooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Pure-ASGI limit on request body size. Checks Content-Length up front and also counts
    streamed bytes, so chunked uploads can't slip past."""

    def __init__(self, app, max_bytes: int = 256 * 1024):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        for key, value in scope.get("headers", []):
            if key == b"content-length":
                try:
                    if int(value) > self.max_bytes:
                        return await self._reject(send, 413)
                except ValueError:
                    return await self._reject(send, 400)

        received = 0
        started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _TooLarge()
            return message

        async def tracking_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _TooLarge:
            if not started:
                await self._reject(send, 413)

    @staticmethod
    async def _reject(send, status: int):
        body = b'{"detail":"Request body too large."}' if status == 413 else b'{"detail":"Bad request."}'
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})