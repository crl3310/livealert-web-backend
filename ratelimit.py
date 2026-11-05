"""
LiveAlert — Rate limiting
=========================
Two protections, both in memory (fine for ONE uvicorn worker, which is how this
app runs; see the notes in auth.py before ever running more than one):

  1. `limiter` (slowapi): max requests per minute.
       - Callers with a VALID Firebase ID token are counted per USER, so a whole
         station sharing one office IP is not treated as a single person.
       - Everyone else (login, refresh, missing/forged/expired tokens, ...) is
         counted per client IP, so nobody can pick their own bucket by making
         up a token.
       - A blanket default applies to every route; stricter limits are set with
         @limiter.limit("N/minute") on the sensitive ones.
  2. `acquire_stream` / `release_stream`: max open live (SSE) connections per
     user, because each one holds a Firestore listener.

Environment settings (all optional):
  RATE_LIMIT_ENABLED=0           switch slowapi off (emergency kill switch)
  RATE_LIMIT_DEFAULT=240/minute  default for routes with no limit of their own
  MAX_STREAMS_PER_USER=5         open live connections per user, per stream type
"""

import hashlib
import os
import threading
import time
import uuid
from collections import OrderedDict

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

ENABLED = os.getenv("RATE_LIMIT_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
DEFAULT_LIMIT = os.getenv("RATE_LIMIT_DEFAULT", "240/minute")
MAX_STREAMS_PER_USER = int(os.getenv("MAX_STREAMS_PER_USER", "5"))


# Verified-token cache: sha256(token) -> (expires_at, uid or None).
# Signature checks are local (Google's public certs are cached by the SDK) but
# not free, and the same token arrives on every request, so remember the answer
# briefly. Failures are cached too, so a flood of junk tokens can't burn CPU.
_KEY_CACHE_TTL_SEC = 30
_KEY_CACHE_MAX = 4096
_MAX_TOKEN_LEN = 4096
_key_cache: "OrderedDict[str, tuple[float, str | None]]" = OrderedDict()
_key_cache_lock = threading.Lock()


def _verify_signature(token: str) -> str | None:
    """
    The ONLY place that talks to Firebase. Returns the verified uid, or None.
    check_revoked is off on purpose: that is a network call per request, and
    security.verify_token does it anyway on the real route. Here we only need
    "is this really signed by Google, and for whom" in order to pick a bucket.
    """
    try:
        from firebase_admin import auth as firebase_auth
        from firebase import get_db

        get_db()  # makes sure the default Firebase app is initialised
        claims = firebase_auth.verify_id_token(token, check_revoked=False, clock_skew_seconds=10)
        uid = claims.get("uid")
        return uid if isinstance(uid, str) and 0 < len(uid) <= 128 else None
    except Exception:
        return None


def _verified_uid(token: str) -> str | None:
    if len(token) > _MAX_TOKEN_LEN or token.count(".") != 2:
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    now = time.time()
    with _key_cache_lock:
        hit = _key_cache.get(digest)
        if hit is not None and hit[0] > now:
            _key_cache.move_to_end(digest)
            return hit[1]
    uid = _verify_signature(token)          # outside the lock: may be slow
    with _key_cache_lock:
        _key_cache[digest] = (now + _KEY_CACHE_TTL_SEC, uid)
        _key_cache.move_to_end(digest)
        while len(_key_cache) > _KEY_CACHE_MAX:
            _key_cache.popitem(last=False)
    return uid


def rate_limit_key(request: Request) -> str:
    """
    Who is being counted: the logged-in user if the Bearer token is genuinely
    signed by Firebase, else the client IP.

    The signature IS checked here. If it weren't, a caller could invent a user id
    per request (never being throttled) or use someone else's id to burn that
    person's allowance and lock them out.
    """
    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        uid = _verified_uid(token.strip())
        if uid:
            return f"user:{uid}"
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=rate_limit_key, default_limits=[DEFAULT_LIMIT], enabled=ENABLED)


def _retry_after(exc: RateLimitExceeded) -> int:
    try:
        return int(exc.limit.limit.get_expiry())   # the length of the window, e.g. 60
    except Exception:
        return 60


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Same {"detail": ...} shape as every other error, so the frontend shows it normally."""
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please slow down and try again in a moment."},
        headers={"Retry-After": str(_retry_after(exc))},
    )


# ---------------------------------------------------------------------------
# Open live-connection (SSE) cap
# ---------------------------------------------------------------------------
_STREAM_MAX_AGE_SEC = 8 * 60 * 60   # a slot older than this is assumed leaked and forgotten
_open_streams: dict[str, dict[str, float]] = {}
_streams_lock = threading.Lock()


def acquire_stream(uid: str, name: str, max_open: int | None = None) -> str:
    """
    Reserve one open-connection slot for this user and stream type, or raise 429.
    Returns a slot id to hand back to release_stream() when the connection closes.
    """
    limit = max_open if max_open is not None else MAX_STREAMS_PER_USER
    key = f"{name}:{uid}"
    now = time.time()
    with _streams_lock:
        slots = _open_streams.setdefault(key, {})
        for slot_id in [s for s, started in slots.items() if now - started > _STREAM_MAX_AGE_SEC]:
            del slots[slot_id]
        if len(slots) >= limit:
            raise HTTPException(
                status_code=429,
                detail="Too many live connections are open for this account. Close other tabs and try again.",
                headers={"Retry-After": "10"},
            )
        slot_id = uuid.uuid4().hex
        slots[slot_id] = now
        return slot_id


def release_stream(uid: str, name: str, slot_id: str) -> None:
    key = f"{name}:{uid}"
    with _streams_lock:
        slots = _open_streams.get(key)
        if slots is not None:
            slots.pop(slot_id, None)
            if not slots:
                _open_streams.pop(key, None)