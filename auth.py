"""
LiveAlert — Auth Router
========================
Implements backend endpoints for frontend auth API calls.
Uses Firebase Identity Toolkit REST endpoints to sign users in and sign them up.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from firebase_admin import auth as firebase_auth
from google.api_core.exceptions import GoogleAPICallError

from accounts import set_admin_duty
from config import FIREBASE_AUTH_URL, API_KEY
from firebase import get_db
from security import get_verified_token
from validation import (
    IdTokenStr, LoginIdentifier, LoginPassword, NewPassword, RefreshTokenStr, StrictModel, is_valid_badge,
)

MAX_LOGIN_BODY_BYTES = 8 * 1024

router = APIRouter(prefix="/auth", tags=["Auth"])


# ---------------------------------------------------------------------------
# Login brute-force protection
#
# Failed logins (wrong password / unknown user) are counted per ACCOUNT and per
# client IP inside a sliding window. Hitting a limit locks that key out for
# LOGIN_LOCKOUT_SEC and /auth/login answers 429 + Retry-After without even
# contacting Firebase. A successful login clears the account's counter (never
# the IP's, so someone with one valid account can't reset their own attack on
# others). State is in memory: it resets on restart and isn't shared between
# multiple worker processes (see notes if you ever run more than one).
# ---------------------------------------------------------------------------
LOGIN_MAX_FAILS_PER_ACCOUNT = 5     # wrong attempts allowed per account...
LOGIN_MAX_FAILS_PER_IP = 20         # ...and per IP (higher: offices share one IP)
LOGIN_WINDOW_SEC = 15 * 60          # failures older than this are forgotten
LOGIN_LOCKOUT_SEC = 15 * 60         # how long a locked key stays blocked

# Only set LOGIN_TRUST_PROXY=1 when the API sits behind a reverse proxy you
# control (nginx, Cloudflare, ...). X-Forwarded-For is trivially spoofable
# otherwise, which would let an attacker dodge the per-IP limit.
TRUST_PROXY_HEADERS = os.getenv("LOGIN_TRUST_PROXY", "0").lower() in {"1", "true", "yes", "on"}

# Admins now sign in with their BADGE NUMBER (sent in the same JSON field the
# page always used, "usernameOrEmail"). Email login is switched off by default.
# Set LOGIN_ALLOW_EMAIL=1 only if something (e.g. the mobile app) still signs in
# with an email through this endpoint while you migrate it.
ALLOW_EMAIL_LOGIN = os.getenv("LOGIN_ALLOW_EMAIL", "0").lower() in {"1", "true", "yes", "on"}

# The ONE error every failed login gets, whether the badge doesn't exist, the
# password is wrong, or the account has no email. Same status + same text means
# nobody can probe which badge numbers are real.
_GENERIC_BAD_LOGIN = "INVALID_LOGIN_CREDENTIALS"

# Firebase error codes that mean "these credentials are wrong".
_BAD_CREDENTIAL_CODES = ("INVALID_PASSWORD", "EMAIL_NOT_FOUND", "INVALID_LOGIN_CREDENTIALS")


class _LoginLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fails: dict[str, deque] = defaultdict(deque)  # key -> failure timestamps
        self._locked_until: dict[str, float] = {}            # key -> unix time

    def retry_after(self, key: str) -> int:
        """Seconds until `key` may try again (0 = not locked)."""
        now = time.time()
        with self._lock:
            until = self._locked_until.get(key, 0.0)
            if until > now:
                return int(until - now) + 1
            self._locked_until.pop(key, None)
            return 0

    def record_failure(self, key: str, limit: int) -> None:
        now = time.time()
        with self._lock:
            if len(self._fails) > 5000:
                self._purge(now)
            q = self._fails[key]
            while q and now - q[0] > LOGIN_WINDOW_SEC:
                q.popleft()
            q.append(now)
            if len(q) >= limit:
                self._locked_until[key] = now + LOGIN_LOCKOUT_SEC
                q.clear()

    def reset(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)
            self._locked_until.pop(key, None)

    def _purge(self, now: float) -> None:
        """Drop expired entries so random-username spam can't grow memory forever."""
        for k in [k for k, q in self._fails.items() if not q or now - q[-1] > LOGIN_WINDOW_SEC]:
            self._fails.pop(k, None)
        for k in [k for k, t in self._locked_until.items() if t <= now]:
            self._locked_until.pop(k, None)


_login_limiter = _LoginLimiter()


def _client_ip(request: Request) -> str:
    if TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _login_limit_keys(request: Request, identifier: str) -> list[tuple[str, int]]:
    return [
        (f"ip:{_client_ip(request)}", LOGIN_MAX_FAILS_PER_IP),
        (f"acct:{identifier.strip().lower()}", LOGIN_MAX_FAILS_PER_ACCOUNT),
    ]


def _enforce_login_limits(keys: list[tuple[str, int]]) -> None:
    """Raise 429 if any of these keys is currently locked out."""
    wait = max((_login_limiter.retry_after(k) for k, _ in keys), default=0)
    if wait > 0:
        minutes = (wait + 59) // 60
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed login attempts. Try again in {minutes} minute{'s' if minutes != 1 else ''}.",
            headers={"Retry-After": str(wait)},
        )


def _record_login_failure(keys: list[tuple[str, int]]) -> None:
    for key, limit in keys:
        _login_limiter.record_failure(key, limit)


def _reset_login_failures(keys: list[tuple[str, int]]) -> None:
    for key, _ in keys:
        if key.startswith("acct:"):  # never reset the IP counter on success
            _login_limiter.reset(key)


def _raise_firestore_unavailable(exc: Exception) -> None:
    """
    Firestore calls don't go through a single choke point the way
    _firebase_auth_request's REST calls do, so a hiccup here — a read/write
    quota hit, a transient network blip, an IAM misconfiguration — previously
    bubbled all the way up as a raw, unhandled 500 with a full stack trace
    instead of a clean error the frontend could actually show someone. This
    turns it into a 503 the frontend's existing try/catch blocks already
    know how to display. Always raises; never returns.
    """

    detail = "Database temporarily unavailable. Please try again in a moment."
    if isinstance(exc, GoogleAPICallError):
        detail = f"Database temporarily unavailable ({exc.__class__.__name__}). Please try again in a moment."
    raise HTTPException(status_code=503, detail=detail) from exc

# Universal default password for admin-created responder accounts. The
# Accounts page doesn't send a password at all when registering an officer —
# the officer isn't choosing it, and the admin isn't the one using it, so
# there's nothing for the frontend to actually own here. This is the single
# source of truth for that value; is_new (see login()/change-password below)
# is what forces it to actually get changed once the responder logs in.
DEFAULT_RESPONDER_PASSWORD = "default123"


class LoginRequest(StrictModel):
    usernameOrEmail: LoginIdentifier   # bounded, no control characters; shape is checked in login()
    password: LoginPassword            # 1-1024 chars, never stripped


class ChangePasswordRequest(StrictModel):
    idToken: IdTokenStr
    newPassword: NewPassword


class CodeRequest(StrictModel):
    email: LoginIdentifier | None = None
    usernameOrEmail: LoginIdentifier | None = None


def _firebase_auth_request(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{FIREBASE_AUTH_URL}:{endpoint}?key={API_KEY}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload_text = exc.read().decode("utf-8", errors="ignore")
        try:
            err = json.loads(payload_text)
            message = err.get("error", {}).get("message", "Firebase Auth request failed.")
        except json.JSONDecodeError:
            # Non-JSON body (e.g. an HTML error page) - log it server-side,
            # but only send a short generic message to the client.
            print(f"[auth._firebase_auth_request] {endpoint} returned HTTP {exc.code} with a non-JSON body")
            message = "Firebase Auth request failed."
        raise HTTPException(status_code=400, detail=message)
    except Exception as exc:
        # str(exc) can contain internals (hostnames, request URLs); log it, don't send it.
        print(f"[auth._firebase_auth_request] {endpoint}: {exc.__class__.__name__}: {exc}")
        raise HTTPException(status_code=503, detail="Authentication service is unavailable. Please try again.")


def _find_user_by_username_or_email(value: str) -> dict[str, Any] | None:
    db = get_db()
    # There's only one collection now — super_admin, responder_admin, and
    # plain responder accounts all live in Responders.
    users = db.collection("Responders")
    matches = list(users.where("username", "==", value).limit(1).stream())
    if matches:
        return {"id": matches[0].id, **(matches[0].to_dict() or {})}
    matches = list(users.where("email", "==", value).limit(1).stream())
    if matches:
        return {"id": matches[0].id, **(matches[0].to_dict() or {})}
    return None


def _find_user_by_badge(badge: str) -> dict[str, Any] | None:
    """
    Exact-match lookup of an account by badge number. Returns None when there
    is no match, AND when more than one account shares the badge (a data
    problem: refusing is safer than guessing which account was meant).
    """
    # Badges are saved in upper case, so "pnp-40921" typed by hand still finds
    # PNP-40921 (the typed form is tried too, for any older lower-case badge).
    candidates = list({badge, badge.upper()})
    matches = list(get_db().collection("Responders").where("badge", "in", candidates).limit(2).stream())
    if len(matches) > 1:
        print("[auth.login] More than one account shares a badge number; refusing login. Fix the duplicate in Responders.")
        return None
    if matches:
        return {"id": matches[0].id, **(matches[0].to_dict() or {})}
    return None


def _create_responder_for_user(user_id: str, username: str | None, email: str) -> None:
    db = get_db()
    ref = db.collection("Responders").document(user_id)
    if ref.get().exists:
        return

    # Matches the schema accounts.py / the Accounts + Units pages actually
    # use (firstName/lastName/stationId/unitId/is_new) — this used to write
    # an older shape (name/group) that predates that refactor and would
    # otherwise make self-registered accounts show up broken/unassignable
    # in those pages. "username" is included so _find_user_by_username_or_email
    # (which now queries this same collection) can still resolve login-by-
    # username, now that there's no separate Users doc to hold it.
    display_name = username or email.split("@")[0]
    first_name, _, last_name = display_name.partition(".")
    responder_data = {
        "username": username or "",
        "firstName": first_name or display_name,
        "lastName": last_name,
        "email": email,
        "badge": "",
        "role": "Responder",
        "stationId": "",
        "unitId": "",
        "status": "on",
        "is_new": False,  # they just set their own password during signup — no forced reset needed
        "createdAt": datetime.now(timezone.utc),
    }
    ref.set(responder_data)


@router.post("/login")
async def login(request: Request, payload: LoginRequest | None = Body(default=None)):
    """Authenticate an admin with badge number and password."""
    if payload is None:
        try:
            raw = await request.body()
            if len(raw) > MAX_LOGIN_BODY_BYTES:
                raise ValueError("body too large")
            data = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(data, dict):
                raise ValueError("body must be an object")
            payload = LoginRequest(**data)
        except Exception:
            # Covers bad JSON, wrong shape and failed validation alike (a raw
            # ValidationError here used to escape as a 500), and never echoes input.
            raise HTTPException(status_code=400, detail="Request body must be valid JSON with usernameOrEmail and password.")

    # Brute-force protection: refuse before doing any work if this IP or
    # account is locked out.
    identifier = payload.usernameOrEmail.strip()
    limit_keys = _login_limit_keys(request, identifier)
    _enforce_login_limits(limit_keys)

    # Resolve the badge number to the account's email on the SERVER. The email
    # is only ever used to talk to Firebase; it is never sent back to the browser.
    user = None
    email = None
    try:
        if "@" in identifier:
            if ALLOW_EMAIL_LOGIN:
                user = _find_user_by_username_or_email(identifier)
                email = identifier
        elif is_valid_badge(identifier):
            user = _find_user_by_badge(identifier)
            email = (user or {}).get("email") or None
        # Anything else (wrong shape) falls through to the generic failure below,
        # so it is indistinguishable from a wrong password and still rate-limited.
    except HTTPException:
        raise
    except Exception as exc:
        _raise_firestore_unavailable(exc)

  

    if not email:
        # Unknown badge, no email on file, or email login switched off: answer
        # exactly like a wrong password.
        _record_login_failure(limit_keys)
        raise HTTPException(status_code=400, detail=_GENERIC_BAD_LOGIN)

    # Also track by the resolved email, so alternating between badge and
    # email for the same account can't dodge the counter.
    email_key = (f"acct:{email.strip().lower()}", LOGIN_MAX_FAILS_PER_ACCOUNT)
    if email_key not in limit_keys:
        limit_keys.append(email_key)
        _enforce_login_limits([email_key])

    try:
        auth_data = _firebase_auth_request("signInWithPassword", {
            "email": email,
            "password": payload.password,
            "returnSecureToken": True,
        })
    except HTTPException as exc:
        detail = str(exc.detail)
        if detail.startswith(_BAD_CREDENTIAL_CODES):
            _record_login_failure(limit_keys)
            # Older Firebase projects answer INVALID_PASSWORD vs EMAIL_NOT_FOUND;
            # collapse them so the answer never says which part was wrong.
            raise HTTPException(status_code=400, detail=_GENERIC_BAD_LOGIN) from exc
        elif detail.startswith("TOO_MANY_ATTEMPTS_TRY_LATER"):
            # Firebase's own throttle kicked in; pass that on as a proper 429.
            raise HTTPException(
                status_code=429,
                detail="Too many attempts. Please try again later.",
                headers={"Retry-After": "60"},
            ) from exc
        raise

    _reset_login_failures(limit_keys)

    auth_user_id = auth_data.get("localId")
    is_new = False
    if auth_user_id:
        # Firebase Auth already confirmed the credentials are correct by this
        # point, so a Firestore hiccup here should degrade gracefully (log it,
        # fall back to is_new=False) rather than fail a login that was
        # otherwise genuinely valid. Worst case, a stale is_new gets caught by
        # a later page load's own profile check instead of right here.
        try:
            username = user.get("username") if user else None

            # Only one collection now, so this is a straightforward existence
            # check: bootstrap a plain-responder doc only if this account has
            # no profile at all yet. Any account with a role already assigned
            # (super_admin via /auth/signup, responder_admin/responder via
            # accounts.py) already has a doc here, so this only fires for a
            # bare self-registration that never got a follow-up role.
            responder_snap = get_db().collection("Responders").document(auth_user_id).get()
            if not responder_snap.exists:
                _create_responder_for_user(auth_user_id, username, email)
                responder_snap = get_db().collection("Responders").document(auth_user_id).get()

            # Surface whether this account still has a default/temporary
            # password so the frontend can force a password-change screen
            # before letting them do anything else.
            is_new = bool((responder_snap.to_dict() or {}).get("is_new", False))
        except Exception as exc:
            print(f"[auth.login] Firestore bookkeeping failed for {auth_user_id}, continuing with defaults: {exc}")

        # Admins go ON duty when they log in, so the report queue knows who can
        # receive reports. Plain responders (mobile app) are ignored by
        # set_admin_duty. Someone still on the temporary password has not
        # finished setting up, so they are not put on duty yet. A failure here
        # must never fail an otherwise valid login.
        if not is_new:
            try:
                set_admin_duty(get_db(), auth_user_id, "on_duty")
            except Exception as exc:
                print(f"[auth.login] Could not set on_duty for {auth_user_id}: {exc}")

    return {
        "status": "success",
        "message": "Authentication successful.",
        # No email / displayName here on purpose: the browser only needs the id
        # and is_new. (The profile chip gets name + role from /accounts/me.)
        "user": {
            "id": auth_user_id,
            "is_new": is_new,
        },
        "token": auth_data.get("idToken"),
        "refreshToken": auth_data.get("refreshToken"),
    }


@router.post("/change-password")
async def change_password(payload: ChangePasswordRequest):
    """
    Sets a new password for the currently-authenticated user, identified by
    the Firebase idToken returned from their login() call. Also clears
    is_new on their Responders doc so they aren't forced through the
    change-password flow again next time.
    """
    if len(payload.newPassword) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long.")
    if payload.newPassword == DEFAULT_RESPONDER_PASSWORD:
        raise HTTPException(status_code=400, detail="Choose a password other than the default one.")

    id_token = payload.idToken
    try:
        lookup = _firebase_auth_request("lookup", {"idToken": payload.idToken})
        email = ((lookup.get("users") or [{}])[0]).get("email")
        if email:
            fresh = _firebase_auth_request("signInWithPassword", {
                "email": email,
                "password": DEFAULT_RESPONDER_PASSWORD,
                "returnSecureToken": True,
            })
            id_token = fresh.get("idToken") or id_token
    except HTTPException as exc:
        print(f"[auth.change_password] Could not refresh idToken, using the original: {exc.detail}")

    try:
        auth_data = _firebase_auth_request("update", {
            "idToken": id_token,
            "password": payload.newPassword,
            "returnSecureToken": True,
        })
    except HTTPException as exc:
        friendly = {
            "CREDENTIAL_TOO_OLD_LOGIN_AGAIN": "Your session is too old. Please log in again.",
            "INVALID_ID_TOKEN": "Your session is no longer valid. Please log in again.",
            "TOKEN_EXPIRED": "Your session has expired. Please log in again.",
            "WEAK_PASSWORD": "That password is too weak. Try a longer one.",
        }
        detail = str(exc.detail)
        for code, message in friendly.items():
            if detail.startswith(code):
                raise HTTPException(status_code=400, detail=message)
        raise

    user_id = auth_data.get("localId")
    if user_id:
        try:
            db = get_db()
            db.collection("Responders").document(user_id).set({"is_new": False}, merge=True)
        except Exception as exc:
            # The password itself already changed in Firebase Auth by this
            # point — don't fail the whole request over is_new bookkeeping.
            # Worst case, the person gets sent through the change-password
            # screen once more than necessary.
            print(f"[auth.change_password] Could not clear is_new for {user_id}: {exc}")

        # login() skips the on-duty step while is_new is set, and the frontend
        # continues straight to the dashboard from here, so put admins on duty now.
        try:
            set_admin_duty(get_db(), user_id, "on_duty")
        except Exception as exc:
            print(f"[auth.change_password] Could not set on_duty for {user_id}: {exc}")

    return {
        "status": "success",
        "message": "Password updated.",
        "token": auth_data.get("idToken"),
        "refreshToken": auth_data.get("refreshToken"),
    }


class RefreshRequest(StrictModel):
    refreshToken: RefreshTokenStr


@router.post("/refresh")
def refresh_session(payload: RefreshRequest):
    """
    Swaps a refresh token for a fresh ID token. ID tokens only live for an
    hour, so the website calls this in the background to keep people signed in.
    Fails with 401 once the refresh token is no longer valid (password changed,
    account disabled/deleted, or sessions revoked).
    """
    body = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "refresh_token": payload.refreshToken,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"https://securetoken.googleapis.com/v1/token?key={API_KEY}",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"[auth.refresh] Firebase rejected the refresh token (HTTP {exc.code})")
        raise HTTPException(status_code=401, detail="Your session has ended. Please log in again.")
    except Exception as exc:
        print(f"[auth.refresh] {exc.__class__.__name__}: {exc}")
        raise HTTPException(status_code=503, detail="Could not renew the session right now.")

    return {
        "status": "success",
        "token": data.get("id_token"),
        "refreshToken": data.get("refresh_token"),
    }


@router.post("/logout")
def logout(token: dict = Depends(get_verified_token)):
    """
    Ends the caller's session server-side.

    Revokes ALL of this user's Firebase refresh tokens. From that moment:
      - /auth/refresh fails with 401, so the site can't silently renew.
      - Any ID token that was issued before now is rejected by
        security.verify_token (needs AUTH_CHECK_REVOKED on, which is the
        default now), so pressing Back to a cached page can't call the API.

    The frontend must still wipe its own copy of the tokens (see notes).
    """
    uid = token["uid"]
    try:
        firebase_auth.revoke_refresh_tokens(uid)
    except Exception as exc:
        print(f"[auth.logout] Could not revoke tokens for {uid}: {exc.__class__.__name__}: {exc}")
        raise HTTPException(status_code=503, detail="Could not log out right now. Please try again.")

    # Admins go OFF duty on logout (no-op for plain responders). Done after the
    # revoke succeeds, and never allowed to fail the logout itself.
    try:
        set_admin_duty(get_db(), uid, "off_duty")
    except Exception as exc:
        print(f"[auth.logout] Could not set off_duty for {uid}: {exc}")

    return {"status": "success", "message": "Logged out."}


@router.post("/send-code")
def send_code(payload: CodeRequest):
    """Mock sending a verification code for passwordless or MFA flows."""
    email = payload.email or payload.usernameOrEmail
    if not email:
        raise HTTPException(status_code=400, detail="Email address is required")

    # Verification is disabled in this mock setup; we preserve the email field.
    return {
        "status": "success",
        "message": f"Verification email prepared for {email}.",
        "email": email,
    }


@router.post("/verify-code")
def verify_code(payload: CodeRequest):
    """Mock verification endpoint without code validation.
    SECURITY: this always reports success. Do not let anything trust it, and
    remove it (or make it real) before production."""
    email = payload.email or payload.usernameOrEmail
    if not email:
        raise HTTPException(status_code=400, detail="Email address is required")

    return {
        "status": "success",
        "message": f"Verification step skipped for {email}.",
        "email": email,
    }