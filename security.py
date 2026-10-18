"""
LiveAlert — Token verification
===============================
Every protected route identifies the caller from a Firebase ID token sent as

    Authorization: Bearer <idToken>

The token is the `token` value returned by POST /auth/login. It is signed by
Google, so it can't be forged or edited the way a plain account id could.
The caller's identity is the verified `uid` inside it, which is also the
document id in the Responders collection.
"""

import os

from fastapi import Header, HTTPException
from firebase_admin import auth as firebase_auth

from firebase import get_db

# Also ask Firebase whether the token was revoked (password reset, disabled
# user, revokeRefreshTokens). Safer, but adds one network call per request.
# Off by default; set AUTH_CHECK_REVOKED=1 to turn it on.
CHECK_REVOKED = os.getenv("AUTH_CHECK_REVOKED", "0").lower() in {"1", "true", "yes", "on"}


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def verify_token(token: str) -> dict:
    """Returns the decoded Firebase token (contains 'uid', 'email', ...) or raises 401."""
    get_db()  # makes sure the default Firebase app is initialised
    try:
        return firebase_auth.verify_id_token(
            token, check_revoked=CHECK_REVOKED, clock_skew_seconds=10
        )
    except firebase_auth.ExpiredIdTokenError:
        raise _unauthorized("Your session has expired. Please log in again.")
    except firebase_auth.RevokedIdTokenError:
        raise _unauthorized("Your session was revoked. Please log in again.")
    except firebase_auth.CertificateFetchError as exc:
        raise HTTPException(status_code=503, detail="Could not verify the session right now.") from exc
    except (firebase_auth.InvalidIdTokenError, ValueError):
        raise _unauthorized("Invalid token.")
    except Exception as exc:
        print(f"[security.verify_token] unexpected {exc.__class__.__name__}: {exc}")
        raise _unauthorized("Invalid token.")


def get_verified_token(authorization: str | None = Header(default=None)) -> dict:
    """
    FastAPI dependency: any valid Firebase login (responder OR citizen).
    Use this when the route doesn't need a Responders profile.
    """
    if not authorization:
        raise _unauthorized("Missing Authorization header.")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _unauthorized("Authorization header must be 'Bearer <token>'.")
    return verify_token(token.strip())