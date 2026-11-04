import re
from datetime import datetime, timezone
from math import atan2, cos, radians, sin, sqrt
from typing import Any, ClassVar
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query

from firebase import get_db
from security import get_verified_token
from validation import (
    ACCOUNT_STATUS_VALUES, BADGE_MESSAGE, DUTY_VALUES, ROLE_VALUES, AccountStatusValue, Badge, DocIdOrBlank,
    DocIdPath, DutyValue, EmailAddress, OptionalDocId, PatchModel, PersonName, RoleValue, StrictModel,
    clean_badge, require_choice,
)

router = APIRouter(prefix="/accounts", tags=["Accounts"])

# ---------------------------------------------------------------------------
# Roles & page permissions
#
# Three roles now, all living in the single Responders collection (see
# "Collection routing" below):
#
#   super_admin      -> exactly the Admin's pages (dashboard, report, history,
#                        units, accounts) PLUS stations. Its DATA is scoped
#                        to its own station like any other admin. Assign it from the Accounts page by
#                        picking "Super Admin" as the role and the main
#                        station (HQ) as the station. Only an existing
#                        super_admin may grant the role.
#   responder_admin   -> exactly dashboard, report, history, units, accounts.
#                        No Stations. On the Accounts page they can only MANAGE
#                        responders of their own station (assign role/rank,
#                        assign unit, set status) — never create, delete or
#                        reset accounts. Created ONLY via this file's
#                        create_account() (the Accounts page). Shown in that
#                        page's role dropdown as "Admin" — the stored value
#                        is still "responder_admin"; only the label changed.
#   responder         -> mobile app only. Not a web-login role at all: gets
#                        ZERO pages here, not just the restricted set. This
#                        project doesn't gate at /auth/login itself (that
#                        endpoint is shared with the mobile app), so the
#                        block happens here, at the page-permission layer.
#
# Identity: every request must carry the Firebase ID token returned by
# /auth/login, sent as `Authorization: Bearer <token>`. The token is verified
# server-side (see security.py) and the caller is the verified uid, which is
# also the Responders document id. A client can no longer claim to be someone
# else by sending an account id; the old X-Account-Id header is ignored.
#
# Default-DENY: a missing/blank role falls back to "Responder" (see
# get_current_account / _serialize), which is mobile-only, i.e. zero pages —
# not unrestricted and not even the four-page restricted set.
# ---------------------------------------------------------------------------
FULL_ACCESS_ROLES = {"super_admin"}  # name kept: other modules may import it. Means "the super-admin role".
MOBILE_ONLY_ROLES = {"responder"}
RESTRICTED_PAGES = {"dashboard", "report", "history", "units", "accounts"}
SUPER_ADMIN_EXTRA_PAGES = {"stations"}

# Station admins ("responder_admin") managing accounts on the Accounts page.
ADMIN_LEVEL_ROLES = {"super_admin", "responder_admin"}
# The only fields a station admin may change on someone else's account.
STATION_ADMIN_EDITABLE_FIELDS = {"role", "status", "unitId"}
# Roles that may be given to a brand-new account (lower-cased).
ALLOWED_NEW_ACCOUNT_ROLES = {"responder", "responder_admin", "super_admin"}
# Roles a station admin may hand out (lower-cased): Responder or Admin, never Super Admin.
STATION_ADMIN_ASSIGNABLE_ROLES = {"responder", "responder_admin"}


def get_allowed_pages(role: str) -> set[str]:
    """
    An empty set means no web pages at all (plain field responders — mobile
    app only); responder_admin gets the fixed four-page allow-list; super_admin
    gets those same four pages plus stations and accounts. Nobody is
    unrestricted any more, so this never returns None.
    """
    normalized = (role or "").strip().lower()
    if normalized in FULL_ACCESS_ROLES:
        return RESTRICTED_PAGES | SUPER_ADMIN_EXTRA_PAGES
    if normalized in MOBILE_ONLY_ROLES:
        return set()
    return RESTRICTED_PAGES


def get_current_account(token: dict = Depends(get_verified_token)) -> dict:
    """
    Resolves the calling account from the VERIFIED Firebase ID token in the
    Authorization header. The uid inside the token is the Responders doc id.

    Raises 401 if the token is missing/invalid/expired (see security.py) and
    403 if the login is valid but has no account profile (e.g. a citizen).
    """
    account_id = token["uid"]

    db = get_db()
    snapshot = db.collection(RESPONDERS).document(account_id).get()
    if not snapshot.exists:
        raise HTTPException(status_code=403, detail="No account profile for this login")

    data = snapshot.to_dict() or {}
    if data.get("disabled") is True:   # "deleted" accounts are only flagged, see delete_account
        raise HTTPException(status_code=403, detail="This account has been disabled.")
    _touch_admin_session(snapshot.reference, data)
    return {
        "id": account_id,
        "collection": RESPONDERS,
        "role": data.get("role") or data.get("position") or "Responder",
        "status": data.get("status"),
        "stationId": str(data.get("stationId") or "").strip(),
    }


# Returned by get_station_scope for a station-bound admin whose account has no
# stationId set. It matches no report, so they see nothing (fail closed)
# instead of everything.
NO_STATION_SCOPE = "__no_station_assigned__"


def get_station_scope(account: dict = Depends(get_current_account)) -> str:
    """
    Which station's data the caller may see, resolved server-side from the
    the verified token (never from a client-supplied query param).

      "ST002"         -> every account, super_admin included: only reports/units
                         routed to that station.
      NO_STATION_SCOPE -> account with no stationId: sees nothing.

    There is no unrestricted scope any more, so this never returns None.
    """
    return account.get("stationId") or NO_STATION_SCOPE


# ---------------------------------------------------------------------------
# Nearest-station auto-routing for ActiveCalls
#
# Nothing else in the backend sets `nearestStationId` when a report is created
# (only the manual PATCH in dashboard.py does), so a station-scoped admin would
# never see a brand-new report. route_unassigned() fills that gap: any call
# with no station yet is routed to the closest station that has coordinates and
# the result is saved to Firestore (nearestStationId + assignedStation).
# Lives here because report.py, dashboard.py, history.py and stream.py already
# import from this module.
# ---------------------------------------------------------------------------

def _haversine_km(lat1, lng1, lat2, lng2) -> float:
    r = 6371.0
    dlat, dlng = radians(lat2 - lat1), radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return r * 2 * atan2(sqrt(a), sqrt(1 - a))


def _pair(lat: Any, lng: Any) -> tuple[float, float] | None:
    try:
        if lat is None or lng is None:
            return None
        return float(lat), float(lng)
    except (TypeError, ValueError):
        return None


def _coords(loc: Any) -> tuple[float, float] | None:
    """GeoPoint, or a dict with latitude/longitude or lat/lng."""
    if loc is None:
        return None
    if hasattr(loc, "latitude") and hasattr(loc, "longitude"):
        return _pair(loc.latitude, loc.longitude)
    if isinstance(loc, dict):
        return _pair(loc.get("latitude", loc.get("lat")), loc.get("longitude", loc.get("lng")))
    return None


def call_coords(data: dict[str, Any]) -> tuple[float, float] | None:
    return _coords(data.get("location")) or _pair(
        data.get("latitude", data.get("lat")), data.get("longitude", data.get("lng"))
    )


def station_coords(sd: dict[str, Any]) -> tuple[float, float] | None:
    # Firestore field names are case-sensitive and Stations uses "Location".
    loc = sd.get("Location")
    return _coords(loc if loc is not None else sd.get("location"))


def has_station(data: dict[str, Any]) -> bool:
    assigned = data.get("assignedStation") or {}
    return bool(str(data.get("nearestStationId") or assigned.get("stationId") or "").strip())


class _RoutedDoc:
    """Stand-in for a Firestore snapshot carrying the freshly-routed data."""

    def __init__(self, doc_id: str, data: dict[str, Any]):
        self.id = doc_id
        self._data = data
        self.exists = True

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


# Roles that actually receive emergency calls, and the only duty value that
# counts as "working". A missing/unknown duty value means NOT on duty, matching
# duty.py (so a station is never assigned a call just because a responder doc
# is missing the field).
_CALL_ROLES = {"responder_admin", "super_admin"}


def _on_duty_station_ids(db: Any) -> set[str]:
    """Upper-cased ids of stations with at least one call-receiving responder
    who is explicitly on duty."""
    ids: set[str] = set()
    for doc in db.collection("Responders").stream():
        d = doc.to_dict() or {}
        if d.get("disabled") is True:
            continue
        if str(d.get("role") or "").strip().lower() not in _CALL_ROLES:
            continue
        if str(d.get("duty") or "").strip().lower() != "on_duty":
            continue
        if not admin_session_is_live(d):   # on_duty but not seen recently = abandoned session
            continue
        sid = str(d.get("stationId") or "").strip().upper()
        if sid:
            ids.add(sid)
    return ids


def route_unassigned(db: Any, docs: list) -> list:
    """Return docs, with any station-less ones routed to their nearest station
    (saved to Firestore). Docs that can't be routed (no coordinates, no station
    with coordinates) come back unchanged."""
    stations: list[tuple[str, dict[str, Any], tuple[float, float]]] | None = None
    out = []
    for doc in docs:
        data = doc.to_dict() or {}
        if has_station(data):
            out.append(doc)
            continue
        point = call_coords(data)
        if point is None:
            out.append(doc)
            continue
        if stations is None:
            on_duty_ids = _on_duty_station_ids(db)
            stations = []
            for s in db.collection("Stations").stream():
                # Skip stations with nobody on duty; the call stays unassigned
                # (and is retried on the next poll) if none qualify.
                if s.id.strip().upper() not in on_duty_ids:
                    continue
                sd = s.to_dict() or {}
                if sd.get("disabled") is True:
                    continue
                sc = station_coords(sd)
                if sc is not None:
                    stations.append((s.id, sd, sc))
        if not stations:
            out.append(doc)
            continue
        sid, sd, _ = min(stations, key=lambda s: _haversine_km(point[0], point[1], s[2][0], s[2][1]))
        assigned = dict(data.get("assignedStation") or {})
        assigned.update({
            "stationId": sid,
            "stationName": sd.get("stationName", ""),
            "commanderName": sd.get("commanderName") or sd.get("chiefName", ""),
            "dispatchStatus": "PENDING",
        })
        try:
            db.collection("ActiveCalls").document(doc.id).update(
                {"nearestStationId": sid, "assignedStation": assigned}
            )
        except Exception:
            out.append(doc)
            continue
        out.append(_RoutedDoc(doc.id, {**data, "nearestStationId": sid, "assignedStation": assigned}))
    return out


def require_page(page: str):
    """
    Dependency factory for guarding routes in OTHER routers, e.g. in your
    units router:

        from accounts import require_page

        @router.get("/units", dependencies=[Depends(require_page("units"))])
        def list_units():
            ...

    Raises 403 if the caller's role isn't allowed on that page.
    """

    def _check(account: dict = Depends(get_current_account)) -> dict:
        allowed = get_allowed_pages(account["role"])
        if allowed is not None and page not in allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Your role does not have access to '{page}'",
            )
        return account

    return _check


def _role_gate(*roles: str):
    allowed = {r.lower() for r in roles}

    def _check(account: dict = Depends(get_current_account)) -> dict:
        if str(account["role"]).strip().lower() not in allowed:
            raise HTTPException(status_code=403, detail="Not allowed.")
        return account

    return _check


# For routers outside this file: Depends(admin_only) = super_admin or
# responder_admin; Depends(super_admin_only) = super_admin.
admin_only = _role_gate(*ADMIN_LEVEL_ROLES)
super_admin_only = _role_gate(*FULL_ACCESS_ROLES)


# ---------------------------------------------------------------------------
# Collection
#
# Every account — super_admin, responder_admin, and plain responder — lives
# in this single collection. There is no more separate Users collection;
# routing between two collections by role is gone, so an account is always
# found (and always written) in exactly one place.
#
# Two independent fields, do not conflate them:
#   status -> "on" / "off" / "inactive": account status, as before.
#   duty   -> "on_duty" / "off_duty": is this responder CURRENTLY on shift?
#             Only meaningful for plain field responders (see _has_duty) —
#             super_admin and responder_admin accounts don't track a shift.
# ---------------------------------------------------------------------------

RESPONDERS = "Responders"

# Same default auth.py already applies when signup() is called with no
# password (see the comment in accounts.js about DEFAULT_RESPONDER_PASSWORD).
# Kept here too so /accounts/{id}/reset-password can put an account back to
# the same starting point without needing to import from auth.py.
DEFAULT_RESPONDER_PASSWORD = "default123"

try:
    from firebase_admin import auth as firebase_auth
except ImportError:  # pragma: no cover - firebase_admin should already be a
    firebase_auth = None  # project dependency; this only guards import order


# Badge numbers are the web login name, so they must be present, tidy and unique.
def _clean_badge(raw: str | None) -> str:
    """Trim, validate and UPPER-CASE a badge so PNP-123 and pnp-123 are one badge."""
    try:
        return clean_badge(raw or "")
    except ValueError:
        raise HTTPException(status_code=400, detail=BADGE_MESSAGE)


def _assert_station_exists(db, station_id: str | None) -> None:
    """A station id coming from a request must point at a real, enabled station."""
    if not station_id:
        return
    snap = db.collection("Stations").document(station_id).get()
    if not snap.exists or (snap.to_dict() or {}).get("disabled") is True:
        raise HTTPException(status_code=400, detail="That station does not exist.")


def _assert_unit_in_station(db, unit_id: str | None, station_id: str | None) -> None:
    """A unit id from a request must exist, be enabled, and belong to the account's station.
    Without this an admin could put a responder on another station's unit."""
    if not unit_id:
        return
    snap = db.collection("Units").document(unit_id).get()
    unit = (snap.to_dict() or {}) if snap.exists else {}
    if (not snap.exists or unit.get("disabled") is True
            or str(unit.get("stationId") or "").strip() != str(station_id or "").strip()):
        raise HTTPException(status_code=400, detail="That unit does not exist in this account's station.")


def _assert_email_free(db, email: str, own_id: str | None = None) -> None:
    snaps = db.collection(RESPONDERS).where("email", "==", email).limit(5).stream()
    if any(d.id != own_id for d in snaps):
        raise HTTPException(status_code=400, detail="Email already registered.")


def _assert_badge_free(db, badge: str, own_id: str | None = None) -> None:
    """
    Refuses a badge that any OTHER account already has (every station, every
    role, disabled accounts included). Looks for the upper-case form and the
    lower-case form so an older account saved in lower case still counts.
    own_id lets an account keep its own badge when it is edited.
    """
    variants = list({badge, badge.upper(), badge.lower()})
    snaps = db.collection("Responders").where("badge", "in", variants).limit(5).stream()
    if any(d.id != own_id for d in snaps):
        raise HTTPException(status_code=400, detail="That badge number is already in use.")


class AccountCreate(StrictModel):
    firstName: PersonName = ""
    lastName: PersonName = ""
    badge: Badge = ""
    email: EmailAddress = ""
    role: RoleValue = ""
    stationId: DocIdOrBlank = ""
    status: AccountStatusValue = ""   # "on" / "off" / "inactive", as before
    duty: DutyValue = "off_duty"      # shift state; new accounts start off shift
    unitId: OptionalDocId = None
    is_new: bool = True


class AccountUpdate(PatchModel):
    # unitId may be sent as null/"" to clear a unit; for every other field an
    # explicit null means "not sent" (before, {"status": null} silently became "on").
    nullable_fields: ClassVar[frozenset[str]] = frozenset({"unitId"})

    firstName: PersonName | None = None
    lastName: PersonName | None = None
    badge: Badge | None = None
    email: EmailAddress | None = None
    role: RoleValue | None = None
    stationId: DocIdOrBlank | None = None
    status: AccountStatusValue | None = None
    duty: DutyValue | None = None
    unitId: OptionalDocId = None
    is_new: bool | None = None


class StatusUpdate(StrictModel):
    status: AccountStatusValue


class DutyUpdate(StrictModel):
    duty: DutyValue | None = None  # omit to flip whatever is currently stored


# Admin-level roles are put on duty automatically when they log in to the web
# dashboard and off duty when they log out (done in auth.py's login/logout via
# set_admin_duty below). That is how the report-assignment logic knows which
# admins in a station can receive a report.
ADMIN_DUTY_ROLES = {"responder_admin", "super_admin"}


def _has_duty(role: str) -> bool:
    """
    Every account that can be on shift tracks a duty state: plain field
    responders (toggled from the mobile app) and admins (toggled by web
    login/logout). A blank role defaults to Responder, matching the list
    endpoint's fallback.
    """
    normalized = (role or "Responder").strip().lower()
    return normalized == "responder" or normalized in ADMIN_DUTY_ROLES


# An admin who closes the tab, loses power or whose 24h browser session lapses
# never calls /auth/logout, so "on_duty" alone can't be trusted. Every
# authenticated request records lastSeenAt (at most once a minute), and an admin
# only counts as available while that is recent. The dashboard's normal polling
# keeps it fresh, so the frontend needs no separate heartbeat.
SESSION_TOUCH_SECONDS = 60
SESSION_STALE_SECONDS = 180


def _is_admin_role(role) -> bool:
    return str(role or "").strip().lower() in ADMIN_DUTY_ROLES


def _touch_admin_session(ref, data: dict) -> None:
    if not _is_admin_role(data.get("role") or data.get("position")):
        return
    now = datetime.now(timezone.utc)
    last = data.get("lastSeenAt")
    if isinstance(last, datetime) and (now - last).total_seconds() < SESSION_TOUCH_SECONDS:
        return
    try:
        ref.set({"lastSeenAt": now}, merge=True)
    except Exception as exc:  # never fail a request over bookkeeping
        print(f"[accounts] Could not record lastSeenAt: {exc}")


def admin_session_is_live(data: dict) -> bool:
    """True if this admin has made a request in the last SESSION_STALE_SECONDS."""
    last = data.get("lastSeenAt")
    if not isinstance(last, datetime):
        return False
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - last).total_seconds() <= SESSION_STALE_SECONDS


def is_admin_available(data: dict) -> bool:
    """Admin is explicitly on duty AND still has a live session. Use this (not
    the raw duty field) when deciding who can receive a report."""
    return (
        _is_admin_role(data.get("role") or data.get("position"))
        and _normalize_duty(data.get("duty")) == "on_duty"
        and admin_session_is_live(data)
        and data.get("disabled") is not True
    )


def set_admin_duty(db, account_id: str, duty: str) -> dict:
    """
    Single place that flips an admin's duty on login/logout. auth.py can import
    and call this directly from its login/logout handlers.

    Plain responders are left alone (they manage their own shift from the
    mobile app). Going ON duty is refused for deactivated accounts, same rule
    as PATCH /{id}/duty; going OFF duty is always allowed.
    """
    duty = _normalize_duty(duty)
    ref = db.collection(RESPONDERS).document(account_id)
    snap = ref.get()
    if not snap.exists:
        raise HTTPException(status_code=403, detail="No account profile for this login")
    data = snap.to_dict() or {}
    role = str(data.get("role") or data.get("position") or "Responder").strip().lower()
    if role not in ADMIN_DUTY_ROLES:
        return {"accountId": account_id, "duty": _normalize_duty(data.get("duty")), "changed": False}

    previous = _normalize_duty(data.get("duty"))
    if duty == "on_duty" and _normalize_status(data.get("status")) == "inactive":
        raise HTTPException(status_code=409, detail="Account is deactivated and cannot be placed on duty")

    changed_at = datetime.now(timezone.utc)
    update = {"duty": duty, "dutyChangedAt": changed_at}
    if duty == "on_duty":
        update["lastSeenAt"] = changed_at
    ref.set(update, merge=True)
    return {
        "accountId": account_id,
        "duty": duty,
        "previousDuty": previous,
        "changed": duty != previous,
        "dutyChangedAt": changed_at,
    }


def _normalize_status(raw_status: str) -> str:
    """
    Normalizes a responder's stored status into one of "on" / "off" / "inactive".

    Kept exactly as it was before duty was introduced: the Accounts admin UI
    writes "on" / "off" / "inactive" directly (from its status <select>).
    Other write paths (e.g. a mobile self-check-in flow) may still use
    "available" / "active" / "deployed" / "standby", so those are mapped for
    backward compatibility. This is unrelated to duty — duty tracks whether a
    responder is currently on shift; status is the account state.
    """
    value = (raw_status or "available").strip().lower()
    if value in ("on", "active", "available"):
        return "on"
    if value in ("off", "deployed", "standby"):
        return "off"
    return "inactive"


def _normalize_duty(raw_duty) -> str:
    """
    Normalizes a responder's shift state into "on_duty" / "off_duty".

    Firestore stores "on_duty" / "off_duty" directly (matching the sample
    doc), but shorthand from other write paths is tolerated.
    """
    if isinstance(raw_duty, bool):
        return "on_duty" if raw_duty else "off_duty"
    value = str(raw_duty or "off_duty").strip().lower()
    if value in ("on_duty", "onduty", "on", "duty", "deployed", "true", "1"):
        return "on_duty"
    return "off_duty"


def _serialize(doc_id: str, data: dict) -> dict:
    """Shared response shape for a Responders-collection doc."""
    role = data.get("role") or data.get("position") or "Responder"
    has_duty = _has_duty(role)

    account = {
        "id": doc_id,
        "collection": RESPONDERS,
        "firstName": data.get("firstName", ""),
        "lastName": data.get("lastName", ""),
        "name": f"{data.get('firstName', '')} {data.get('lastName', '')}".strip() or data.get("name") or "Unknown",
        "badge": data.get("badge", ""),
        "email": data.get("email", ""),
        "role": role,
        "stationId": data.get("stationId", ""),
        "unitId": data.get("unitId") or "",
        "status": _normalize_status(data.get("status")),
        "is_new": data.get("is_new", False),
    }

    # duty is reported for responders and admins (admins: set by web login/logout).
    account["duty"] = _normalize_duty(data.get("duty")) if has_duty else None
    if account["duty"] == "on_duty" and _is_admin_role(role) and not admin_session_is_live(data):
        account["duty"] = "off_duty"   # stored on_duty, but the session was abandoned
    account["dutyChangedAt"] = data.get("dutyChangedAt") if has_duty else None

    return account


def _require_super_admin_caller(db, account_id: str | None) -> None:
    """Granting the super_admin role is itself restricted to a super_admin."""
    if not account_id:
        raise HTTPException(status_code=401, detail="Login required.")
    snapshot = db.collection(RESPONDERS).document(account_id).get()
    caller = snapshot.to_dict() or {} if snapshot.exists else {}
    caller_role = (caller.get("role") or caller.get("position") or "").strip().lower()
    if caller_role not in FULL_ACCESS_ROLES:
        raise HTTPException(status_code=403, detail="Only a Super Admin can assign the Super Admin role.")


def _role_of(data: dict) -> str:
    return (data.get("role") or data.get("position") or "").strip().lower()


def _get_caller(db, account_id: str | None) -> dict | None:
    """Caller resolved from the verified account id, or None if unknown."""
    if not account_id:
        return None
    snapshot = db.collection(RESPONDERS).document(account_id).get()
    if not snapshot.exists:
        return None
    data = snapshot.to_dict() or {}
    return {
        "id": account_id,
        "role": _role_of(data),
        "stationId": str(data.get("stationId") or "").strip(),
    }


def _require_admin_caller(db, account_id: str | None) -> dict:
    """Login required: caller must exist and be a super_admin or responder_admin."""
    caller = _get_caller(db, account_id)
    if caller is None:
        raise HTTPException(status_code=401, detail="Login required.")
    if caller["role"] not in ADMIN_LEVEL_ROLES:
        raise HTTPException(status_code=403, detail="Not allowed.")
    return caller


def _require_super_admin(db, account_id: str | None, action: str) -> dict:
    caller = _get_caller(db, account_id)
    if caller is None:
        raise HTTPException(status_code=401, detail="Login required.")
    if caller["role"] not in FULL_ACCESS_ROLES:
        raise HTTPException(status_code=403, detail=f"Only a Super Admin can {action}.")
    return caller


def _assert_may_assign_in_own_station(caller: dict, effective_station: str, role_change: bool, unit_change: bool) -> None:
    """
    A super admin may assign a role/rank or a unit only for accounts in THEIR
    station. Clearing a unit (unitId null) is always allowed.
    """
    if not (role_change or unit_change):
        return
    effective_station = (effective_station or "").strip()
    if not caller["stationId"] or effective_station != caller["stationId"]:
        raise HTTPException(
            status_code=403,
            detail="You can only assign roles and units for accounts in your own station.",
        )


def _assert_station_admin_may_manage(caller: dict, target: dict) -> None:
    """A station admin may only touch non-admin accounts in their own station."""
    target_station = str(target.get("stationId") or "").strip()
    if not caller["stationId"] or target_station != caller["stationId"]:
        raise HTTPException(status_code=403, detail="You can only manage accounts in your own station.")
    if _role_of(target) in ADMIN_LEVEL_ROLES:
        raise HTTPException(status_code=403, detail="Admins cannot manage Admin or Super Admin accounts.")


def _find_account(db, account_id: str):
    """Returns (doc_ref, snapshot) or raises 404."""
    ref = db.collection(RESPONDERS).document(account_id)
    snapshot = ref.get()
    if snapshot.exists:
        return ref, snapshot
    raise HTTPException(status_code=404, detail="Account not found")


@router.get("/me")
def get_me(account: dict = Depends(get_current_account)):
    """
    Full profile for the logged-in caller, resolved from the verified
    Bearer token. Frontend calls this right after login to know who it's talking to.
    """
    db = get_db()
    ref = db.collection(RESPONDERS).document(account["id"])
    data = ref.get().to_dict() or {}
    return _serialize(account["id"], data)


@router.get("/me/permissions")
def get_my_permissions(account: dict = Depends(get_current_account)):
    """
    Which pages the logged-in caller may see. `pages: []` means no web pages
    at all (plain field responders — mobile only); otherwise the exact
    allow-list: responder_admin gets ["accounts", "dashboard", "history",
    "report", "units"], and super_admin gets those plus "stations".

    Frontend calls this after login and hides any nav item not in the list.
    The backend still enforces this on each route via require_page(), so
    hiding the nav link is a convenience, not the actual security boundary.
    """
    allowed = get_allowed_pages(account["role"])
    return {
        "role": account["role"],
        "pages": sorted(allowed) if allowed is not None else None,
    }


@router.get("")
def list_accounts(
    duty: str | None = Query(None, max_length=20),
    status: str | None = Query(None, max_length=20),
    role: str | None = Query(None, max_length=40),
    account: dict = Depends(get_current_account),
):
    """
    Frontend accounts page: the full roster (super_admin, responder_admin,
    and plain responders all live in the one Responders collection now).

    Optional filters, normalized the same way as the stored values so
    ?duty=on and ?duty=on_duty behave identically:
      /accounts?duty=on_duty         -> responders currently on shift
      /accounts?status=on            -> only accounts with status "on"
      /accounts?role=responder_admin -> only Admin accounts
    """
    db = get_db()

    duty = require_choice(duty, DUTY_VALUES, "duty")
    status = require_choice(status, ACCOUNT_STATUS_VALUES, "status")
    role = require_choice(role, ROLE_VALUES, "role")
    duty_filter = _normalize_duty(duty) if duty else None
    status_filter = _normalize_status(status) if status else None
    role_filter = role.strip().lower() if role else None

    # A station admin only gets accounts from their own station (responders AND
    # the other admins of that station, themselves included) — never other stations.
    caller = _require_admin_caller(db, account["id"])
    station_admin_scope = caller["stationId"] if caller and caller["role"] == "responder_admin" else None
    scope_active = caller is not None and caller["role"] == "responder_admin"

    accounts = []
    for doc in db.collection(RESPONDERS).stream():
        if (doc.to_dict() or {}).get("disabled") is True:
            continue   # "deleted" accounts stay in the database but are hidden
        item = _serialize(doc.id, doc.to_dict() or {})   # not "account": that is the caller

        if scope_active:
            if not station_admin_scope or str(item["stationId"] or "").strip() != station_admin_scope:
                continue

        if role_filter and item["role"].strip().lower() != role_filter:
            continue
        if status_filter and item["status"] != status_filter:
            continue
        # (duty filter now matches admins as well as responders)
        if duty_filter and item["duty"] != duty_filter:
            continue

        accounts.append(item)

    # Tiles keep describing field responders only; admins now carry a duty value too.
    field_responders = [a for a in accounts if a["role"].strip().lower() == "responder"]
    on_duty = [a for a in field_responders if a["duty"] == "on_duty"]
    responder_admins = [a for a in accounts if a["role"].strip().lower() == "responder_admin"]
    super_admins = [a for a in accounts if a["role"].strip().lower() == "super_admin"]

    return {
        "accounts": accounts,
        "counts": {
            "total": len(accounts),
            "responders": len(field_responders),
            "responderAdmins": len(responder_admins),
            "superAdmins": len(super_admins),
            "onDuty": len(on_duty),
            "offDuty": len(field_responders) - len(on_duty),
        },
    }


@router.post("", status_code=201)
def create_account(payload: AccountCreate, account: dict = Depends(get_current_account)):
    """
    Creates a plain responder, a responder_admin ("Admin" in the Accounts page
    dropdown; the stored role string is still "responder_admin"), or a
    super_admin ("Super Admin" in the dropdown; only a super_admin may do this).

    This is the ONLY place accounts are created. It makes the Firebase login
    (email + the default temporary password) AND the Responders profile in one
    request, so there is no public signup endpoint any more. New accounts are
    always flagged is_new so the person must change the default password.
    """
    requested_role = (payload.role or "Responder").strip()
    if requested_role.lower() not in ALLOWED_NEW_ACCOUNT_ROLES:
        raise HTTPException(status_code=400, detail="Invalid role.")

    db = get_db()
    # Adding accounts is a Super Admin action. Station admins can only manage
    # existing accounts (see update_account).
    caller = _require_super_admin(db, account["id"], "create accounts")
    _assert_may_assign_in_own_station(
        caller,
        payload.stationId,
        role_change=requested_role.lower() != "responder",
        unit_change=bool(payload.unitId),
    )
    if requested_role.lower() == "super_admin":
        # Same rule update_account applies when granting the role.
        _require_super_admin_caller(db, account["id"])

    if firebase_auth is None:
        raise HTTPException(status_code=500, detail="firebase_admin auth is not available")

    first_name = (payload.firstName or "").strip()
    last_name = (payload.lastName or "").strip()
    email = (payload.email or "").strip()
    if not first_name or not last_name:
        raise HTTPException(status_code=400, detail="First and last name are required.")
    if not email:
        raise HTTPException(status_code=400, detail="Email is required.")
    badge = _clean_badge(payload.badge)
    _assert_station_exists(db, payload.stationId)
    _assert_unit_in_station(db, payload.unitId, payload.stationId)

    # Same username rule the Accounts page has always used (first.last, lower-case,
    # no spaces); login-by-username depends on this field being present.
    username = re.sub(r"\s+", "", f"{first_name}.{last_name}".lower())

    users = db.collection(RESPONDERS)
    if list(users.where("username", "==", username).limit(1).stream()):
        raise HTTPException(status_code=400, detail="Username already exists.")
    if list(users.where("email", "==", email).limit(1).stream()):
        raise HTTPException(status_code=400, detail="Email already registered.")
    _assert_badge_free(db, badge)

    try:
        user_record = firebase_auth.create_user(email=email, password=DEFAULT_RESPONDER_PASSWORD)
    except firebase_auth.EmailAlreadyExistsError:
        raise HTTPException(status_code=400, detail="Email already registered.")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid email address.")
    except Exception as err:
        print(f"[accounts.create_account] Firebase create_user failed: {err}")
        raise HTTPException(status_code=502, detail="Could not create the login for this account.")
    account_id = user_record.uid

    has_duty = _has_duty(requested_role)

    account_data = {
        "firstName": first_name,
        "lastName": last_name,
        "badge": badge,
        "email": email,
        "role": requested_role,
        "stationId": payload.stationId,
        "status": _normalize_status(payload.status),
        "unitId": payload.unitId,
        "is_new": True,  # always: the account starts on the default password
    }

    if has_duty:
        account_data["duty"] = _normalize_duty(payload.duty)
        account_data["dutyChangedAt"] = datetime.now(timezone.utc)

    try:
        users.document(account_id).set({
            **account_data,
            "username": username,
            "createdAt": datetime.now(timezone.utc),
        })
    except Exception as err:
        # Don't leave a login behind that has no profile.
        print(f"[accounts.create_account] Firestore write failed for {account_id}: {err}")
        try:
            firebase_auth.delete_user(account_id)
        except Exception as cleanup_err:
            print(f"[accounts.create_account] Could not remove orphan login {account_id}: {cleanup_err}")
        raise HTTPException(status_code=503, detail="Could not save the account. Nothing was created.")

    return {"id": account_id, "collection": RESPONDERS, **account_data}


@router.patch("/{account_id}")
def update_account(account_id: DocIdPath, payload: AccountUpdate, account: dict = Depends(get_current_account)):
    db = get_db()
    ref, snapshot = _find_account(db, account_id)
    existing = snapshot.to_dict() or {}

    data = payload.model_dump(exclude_unset=True)

    caller = _require_admin_caller(db, account["id"])
    if caller and caller["role"] == "responder_admin":
        if caller["id"] == account_id:
            # A station admin cannot edit their own account from the Accounts page.
            # The one exception is clearing is_new after a password change.
            data = {k: v for k, v in data.items() if k == "is_new"}
        else:
            _assert_station_admin_may_manage(caller, existing)
            # Only role / unit / status — identity, station and password flags are ignored.
            data = {k: v for k, v in data.items() if k in STATION_ADMIN_EDITABLE_FIELDS}
            if "role" in data:
                new_role = _role_of({"role": data["role"]})
                if new_role == _role_of(existing):
                    data.pop("role")   # unchanged (e.g. an old rank being re-sent) — nothing to assign
                elif new_role not in STATION_ADMIN_ASSIGNABLE_ROLES:
                    raise HTTPException(status_code=403, detail="Admins can only assign the Responder or Admin role.")

    if caller and caller["role"] in FULL_ACCESS_ROLES:
        effective_station = str(data.get("stationId", existing.get("stationId")) or "")
        role_change = "role" in data and _role_of({"role": data["role"]}) != _role_of(existing)
        unit_change = bool(data.get("unitId")) and (data.get("unitId") or "") != (existing.get("unitId") or "")
        _assert_may_assign_in_own_station(caller, effective_station, role_change, unit_change)
        if role_change and caller["id"] == account_id:
            raise HTTPException(status_code=403, detail="You cannot change your own role.")

    # Granting super_admin, or changing the role of an existing super_admin
    # (demoting one), is restricted to a super_admin.
    existing_role = (existing.get("role") or existing.get("position") or "").strip().lower()
    if "role" in data and (
        (data["role"] or "").strip().lower() == "super_admin"
        or (existing_role == "super_admin" and (data["role"] or "").strip().lower() != "super_admin")
    ):
        _require_super_admin_caller(db, account["id"])

    if "badge" in data:
        data["badge"] = _clean_badge(data["badge"])
        _assert_badge_free(db, data["badge"], own_id=account_id)
    if "email" in data:
        _assert_email_free(db, data["email"], own_id=account_id)

    # Ids from the request must point at real things, and the unit must sit in
    # the account's (new) station.
    target_station = str(data.get("stationId", existing.get("stationId")) or "").strip()
    if data.get("stationId"):
        _assert_station_exists(db, data["stationId"])
    if data.get("unitId"):
        _assert_unit_in_station(db, data["unitId"], target_station)

    # Normalize before writing so an admin UI sending "On Duty" or a mobile
    # client sending "on" can't put an unrecognized string into Firestore.
    if "status" in data:
        data["status"] = _normalize_status(data["status"])
    if "duty" in data:
        data["duty"] = _normalize_duty(data["duty"])
        data["dutyChangedAt"] = datetime.now(timezone.utc)

    if not data:
        return {"updated": True, "accountId": account_id, "collection": RESPONDERS}

    # A role change can move an account into or out of duty-tracking, since
    # only plain field responders have a shift state. Keep the duty fields
    # consistent rather than leaving stale/missing ones behind.
    if "role" in data:
        if _has_duty(data["role"]):
            data.setdefault("duty", existing.get("duty", "off_duty"))
            data.setdefault("dutyChangedAt", existing.get("dutyChangedAt") or datetime.now(timezone.utc))
        else:
            # Leaving the plain-responder role ends any open shift.
            data["duty"] = None
            data["dutyChangedAt"] = None

    # The doc is guaranteed to exist (_find_account raised otherwise), but
    # set(..., merge=True) is still preferred over .update(): update() throws
    # google.api_core.exceptions.NotFound if the doc disappears between the
    # read and the write, which previously bubbled up as an unhandled 500 and
    # produced the "changed in Firestore but the frontend looks stale" symptom.
    ref.set(data, merge=True)

    return {"updated": True, "accountId": account_id, "collection": RESPONDERS, **data}


@router.patch("/{account_id}/duty")
def set_duty(account_id: DocIdPath, payload: DutyUpdate, account: dict = Depends(get_current_account)):
    """
    Shift toggle. Send {"duty": "on_duty"} / {"duty": "off_duty"} to set an
    explicit state, or an empty body to flip whatever is currently stored.

    Plain field responders only. A deactivated account cannot go on duty —
    that is the one place where status and duty interact.
    """
    db = get_db()
    ref, snapshot = _find_account(db, account_id)
    existing_role = (snapshot.to_dict() or {}).get("role")

    # Only the account itself, or an admin of the same station, may change a duty state.
    if account["id"] != account_id:
        caller = _require_admin_caller(db, account["id"])
        target_station = str((snapshot.to_dict() or {}).get("stationId") or "").strip()
        if not caller["stationId"] or caller["stationId"] != target_station:
            raise HTTPException(status_code=403, detail="You can only change duty for accounts in your own station.")

    if not _has_duty(existing_role):
        raise HTTPException(status_code=409, detail="Only responders have a duty state")

    existing = snapshot.to_dict() or {}
    current_duty = _normalize_duty(existing.get("duty"))
    account_status = _normalize_status(existing.get("status"))

    if payload.duty is None:
        new_duty = "off_duty" if current_duty == "on_duty" else "on_duty"
    else:
        new_duty = _normalize_duty(payload.duty)

    if new_duty == "on_duty" and account_status == "inactive":
        raise HTTPException(
            status_code=409,
            detail="Account is deactivated and cannot be placed on duty",
        )

    changed_at = datetime.now(timezone.utc)
    ref.set({"duty": new_duty, "dutyChangedAt": changed_at}, merge=True)

    return {
        "accountId": account_id,
        "duty": new_duty,
        "previousDuty": current_duty,
        "changed": new_duty != current_duty,
        "dutyChangedAt": changed_at,
    }


@router.patch("/{account_id}/status")
def set_status(account_id: DocIdPath, payload: StatusUpdate, account: dict = Depends(get_current_account)):
    """
    Set account status. Send {"status": "on"}, {"status": "off"}, or
    {"status": "inactive"} — same values the Accounts admin UI has always
    used.

    Deactivating a responder (status "inactive") force-ends any open shift,
    so a disabled account can't be left sitting in the on-duty roster.
    """
    db = get_db()
    ref, snapshot = _find_account(db, account_id)
    existing = snapshot.to_dict() or {}

    caller = _require_admin_caller(db, account["id"])
    if caller and caller["role"] == "responder_admin":
        if caller["id"] == account_id:
            raise HTTPException(status_code=403, detail="You cannot change your own status.")
        _assert_station_admin_may_manage(caller, existing)

    new_status = _normalize_status(payload.status)
    status_data = {"status": new_status}

    if new_status == "inactive" and _has_duty(existing.get("role")):
        if _normalize_duty(existing.get("duty")) == "on_duty":
            status_data["duty"] = "off_duty"
            status_data["dutyChangedAt"] = datetime.now(timezone.utc)

    ref.set(status_data, merge=True)

    return {"accountId": account_id, "collection": RESPONDERS, **status_data}


def _resolve_auth_uid(account_id: str, email: str) -> str | None:
    """Firebase Auth uid for this account: the doc id if it is a real uid, else the user with the profile's email."""
    try:
        return firebase_auth.get_user(account_id).uid
    except firebase_auth.UserNotFoundError:
        pass
    except Exception as err:
        print(f"[accounts._resolve_auth_uid] get_user failed: {err.__class__.__name__}: {err}")
    if email:
        try:
            return firebase_auth.get_user_by_email(email).uid
        except firebase_auth.UserNotFoundError:
            pass
        except Exception as err:
            print(f"[accounts._resolve_auth_uid] get_user_by_email failed: {err.__class__.__name__}: {err}")
    return None


@router.post("/{account_id}/reset-password")
def reset_password(account_id: DocIdPath, account: dict = Depends(get_current_account)):
    """
    Resets an account's login password back to the same default new accounts
    get (see DEFAULT_RESPONDER_PASSWORD) and re-arms is_new so they're forced
    to change it on next login — same effect as the "Force password reset"
    toggle in the Accounts UI, but this is what actually changes the
    Firebase Auth credential; the toggle alone only flips the is_new flag on
    the Firestore doc.
    """
    if firebase_auth is None:
        raise HTTPException(status_code=500, detail="firebase_admin auth is not available")

    db = get_db()
    _require_super_admin(db, account["id"], "reset passwords")
    ref, snapshot = _find_account(db, account_id)
    email = str((snapshot.to_dict() or {}).get("email") or "").strip()

    # Find the REAL Firebase Auth user: by doc id first, then by the profile's
    # email, instead of assuming the two always line up.
    auth_uid = _resolve_auth_uid(account_id, email)
    if auth_uid is None:
        raise HTTPException(
            status_code=404,
            detail="No Firebase login exists for this account's id or email. Check the account's email.",
        )

    changes = {"password": DEFAULT_RESPONDER_PASSWORD}
    if email:
        # Keep the Auth email in step with the profile, because /auth/login signs in
        # with the email stored on the profile.
        changes["email"] = email

    try:
        firebase_auth.update_user(auth_uid, **changes)
    except Exception as err:
        # Covers firebase_admin.auth.UserNotFoundError (doc exists in
        # Firestore but has no matching Auth credential — e.g. a manually
        # seeded record) and any other Admin SDK failure.
        print(f"[accounts.reset_password] {err.__class__.__name__}: {err}")  # keep details server-side
        raise HTTPException(status_code=502, detail="Could not reset the password for this account.")

    ref.set({"is_new": True}, merge=True)

    return {"accountId": account_id, "collection": RESPONDERS, "reset": True, "is_new": True}


@router.delete("/{account_id}")
def delete_account(account_id: DocIdPath, account: dict = Depends(get_current_account)):
    db = get_db()
    _require_super_admin(db, account["id"], "revoke accounts")
    ref, _ = _find_account(db, account_id)

    # Soft delete: nothing is removed. The account is flagged, then hidden from
    # the roster and refused by get_current_account. To bring one back, set
    # disabled to false on its Responders document.
    ref.set({"disabled": True, "disabledAt": datetime.now(timezone.utc)}, merge=True)

    return {"deleted": True, "accountId": account_id, "collection": RESPONDERS}