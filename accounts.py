from datetime import datetime, timezone
from math import atan2, cos, radians, sin, sqrt
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from firebase import get_db

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
# Identity: this project doesn't verify a signed token on every request, so
# the caller is identified the same way the rest of this file already treats
# accounts — by Firestore doc id. The frontend sends that id back on every
# request as the X-Account-Id header (set right after login). That's fine
# for gating which *pages* someone sees, but it's UX, not real security —
# anyone who can set a request header can claim to be any account id. Put a
# real auth check (signed token / session cookie) in front of this once one
# exists.
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


def get_current_account(x_account_id: str = Header(None, alias="X-Account-Id")) -> dict:
    """
    Resolves the calling account from the X-Account-Id header, which the
    frontend sets to the id it got back from login/create_account.

    Raises 401 if the header is missing, 404 if no account with that id
    exists in either collection.
    """
    if not x_account_id:
        raise HTTPException(status_code=401, detail="Missing X-Account-Id header")

    db = get_db()
    snapshot = db.collection(RESPONDERS).document(x_account_id).get()
    if not snapshot.exists:
        raise HTTPException(status_code=404, detail="Account not found")

    data = snapshot.to_dict() or {}
    return {
        "id": x_account_id,
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
    X-Account-Id header (never from a client-supplied query param).

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
        if str(d.get("role") or "").strip().lower() not in _CALL_ROLES:
            continue
        if str(d.get("duty") or "").strip().lower() != "on_duty":
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


class AccountCreate(BaseModel):
    id: str | None = None  # if provided, reuses this doc id (e.g. a Firebase Auth UID from /auth/signup) instead of generating a new one — see create_account() below
    firstName: str = ""
    lastName: str = ""
    badge: str = ""
    email: str = ""
    role: str = ""
    stationId: str = ""
    status: str = ""            # "on" / "off" / "inactive", as before
    duty: str = "off_duty"      # shift state; new accounts start off shift
    unitId: str | None = None
    is_new: bool = True


class AccountUpdate(BaseModel):
    firstName: str | None = None
    lastName: str | None = None
    badge: str | None = None
    email: str | None = None
    role: str | None = None
    stationId: str | None = None
    status: str | None = None
    duty: str | None = None
    unitId: str | None = None
    is_new: bool | None = None


class DutyUpdate(BaseModel):
    duty: str | None = None  # omit to flip whatever is currently stored


def _has_duty(role: str) -> bool:
    """
    Only plain field responders track a duty/shift state — super_admin and
    responder_admin accounts don't go on/off shift. A blank role defaults to
    Responder, matching the list endpoint's fallback.
    """
    return (role or "Responder").strip().lower() == "responder"


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

    # duty is a plain-responder-only concept; super_admin/responder_admin
    # report null so the UI can render a dash instead of a misleading
    # "Off duty" badge.
    account["duty"] = _normalize_duty(data.get("duty")) if has_duty else None
    account["dutyChangedAt"] = data.get("dutyChangedAt") if has_duty else None

    return account


def _require_super_admin_caller(db, x_account_id: str | None) -> None:
    """Granting the super_admin role is itself restricted to a super_admin."""
    if not x_account_id:
        raise HTTPException(status_code=401, detail="Missing X-Account-Id header")
    snapshot = db.collection(RESPONDERS).document(x_account_id).get()
    caller = snapshot.to_dict() or {} if snapshot.exists else {}
    caller_role = (caller.get("role") or caller.get("position") or "").strip().lower()
    if caller_role not in FULL_ACCESS_ROLES:
        raise HTTPException(status_code=403, detail="Only a Super Admin can assign the Super Admin role.")


def _role_of(data: dict) -> str:
    return (data.get("role") or data.get("position") or "").strip().lower()


def _get_caller(db, x_account_id: str | None) -> dict | None:
    """Caller resolved from X-Account-Id, or None if missing / unknown."""
    if not x_account_id:
        return None
    snapshot = db.collection(RESPONDERS).document(x_account_id).get()
    if not snapshot.exists:
        return None
    data = snapshot.to_dict() or {}
    return {
        "id": x_account_id,
        "role": _role_of(data),
        "stationId": str(data.get("stationId") or "").strip(),
    }


def _require_super_admin(db, x_account_id: str | None, action: str) -> dict:
    caller = _get_caller(db, x_account_id)
    if caller is None:
        raise HTTPException(status_code=401, detail="Missing or unknown X-Account-Id header")
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
    Full profile for the logged-in caller, resolved from the X-Account-Id
    header. Frontend calls this right after login to know who it's talking to.
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
    duty: str | None = None,
    status: str | None = None,
    role: str | None = None,
    x_account_id: str = Header(None, alias="X-Account-Id"),
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

    duty_filter = _normalize_duty(duty) if duty else None
    status_filter = _normalize_status(status) if status else None
    role_filter = role.strip().lower() if role else None

    # A station admin only gets responders from their own station — never
    # Admin / Super Admin accounts and never other stations.
    caller = _get_caller(db, x_account_id)
    station_admin_scope = caller["stationId"] if caller and caller["role"] == "responder_admin" else None
    scope_active = caller is not None and caller["role"] == "responder_admin"

    accounts = []
    for doc in db.collection(RESPONDERS).stream():
        account = _serialize(doc.id, doc.to_dict() or {})

        if scope_active:
            if not station_admin_scope or str(account["stationId"] or "").strip() != station_admin_scope:
                continue
            if account["role"].strip().lower() in ADMIN_LEVEL_ROLES:
                continue

        if role_filter and account["role"].strip().lower() != role_filter:
            continue
        if status_filter and account["status"] != status_filter:
            continue
        # A duty filter only ever matches plain responders, since duty is
        # None for super_admin/responder_admin.
        if duty_filter and account["duty"] != duty_filter:
            continue

        accounts.append(account)

    field_responders = [a for a in accounts if a["duty"] is not None]
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
def create_account(payload: AccountCreate, x_account_id: str = Header(None, alias="X-Account-Id")):
    """
    Creates a plain responder, a responder_admin ("Admin" in the Accounts page
    dropdown; the stored role string is still "responder_admin"), or a
    super_admin ("Super Admin" in the dropdown; only a super_admin may do this).
    """
    requested_role = (payload.role or "Responder").strip()

    db = get_db()
    # Adding accounts is a Super Admin action. Station admins can only manage
    # existing accounts (see update_account).
    caller = _require_super_admin(db, x_account_id, "create accounts")
    _assert_may_assign_in_own_station(
        caller,
        payload.stationId,
        role_change=requested_role.lower() != "responder",
        unit_change=bool(payload.unitId),
    )

    # If this account was created via /auth/signup first (the normal flow from
    # the Accounts page — see accounts.js), payload.id is that Firebase Auth
    # UID, and signup() has already written a doc under it (with an older,
    # different field shape). Reusing that same id here means the write below
    # correctly fills in the real profile schema, rather than creating a
    # second, disconnected record that isn't tied to any login credential.
    account_id = payload.id or str(uuid4())

    has_duty = _has_duty(requested_role)

    account_data = {
        "firstName": payload.firstName,
        "lastName": payload.lastName,
        "badge": payload.badge,
        "email": payload.email,
        "role": requested_role,
        "stationId": payload.stationId,
        "status": _normalize_status(payload.status),
        "unitId": payload.unitId,
        "is_new": payload.is_new,
    }

    if has_duty:
        account_data["duty"] = _normalize_duty(payload.duty)
        account_data["dutyChangedAt"] = datetime.now(timezone.utc)

    # merge=True (not a plain .set()) so this doesn't wipe out the "username"
    # field that /auth/signup already wrote for this same doc — login-by-
    # username depends on that field still being present.
    db.collection(RESPONDERS).document(account_id).set({
        **account_data,
        "createdAt": datetime.now(timezone.utc),
    }, merge=True)

    return {"id": account_id, "collection": RESPONDERS, **account_data}


@router.patch("/{account_id}")
def update_account(account_id: str, payload: AccountUpdate, x_account_id: str = Header(None, alias="X-Account-Id")):
    db = get_db()
    ref, snapshot = _find_account(db, account_id)
    existing = snapshot.to_dict() or {}

    data = payload.model_dump(exclude_unset=True)

    caller = _get_caller(db, x_account_id)
    if caller and caller["role"] == "responder_admin":
        if caller["id"] == account_id:
            # Editing yourself (e.g. clearing is_new after a password change) is
            # fine, but an admin can't change their own role, station or status.
            for field in ("role", "stationId", "status"):
                data.pop(field, None)
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
        _require_super_admin_caller(db, x_account_id)

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
def set_duty(account_id: str, payload: DutyUpdate):
    """
    Shift toggle. Send {"duty": "on_duty"} / {"duty": "off_duty"} to set an
    explicit state, or an empty body to flip whatever is currently stored.

    Plain field responders only. A deactivated account cannot go on duty —
    that is the one place where status and duty interact.
    """
    db = get_db()
    ref, snapshot = _find_account(db, account_id)
    existing_role = (snapshot.to_dict() or {}).get("role")

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
def set_status(account_id: str, payload: AccountUpdate, x_account_id: str = Header(None, alias="X-Account-Id")):
    """
    Set account status. Send {"status": "on"}, {"status": "off"}, or
    {"status": "inactive"} — same values the Accounts admin UI has always
    used.

    Deactivating a responder (status "inactive") force-ends any open shift,
    so a disabled account can't be left sitting in the on-duty roster.
    """
    if payload.status is None:
        raise HTTPException(status_code=422, detail="status is required")

    db = get_db()
    ref, snapshot = _find_account(db, account_id)
    existing = snapshot.to_dict() or {}

    caller = _get_caller(db, x_account_id)
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


@router.post("/{account_id}/reset-password")
def reset_password(account_id: str, x_account_id: str = Header(None, alias="X-Account-Id")):
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
    _require_super_admin(db, x_account_id, "reset passwords")
    ref, _ = _find_account(db, account_id)

    try:
        firebase_auth.update_user(account_id, password=DEFAULT_RESPONDER_PASSWORD)
    except Exception as err:
        # Covers firebase_admin.auth.UserNotFoundError (doc exists in
        # Firestore but has no matching Auth credential — e.g. a manually
        # seeded record) and any other Admin SDK failure.
        raise HTTPException(status_code=502, detail=f"Could not reset password: {err}")

    ref.set({"is_new": True}, merge=True)

    return {"accountId": account_id, "collection": RESPONDERS, "reset": True, "is_new": True}


@router.delete("/{account_id}")
def delete_account(account_id: str, x_account_id: str = Header(None, alias="X-Account-Id")):
    db = get_db()
    _require_super_admin(db, x_account_id, "revoke accounts")
    ref, _ = _find_account(db, account_id)
    ref.delete()

    return {"deleted": True, "accountId": account_id, "collection": RESPONDERS}