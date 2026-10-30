import base64
import json
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from accounts import admin_only, get_station_scope
from firebase import get_db
from validation import (
    DocIdPath, DocIdStr, Jurisdiction, PatchModel, StrictModel, UnitName, UnitStatusValue, optional_doc_id,
)

router = APIRouter(prefix="/units", tags=["Units"])


class UnitCreate(StrictModel):
    stationId: DocIdStr
    unitName: UnitName
    jurisdiction: Jurisdiction = "General Area"
    status: UnitStatusValue = "on"


class UnitUpdate(PatchModel):
    # An explicit null is treated as "not sent" (PatchModel), so {"unitName": null}
    # can no longer blank a unit's name.
    stationId: DocIdStr | None = None
    unitName: UnitName | None = None
    jurisdiction: Jurisdiction | None = None
    status: UnitStatusValue | None = None
    # NOTE: PATCH disabled=true hides a unit WITHOUT releasing its personnel the way
    # DELETE does. If the frontend only uses DELETE, remove this field.
    disabled: bool | None = None


def _assert_own_station(account: dict, station_id: str | None) -> None:
    """Admins may only create/change units in their own station."""
    if not account["stationId"] or str(station_id or "").strip() != account["stationId"]:
        raise HTTPException(status_code=403, detail="You can only manage units in your own station.")


def _is_deleted(doc) -> bool:
    """Deleted units are kept in Firestore with disabled=True and hidden everywhere."""
    return (doc.to_dict() or {}).get("disabled") is True


def _serialize_unit(doc) -> dict:
    data = doc.to_dict() or {}
    return {
        "id": doc.id,
        "stationId": data.get("stationId", ""),
        "unitName": data.get("unitName", ""),
        "jurisdiction": data.get("jurisdiction", "General Area"),
        "status": data.get("status", "on"),
    }


_DUTY_FIELDS = ("duty", "dutyStatus", "onDuty")
_DUTY_ON_VALUES = {"on_duty", "on-duty", "on duty", "onduty", "on", "online", "true"}
_ID_FIELDS = ("userId", "accountId", "uid", "responderId", "email")


def _is_on_duty(data: dict) -> bool:
    for key in _DUTY_FIELDS:
        v = data.get(key)
        if v is True or (isinstance(v, str) and v.strip().lower() in _DUTY_ON_VALUES):
            return True
    # "status" is also used for account state elsewhere, so only the explicit
    # on-duty spellings count here (not "active"/"on").
    status = data.get("status")
    return isinstance(status, str) and status.strip().lower() in {"on_duty", "on-duty", "on duty"}


# The responder docs (badge, duty, unitId, ...) may live in any of these
# collections depending on how accounts are stored; empty/missing ones are
# simply skipped.
_RESPONDER_COLLECTIONS = ("Responders", "Accounts", "Users")


def _on_duty_responders(db, station_id: str | None = None) -> list[dict]:
    """On-duty responders as small records the frontend can match on.

    station_id=None means every station (used internally, e.g. by report.py);
    the Units page passes the caller's station so it only sees its own people."""
    out, seen = [], set()
    for coll in _RESPONDER_COLLECTIONS:
        for doc in db.collection(coll).stream():
            if doc.id in seen:
                continue
            data = doc.to_dict() or {}
            if data.get("disabled") is True:
                continue
            if station_id is not None and str(data.get("stationId") or "").strip() != station_id:
                continue
            if not _is_on_duty(data):
                continue
            seen.add(doc.id)
            out.append({
                "id": doc.id,
                "email": (data.get("email") or "").strip().lower(),
                "unitId": data.get("unitId") or None,
                "name": f"{data.get('firstName', '')} {data.get('lastName', '')}".strip().lower(),
            })
    return out


# ---------------------------------------------------------------------------
# Paged listing
#
# The Units page ranks units (can take reports -> staffed but all offline ->
# greyed out), shows four tile counts, and searches by unit OR personnel name.
# All of that depends on each unit's personnel and who is on duty, so it is
# computed here, on the server, and the page only receives one slice at a time.
#
#   GET /units?limit=20&cursor=<token>&q=<text>&filter=all|active|available|inactive
#   -> { "units":  [ {id, stationId, unitName, jurisdiction, status,
#                     personnel: [{id, name, online}], personnelCount, onlineCount,
#                     state: live|standby|empty|inactive, greyed, forwardable}, ... ],
#        "nextCursor": "<token>" | null,
#        "matched": N,                                  # units matching q + filter
#        "counts":  { total, active, inactive, available },   # the four tiles
#        "onDutyResponders": [...] }
#
# Without `limit` the answer is the old one: {"units": [...], "onDutyResponders": [...]},
# so other pages that call GET /units keep working.
# ---------------------------------------------------------------------------
_UNIT_PERSONNEL_ROLES = {"responder", "responder_admin"}   # super_admin is not a stationed role
MAX_PAGE_SIZE = 200


def _person_name(data: dict) -> str:
    name = f"{data.get('firstName', '')} {data.get('lastName', '')}".strip()
    return name or str(data.get("email") or "").strip() or "Unnamed"


def _station_personnel(db, station_id: str) -> list[dict]:
    """Every responder / responder_admin in the station, as small records."""
    people = []
    for doc in db.collection("Responders").stream():
        data = doc.to_dict() or {}
        if data.get("disabled") is True:
            continue
        if str(data.get("stationId") or "").strip() != station_id:
            continue
        role = str(data.get("role") or data.get("position") or "Responder").strip().lower()
        if role not in _UNIT_PERSONNEL_ROLES:
            continue
        people.append({
            "id": doc.id,
            "name": _person_name(data),
            "unitId": data.get("unitId") or None,
            "online": _is_on_duty(data),
        })
    people.sort(key=lambda p: p["name"].lower())
    return people


def _rank(row: dict) -> int:
    """0 = can take reports, 1 = staffed but everyone offline, 2 = greyed out."""
    if row["greyed"]:
        return 2
    return 0 if row["forwardable"] else 1


def _unit_row(doc, people_by_unit: dict[str, list[dict]]) -> dict:
    unit = _serialize_unit(doc)
    people = people_by_unit.get(doc.id, [])
    online = sum(1 for p in people if p["online"])
    active = unit["status"] == "on"
    forwardable = active and online > 0
    unit.update({
        "personnel": [{"id": p["id"], "name": p["name"], "online": p["online"]} for p in people],
        "personnelCount": len(people),
        "onlineCount": online,
        "state": "inactive" if not active else "empty" if not people else "live" if forwardable else "standby",
        "greyed": (not active) or not people,
        "forwardable": forwardable,
    })
    return unit


def _sort_key(row: dict) -> tuple:
    return (_rank(row), str(row["unitName"] or "").lower(), row["id"])


def _encode_cursor(key: tuple) -> str:
    return base64.urlsafe_b64encode(json.dumps(list(key)).encode()).decode()


def _decode_cursor(cursor: str | None) -> tuple | None:
    if not cursor:
        return None
    try:
        if len(cursor) > 512:
            raise ValueError("cursor too long")
        parts = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
        if not isinstance(parts, list) or len(parts) != 3:
            raise ValueError("bad shape")
        rank, name, uid = parts
        if not (isinstance(rank, int) and not isinstance(rank, bool) and 0 <= rank <= 2):
            raise ValueError("bad rank")
        if not isinstance(name, str) or not isinstance(uid, str):
            raise ValueError("bad types")
        return rank, name, uid
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid cursor.")


@router.get("")
def list_units(
    limit: int | None = Query(None, ge=1, le=MAX_PAGE_SIZE),
    cursor: str | None = Query(None, max_length=512),
    q: str | None = Query(None, max_length=100),
    filter: Literal["all", "active", "available", "inactive"] | None = Query(None),
    station_id: str = Depends(get_station_scope),
    _: dict = Depends(admin_only),
):
    """Frontend units page: the caller's own station's units (filtered server-side)."""
    db = get_db()
    docs = [
        d for d in db.collection("Units").stream()
        if str((d.to_dict() or {}).get("stationId") or "").strip() == station_id
        and not _is_deleted(d)
    ]

    # ----- Legacy (unpaged) answer: unchanged behaviour ---------------------
    if limit is None:
        units = [_serialize_unit(doc) for doc in docs]
        return {"units": units, "onDutyResponders": _on_duty_responders(db, station_id)}

    # ----- Paged answer ------------------------------------------------------
    page_size = limit
    after = _decode_cursor(cursor)

    people_by_unit: dict[str, list[dict]] = {}
    for person in _station_personnel(db, station_id):
        if person["unitId"]:
            people_by_unit.setdefault(person["unitId"], []).append(person)

    rows = sorted((_unit_row(doc, people_by_unit) for doc in docs), key=_sort_key)

    # The four tiles describe the whole station, ignoring search and the selected tile.
    counts = {
        "total": len(rows),
        "active": sum(1 for r in rows if not r["greyed"]),
        "inactive": sum(1 for r in rows if r["greyed"]),
        "available": sum(1 for r in rows if r["forwardable"]),
    }

    mode = (filter or "all").strip().lower()
    if mode == "active":
        rows = [r for r in rows if not r["greyed"]]
    elif mode == "inactive":
        rows = [r for r in rows if r["greyed"]]
    elif mode == "available":
        rows = [r for r in rows if r["forwardable"]]

    query = (q or "").strip().lower()
    if query:
        rows = [
            r for r in rows
            if query in str(r["unitName"] or "").lower()
            or any(query in p["name"].lower() for p in r["personnel"])
        ]

    matched = len(rows)
    if after is not None:
        rows = [r for r in rows if _sort_key(r) > after]
    page = rows[:page_size]
    next_cursor = _encode_cursor(_sort_key(page[-1])) if len(rows) > page_size and page else None

    return {"units": page, "nextCursor": next_cursor, "matched": matched, "counts": counts}


@router.get("/personnel")
def list_assignable_personnel(
    q: str | None = Query(None, max_length=100),
    unitId: str | None = Query(None, max_length=128),
    limit: int = Query(50, ge=1, le=100),
    station_id: str = Depends(get_station_scope),
    _: dict = Depends(admin_only),
):
    """Responders of the caller's station who can be put on a unit: nobody's unit yet, or
    the unit being edited (unitId). Used by the "+" picker in the Add/Edit Unit window."""
    query = (q or "").strip().lower()
    unitId = optional_doc_id(unitId, "unitId")
    out = []
    for person in _station_personnel(get_db(), station_id):
        if person["unitId"] and person["unitId"] != unitId:
            continue
        if query and query not in person["name"].lower():
            continue
        out.append({"id": person["id"], "name": person["name"], "online": person["online"]})
        if len(out) >= max(1, min(limit, 100)):
            break
    return {"personnel": out}


@router.post("", status_code=201)
def create_unit(payload: UnitCreate, account: dict = Depends(admin_only)):
    db = get_db()
    _assert_own_station(account, payload.stationId)

    # Units live under a station — fail fast with a clear error instead of
    # silently writing a unit that references a station that doesn't exist,
    # which would otherwise show up as a "phantom" unit nothing can reach.
    station_snap = db.collection("Stations").document(payload.stationId).get()
    if not station_snap.exists or (station_snap.to_dict() or {}).get("disabled") is True:
        raise HTTPException(status_code=404, detail=f"Station '{payload.stationId}' not found")

    unit_data = {
        "stationId": payload.stationId,
        "unitName": payload.unitName,
        "jurisdiction": payload.jurisdiction,
        "status": payload.status,
        "disabled": False,
    }
    update_time, doc_ref = db.collection("Units").add(unit_data)
    return {"id": doc_ref.id, **unit_data}


@router.patch("/{unit_id}")
def update_unit(unit_id: DocIdPath, payload: UnitUpdate, account: dict = Depends(admin_only)):
    db = get_db()
    ref = db.collection("Units").document(unit_id)
    snap = ref.get()
    if not snap.exists or (snap.to_dict() or {}).get("disabled") is True:
        raise HTTPException(status_code=404, detail="Unit not found")
    _assert_own_station(account, (snap.to_dict() or {}).get("stationId"))

    update_data = payload.model_dump(exclude_unset=True)
    if "stationId" in update_data:
        _assert_own_station(account, update_data["stationId"])
        station_snap = db.collection("Stations").document(update_data["stationId"]).get()
        if not station_snap.exists or (station_snap.to_dict() or {}).get("disabled") is True:
            raise HTTPException(status_code=404, detail=f"Station '{update_data['stationId']}' not found")

    if update_data:
        ref.update(update_data)

    return {"updated": True, "unitId": unit_id}


@router.delete("/{unit_id}")
def delete_unit(unit_id: DocIdPath, account: dict = Depends(admin_only)):
    """Soft delete: the unit stays in Firestore with disabled=True and disappears from the page."""
    db = get_db()
    ref = db.collection("Units").document(unit_id)
    snap = ref.get()
    if not snap.exists or (snap.to_dict() or {}).get("disabled") is True:
        raise HTTPException(status_code=404, detail="Unit not found")
    _assert_own_station(account, (snap.to_dict() or {}).get("stationId"))

    # Release everyone assigned to this unit (unitId -> None), in every collection
    # that can hold responder docs, then mark the unit deleted.
    released = 0
    for coll in _RESPONDER_COLLECTIONS:
        for doc in db.collection(coll).stream():
            if str((doc.to_dict() or {}).get("unitId") or "") == unit_id:
                doc.reference.update({"unitId": None})
                released += 1

    ref.update({"disabled": True})
    return {"deleted": True, "unitId": unit_id, "releasedResponders": released}