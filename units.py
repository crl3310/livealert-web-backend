from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from accounts import admin_only, get_station_scope
from firebase import get_db

router = APIRouter(prefix="/units", tags=["Units"])


class UnitCreate(BaseModel):
    stationId: str
    unitName: str
    jurisdiction: str = "General Area"
    status: str = "on"


class UnitUpdate(BaseModel):
    stationId: str | None = None
    unitName: str | None = None
    jurisdiction: str | None = None
    status: str | None = None


def _assert_own_station(account: dict, station_id: str | None) -> None:
    """Admins may only create/change units in their own station."""
    if not account["stationId"] or str(station_id or "").strip() != account["stationId"]:
        raise HTTPException(status_code=403, detail="You can only manage units in your own station.")


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


def _on_duty_responders(db, station_id: str) -> list[dict]:
    """On-duty responders of ONE station, as small records the frontend can match on."""
    out, seen = [], set()
    for coll in _RESPONDER_COLLECTIONS:
        for doc in db.collection(coll).stream():
            if doc.id in seen:
                continue
            data = doc.to_dict() or {}
            if str(data.get("stationId") or "").strip() != station_id:
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


@router.get("")
def list_units(station_id: str = Depends(get_station_scope), _: dict = Depends(admin_only)):
    """Frontend units page: the caller's own station's units (filtered server-side)."""
    db = get_db()
    docs = [
        d for d in db.collection("Units").stream()
        if str((d.to_dict() or {}).get("stationId") or "").strip() == station_id
    ]
    units = [_serialize_unit(doc) for doc in docs]
    return {"units": units, "onDutyResponders": _on_duty_responders(db, station_id)}


@router.post("", status_code=201)
def create_unit(payload: UnitCreate, account: dict = Depends(admin_only)):
    db = get_db()
    _assert_own_station(account, payload.stationId)

    # Units live under a station — fail fast with a clear error instead of
    # silently writing a unit that references a station that doesn't exist,
    # which would otherwise show up as a "phantom" unit nothing can reach.
    station_ref = db.collection("Stations").document(payload.stationId)
    if not station_ref.get().exists:
        raise HTTPException(status_code=404, detail=f"Station '{payload.stationId}' not found")

    unit_data = {
        "stationId": payload.stationId,
        "unitName": payload.unitName,
        "jurisdiction": payload.jurisdiction,
        "status": payload.status,
    }
    update_time, doc_ref = db.collection("Units").add(unit_data)
    return {"id": doc_ref.id, **unit_data}


@router.patch("/{unit_id}")
def update_unit(unit_id: str, payload: UnitUpdate, account: dict = Depends(admin_only)):
    db = get_db()
    ref = db.collection("Units").document(unit_id)
    snap = ref.get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="Unit not found")
    _assert_own_station(account, (snap.to_dict() or {}).get("stationId"))

    update_data = payload.model_dump(exclude_unset=True)
    if "stationId" in update_data:
        _assert_own_station(account, update_data["stationId"])
        station_ref = db.collection("Stations").document(update_data["stationId"])
        if not station_ref.get().exists:
            raise HTTPException(status_code=404, detail=f"Station '{update_data['stationId']}' not found")

    if update_data:
        ref.update(update_data)

    return {"updated": True, "unitId": unit_id}