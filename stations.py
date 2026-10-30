from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from google.api_core.exceptions import AlreadyExists

from accounts import admin_only, super_admin_only
from firebase import get_db
from validation import CommanderName, DocIdPath, Latitude, Longitude, PatchModel, StationName, StrictModel

try:
    # firebase_admin's Firestore client exposes GeoPoint here.
    from firebase_admin.firestore import GeoPoint
except ImportError:  # pragma: no cover - fallback if using google-cloud-firestore directly
    from google.cloud.firestore_v1 import GeoPoint

router = APIRouter(prefix="/stations", tags=["Stations"])


class Location(StrictModel):
    latitude: Latitude      # -90..90, never NaN/Infinity (those would reach GeoPoint)
    longitude: Longitude    # -180..180


class StationCreate(StrictModel):
    stationName: StationName
    commanderName: CommanderName = ""
    location: Location


class StationUpdate(PatchModel):
    stationName: StationName | None = None
    commanderName: CommanderName | None = None
    location: Location | None = None


def _next_station_id(db) -> str:
    """Generates the next sequential 'ST00N' id based on existing station docs."""
    docs = list(db.collection("Stations").stream())
    nums = []
    for doc in docs:
        if doc.id.startswith("ST") and doc.id[2:].isdigit():
            nums.append(int(doc.id[2:]))
    next_num = (max(nums) + 1) if nums else 1
    return f"ST{next_num:03d}"


def _is_disabled(doc) -> bool:
    """A "deleted" station is only flagged disabled; its document stays in Firestore."""
    return (doc.to_dict() or {}).get("disabled") is True


def _serialize_station(doc) -> dict:
    data = doc.to_dict() or {}
    raw_location = data.get("Location")

    if raw_location is not None and hasattr(raw_location, "latitude"):
        location = {"latitude": raw_location.latitude, "longitude": raw_location.longitude}
    elif isinstance(raw_location, dict):
        location = {
            "latitude": raw_location.get("latitude", 0),
            "longitude": raw_location.get("longitude", 0),
        }
    else:
        location = {"latitude": 0, "longitude": 0}

    return {
        "id": doc.id,
        "stationName": data.get("stationName", ""),
        "commanderName": data.get("commanderName", ""),
        "location": location,
    }


@router.get("", dependencies=[Depends(admin_only)])
def list_stations():
    """Frontend accounts page: station registry for dropdowns/filters."""
    db = get_db()
    docs = list(db.collection("Stations").stream())
    stations = [_serialize_station(doc) for doc in docs if not _is_disabled(doc)]
    return {"stations": stations}


@router.post("", status_code=201, dependencies=[Depends(super_admin_only)])
def create_station(payload: StationCreate):
    db = get_db()

    # create() fails if the document already exists, so two admins creating a
    # station at the same moment can't silently overwrite each other (set() did).
    for _ in range(5):
        station_id = _next_station_id(db)
        try:
            db.collection("Stations").document(station_id).create({
                "stationName": payload.stationName,
                "commanderName": payload.commanderName,
                "Location": GeoPoint(payload.location.latitude, payload.location.longitude),
            })
            break
        except AlreadyExists:
            continue
    else:
        raise HTTPException(status_code=503, detail="Could not allocate a station id. Please try again.")

    return {
        "id": station_id,
        "stationName": payload.stationName,
        "commanderName": payload.commanderName,
        "location": payload.location.model_dump(),
    }


@router.patch("/{station_id}", dependencies=[Depends(super_admin_only)])
def update_station(station_id: DocIdPath, payload: StationUpdate):
    db = get_db()
    station_ref = db.collection("Stations").document(station_id)
    snapshot = station_ref.get()
    if not snapshot.exists or _is_disabled(snapshot):
        raise HTTPException(status_code=404, detail="Station not found")

    update_data = {}
    if payload.stationName is not None:
        update_data["stationName"] = payload.stationName
    if payload.commanderName is not None:
        update_data["commanderName"] = payload.commanderName
    if payload.location is not None:
        update_data["Location"] = GeoPoint(payload.location.latitude, payload.location.longitude)

    if update_data:
        station_ref.update(update_data)

    return {"updated": True, "stationId": station_id}


@router.delete("/{station_id}", dependencies=[Depends(super_admin_only)])
def delete_station(station_id: DocIdPath):
    db = get_db()
    station_ref = db.collection("Stations").document(station_id)
    snapshot = station_ref.get()
    if not snapshot.exists or _is_disabled(snapshot):
        raise HTTPException(status_code=404, detail="Station not found")

    # Soft delete: nothing is removed. The station is flagged and then hidden.
    # To bring one back, set disabled to false on its Stations document.
    station_ref.set({"disabled": True, "disabledAt": datetime.now(timezone.utc)}, merge=True)
    return {"deleted": True, "stationId": station_id}