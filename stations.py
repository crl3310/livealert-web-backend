from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from accounts import admin_only, super_admin_only
from firebase import get_db

try:
    # firebase_admin's Firestore client exposes GeoPoint here.
    from firebase_admin.firestore import GeoPoint
except ImportError:  # pragma: no cover - fallback if using google-cloud-firestore directly
    from google.cloud.firestore_v1 import GeoPoint

router = APIRouter(prefix="/stations", tags=["Stations"])


class Location(BaseModel):
    latitude: float
    longitude: float


class StationCreate(BaseModel):
    stationName: str
    commanderName: str = ""
    location: Location


class StationUpdate(BaseModel):
    stationName: str | None = None
    commanderName: str | None = None
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
    stations = [_serialize_station(doc) for doc in docs]
    return {"stations": stations}


@router.post("", status_code=201, dependencies=[Depends(super_admin_only)])
def create_station(payload: StationCreate):
    db = get_db()
    station_id = _next_station_id(db)

    db.collection("Stations").document(station_id).set({
        "stationName": payload.stationName,
        "commanderName": payload.commanderName,
        "Location": GeoPoint(payload.location.latitude, payload.location.longitude),
    })

    return {
        "id": station_id,
        "stationName": payload.stationName,
        "commanderName": payload.commanderName,
        "location": payload.location.model_dump(),
    }


@router.patch("/{station_id}", dependencies=[Depends(super_admin_only)])
def update_station(station_id: str, payload: StationUpdate):
    db = get_db()
    station_ref = db.collection("Stations").document(station_id)
    if not station_ref.get().exists:
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
def delete_station(station_id: str):
    db = get_db()
    station_ref = db.collection("Stations").document(station_id)
    if not station_ref.get().exists:
        raise HTTPException(status_code=404, detail="Station not found")
    station_ref.delete()
    return {"deleted": True, "stationId": station_id}