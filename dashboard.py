from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
from math import radians, sin, cos, sqrt, atan2
from accounts import get_station_scope, route_unassigned
from firebase import get_db

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

def _created_datetime(data: dict[str, Any]) -> datetime | None:
    return _to_dt(data.get("timestamp") or data.get("createdAt"))


# A report is live until it has ended or a day has passed; after that it
# belongs in History. (report.py, dashboard.py and history.py share this rule.)
ARCHIVE_AFTER = timedelta(hours=24)
ENDED_STATUSES = frozenset({"resolved", "closed", "completed", "cancelled", "ended"})


def _has_ended(data: dict[str, Any]) -> bool:
    return str(data.get("status") or "").strip().lower() in ENDED_STATUSES


def _is_archived(data: dict[str, Any]) -> bool:
    if _has_ended(data):
        return True
    created = _created_datetime(data)
    return created is not None and datetime.now(timezone.utc) - created >= ARCHIVE_AFTER


def _report_station_id(data: dict[str, Any]) -> str:
    """The station a report was routed to (same precedence the payloads use)."""
    assigned = data.get("assignedStation") or {}
    return str(data.get("nearestStationId") or assigned.get("stationId") or "").strip()


def _matches_station(data: dict[str, Any], station_id: str | None) -> bool:
    """True if the report belongs to station_id. No station_id = no filtering."""
    if not station_id or not station_id.strip():
        return True
    return _report_station_id(data) == station_id.strip()



class AccountCreate(BaseModel):
    name: str
    role: str
    group: str = "Alpha 1"
    status: str = "on"


class AccountUpdate(BaseModel):
    name: str | None = None
    role: str | None = None
    group: str | None = None
    status: str | None = None


class ReportUpdate(BaseModel):
    status: str | None = None
    level: str | None = None
    incidentType: str | None = None
    description: str | None = None
    nearestStationId: str | None = None


def _to_dt(firestore_ts):
    if firestore_ts is None:
        return None
    if hasattr(firestore_ts, "ToDatetime"):
        return firestore_ts.ToDatetime(tzinfo=timezone.utc)
    if isinstance(firestore_ts, datetime):
        return firestore_ts if firestore_ts.tzinfo else firestore_ts.replace(tzinfo=timezone.utc)
    return None


def _haversine(lat1, lng1, lat2, lng2) -> float:
    """Calculate distance in km between two coordinates."""
    R = 6371
    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _location_to_string(location: Any) -> str:
    if isinstance(location, str):
        return location
    if isinstance(location, dict):
        address = location.get("address") or location.get("name") or ""
        if address:
            return address
        lat = location.get("lat")
        lng = location.get("lng")
        if lat is not None and lng is not None:
            return f"{lat}, {lng}"
    if hasattr(location, "latitude"):
        return f"{location.latitude}, {location.longitude}"
    return "Unknown location"


def _normalize_level(level: Any) -> str:
    if isinstance(level, str) and level.lower().startswith("level"):
        return level
    if level in (None, ""):
        return "Level 3"
    return f"Level {level}"


def _format_time(created_dt: datetime | None) -> str:
    if not created_dt:
        return "Just now"
    now = datetime.now(timezone.utc)
    delta = now - created_dt
    minutes = int(delta.total_seconds() // 60)
    if minutes < 1:
        return "Just now"
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days == 1:
        return "Yesterday"
    return created_dt.strftime("%b %d")


def _history_label(created_dt: datetime | None) -> str:
    if not created_dt:
        return "Today"
    now = datetime.now(timezone.utc)
    delta = now - created_dt
    days = int(delta.total_seconds() // 86400)
    if days <= 0:
        return "Today"
    if days == 1:
        return "Yesterday"
    return created_dt.strftime("%B %d")


def _get_reporter_name(db: Any, reporter_id: str | None) -> str:
    """Try to resolve a reporter UUID to a user's name; fall back to the raw value."""
    if not reporter_id:
        return "Unknown"
    try:
        user_doc = db.collection("Users").document(reporter_id).get()
        if user_doc.exists:
            data = user_doc.to_dict() or {}
            first = data.get("firstName", "")
            last = data.get("lastName", "")
            if first or last:
                return f"{first} {last}".strip()
            if data.get("name"):
                return data["name"]
    except Exception:
        pass
    # ActiveCalls.reporterUuid isn't always a real Users doc id (e.g. test data),
    # so fall back to showing the raw value instead of "Unknown".
    return reporter_id


def _active_call_lat_lng(location: Any) -> tuple[Any, Any]:
    if isinstance(location, dict):
        lat = location.get("latitude", location.get("lat"))
        lng = location.get("longitude", location.get("lng"))
        return lat, lng
    if hasattr(location, "latitude"):
        return location.latitude, location.longitude
    return None, None


def _build_report_payload(doc_id: str, data: dict[str, Any], db: Any) -> dict[str, Any]:
    """Builds the shared report shape from an ActiveCalls document."""
    created_dt = _to_dt(data.get("timestamp") or data.get("createdAt"))
    location = data.get("location") if isinstance(data.get("location"), dict) else {}
    location_str = _location_to_string(data.get("location"))

    assigned_station = data.get("assignedStation") or {}
    incident_type = data.get("incidentType") or data.get("channelName") or "Unknown"
    description = (
        data.get("description")
        or data.get("ai_assessment")
        or f"{incident_type} reported at {location_str}."
    )
    lat, lng = _active_call_lat_lng(location)

    return {
        "id": doc_id,
        "type": incident_type,
        "location": location_str,
        "time": _format_time(created_dt),
        "urgency": _normalize_level(data.get("level")),
        "reporter": _get_reporter_name(db, data.get("reporterUuid") or data.get("reportedBy")),
        "description": description,
        "lat": data.get("lat") or lat,
        "lng": data.get("lng") or lng,
        "date": _history_label(created_dt),
        "status": data.get("status", "pending"),
        "nearestStationId": data.get("nearestStationId") or assigned_station.get("stationId", ""),
        "streamChannelId": data.get("streamChannelId") or data.get("channelName", ""),
        # Stamped server-side by report.py's PATCH /reports/{id} the moment
        # status crosses into ENDED_STATUSES — surfaced here too since this
        # payload shape mirrors report.py's.
        "endedAt": _to_dt(data.get("endedAt")).isoformat() if _to_dt(data.get("endedAt")) else None,
    }


# ── Stats ─────────────────────────────────────────────────────────────────────

@router.get("/stats")
def get_stats(station_id: str | None = Depends(get_station_scope)):
    """
    Returns all 4 dashboard stat cards:
      todayReports, unitsAvailable, unitsDeployed, reportedAreas, recentReports
    """
    db = get_db()
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    call_docs = route_unassigned(db, list(db.collection("ActiveCalls").stream()))

    today_reports = 0
    reported_station_ids = set()
    recent_reports = []

    for doc in call_docs:
        data = doc.to_dict() or {}
        if not _matches_station(data, station_id):
            continue
        created_dt = _to_dt(data.get("timestamp") or data.get("createdAt"))

        if created_dt and created_dt >= today_start:
            today_reports += 1

        assigned_station = data.get("assignedStation") or {}
        sid = (data.get("nearestStationId") or assigned_station.get("stationId", "")).strip()
        if sid and not _is_archived(data):
            reported_station_ids.add(sid)

        # Ended or day-old reports live in History now, not the dashboard feed.
        if _is_archived(data):
            continue

        recent_reports.append({
            "id": doc.id,
            "incidentType": data.get("incidentType") or data.get("channelName") or "Unknown",
            "nearestStationId": data.get("nearestStationId") or assigned_station.get("stationId", ""),
            "status": data.get("status", "pending"),
            "createdAt": created_dt.isoformat() if created_dt else None,
            "endedAt": _to_dt(data.get("endedAt")).isoformat() if _to_dt(data.get("endedAt")) else None,
        })

    recent_reports.sort(key=lambda r: r["createdAt"] or "", reverse=True)
    recent_reports = recent_reports[:10]

    responder_docs = list(db.collection("Responders").stream())
    if station_id:
        responder_docs = [d for d in responder_docs if (d.to_dict() or {}).get("stationId") == station_id]
    units_available = sum(1 for d in responder_docs if d.to_dict().get("status") == "available")
    units_deployed  = sum(1 for d in responder_docs if d.to_dict().get("status") == "deployed")

    return {
        "todayReports": today_reports,
        "unitsAvailable": units_available,
        "unitsDeployed": units_deployed,
        "reportedAreas": len(reported_station_ids),
        "recentReports": recent_reports,
    }


# ── Analytics (stats cards, charts and heatmap) ──────────────────────────────

# Unit status values (lower-cased) that mean "not able to take a call".
_UNAVAILABLE_UNIT_STATUSES = frozenset({"off", "offline", "inactive", "unavailable", "maintenance", "out of service"})
_DEPLOYED_UNIT_STATUSES = frozenset({"deployed", "dispatched", "on call", "on_call", "busy", "responding", "en route"})


def _to_coord(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _call_coordinates(data: dict[str, Any]) -> tuple[float | None, float | None]:
    """Latitude/longitude of an ActiveCalls document, wherever it was saved:
    top-level lat/lng (or latitude/longitude), or inside `location` as a map
    or a Firestore GeoPoint."""
    loc_lat, loc_lng = _active_call_lat_lng(data.get("location"))
    lat = _to_coord(data.get("lat", data.get("latitude")))
    lng = _to_coord(data.get("lng", data.get("longitude")))
    if lat is None or lng is None:
        lat, lng = _to_coord(loc_lat), _to_coord(loc_lng)
    if lat is None or lng is None or abs(lat) > 90 or abs(lng) > 180 or (lat == 0 and lng == 0):
        return None, None
    return lat, lng


def _call_unit_ids(data: dict[str, Any]) -> list[str]:
    assigned = data.get("assignedStation") or {}
    raw = data.get("unitIds") if isinstance(data.get("unitIds"), list) else assigned.get("unitIds")
    return [str(u) for u in (raw or []) if u]


@router.get("/analytics")
def get_analytics(
    days: int = 366,
    tz_offset: int = 0,
    station_id: str | None = Depends(get_station_scope),
):
    """One call for the whole dashboard: every ActiveCalls report from the last
    `days` days (with coordinates for the heatmap), plus unit / responder /
    station numbers from the Units, Responders and Stations collections.

    tz_offset = the browser's offset from UTC in minutes (Manila = 480) so
    "today" means the viewer's day, not the UTC day."""
    db = get_db()
    days = max(1, min(days, 800))
    now = datetime.now(timezone.utc)
    offset = timedelta(minutes=max(-840, min(tz_offset, 840)))
    today_start = (now + offset).replace(hour=0, minute=0, second=0, microsecond=0) - offset
    yesterday_start = today_start - timedelta(days=1)
    cutoff = now - timedelta(days=days)

    # Stations: id -> display name
    station_docs = list(db.collection("Stations").stream())
    station_names = {}
    for d in station_docs:
        sd = d.to_dict() or {}
        station_names[d.id] = sd.get("stationName") or sd.get("name") or d.id

    # ActiveCalls: every report in range, plus which units are out on live calls
    reports = []
    today_reports = yesterday_reports = 0
    busy_unit_ids: set[str] = set()
    stations_with_live_calls: set[str] = set()
    call_docs = route_unassigned(db, list(db.collection("ActiveCalls").stream()))
    for doc in call_docs:
        data = doc.to_dict() or {}
        if not _matches_station(data, station_id):
            continue
        created_dt = _created_datetime(data)
        sid = _report_station_id(data) or str(data.get("stationId") or "").strip()

        if not _is_archived(data):
            busy_unit_ids.update(_call_unit_ids(data))
            if sid:
                stations_with_live_calls.add(sid)

        if created_dt is None:
            continue
        if created_dt >= today_start:
            today_reports += 1
        elif created_dt >= yesterday_start:
            yesterday_reports += 1
        if created_dt < cutoff:
            continue

        lat, lng = _call_coordinates(data)
        ended_dt = _to_dt(data.get("endedAt"))
        reports.append({
            "id": doc.id,
            "type": data.get("incidentType") or data.get("channelName") or "Unknown",
            "urgency": _normalize_level(data.get("level")),
            "status": data.get("status", "pending"),
            "createdAt": created_dt.isoformat(),
            "endedAt": ended_dt.isoformat() if ended_dt else None,
            "stationId": sid,
            "stationName": data.get("stationName") or station_names.get(sid, sid),
            "location": _location_to_string(data.get("location")),
            "lat": lat,
            "lng": lng,
            "unitCount": len(_call_unit_ids(data)),
        })
    reports.sort(key=lambda r: r["createdAt"], reverse=True)

    # Units: deployed = out on a live call (or marked deployed); available = the rest that are on duty
    unit_docs = list(db.collection("Units").stream())
    if station_id:
        unit_docs = [d for d in unit_docs if str((d.to_dict() or {}).get("stationId") or "").strip() == station_id.strip()]
    units_total = units_deployed = units_available = 0
    for d in unit_docs:
        status = str((d.to_dict() or {}).get("status") or "").strip().lower()
        if status in _UNAVAILABLE_UNIT_STATUSES:
            continue
        units_total += 1
        if d.id in busy_unit_ids or status in _DEPLOYED_UNIT_STATUSES:
            units_deployed += 1
        else:
            units_available += 1

    # Responders (the people): same status field the stats card used before
    responder_docs = list(db.collection("Responders").stream())
    if station_id:
        responder_docs = [d for d in responder_docs if (d.to_dict() or {}).get("stationId") == station_id]
    responder_status = [str((d.to_dict() or {}).get("status") or "").strip().lower() for d in responder_docs]

    return {
        "generatedAt": now.isoformat(),
        "todayReports": today_reports,
        "yesterdayReports": yesterday_reports,
        "reports": reports,
        "units": {"total": units_total, "available": units_available, "deployed": units_deployed},
        "responders": {
            "total": len(responder_docs),
            "available": responder_status.count("available"),
            "deployed": responder_status.count("deployed"),
        },
        "stations": {
            "total": len(station_docs) if not station_id else 1,
            "withLiveReports": len(stations_with_live_calls),
        },
    }


# ── Active Emergencies ────────────────────────────────────────────────────────

@router.get("/active")
def get_active(station_id: str | None = Depends(get_station_scope)):
    """Left panel — active calls. Called directly by the frontend's
    getActiveEmergencies(), which is what report.js polls to find the next
    incident to open. Note: since the ActiveCalls collection only ever
    contains calls that are currently in progress, we don't filter by a
    "pending"/"active" status the way the old Emergencies-based version did
    (ActiveCalls documents use their own status values, e.g. "evaluating").
    We just exclude anything explicitly marked as finished."""
    db = get_db()
    docs = route_unassigned(db, list(db.collection("ActiveCalls").stream()))

    active = []
    for doc in docs:
        data = doc.to_dict() or {}
        if _is_archived(data) or not _matches_station(data, station_id):
            continue
        assigned_station = data.get("assignedStation") or {}
        created_dt = _to_dt(data.get("timestamp") or data.get("createdAt"))
        lat, lng = _active_call_lat_lng(data.get("location"))
        active.append({
            "id": doc.id,
            "incidentType": data.get("incidentType") or data.get("channelName") or "Unknown",
            "location": _location_to_string(data.get("location")),
            "lat": data.get("lat") or lat,
            "lng": data.get("lng") or lng,
            "nearestStationId": data.get("nearestStationId") or assigned_station.get("stationId", ""),
            "reportedBy": data.get("reporterUuid") or data.get("reportedBy", ""),
            "respondersInvolved": data.get("respondersInvolved", []),
            "status": data.get("status", "pending"),
            "streamChannelId": data.get("streamChannelId") or data.get("channelName", ""),
            "createdAt": created_dt,
            # Always None here in practice — _is_archived() already filters out
            # anything with an ended status before it reaches this list — but
            # kept for shape parity with the other report payloads.
            "endedAt": _to_dt(data.get("endedAt")).isoformat() if _to_dt(data.get("endedAt")) else None,
        })

    # Previously this returned whatever order Firestore happened to stream
    # documents in — which is arbitrary, not chronological. The frontend
    # (report.js, dashboard.js) assumes index 0 is the most recent active
    # call, so an old/stale doc could keep "winning" indefinitely just by
    # virtue of Firestore's internal ordering, regardless of what's actually
    # newest. Sort by real timestamp, newest first, so that assumption holds.
    active.sort(key=lambda item: item["createdAt"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    return {"active": active, "count": len(active)}


# ── Recent Reports ────────────────────────────────────────────────────────────

@router.get("/recent")
def get_recent(limit: int = 10, station_id: str | None = Depends(get_station_scope)):
    """Right panel — most recent reports sorted by createdAt."""
    db = get_db()
    # No .limit() on the query: ended / day-old reports are skipped below, so
    # the limit has to be applied after filtering or the feed would come up short.
    docs = route_unassigned(db, list(
        db.collection("ActiveCalls")
        .order_by("timestamp", direction="DESCENDING")
        .stream()
    ))

    reports = []
    for doc in docs:
        data = doc.to_dict() or {}
        if _is_archived(data) or not _matches_station(data, station_id):
            continue
        created_dt = _to_dt(data.get("timestamp") or data.get("createdAt"))
        assigned_station = data.get("assignedStation") or {}
        reports.append({
            "id": doc.id,
            "incidentType": data.get("incidentType") or data.get("channelName") or "Unknown",
            "nearestStationId": data.get("nearestStationId") or assigned_station.get("stationId", ""),
            "status": data.get("status", "pending"),
            "createdAt": created_dt.isoformat() if created_dt else None,
            # Same parity note as get_active(): filtered to non-archived docs,
            # so this is always None today, but kept for a consistent shape.
            "endedAt": _to_dt(data.get("endedAt")).isoformat() if _to_dt(data.get("endedAt")) else None,
        })
        if len(reports) >= limit:
            break

    return {"reports": reports}

# NOTE: this file used to also define /dashboard/reports/recent,
# /dashboard/reports/{report_id}, /dashboard/history, and a PATCH on
# /dashboard/reports/{report_id}. Those were dead code left over from before
# report.py and history.py existed as their own routers (this router's prefix
# is "/dashboard", so they never actually collided with /reports/... or
# /history, they just duplicated stale Emergencies-based logic that nothing
# called). Removed here — use the routes in report.py / history.py instead,
# which already read from ActiveCalls.


@router.get("/accounts")
def list_accounts():
    """Frontend accounts page: operational unit roster."""
    db = get_db()
    docs = list(db.collection("Responders").stream())

    accounts = []
    for doc in docs:
        data = doc.to_dict() or {}
        status = data.get("status", "available")
        if status in ("available", "active"):
            normalized_status = "on"
        elif status == "deployed":
            normalized_status = "off"
        else:
            normalized_status = "inactive"

        accounts.append({
            "id": doc.id,
            "name": data.get("name") or f"{data.get('firstName', '')} {data.get('lastName', '')}".strip() or "Unknown",
            "role": data.get("role") or data.get("position") or "Firefighter",
            "group": data.get("group") or data.get("station") or "Alpha 1",
            "status": normalized_status,
        })

    return {"accounts": accounts[:20]}


@router.post("/accounts", status_code=201)
def create_account(payload: AccountCreate):
    db = get_db()
    account_id = str(uuid4())
    account_data = {
        "name": payload.name,
        "role": payload.role,
        "group": payload.group,
        "status": payload.status,
    }
    db.collection("Responders").document(account_id).set(account_data)
    return {"id": account_id, **account_data}


@router.patch("/accounts/{account_id}")
def update_account(account_id: str, payload: AccountUpdate):
    db = get_db()
    ref = db.collection("Responders").document(account_id)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="Account not found")
    data = payload.model_dump(exclude_unset=True)
    if data:
        ref.update(data)
    return {"updated": True, "accountId": account_id}


@router.delete("/accounts/{account_id}")
def delete_account(account_id: str):
    db = get_db()
    ref = db.collection("Responders").document(account_id)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="Account not found")
    ref.delete()
    return {"deleted": True, "accountId": account_id}


# ── Modify Nearest Station Modal ──────────────────────────────────────────────

@router.get("/stations/{emergency_id}")
def get_stations_for_emergency(emergency_id: str):
    """
    Returns all stations with their distance from the emergency location.
    Used to populate the 'Modify Nearest Station' modal map.
    """
    db = get_db()

    # Get the active call
    doc = db.collection("ActiveCalls").document(emergency_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Active call not found")

    data = doc.to_dict()
    assigned_station = data.get("assignedStation") or {}
    current_nearest = data.get("nearestStationId") or assigned_station.get("stationId", "")
    emergency_location = data.get("location")

    # Get coordinates if available (ActiveCalls stores latitude/longitude)
    e_lat, e_lng = None, None
    if emergency_location:
        if hasattr(emergency_location, "latitude"):
            e_lat, e_lng = emergency_location.latitude, emergency_location.longitude
        elif isinstance(emergency_location, dict):
            e_lat = emergency_location.get("latitude", emergency_location.get("lat"))
            e_lng = emergency_location.get("longitude", emergency_location.get("lng"))

    # Get all stations
    station_docs = list(db.collection("Stations").stream())
    stations = []

    for s in station_docs:
        sd = s.to_dict()
        loc = sd.get("Location") if sd.get("Location") is not None else sd.get("location")

        s_lat, s_lng = None, None
        if loc:
            if hasattr(loc, "latitude"):
                s_lat, s_lng = loc.latitude, loc.longitude
            elif isinstance(loc, dict):
                s_lat = loc.get("lat")
                s_lng = loc.get("lng")

        # Calculate distance if both coordinates available
        distance_km = None
        if e_lat and e_lng and s_lat and s_lng:
            distance_km = round(_haversine(e_lat, e_lng, s_lat, s_lng), 1)

        stations.append({
            "id": s.id,
            "stationName": sd.get("stationName", "Unknown"),
            "chiefName": sd.get("chiefName", ""),
            "lat": s_lat,
            "lng": s_lng,
            "distanceKm": distance_km,
            "isNearest": s.id == current_nearest,
        })

    # Sort by distance
    stations.sort(key=lambda x: x["distanceKm"] if x["distanceKm"] is not None else 9999)

    return {
        "emergencyId": emergency_id,
        "currentNearestStationId": current_nearest,
        "stations": stations,
    }


class UpdateNearestStation(BaseModel):
    nearestStationId: str


@router.patch("/stations/{emergency_id}")
def update_nearest_station(emergency_id: str, body: UpdateNearestStation):
    """
    Updates the nearest station for an active call.
    Called when user confirms a new station in the modal.
    """
    db = get_db()

    ref = db.collection("ActiveCalls").document(emergency_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Active call not found")

    # Verify the station exists
    station_ref = db.collection("Stations").document(body.nearestStationId)
    station_doc = station_ref.get()
    if not station_doc.exists:
        raise HTTPException(status_code=404, detail="Station not found")

    station_data = station_doc.to_dict() or {}

    # Write both the flat field (used as a fallback by report payload builders)
    # and update the nested assignedStation, which is ActiveCalls' native shape.
    existing = doc.to_dict() or {}
    assigned_station = dict(existing.get("assignedStation") or {})
    assigned_station["stationId"] = body.nearestStationId
    assigned_station["stationName"] = station_data.get("stationName", "Unknown")
    assigned_station["commanderName"] = station_data.get("chiefName", assigned_station.get("commanderName", ""))

    ref.update({
        "nearestStationId": body.nearestStationId,
        "assignedStation": assigned_station,
    })

    return {
        "emergencyId": emergency_id,
        "updatedNearestStationId": body.nearestStationId,
        "stationName": station_data.get("stationName", "Unknown"),
    }