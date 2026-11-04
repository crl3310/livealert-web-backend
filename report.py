from datetime import datetime, timedelta, timezone
import math
import os
import re
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import Field

from accounts import get_station_scope, require_page, route_unassigned
from firebase import get_db
from history import (
    _bucket_video_urls, _find_incident_report_raw, _has_real_ai, _incident_report_out,
    _remarks_out, _video_urls,
)
from units import _on_duty_responders
from validation import (
    Description, DocIdOrBlank, DocIdPath, DocIdStr, IncidentType, OptionalDocId, PatchModel,
    ReportLevelValue, ReportStatusValue, ShortLabel, StrictModel,
)

try:
    from google.cloud import firestore  # type: ignore
except ImportError:  # pragma: no cover - depends on which SDK the project uses
    from firebase_admin import firestore  # type: ignore

router = APIRouter(prefix="/reports", tags=["Reports"])

# forward-options?debug=true dumps internals (every station's free units, field names).
# Off unless the server is started with API_DEBUG_OUTPUT=1.
ALLOW_DEBUG_OUTPUT = os.getenv("API_DEBUG_OUTPUT", "0").lower() in {"1", "true", "yes", "on"}

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



class ResponderSnapshot(StrictModel):
    id: DocIdOrBlank = ""
    name: ShortLabel = ""
    unitName: ShortLabel = ""


class ForwardRequest(StrictModel):
    # The station the admin was looking at when they pressed Forward. If the report has
    # moved or changed since, the request is rejected (409) instead of forwarding twice.
    fromStationId: OptionalDocId = None
    # The station the admin picked from the list. Omitted = the nearest one that has a free unit.
    toStationId: OptionalDocId = None


# Caps stop one request from fanning out into hundreds of Firestore reads/writes.
MAX_UNITS_PER_REPORT = 20
MAX_RESPONDER_SNAPSHOTS = 50


class ReportUpdate(PatchModel):
    # Explicit nulls count as "not sent" (PatchModel). Before, {"unitIds": null} unassigned every unit.
    status: ReportStatusValue | None = None
    level: ReportLevelValue | None = None
    incidentType: IncidentType | None = None
    description: Description | None = None
    nearestStationId: DocIdStr | None = None
    # full list of assigned units (replaces existing; [] unassigns all)
    unitIds: Annotated[list[DocIdStr], Field(max_length=MAX_UNITS_PER_REPORT)] | None = None
    # Sent by End Dispatch together with status="resolved"; only stored on that first transition.
    resolvedBy: ShortLabel | None = None
    resolvedById: DocIdOrBlank | None = None
    deployedResponders: Annotated[list[ResponderSnapshot], Field(max_length=MAX_RESPONDER_SNAPSHOTS)] | None = None


def _to_dt(firestore_ts):
    if firestore_ts is None:
        return None
    if hasattr(firestore_ts, "ToDatetime"):
        return firestore_ts.ToDatetime(tzinfo=timezone.utc)
    if isinstance(firestore_ts, datetime):
        return firestore_ts if firestore_ts.tzinfo else firestore_ts.replace(tzinfo=timezone.utc)
    return None


def _iso(value: Any) -> str | None:
    moment = _to_dt(value)
    return moment.isoformat() if moment else None


def _location_to_string(location: Any) -> str:
    if isinstance(location, str):
        return location
    if isinstance(location, dict):
        address = location.get("address") or location.get("name") or ""
        if address:
            return address
        lat = location.get("latitude", location.get("lat"))
        lng = location.get("longitude", location.get("lng"))
        if lat is not None and lng is not None:
            return f"{lat}, {lng}"
    if hasattr(location, "latitude"):
        return f"{location.latitude}, {location.longitude}"
    return "Unknown location"


def _normalize_threat(threat: Any) -> str:
    """Threat level only. Missing -> "Unknown" (never a made-up level)."""
    if threat in (None, ""):
        return "Unknown"
    if isinstance(threat, str) and threat.lower().startswith("level"):
        return threat
    if str(threat).strip().isdigit():
        return f"Level {str(threat).strip()}"
    return str(threat)


def _resolve_live_threat(data: dict[str, Any]) -> tuple[str, Any]:
    """(label, threatLevel to send) for the live report page. Decided by `threatLevel` alone:
    "fake" -> Fake | 0 / empty / missing -> Unknown | 1-4 -> "Level n"."""
    raw = data.get("threatLevel")
    if isinstance(raw, float) and raw.is_integer():
        raw = int(raw)
    text = "" if raw is None else str(raw).strip()
    if text.lower() == "fake":
        return "Fake", "fake"
    if not text or re.fullmatch(r"(level\s*)?0+", text, re.IGNORECASE) or text.lower() == "unknown":
        return "Unknown", None
    return _normalize_threat(raw), raw


# The threatLevel stored on a new report is only a default (it was showing "Level 2" at once). Until the
# AI has written its assessment the level is not real, so the payload carries an `aiPending` flag the
# frontend uses to show "Pending" / "Analyzing..." instead of the level.
AI_PENDING_MAX = timedelta(minutes=2)  # if the AI never answers, stop waiting and show the stored level


def _is_ai_pending(data: dict[str, Any], created_dt: datetime | None) -> bool:
    if _has_ended(data):
        return False
    if _has_real_ai(data):
        return False
    if created_dt is not None and datetime.now(timezone.utc) - created_dt > AI_PENDING_MAX:
        return False
    return True


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


def _get_unit_name(db: Any, unit_id: str | None) -> str | None:
    """Resolve an assigned unit id to its display name, for report detail views."""
    if not unit_id:
        return None
    try:
        unit_doc = db.collection("Units").document(unit_id).get()
        if unit_doc.exists:
            return (unit_doc.to_dict() or {}).get("unitName")
    except Exception:
        pass
    return None


def _extract_unit_ids(data: dict[str, Any]) -> list[str]:
    """All units assigned to a report, de-duplicated, order preserved.
    Reads assignedStation.unitIds, falling back to the older single
    assignedStation.unitId / top-level unitId for docs written before
    multi-unit assignment existed."""
    assigned = data.get("assignedStation") or {}
    raw = assigned.get("unitIds")
    if not isinstance(raw, list):
        raw = [assigned.get("unitId") or data.get("unitId")]
    unit_ids: list[str] = []
    for uid in raw:
        if uid and uid not in unit_ids:
            unit_ids.append(uid)
    return unit_ids


def _build_report_payload(doc_id: str, data: dict[str, Any], db: Any, sign_media: bool = False) -> dict[str, Any]:
    created_dt = _to_dt(data.get("timestamp") or data.get("createdAt"))
    location = data.get("location") if isinstance(data.get("location"), dict) else {}
    location_str = _location_to_string(data.get("location"))

    assigned_station = data.get("assignedStation") or {}
    # Units live inside assignedStation.unitIds (the source of truth).
    unit_ids = _extract_unit_ids(data)
    units = [{"id": uid, "unitName": _get_unit_name(db, uid)} for uid in unit_ids]
    incident_type = (
        data.get("incidentType")
        or data.get("channelName")
        or "Unknown"
    )
    description = (
        data.get("description")
        or data.get("ai_summary")
        or data.get("ai_assessment")
        or f"{incident_type} reported at {location_str}."
    )

    # The level stored on a report is only a default until the AI has assessed it, so it is not shown as-is:
    # Fake / Unknown / real level, decided by threatLevel alone (see _resolve_live_threat).
    threat_label, threat_level = _resolve_live_threat(data)

    # The responder's IncidentReport (a map on the report document). Only on the detail view: it looks up unit names.
    incident_report = _incident_report_out(_find_incident_report_raw(None, data), db) if sign_media else None

    return {
        "id": doc_id,
        "type": incident_type,
        "location": location_str,
        "time": _format_time(created_dt),
        "threat": threat_label,
        "reporter": _get_reporter_name(db, data.get("reporterUuid") or data.get("reportedBy")),
        "description": description,
        "lat": data.get("lat") or location.get("latitude") or location.get("lat"),
        "lng": data.get("lng") or location.get("longitude") or location.get("lng"),
        "date": _history_label(created_dt),
        "status": data.get("status", "pending"),
        "aiSummary": data.get("ai_summary"),
        "aiReasoning": data.get("ai_reasoning"),
        # AI confidence, passed through exactly as stored so the frontend (LiveAlertAI.getConfidence)
        # can read it. Do NOT use `or` fallbacks here: a confidence of 0 is a real value.
        "aiEmergencyConfidence": data.get("aiEmergencyConfidence"),
        "aiMode": data.get("aiMode"),
        "aiAssessment": data.get("ai_assessment"),
        "nearestStationId": data.get("nearestStationId") or assigned_station.get("stationId", ""),
        # Alias of nearestStationId under the name the frontend's unit-assignment
        # dropdown (report.js) actually reads, so a report can be filtered down
        # to the units that belong to its own station.
        "stationId": data.get("nearestStationId") or assigned_station.get("stationId", ""),
        "unitIds": unit_ids,
        "units": units,
        "assignedStation": {
            **{k: v for k, v in assigned_station.items() if k != "unitId"},
            "unitIds": unit_ids,
        },
        "streamChannelId": data.get("streamChannelId") or data.get("channelName", ""),
        # Set once, server-side, the moment a PATCH moves status into
        # ENDED_STATUSES (see update_report below) — not client-writable.
        "endedAt": _to_dt(data.get("endedAt")).isoformat() if _to_dt(data.get("endedAt")) else None,
        "resolvedAt": _to_dt(data.get("endedAt")).isoformat() if _to_dt(data.get("endedAt")) else None,
        "threatLevel": threat_level,
        # Creation time as an ISO string WITH timezone (the frontend's Pending stage counts from it),
        # and whether the AI is still working on this report (the stored threat level is only a default until then).
        "createdAt": created_dt.isoformat() if created_dt else None,
        "aiPending": _is_ai_pending(data, created_dt),
        "resolvedBy": data.get("resolvedBy") or "",
        "resolvedById": data.get("resolvedById") or "",
        # Remarks: False until the admin writes them (or if they chose "later").
        "haveRemarks": bool(data.get("haveRemarks")),
        "remarks": _remarks_out(data),
        "deployedUnits": data.get("deployedUnits") or [],
        # Set when another station passed this report on (from the last forwardHistory entry).
        "forwardedFrom": _forward_out(data),
        # Which admin has this report ("" = nobody yet) and whether it is waiting in the station queue.
        # queuePosition (1 = next up) is filled in by /reports/recent, where the whole queue is known.
        "assignedAdminId": assigned_admin_id(data),
        "assignedAdminName": assigned_station.get("assignedAdminName") or "",
        "queueState": data.get("queueState") or "",
        "queuedAt": _iso(data.get("queuedAt")),
        "queuePosition": None,
        # Set when no station could take this report (every station is out of units).
        "capacityAlert": {"at": _iso((data.get("capacityAlert") or {}).get("at")),
                          "reason": (data.get("capacityAlert") or {}).get("reason", "")}
                         if isinstance(data.get("capacityAlert"), dict) else None,
        "deployedResponders": data.get("deployedResponders") or [],
        "incidentReport": incident_report,
        "videoUrls": _video_urls(data, sign=sign_media),
        "evidenceUploadedAt": data.get("evidence_uploaded_at") and _to_dt(data.get("evidence_uploaded_at")) and _to_dt(data.get("evidence_uploaded_at")).isoformat(),
    }


# ---------------------------------------------------------------------------
# Report queue (see dispatch_queue.py)
# ---------------------------------------------------------------------------
# Queue order: highest threat level first (4 down to 1), then the report that has waited longest.
# A report whose AI assessment has not arrived yet, or whose level is unknown, ranks as this level
# (it is the same default a new report is stored with). "Fake" ranks below level 1.
UNKNOWN_THREAT_PRIORITY = 2

# Which admin has the report lives INSIDE assignedStation, next to the station it belongs to, so it is
# replaced together with the station when the report is forwarded: assignedStation.{assignedAdminId,
# assignedAdminName, assignedAt}. The waiting-list state stays on the document itself.
ADMIN_KEYS = ("assignedAdminId", "assignedAdminName", "assignedAt")
QUEUE_FIELDS = ("queueState", "queuedAt", "capacityAlert")


def assigned_admin_id(data: dict[str, Any]) -> str:
    return str((data.get("assignedStation") or {}).get("assignedAdminId") or "")


def admin_clear_fields() -> dict[str, Any]:
    """Update paths that remove the admin from assignedStation without touching its other keys."""
    return {f"assignedStation.{key}": firestore.DELETE_FIELD for key in ADMIN_KEYS}


def queue_priority(data: dict[str, Any]) -> int:
    """4 (most urgent) .. 1, 0 = fake. Unknown / still being assessed = UNKNOWN_THREAT_PRIORITY."""
    if _is_ai_pending(data, _created_datetime(data)):
        return UNKNOWN_THREAT_PRIORITY
    label, _ = _resolve_live_threat(data)
    if label == "Fake":
        return 0
    match = re.search(r"\d+", label)
    level = int(match.group()) if match else None
    return level if level in (1, 2, 3, 4) else UNKNOWN_THREAT_PRIORITY


def queue_sort_key(doc_id: str, data: dict[str, Any]) -> tuple:
    """Smaller sorts first: higher threat, then older report, then id (so the order is stable)."""
    created = _created_datetime(data) or datetime.max.replace(tzinfo=timezone.utc)
    return (-queue_priority(data), created, doc_id)


def _queue_reset_fields() -> dict[str, Any]:
    """Wipes the admin assignment / queue state, e.g. when a report moves to another station."""
    return {**admin_clear_fields(), **{name: firestore.DELETE_FIELD for name in QUEUE_FIELDS}}


# ---------------------------------------------------------------------------
# Forward a report to the next nearest station
# ---------------------------------------------------------------------------
# The Forward option exists only while the station that received the report has AT MOST this many
# free units left (a free unit = active, has a responder on duty, and is not already on another
# live emergency). 1 = "last unit" or "none left". Set it to 0 to forward only when nothing is left.
FORWARD_WHEN_AT_MOST_UNITS = 1

# A station can RECEIVE a forwarded report only when it has MORE free units than the limit above, so it would not
# immediately be a "last unit" station itself. Otherwise the report could bounce between stations in a loop.
# With the limit at 1 this means 2 or more free units.
FORWARD_MIN_FREE_UNITS_TO_RECEIVE = FORWARD_WHEN_AT_MOST_UNITS + 1

_FORWARD_REASONS = {
    "ended": "This report has already ended.",
    "no_station": "This report has no station yet.",
    "unit_assigned": "A unit is already assigned to this report.",
    "not_last_unit": "This station still has enough free units, so it can't be forwarded.",
    "no_location": "Can't find the next station: no location for the report or station.",
    "no_station_available": "No other station has enough free units right now.",
}


_COORD_TEXT = re.compile(r"^\s*(-?\d{1,3}(?:\.\d+)?)\s*[,;\s]\s*(-?\d{1,3}(?:\.\d+)?)\s*$")
_REPORT_POINT_KEYS = ("location", "coordinates", "geo", "position", "geoPoint", "gps", "reporterLocation",
                      "userLocation", "currentLocation", "lastKnownLocation", "incidentLocation", "locationRaw")


def _first(obj: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if obj.get(key) is not None:
            return obj[key]
    return None


def _coords(obj: Any) -> tuple[float, float] | None:
    """Lat/lng out of whatever the app stored: a GeoPoint, "14.49, 121.05" text, {coordinates: [lng, lat]},
    or a dict using lat/latitude/_latitude and lng/lon/long/longitude/_longitude."""
    if obj is None:
        return None
    if isinstance(obj, str):
        m = _COORD_TEXT.match(obj)
        if not m:
            return None
        lat, lng = m.group(1), m.group(2)
    elif hasattr(obj, "latitude") and hasattr(obj, "longitude"):
        lat, lng = obj.latitude, obj.longitude
    elif isinstance(obj, dict):
        pair = obj.get("coordinates")
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            lng, lat = pair[0], pair[1]
        else:
            lat = _first(obj, ("lat", "latitude", "_latitude"))
            lng = _first(obj, ("lng", "lon", "long", "longitude", "_longitude"))
    else:
        return None
    try:
        lat, lng = float(lat), float(lng)
    except (TypeError, ValueError):
        return None
    if (lat == 0 and lng == 0) or abs(lat) > 90 or abs(lng) > 180:
        return None
    return lat, lng


def _report_origin(data: dict[str, Any]) -> tuple[float, float] | None:
    """Where the incident is: top-level lat/lng first, then any of the places a report may carry a position."""
    found = _coords(data)
    if found:
        return found
    for key in _REPORT_POINT_KEYS:
        found = _coords(data.get(key))
        if found:
            return found
    return None


# The Stations collection stores the point as "Location" (capital L); the summary endpoint in stream.py reads it
# the same way. Reading only "location" left every station without a position, so every distance was unknown.
_STATION_POINT_KEYS = ("Location", "location", "stationLocation", "coordinates", "geo", "position", "geoPoint")


def _station_point(st: dict[str, Any]) -> Any:
    """The raw location value a station carries, whichever spelling it uses."""
    for key in _STATION_POINT_KEYS:
        if st.get(key) is not None and _coords(st.get(key)):
            return st[key]
    return None


def _station_spot(st: dict[str, Any]) -> tuple[float, float] | None:
    return _coords(_station_point(st)) or _coords(st)


def _distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    rad = math.radians
    d_lat, d_lng = rad(b[0] - a[0]), rad(b[1] - a[1])
    h = math.sin(d_lat / 2) ** 2 + math.cos(rad(a[0])) * math.cos(rad(b[0])) * math.sin(d_lng / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def _busy_unit_ids(db: Any, exclude_report_id: str) -> set[str]:
    """Units already committed to some OTHER live emergency."""
    busy: set[str] = set()
    for doc in db.collection("ActiveCalls").stream():
        if doc.id == exclude_report_id:
            continue
        data = doc.to_dict() or {}
        if _is_archived(data) or _has_ended(data):   # finished reports no longer tie up their units
            continue
        busy.update(_extract_unit_ids(data))
    return busy


def _free_units_by_station(db: Any, busy: set[str]) -> dict[str, list[str]]:
    """{stationId: [unit ids that can take a report right now]}.
    Same availability rule as the Units page / assign modal (active + a responder on duty),
    minus units that are already out on another emergency."""
    on_duty = {r["unitId"] for r in _on_duty_responders(db) if r.get("unitId")}
    free: dict[str, list[str]] = {}
    for unit_doc in db.collection("Units").stream():
        unit = unit_doc.to_dict() or {}
        station = unit.get("stationId")
        if not station or unit.get("status", "on") != "on":
            continue
        if unit_doc.id not in on_duty or unit_doc.id in busy:
            continue
        free.setdefault(station, []).append(unit_doc.id)
    return free


def _forward_options(db: Any, report_id: str, data: dict[str, Any], debug: bool = False) -> dict[str, Any]:
    current = _report_station_id(data)
    out: dict[str, Any] = {
        "canForward": False, "remainingUnits": 0, "currentStationId": current,
        "nextStation": None, "reason": None, "minFreeUnits": FORWARD_MIN_FREE_UNITS_TO_RECEIVE,
    }

    def no(reason: str) -> dict[str, Any]:
        out["reason"] = reason
        return out

    if _has_ended(data):
        return no("ended")
    if not current:
        return no("no_station")
    if _extract_unit_ids(data):          # someone is already on the way: too late to forward
        return no("unit_assigned")

    busy = _busy_unit_ids(db, report_id)
    free = _free_units_by_station(db, busy)
    if debug:
        out["debug"] = {"busyUnitIds": sorted(busy), "freeUnitsByStationId": free}
    out["remainingUnits"] = len(free.get(current, []))
    if out["remainingUnits"] > FORWARD_WHEN_AT_MOST_UNITS:
        return no("not_last_unit")

    # Never bounce a report back to a station it already came from / was forwarded to.
    tried = {current}
    for hop in data.get("forwardHistory") or []:
        if isinstance(hop, dict):
            tried.update(s for s in (hop.get("fromStationId"), hop.get("toStationId")) if s)

    origin = _report_origin(data)
    stations = {s.id: (s.to_dict() or {}) for s in db.collection("Stations").stream()}
    if origin is None:   # no incident coordinates: measure from the current station instead
        origin = _station_spot(stations.get(current, {}))

    # Every other station: ones that can take the report first (nearest at the top), then greyed-out ones.
    # A station with disabled: true is closed and is not listed at all. An enabled station without enough free
    # units (FORWARD_MIN_FREE_UNITS_TO_RECEIVE) is listed greyed out; one whose distance can't be worked out
    # (no coordinates) goes at the bottom.
    rows: list[dict[str, Any]] = []
    for sid, st in stations.items():
        if sid in tried:
            continue
        if st.get("disabled") is True:      # closed station (Stations.disabled): never shown; update_report refuses it too
            continue
        spot = _station_spot(st)
        dist = round(_distance_km(origin, spot), 1) if origin and spot else None
        n_free = len(free.get(sid, []))
        rows.append({"id": sid, "name": st.get("stationName") or sid, "distanceKm": dist,
                     "availableUnits": n_free, "available": n_free >= FORWARD_MIN_FREE_UNITS_TO_RECEIVE})
    # Stations that can take the report come first, nearest at the very top; greyed-out ones (no free unit) go below.
    rows.sort(key=lambda r: (not r["available"], r["distanceKm"] is None, r["distanceKm"] or 0, r["name"]))
    out["stations"] = rows

    if debug:
        out.setdefault("debug", {}).update({
            "origin": origin, "stationCount": len(stations), "alreadyTried": sorted(tried),
            "stationFields": {sid: sorted(st.keys()) for sid, st in stations.items()},
        })

    nearest = next((r for r in rows if r["available"]), None)
    if nearest is None:
        return no("no_station_available")
    out["canForward"] = True
    out["nextStation"] = nearest
    return out


def _forward_out(data: dict[str, Any]) -> dict[str, Any] | None:
    """Which station relayed this report to us: the last entry of forwardHistory (the only thing stored)."""
    hops = [h for h in (data.get("forwardHistory") or []) if isinstance(h, dict)]
    if not hops:
        return None
    hop = hops[-1]
    at = _to_dt(hop.get("at"))
    return {"stationId": hop.get("fromStationId", ""), "stationName": hop.get("fromStationName", ""),
            "at": at.isoformat() if at else None}


@router.get("/recent")
def get_recent_reports(station_id: str | None = Depends(get_station_scope)):
    """Frontend dashboard page: recent incidents list. Returns the reports that
    are still live, most recent first. Anything that has ended or is a day old
    (see _is_archived) is left out here and shows up in /history."""
    db = get_db()
    docs = [
        doc for doc in route_unassigned(db, list(db.collection("ActiveCalls").stream()))
        if not _is_archived(doc.to_dict() or {})
        and _matches_station(doc.to_dict() or {}, station_id)
    ]

    # Sort by the actual timestamp rather than the formatted "time" string
    # (the old version sorted strings like "5m ago" vs "2h ago" vs "Jul 19",
    # which doesn't sort chronologically). Docs with no timestamp sort last.
    def _sort_key(doc):
        data = doc.to_dict() or {}
        created_dt = _to_dt(data.get("timestamp") or data.get("createdAt"))
        return created_dt or datetime.min.replace(tzinfo=timezone.utc)

    docs.sort(key=_sort_key, reverse=True)

    reports = [_build_report_payload(doc.id, doc.to_dict() or {}, db) for doc in docs]

    # 1 = next report to be handed to a free admin.
    waiting = sorted((d for d in docs if (d.to_dict() or {}).get("queueState") == "queued"),
                     key=lambda d: queue_sort_key(d.id, d.to_dict() or {}))
    position = {d.id: n for n, d in enumerate(waiting, start=1)}
    for report in reports:
        report["queuePosition"] = position.get(report["id"])
    return {"reports": reports}


@router.get("/{report_id}/forward-options")
def get_forward_options(report_id: DocIdPath, debug: bool = False, station_id: str | None = Depends(get_station_scope)):
    """Should this report show a Forward button, and to which station? Computed server-side so the
    page doesn't need to see other stations' units."""
    db = get_db()
    doc = db.collection("ActiveCalls").document(report_id).get()
    if not doc.exists or not _matches_station(doc.to_dict() or {}, station_id):
        raise HTTPException(status_code=404, detail="Report not found")
    return _forward_options(db, report_id, doc.to_dict() or {}, debug=debug and ALLOW_DEBUG_OUTPUT)


@router.post("/{report_id}/forward")
def forward_report(
    report_id: DocIdPath,
    body: ForwardRequest = ForwardRequest(),
    station_id: str | None = Depends(get_station_scope),
    user: dict = Depends(require_page("report")),
):
    """Hand the report to the next nearest station that has enough free units. That station then owns it."""
    db = get_db()
    ref = db.collection("ActiveCalls").document(report_id)
    doc = ref.get()
    existing = doc.to_dict() or {}
    if not doc.exists or not _matches_station(existing, station_id):
        raise HTTPException(status_code=404, detail="Report not found")

    current = _report_station_id(existing)
    if body.fromStationId and body.fromStationId.strip() != current:
        raise HTTPException(status_code=409, detail="This report was already moved to another station.")

    # Re-check everything now; the button may be stale (a unit came back on duty, was assigned, ...).
    options = _forward_options(db, report_id, existing)
    if not options["canForward"]:
        raise HTTPException(status_code=409, detail=_FORWARD_REASONS.get(options["reason"], "This report can't be forwarded right now."))

    chosen = options["nextStation"]
    if body.toStationId:
        chosen = next((r for r in options.get("stations", []) if r["id"] == body.toStationId), None)
        if chosen is None:
            raise HTTPException(status_code=400, detail="That station can't receive this report.")
        if not chosen["available"]:
            raise HTTPException(status_code=409, detail=f"That station needs at least {FORWARD_MIN_FREE_UNITS_TO_RECEIVE} free units to receive this report.")
    target_id = chosen["id"]
    target_doc = db.collection("Stations").document(target_id).get()
    target = target_doc.to_dict() or {}
    now = datetime.now(timezone.utc)
    old_assigned = existing.get("assignedStation") or {}
    from_name = old_assigned.get("stationName") or existing.get("stationName") or current

    # Rebuild assignedStation for the new station (read-merge-write, so unrelated keys such as
    # dispatchStatus survive) and start with no units: the new station picks its own.
    assigned = dict(old_assigned)
    assigned.update({"stationId": target_id, "stationName": target.get("stationName") or target_id, "unitIds": []})
    assigned.pop("unitId", None)
    for key in ADMIN_KEYS:   # the old station's admin does not follow the report
        assigned.pop(key, None)
    if target.get("commanderName"):
        assigned["commanderName"] = target["commanderName"]
    else:
        assigned.pop("commanderName", None)
        assigned.pop("commandName", None)
    target_point = _station_point(target)
    if target_point is not None:
        assigned["location"] = target_point

    hop = {
        "fromStationId": current, "fromStationName": from_name,
        "toStationId": target_id, "toStationName": assigned["stationName"],
        "at": now, "by": str(user.get("id") or user.get("uid") or user.get("email") or ""),
    }
    ref.update({
        "nearestStationId": target_id,
        "assignedStation": assigned,
        "forwardHistory": firestore.ArrayUnion([hop]),
        # The new station's queue starts from scratch: not waiting yet. (The admin was dropped
        # from assignedStation above.)
        **{name: firestore.DELETE_FIELD for name in QUEUE_FIELDS},
    })
    return {"forwarded": True, "reportId": report_id, "toStation": chosen}


@router.get("/{report_id}")
def get_report_detail(report_id: DocIdPath, station_id: str | None = Depends(get_station_scope)):
    """Frontend report detail page."""
    db = get_db()
    doc = db.collection("ActiveCalls").document(report_id).get()
    if doc.exists:
        doc = route_unassigned(db, [doc])[0]
    if not doc.exists or not _matches_station(doc.to_dict() or {}, station_id):
        raise HTTPException(status_code=404, detail="Report not found")
    payload = _build_report_payload(doc.id, doc.to_dict() or {}, db, sign_media=True)
    if not payload["videoUrls"]:  # the evidence upload may have been saved on the legacy Emergencies copy
        legacy = db.collection("Emergencies").document(report_id).get()
        if legacy.exists:
            payload["videoUrls"] = _video_urls(legacy.to_dict() or {}, sign=True)
    if not payload["videoUrls"] and _has_ended(doc.to_dict() or {}):
        payload["videoUrls"] = _bucket_video_urls(report_id)  # <report id>.mp4 in the storage bucket
    return payload


@router.patch("/{report_id}")
def update_report(
    report_id: DocIdPath,
    payload: ReportUpdate,
    background_tasks: BackgroundTasks,
    station_id: str | None = Depends(get_station_scope),
    _: dict = Depends(require_page("report")),
):
    db = get_db()
    ref = db.collection("ActiveCalls").document(report_id)
    doc = ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Report not found")

    data = payload.model_dump(exclude_unset=True)
    existing_data = doc.to_dict() or {}
    if not _matches_station(existing_data, station_id):
        raise HTTPException(status_code=404, detail="Report not found")

    # A report can only be routed to a station that exists and is enabled; otherwise
    # it would vanish from every station's list.
    if data.get("nearestStationId"):
        target = db.collection("Stations").document(data["nearestStationId"]).get()
        if not target.exists or (target.to_dict() or {}).get("disabled") is True:
            raise HTTPException(status_code=400, detail="That station does not exist.")

    # Stamp when a report actually ended, server-side, the first time status
    # crosses into ENDED_STATUSES (e.g. via the "End Dispatch" button) — never
    # trust a client-supplied endedAt. If a previously-ended report is
    # reopened, clear the stale endedAt so it doesn't claim to be ended while
    # status says otherwise.
    if "status" in data:
        new_status_ended = str(data["status"]).strip().lower() in ENDED_STATUSES
        was_ended = _has_ended(existing_data)
        if new_status_ended and not was_ended:
            data["endedAt"] = datetime.now(timezone.utc)
            # Snapshot who handled it. Units come from the report itself (not the client).
            data["haveRemarks"] = bool(existing_data.get("haveRemarks"))
            data["deployedUnits"] = [
                {"id": uid, "name": _get_unit_name(db, uid) or uid}
                for uid in _extract_unit_ids(existing_data)
            ]
            if "deployedResponders" in data:
                data["deployedResponders"] = [r.model_dump() if hasattr(r, "model_dump") else r for r in (data["deployedResponders"] or [])][:50]
            # The admin who answered the emergency: id + name are saved together with the resolved status.
            data["resolvedById"] = (data.get("resolvedById") or "").strip()[:120]
            data["resolvedBy"] = (data.get("resolvedBy") or "").strip()[:120]
            if data["resolvedById"] and not data["resolvedBy"]:
                data["resolvedBy"] = _get_reporter_name(db, data["resolvedById"])  # look the name up from Users
        elif not new_status_ended and was_ended and existing_data.get("endedAt") is not None:
            data["endedAt"] = firestore.DELETE_FIELD

    if not ("status" in data and "endedAt" in data and data["endedAt"] is not firestore.DELETE_FIELD):
        for field in ("resolvedBy", "resolvedById", "deployedResponders"):
            data.pop(field, None)

    # Assigning units — validate every unit exists and belongs to the same
    # station this report is assigned to. Without this, a stale/typo'd unitId
    # would silently attach and the frontend modal (which filters units by
    # stationId) would never show it. Sending an empty list unassigns all units.
    if "unitIds" in data:
        requested = data.pop("unitIds") or []

        unit_ids: list[str] = []
        for uid in requested:
            if uid and uid not in unit_ids:
                unit_ids.append(uid)

        existing_assigned = existing_data.get("assignedStation") or {}
        report_station_id = (
            data.get("nearestStationId")
            or existing_data.get("nearestStationId")
            or existing_assigned.get("stationId")
        )

        # Units newly being added must be available: active AND with at least
        # one responder on duty (same rule as "Can receive reports" on the
        # Units page). Units already on the report are left alone, so saving
        # never fails just because an assigned unit has since gone offline.
        already_assigned = set(_extract_unit_ids(existing_data))
        newly_added = {uid for uid in unit_ids if uid not in already_assigned}
        on_duty_unit_ids = (
            {r["unitId"] for r in _on_duty_responders(db) if r.get("unitId")}
            if newly_added else set()
        )

        for uid in unit_ids:
            unit_doc = db.collection("Units").document(uid).get()
            if not unit_doc.exists:
                raise HTTPException(status_code=404, detail=f"Unit '{uid}' not found")

            unit_data = unit_doc.to_dict() or {}
            if report_station_id and unit_data.get("stationId") != report_station_id:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unit '{uid}' belongs to a different station than this report is assigned to.",
                )

            if uid in newly_added and (
                unit_data.get("status", "on") != "on" or uid not in on_duty_unit_ids
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unit '{unit_data.get('unitName') or uid}' is not available. It must be active with a responder on duty.",
                )

        # Read-merge-write against the live doc (rather than trusting a
        # client-supplied assignedStation object) so we never clobber sibling
        # fields like stationName/commanderName/dispatchStatus.
        # Only the unit keys are written (as update paths), so sibling fields, including the admin
        # the queue assigned, are never overwritten.
        data["assignedStation.unitIds"] = unit_ids
        data["assignedStation.unitId"] = firestore.DELETE_FIELD  # drop the legacy single-unit field

        # Clean up the legacy top-level "unitId" field from older docs.
        if "unitId" in existing_data:
            data["unitId"] = firestore.DELETE_FIELD

    # Manually moving a report to another station hands it to that station's queue.
    new_station = str(data.get("nearestStationId") or "").strip()
    if new_station and new_station != _report_station_id(existing_data):
        data.update(_queue_reset_fields())

    if data:
        ref.update(data)

    # A report that just ended frees its admin: hand the next queued report on without
    # waiting for the next scheduled queue pass.
    if data.get("endedAt") is not None and data["endedAt"] is not firestore.DELETE_FIELD:
        from dispatch_queue import run_dispatch_pass  # imported here: dispatch_queue imports this module
        background_tasks.add_task(run_dispatch_pass)
    return {"updated": True, "reportId": report_id}