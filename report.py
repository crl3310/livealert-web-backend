from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from accounts import get_station_scope, require_page, route_unassigned
from firebase import get_db
from history import _bucket_video_urls, _remarks_out, _video_urls
from units import _on_duty_responders

try:
    from google.cloud import firestore  # type: ignore
except ImportError:  # pragma: no cover - depends on which SDK the project uses
    from firebase_admin import firestore  # type: ignore

router = APIRouter(prefix="/reports", tags=["Reports"])

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



class ResponderSnapshot(BaseModel):
    id: str = ""
    name: str = ""
    unitName: str = ""


class ReportUpdate(BaseModel):
    status: str | None = None
    level: str | None = None
    incidentType: str | None = None
    description: str | None = None
    nearestStationId: str | None = None
    unitIds: list[str] | None = None  # full list of assigned units (replaces existing; [] unassigns all)
    # Sent by End Dispatch together with status="resolved"; only stored on that first transition.
    resolvedBy: str | None = None
    resolvedById: str | None = None
    deployedResponders: list[ResponderSnapshot] | None = None


def _to_dt(firestore_ts):
    if firestore_ts is None:
        return None
    if hasattr(firestore_ts, "ToDatetime"):
        return firestore_ts.ToDatetime(tzinfo=timezone.utc)
    if isinstance(firestore_ts, datetime):
        return firestore_ts if firestore_ts.tzinfo else firestore_ts.replace(tzinfo=timezone.utc)
    return None


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

    return {
        "id": doc_id,
        "type": incident_type,
        "location": location_str,
        "time": _format_time(created_dt),
        "threat": _normalize_threat(data.get("threatLevel")),
        "reporter": _get_reporter_name(db, data.get("reporterUuid") or data.get("reportedBy")),
        "description": description,
        "lat": data.get("lat") or location.get("latitude") or location.get("lat"),
        "lng": data.get("lng") or location.get("longitude") or location.get("lng"),
        "date": _history_label(created_dt),
        "status": data.get("status", "pending"),
        "aiSummary": data.get("ai_summary"),
        "aiReasoning": data.get("ai_reasoning"),
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
        "threatLevel": data.get("threatLevel"),
        "resolvedBy": data.get("resolvedBy") or "",
        "resolvedById": data.get("resolvedById") or "",
        # Remarks: False until the admin writes them (or if they chose "later").
        "haveRemarks": bool(data.get("haveRemarks")),
        "remarks": _remarks_out(data),
        "deployedUnits": data.get("deployedUnits") or [],
        "deployedResponders": data.get("deployedResponders") or [],
        "videoUrls": _video_urls(data, sign=sign_media),
        "evidenceUploadedAt": data.get("evidence_uploaded_at") and _to_dt(data.get("evidence_uploaded_at")) and _to_dt(data.get("evidence_uploaded_at")).isoformat(),
    }


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
    return {"reports": reports}


@router.get("/{report_id}")
def get_report_detail(report_id: str, station_id: str | None = Depends(get_station_scope)):
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
    report_id: str,
    payload: ReportUpdate,
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
        assigned_station = dict(existing_assigned)
        assigned_station["unitIds"] = unit_ids
        assigned_station.pop("unitId", None)  # drop the legacy single-unit field
        data["assignedStation"] = assigned_station

        # Clean up the legacy top-level "unitId" field from older docs.
        if "unitId" in existing_data:
            data["unitId"] = firestore.DELETE_FIELD

    if data:
        ref.update(data)
    return {"updated": True, "reportId": report_id}