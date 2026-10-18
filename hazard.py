"""
LiveAlert — Hazards Router
===========================
Handles fetching, real-time listening, and managing hazard reports from Firestore.
"""

import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from accounts import get_station_scope
from firebase import get_db

router = APIRouter(prefix="/hazards", tags=["Hazards"])

ENDED_HAZARD_STATUSES = frozenset({"resolved", "closed", "completed", "cancelled", "ended"})

# --- Resolution report (required before a hazard can be marked resolved) ---------------------
# Keep these in sync with RESOLVE_RULES and the <select> options in the frontend.
RESOLUTION_OUTCOMES = {
    "resolved_onsite": "Resolved on site",
    "cleared_by_responders": "Cleared by responders",
    "referred": "Referred to another agency",
    "false_report": "False or mistaken report",
    "duplicate": "Duplicate report",
}
MIN_DETAILS_LEN = 10
MIN_NOTES_LEN = 20
MAX_TEXT_LEN = 2000
MAX_EVIDENCE_FILES = 3
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024
ALLOWED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
ALLOWED_VIDEO_TYPES = frozenset({"video/mp4", "video/quicktime", "video/webm"})


def _to_dt(firestore_ts):
    if firestore_ts is None:
        return None
    if hasattr(firestore_ts, "ToDatetime"):
        return firestore_ts.ToDatetime(tzinfo=timezone.utc)
    if isinstance(firestore_ts, datetime):
        return firestore_ts if firestore_ts.tzinfo else firestore_ts.replace(tzinfo=timezone.utc)
    return None


def _normalize_location(loc: Any) -> dict:
    """
    Normalizes location data supporting Firestore GeoPoints, dictionaries
    (latitude/longitude or lat/lng), and missing coordinates.
    """
    if loc is None:
        return {"latitude": None, "longitude": None, "hasCoordinates": False}
    
    if hasattr(loc, "latitude") and hasattr(loc, "longitude"):
        return {
            "latitude": float(loc.latitude),
            "longitude": float(loc.longitude),
            "hasCoordinates": True,
            "address": getattr(loc, "address", "")
        }
    
    if isinstance(loc, dict):
        lat = loc.get("latitude") if loc.get("latitude") is not None else loc.get("lat")
        lng = loc.get("longitude") if loc.get("longitude") is not None else loc.get("lng")
        has_coords = lat is not None and lng is not None
        return {
            "latitude": float(lat) if has_coords else None,
            "longitude": float(lng) if has_coords else None,
            "hasCoordinates": has_coords,
            "address": loc.get("address", loc.get("name", ""))
        }
        
    return {"latitude": None, "longitude": None, "hasCoordinates": False}


_REPORTER_NAME_CACHE: dict[str, str] = {}
_NAME_FIELDS = ("displayName", "fullName", "name", "Name", "username")
_UUID_FIELDS = ("uid", "uuid", "userId", "id")


def _user_display_name(user: dict) -> str:
    for field in _NAME_FIELDS:
        value = str(user.get(field) or "").strip()
        if value:
            return value
    first = str(user.get("firstName") or user.get("first_name") or "").strip()
    last = str(user.get("lastName") or user.get("last_name") or "").strip()
    return f"{first} {last}".strip()


def _reporter_name(db, data: dict) -> str:
    """
    Reporter's display name from the Users collection, looked up by reporterUuid
    (document id first, then a uid/uuid/userId field). A name stored on the hazard
    itself wins if present. Returns "" when nothing is found.
    """
    for key in ("reporterName", "reporter_name"):
        value = str(data.get(key) or "").strip()
        if value:
            return value

    uuid = str(data.get("reporterUuid") or "").strip()
    if not uuid:
        return ""
    if uuid in _REPORTER_NAME_CACHE:
        return _REPORTER_NAME_CACHE[uuid]

    name = ""
    try:
        users = db.collection("Users")
        snap = users.document(uuid).get()
        if snap.exists:
            name = _user_display_name(snap.to_dict() or {})
        if not name:
            for field in _UUID_FIELDS:
                match = next(iter(users.where(field, "==", uuid).limit(1).stream()), None)
                if match:
                    name = _user_display_name(match.to_dict() or {})
                    if name:
                        break
    except Exception:
        name = ""
    if name:  # only cache hits so a newly created profile is picked up later
        _REPORTER_NAME_CACHE[uuid] = name
    return name


def _report_station_id(data: dict) -> str:
    assigned = data.get("assignedStation") or {}
    return str(data.get("nearestStationId") or assigned.get("stationId") or "").strip()


def _matches_station(data: dict, station_id: str | None) -> bool:
    if not station_id or not station_id.strip():
        return True
    return _report_station_id(data) == station_id.strip()


def _is_ended(data: dict) -> bool:
    return str(data.get("status") or "").strip().lower() in ENDED_HAZARD_STATUSES


def _serialize_hazard(db, doc_id: str, data: dict) -> dict:
    """One shape for both GET /hazards and the live stream, so the frontend handles them identically."""
    timestamp_dt = _to_dt(data.get("timestamp"))
    return {
        "id": doc_id,
        "hazardId": data.get("hazardId"),
        "title": data.get("title"),
        "category": data.get("category"),
        "description": data.get("description"),
        "status": data.get("status"),
        "aiAssessment": data.get("ai_assessment"),
        "reporterUuid": data.get("reporterUuid"),
        "reporterName": _reporter_name(db, data),
        "location": _normalize_location(data.get("location")),
        "assignedStation": data.get("assignedStation", {}),
        "imageCount": len([i for i in (data.get("images") or []) if isinstance(i, dict) and i.get("path")]),
        "timestamp": timestamp_dt.isoformat() if timestamp_dt else None,
    }


def _find_hazard_doc(db, hazard_id: str):
    """Looks a hazard up by document id first, then by its own `hazardId` field."""
    ref = db.collection("Hazards").document(hazard_id)
    snap = ref.get()
    if snap.exists:
        return ref, snap
    match = next(iter(db.collection("Hazards").where("hazardId", "==", hazard_id).limit(1).stream()), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Hazard not found: {hazard_id}")
    return match.reference, match


def _required_text(value: str | None, label: str, min_len: int) -> str:
    text = (value or "").strip()
    if len(text) < min_len:
        raise HTTPException(status_code=422, detail=f"{label} must be at least {min_len} characters.")
    if len(text) > MAX_TEXT_LEN:
        raise HTTPException(status_code=422, detail=f"{label} must be {MAX_TEXT_LEN} characters or fewer.")
    return text


def _storage_bucket():
    """Firebase Storage bucket. Needs `storageBucket` in firebase_admin.initialize_app(...) (see firebase.py)."""
    try:
        from firebase_admin import storage
        return storage.bucket()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evidence storage is not configured: {e}")


def _delete_evidence(items: list[dict]) -> None:
    """Best-effort cleanup of files uploaded for a resolve that then failed."""
    if not items:
        return
    try:
        bucket = _storage_bucket()
        for item in items:
            try:
                bucket.blob(item["path"]).delete()
            except Exception:
                pass
    except HTTPException:
        pass


def _store_evidence(hazard_doc_id: str, files: list[UploadFile]) -> list[dict]:
    """
    Validates and uploads the optional photo/video evidence. Every file's type is checked before
    anything is uploaded; if an upload fails midway, the files already uploaded are removed again.
    """
    files = [f for f in (files or []) if f and f.filename]
    if not files:
        return []
    if len(files) > MAX_EVIDENCE_FILES:
        raise HTTPException(status_code=422, detail=f"You can attach up to {MAX_EVIDENCE_FILES} files.")

    checked = []
    for f in files:
        ctype = (f.content_type or "").lower()
        if ctype in ALLOWED_IMAGE_TYPES:
            checked.append((f, "image", ctype, MAX_IMAGE_BYTES))
        elif ctype in ALLOWED_VIDEO_TYPES:
            checked.append((f, "video", ctype, MAX_VIDEO_BYTES))
        else:
            raise HTTPException(
                status_code=415,
                detail=f"{f.filename}: only JPG, PNG, WebP photos and MP4, MOV, WebM videos are allowed.",
            )

    bucket = _storage_bucket()
    stored: list[dict] = []
    try:
        for f, kind, ctype, limit in checked:
            data = f.file.read(limit + 1)
            if len(data) > limit:
                raise HTTPException(
                    status_code=413,
                    detail=f"{f.filename} is too large (max {limit // (1024 * 1024)} MB for a {kind}).",
                )
            ext = re.sub(r"[^a-z0-9]", "", os.path.splitext(f.filename)[1].lower())[:8]
            path = f"hazard-evidence/{hazard_doc_id}/{uuid.uuid4().hex}{'.' + ext if ext else ''}"
            bucket.blob(path).upload_from_string(data, content_type=ctype)
            stored.append({
                "path": path,
                "type": kind,
                "contentType": ctype,
                "name": os.path.basename(f.filename)[:120],
                "size": len(data),
            })
    except Exception:
        _delete_evidence(stored)
        raise
    return stored


@router.get("")
def list_hazards(station_id: str | None = Depends(get_station_scope)):
    """
    Fetches active hazard documents from Firestore, filtered strictly by station scope
    and excluding resolved ones.
    """
    db = get_db()
    try:
        hazards = []
        for doc in db.collection("Hazards").stream():
            data = doc.to_dict() or {}
            if not _matches_station(data, station_id) or _is_ended(data):
                continue
            hazards.append(_serialize_hazard(db, doc.id, data))
        return {"hazards": hazards}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch hazards: {str(e)}")


@router.patch("/{hazard_id}/resolve")
def resolve_hazard(
    hazard_id: str,
    request: Request,
    outcome: str = Form(""),
    hazard_details: str = Form(""),
    resolution_notes: str = Form(""),
    evidence: list[UploadFile] = File(default=[]),
    station_id: str | None = Depends(get_station_scope),
):
    """
    Marks a hazard as resolved. Only hazards inside the caller's station scope can be resolved.

    Expects multipart/form-data:
      outcome            one of RESOLUTION_OUTCOMES (required)
      hazard_details     what was found on site (required)
      resolution_notes   how it was resolved (required)
      evidence           up to 3 photos/videos (optional, repeat the field)

    The resolution report is stored on the hazard under `resolution`. Safe to call twice: if the hazard
    is already resolved nothing is overwritten. After writing, the document is read back: the endpoint
    only reports success when Firestore really holds status == "resolved".
    """
    db = get_db()
    stored: list[dict] = []
    try:
        ref, snap = _find_hazard_doc(db, hazard_id)
        data = snap.to_dict() or {}
        if not _matches_station(data, station_id):
            raise HTTPException(status_code=403, detail="This hazard belongs to another station")

        if str(data.get("status") or "").strip().lower() != "resolved":
            outcome_key = (outcome or "").strip()
            if outcome_key not in RESOLUTION_OUTCOMES:
                raise HTTPException(status_code=422, detail="Choose how this report ended.")
            details = _required_text(hazard_details, "Hazard details", MIN_DETAILS_LEN)
            notes = _required_text(resolution_notes, "Resolution notes", MIN_NOTES_LEN)

            stored = _store_evidence(ref.id, evidence)
            try:
                ref.update({
                    "status": "resolved",
                    "resolvedAt": datetime.now(timezone.utc),
                    "resolution": {
                        "outcome": outcome_key,
                        "outcomeLabel": RESOLUTION_OUTCOMES[outcome_key],
                        "hazardDetails": details,
                        "resolutionNotes": notes,
                        "resolvedBy": (request.headers.get("X-Account-Id") or "").strip(),
                        "evidence": stored,
                    },
                })
            except Exception:
                _delete_evidence(stored)
                raise

        # Verify: read it back instead of trusting that update() worked.
        check = (ref.get().to_dict() or {})
        if str(check.get("status") or "").strip().lower() != "resolved":
            raise HTTPException(status_code=500, detail="Resolve was not saved - the report is still active")

        resolved_at = _to_dt(check.get("resolvedAt"))
        return {
            "id": ref.id,
            "status": "resolved",
            "verified": True,
            "resolvedAt": resolved_at.isoformat() if resolved_at else None,
            "evidenceCount": len((check.get("resolution") or {}).get("evidence") or []),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve hazard: {str(e)}")


@router.get("/{hazard_id}/evidence")
def hazard_evidence(hazard_id: str, station_id: str | None = Depends(get_station_scope)):
    """
    The resolution report of a hazard, with short-lived (30 min) signed links to its photos/videos,
    so a supervisor can validate a resolution. Same station scope rules as resolving.
    """
    db = get_db()
    try:
        ref, snap = _find_hazard_doc(db, hazard_id)
        data = snap.to_dict() or {}
        if not _matches_station(data, station_id):
            raise HTTPException(status_code=403, detail="This hazard belongs to another station")

        resolution = data.get("resolution") or {}
        media = []
        items = [e for e in (resolution.get("evidence") or []) if e.get("path")]
        if items:
            bucket = _storage_bucket()
            for e in items:
                url = bucket.blob(e["path"]).generate_signed_url(
                    version="v4", expiration=timedelta(minutes=30), method="GET"
                )
                media.append({"type": e.get("type"), "name": e.get("name"), "url": url})

        resolved_at = _to_dt(data.get("resolvedAt"))
        return {
            "id": ref.id,
            "status": data.get("status"),
            "resolvedAt": resolved_at.isoformat() if resolved_at else None,
            "outcome": resolution.get("outcome"),
            "outcomeLabel": resolution.get("outcomeLabel"),
            "hazardDetails": resolution.get("hazardDetails"),
            "resolutionNotes": resolution.get("resolutionNotes"),
            "resolvedBy": resolution.get("resolvedBy"),
            "evidence": media,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load resolution report: {str(e)}")


@router.get("/{hazard_id}/images")
def hazard_images(hazard_id: str, station_id: str | None = Depends(get_station_scope)):
    """
    The photos the reporter attached when submitting the hazard, as short-lived (30 min) signed links.
    Links are made on demand (not stored) so they never go stale in the list or the live stream.
    """
    db = get_db()
    try:
        ref, snap = _find_hazard_doc(db, hazard_id)
        data = snap.to_dict() or {}
        if not _matches_station(data, station_id):
            raise HTTPException(status_code=403, detail="This hazard belongs to another station")

        items = [i for i in (data.get("images") or []) if isinstance(i, dict) and i.get("path")]
        images = []
        if items:
            bucket = _storage_bucket()
            for i in items:
                url = bucket.blob(i["path"]).generate_signed_url(
                    version="v4", expiration=timedelta(minutes=30), method="GET"
                )
                images.append({"name": i.get("name"), "url": url})
        return {"id": ref.id, "images": images}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load hazard photos: {str(e)}")


@router.get("/live-updates")
async def hazard_live_updates(request: Request, station_id: str | None = Depends(get_station_scope)):
    """
    SSE endpoint for real-time hazard updates (consumed with fetch streams), filtered by station scope.
    Sends the full hazard for added/modified documents and {"id", "action": "remove"} when a hazard
    is deleted or ends (resolved, closed, ...).
    """
    db = get_db()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def snapshot_callback(col_snapshot, changes, read_time):
        # Runs on a Firestore thread: hand results to the event loop thread-safely.
        for change in changes:
            data = change.document.to_dict() or {}
            doc_id = change.document.id

            if not _matches_station(data, station_id):
                continue

            if change.type.name == "REMOVED" or _is_ended(data):
                payload = {"id": doc_id, "action": "remove"}
            else:
                payload = _serialize_hazard(db, doc_id, data)

            loop.call_soon_threadsafe(queue.put_nowait, payload)

    watch = db.collection("Hazards").on_snapshot(snapshot_callback)

    async def event_stream():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {json.dumps(payload, default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            watch.unsubscribe()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )