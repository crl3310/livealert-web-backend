import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import unquote, urlsplit

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from accounts import get_station_scope, route_unassigned
from firebase import get_db

router = APIRouter(prefix="/history", tags=["History"])
log = logging.getLogger(__name__)

# Storage bucket that holds the recorded evidence videos, named after the report (emergency_<id>.mp4).
# Override with the FIREBASE_STORAGE_BUCKET environment variable if the bucket ever changes.
STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET", "livealert-d12cc.firebasestorage.app")
VIDEO_EXTS = (".mp4", ".mov", ".webm", ".m4v")

def _to_dt(firestore_ts):
    if firestore_ts is None:
        return None
    if hasattr(firestore_ts, "ToDatetime"):
        return firestore_ts.ToDatetime(tzinfo=timezone.utc)
    if isinstance(firestore_ts, datetime):
        return firestore_ts if firestore_ts.tzinfo else firestore_ts.replace(tzinfo=timezone.utc)
    return None

def _iso(value: Any) -> str | None:
    """ISO string for a Firestore timestamp, datetime or an ISO string saved by the frontend."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value
    dt = _to_dt(value)
    return dt.isoformat() if dt else None


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
    """The station a report was routed to (same precedence report.py uses)."""
    assigned = data.get("assignedStation") or {}
    return str(data.get("nearestStationId") or assigned.get("stationId") or "").strip()


def _matches_station(data: dict[str, Any], station_id: str | None) -> bool:
    """True if the report belongs to station_id. No station_id = no filtering."""
    if not station_id or not station_id.strip():
        return True
    return _report_station_id(data) == station_id.strip()



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
    """History shows the threat level only. Missing -> "Unknown" (never a made-up level)."""
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


def _remarks_out(data: dict[str, Any]) -> dict[str, Any] | None:
    remarks = data.get("remarks")
    if not isinstance(remarks, dict):
        return None
    return {
        "additionalDetails": remarks.get("additionalDetails") or "",
        "resolution": remarks.get("resolution") or "",
        "addedBy": remarks.get("addedBy") or "",
        "addedById": remarks.get("addedById") or "",
        "addedAt": _iso(remarks.get("addedAt")),
    }


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic")


def _sign_storage_url(url: str) -> str:
    """Evidence in the storage bucket is normally private, so a bare storage.googleapis.com link won't play.
    Swap it for a short-lived signed URL. Falls back to the original link if signing isn't possible
    (already-signed or token links are returned untouched)."""
    try:
        parts = urlsplit(url)
        if parts.netloc != "storage.googleapis.com" or "X-Goog-Signature" in parts.query:
            return url
        bucket_name, _, blob_name = unquote(parts.path.lstrip("/")).partition("/")
        if not bucket_name or not blob_name:
            return url
        from firebase_admin import storage  # imported here so this file still loads without it
        blob = storage.bucket(bucket_name).blob(blob_name)
        return blob.generate_signed_url(version="v4", expiration=timedelta(hours=2), method="GET")
    except Exception:
        return url


def _bucket_video_urls(report_id: str) -> list[str]:
    """Find a report's recording in the storage bucket by its id (e.g. emergency_b3785350.mp4).
    Used when the report document has no link to the video. Returns signed URLs, or the plain
    public URL if signing isn't available."""
    if not report_id:
        return []
    try:
        from firebase_admin import storage  # imported here so this file still loads without it
        bucket = storage.bucket(STORAGE_BUCKET)
        urls: list[str] = []
        for prefix in (report_id, f"evidence_video/{report_id}", f"evidence/{report_id}"):
            for blob in bucket.list_blobs(prefix=prefix, max_results=10):
                if not blob.name.lower().endswith(VIDEO_EXTS):
                    continue
                try:
                    urls.append(blob.generate_signed_url(version="v4", expiration=timedelta(hours=2), method="GET"))
                except Exception as exc:
                    log.warning("Could not sign %s (%s); using its public URL", blob.name, exc)
                    urls.append(blob.public_url)
            if urls:
                break
        return urls
    except Exception as exc:
        log.warning("Could not look up evidence video for %s in bucket %s: %s", report_id, STORAGE_BUCKET, exc)
        return []


def _video_urls(data: dict[str, Any], sign: bool = False) -> list[str]:
    """Recorded video evidence for the report. Checks every field name a video may have been saved under.
    The report document stores it as evidenceUrl (with evidence_uploaded_at). sign=True turns bucket links into
    signed URLs the browser can play; leave it off for list endpoints (signing can be slow)."""
    urls: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, dict):
            value = value.get("url") or value.get("downloadUrl") or value.get("src")
        if isinstance(value, str) and value:
            value = _sign_storage_url(value) if sign else value
            if value not in urls:
                urls.append(value)

    def is_image(value: Any) -> bool:
        return isinstance(value, str) and value.lower().split("?")[0].endswith(IMAGE_EXTS)

    # Whatever the field is called: any link-valued field whose name mentions evidence / video / recording.
    for key, value in data.items():
        if (isinstance(value, str) and value.startswith(("http://", "https://")) and not is_image(value)
                and any(word in key.lower() for word in ("evidence", "video", "recording"))):
            add(value)
    for key in ("evidenceUrl", "evidence_url", "videoUrl", "recordingUrl", "recording"):
        value = data.get(key)
        if not is_image(value if isinstance(value, str) else (value or {}).get("url") if isinstance(value, dict) else None):
            add(value)
    for key in ("videoUrls", "recordings", "videos"):
        for entry in data.get(key) or []:
            add(entry)
    for entry in data.get("evidence") or []:   # evidence may mix photos and videos: keep videos only
        kind = str(entry.get("type") or entry.get("contentType") or "") if isinstance(entry, dict) else ""
        url = entry.get("url") if isinstance(entry, dict) else entry
        if kind.startswith("video") or (isinstance(url, str) and url.lower().split("?")[0].endswith((".mp4", ".mov", ".webm", ".m4v"))):
            add(entry)
    return urls


def _history_item(doc_id: str, data: dict[str, Any]) -> dict[str, Any]:
    created_dt = _created_datetime(data)
    return {
        "id": doc_id,
        "type": data.get("incidentType") or data.get("channelName") or "Unknown",
        "location": _location_to_string(data.get("location")),
        "time": _format_time(created_dt),
        "threat": _normalize_threat(data.get("threatLevel")),
        "date": _history_label(created_dt),
        "status": data.get("status", "pending"),
        # True when the report actually ended; False when it was archived only
        # because a day passed. The History page's "Resolved" filter uses this.
        "ended": _has_ended(data),
        "createdAt": created_dt.isoformat() if created_dt else None,
        "threatLevel": data.get("threatLevel"),
        "resolvedAt": _iso(data.get("resolvedAt") or data.get("endedAt")),
        "resolvedBy": data.get("resolvedBy") or "",
        "resolvedById": data.get("resolvedById") or "",
        # Remarks: False until an admin writes them (or if they chose "later").
        "haveRemarks": bool(data.get("haveRemarks")),
        "remarks": _remarks_out(data),
    }


def _history_detail(doc_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """The list item plus everything the report page needs to show a finished incident."""
    item = _history_item(doc_id, data)
    item.update({
        "description": data.get("description") or data.get("incidentDescription") or "",
        "deployedUnits": data.get("deployedUnits") or [],
        "deployedResponders": data.get("deployedResponders") or [],
        # The report document's own link first; otherwise look for <report id>.mp4 in the storage bucket.
        "videoUrls": _video_urls(data, sign=True) or (_bucket_video_urls(doc_id) if _has_ended(data) else []),
        "evidenceUploadedAt": _iso(data.get("evidence_uploaded_at")),
    })
    return item


def _find_report(db, report_id: str):
    """(document ref, data) for a report, looking in ActiveCalls first, then the legacy Emergencies collection."""
    for collection in ("ActiveCalls", "Emergencies"):
        ref = db.collection(collection).document(report_id)
        snap = ref.get()
        if snap.exists:
            return ref, snap.to_dict() or {}
    return None, {}


def _merged_report_data(db, report_id: str) -> dict[str, Any]:
    """A report can exist in both ActiveCalls and the legacy Emergencies collection (the evidence upload may have
    landed on either). Merge them, ActiveCalls winning, and ignore empty values so they can't hide real ones."""
    merged: dict[str, Any] = {}
    for collection in ("Emergencies", "ActiveCalls"):
        snap = db.collection(collection).document(report_id).get()
        if snap.exists:
            merged.update({k: v for k, v in (snap.to_dict() or {}).items() if v not in (None, "", [])})
    return merged


@router.get("")
def get_history_reports(limit: int | None = None, station_id: str | None = Depends(get_station_scope)):
    """Frontend history page: every incident that is no longer live, newest first.

    A report lands here once it has ended or a day has passed since it was
    created (see _is_archived). Reports still in the legacy Emergencies
    collection are included too so nothing that used to be in History vanishes;
    an ActiveCalls copy wins if the same id exists in both.
    """
    db = get_db()
    entries: dict[str, tuple[datetime, dict[str, Any]]] = {}
    oldest = datetime.min.replace(tzinfo=timezone.utc)

    for doc in db.collection("Emergencies").stream():
        data = doc.to_dict() or {}
        if not _matches_station(data, station_id):
            continue
        entries[doc.id] = (_created_datetime(data) or oldest, _history_item(doc.id, data))

    for doc in route_unassigned(db, list(db.collection("ActiveCalls").stream())):
        data = doc.to_dict() or {}
        if not _is_archived(data) or not _matches_station(data, station_id):
            continue
        entries[doc.id] = (_created_datetime(data) or oldest, _history_item(doc.id, data))

    ordered = sorted(entries.values(), key=lambda entry: entry[0], reverse=True)
    reports = [item for _, item in ordered]
    if limit:
        reports = reports[:limit]

    return {"reports": reports}


@router.get("/{report_id}")
def get_history_report(report_id: str, station_id: str | None = Depends(get_station_scope)):
    """One finished report with its remarks, deployed units/responders and recorded video."""
    data = _merged_report_data(get_db(), report_id)
    if not data or not _matches_station(data, station_id):
        raise HTTPException(status_code=404, detail="Report not found")
    return _history_detail(report_id, data)


MIN_REMARK_CHARS = 10
MAX_REMARK_CHARS = 2000


class RemarksIn(BaseModel):
    additionalDetails: str
    resolution: str
    adminName: str | None = None
    adminId: str | None = None


@router.patch("/{report_id}/remarks")
def save_report_remarks(report_id: str, body: RemarksIn, station_id: str | None = Depends(get_station_scope)):
    """Saves the admin's remarks on a resolved report and flips haveRemarks to True."""
    details = body.additionalDetails.strip()
    resolution = body.resolution.strip()
    for label, text in (("Additional details", details), ("Resolution", resolution)):
        if len(text) < MIN_REMARK_CHARS:
            raise HTTPException(status_code=422, detail=f"{label} needs at least {MIN_REMARK_CHARS} characters.")
        if len(text) > MAX_REMARK_CHARS:
            raise HTTPException(status_code=422, detail=f"{label} is too long (max {MAX_REMARK_CHARS} characters).")

    ref, data = _find_report(get_db(), report_id)
    if ref is None or not _matches_station(data, station_id):
        raise HTTPException(status_code=404, detail="Report not found")
    if not _has_ended(data):
        raise HTTPException(status_code=409, detail="Remarks can only be added once the report is resolved.")

    ref.update({
        "haveRemarks": True,
        "remarks": {
            "additionalDetails": details,
            "resolution": resolution,
            # The admin who answered the emergency (saved with the resolved status).
            # Only reports resolved before this feature fall back to whoever is typing.
            "addedBy": (data.get("resolvedBy") or body.adminName or "").strip() or "Unknown admin",
            "addedById": (data.get("resolvedById") or body.adminId or "").strip(),
            "addedAt": datetime.now(timezone.utc).isoformat(),
        },
    })
    return _history_item(report_id, ref.get().to_dict() or {})