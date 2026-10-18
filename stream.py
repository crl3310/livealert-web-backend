"""
LiveAlert — Stream Router
==========================
Handles live stream channels and report summaries.
Uses Agora Web SDK for live broadcast and viewing.

Endpoints:
    POST /api/v1/stream/create/{emergency_id}   — create a stream channel
    GET  /api/v1/stream/token/{channel_id}      — get dynamic token for channel
    GET  /api/v1/stream/join/{channel_id}       — broadcaster page
    GET  /api/v1/stream/view/{channel_id}       — viewer page
    GET  /api/v1/stream/{emergency_id}/summary   — full report summary
    GET  /api/v1/stream/live-updates            — real-time SSE database stream
"""

import uuid
import time
import asyncio
import json
from datetime import datetime, timezone
from math import radians, sin, cos, sqrt, atan2

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from agora_token_builder import RtcTokenBuilder

from accounts import get_station_scope, require_page, route_unassigned
from security import get_verified_token
from config import AGORA_APP_ID, AGORA_APP_CERTIFICATE
from firebase import get_db

router = APIRouter(prefix="/stream", tags=["Stream"])


def _to_dt(firestore_ts):
    if firestore_ts is None:
        return None
    if hasattr(firestore_ts, "ToDatetime"):
        return firestore_ts.ToDatetime(tzinfo=timezone.utc)
    if isinstance(firestore_ts, datetime):
        return firestore_ts if firestore_ts.tzinfo else firestore_ts.replace(tzinfo=timezone.utc)
    return None


def _haversine(lat1, lng1, lat2, lng2) -> float:
    R = 6371
    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _get_unit_name(db, unit_id: str | None) -> str | None:
    """Resolve an assigned unit id to its display name, mirroring report.py."""
    if not unit_id:
        return None
    try:
        unit_doc = db.collection("Units").document(unit_id).get()
        if unit_doc.exists:
            return (unit_doc.to_dict() or {}).get("unitName")
    except Exception:
        pass
    return None


def _normalize_threat(threat) -> str:
    """Mirrors report.py's _normalize_threat(). Threat level only: a missing
    value is "Unknown", never a made-up level, and the same formatted value
    is returned under the "threat" key everywhere so a report's displayed
    level does not change after its detail page re-fetches it."""
    if threat in (None, ""):
        return "Unknown"
    if isinstance(threat, str) and threat.lower().startswith("level"):
        return threat
    if str(threat).strip().isdigit():
        return f"Level {str(threat).strip()}"
    return str(threat)


def _generate_agora_token(channel_id: str, uid: int = 0, expiration_seconds: int = 3600) -> str:
    """
    Generate a dynamic token for Agora channel access, using Agora's real
    AccessToken2 spec via the official agora-token-builder library.

    NOTE: This used to be a hand-rolled HMAC-SHA256 token that did NOT match
    Agora's actual token format (which is a "007"-prefixed AccessToken2
    structure with CRC32-hashed channel/uid privileges, not a raw
    version+signature+app_id+channel+timestamp+uid struct). Agora's gateway
    silently rejected that invalid token, which is what caused the
    "CAN_NOT_GET_GATEWAY_SERVER: dynamic use static key" error — Agora
    couldn't parse a valid dynamic token out of it and treated the join as
    tokenless against a certificate-enabled (token-required) project.

    Args:
        channel_id: Agora channel name
        uid: User ID (0 for any user)
        expiration_seconds: Token expiration time in seconds (default 1 hour)

    Returns:
        A valid Agora RTC token string.
    """
    privilege_expire_ts = int(time.time()) + expiration_seconds
    # Role 1 = PUBLISHER. Both broadcaster and viewer pages currently join
    # in publish-capable mode; if you want viewers to be subscribe-only,
    # pass role=2 for the /view page specifically.
    return RtcTokenBuilder.buildTokenWithUid(
        AGORA_APP_ID,
        AGORA_APP_CERTIFICATE,
        channel_id,
        uid,
        1,
        privilege_expire_ts,
    )


# ── Real-Time SSE Updates ─────────────────────────────────────────────────────

@router.get("/live-updates")
async def live_updates(station_id: str = Depends(get_station_scope)):
    """
    Server-Sent Events (SSE) endpoint, scoped to the caller's station.
    Only calls routed to the caller's station are pushed; everything else is
    dropped server-side, so other stations' data never reaches the browser.
    """
    db = get_db()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def snapshot_callback(col_snapshot, changes, read_time):
        # Firestore calls this from its own thread, not the event loop.
        for change in changes:
            if change.type.name not in ("ADDED", "MODIFIED"):
                continue

            doc = change.document
            try:
                # A brand-new call has no station yet. Route it now (same as
                # the list/summary endpoints do) so it reaches the right station
                # immediately instead of being dropped until something else routes it.
                doc = route_unassigned(db, [doc])[0]
            except Exception:
                pass

            data = doc.to_dict() or {}
            assigned = data.get("assignedStation") or {}
            owner = str(data.get("nearestStationId") or assigned.get("stationId") or "").strip()
            if owner != station_id:
                continue  # not this station's call (also true for NO_STATION_SCOPE)

            data["id"] = doc.id
            loop.call_soon_threadsafe(queue.put_nowait, data)

    query_watch = db.collection("ActiveCalls").on_snapshot(snapshot_callback)

    async def event_generator():
        try:
            while True:
                item = await queue.get()
                yield f"data: {json.dumps(item, default=str)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            query_watch.unsubscribe()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Create Stream ─────────────────────────────────────────────────────────────

@router.post("/create/{emergency_id}")
def create_stream(
    emergency_id: str,
    station_id: str = Depends(get_station_scope),
    _: dict = Depends(require_page("dashboard")),
):
    """Creates a unique stream channel for an active call."""
    db = get_db()
    ref = db.collection("ActiveCalls").document(emergency_id)
    doc = ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Active call not found")

    data = doc.to_dict()
    assigned = data.get("assignedStation") or {}
    if str(data.get("nearestStationId") or assigned.get("stationId") or "").strip() != station_id:
        raise HTTPException(status_code=404, detail="Active call not found")
    channel_id = data.get("streamChannelId") or f"livealert-{uuid.uuid4().hex[:10]}"
    ref.update({"streamChannelId": channel_id})

    return {
        "emergencyId": emergency_id,
        "channelId": channel_id,
        "agoraAppId": AGORA_APP_ID,
        "joinUrl": f"/api/v1/stream/join/{channel_id}",
        "viewUrl": f"/api/v1/stream/view/{channel_id}",
    }


# ── Get Token ─────────────────────────────────────────────────────────────────

@router.get("/token/{channel_id}")
def get_stream_token(channel_id: str, _: dict = Depends(get_verified_token)):
    """Get a dynamic token for joining an Agora channel."""
    try:
        token = _generate_agora_token(channel_id)
        return {
            "token": token,
            "appId": AGORA_APP_ID,
            "channelId": channel_id,
            "uid": 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token generation failed: {str(e)}")


# ── Broadcaster Page ──────────────────────────────────────────────────────────

@router.get("/join/{channel_id}", response_class=HTMLResponse)
def join_stream(channel_id: str):
    """Opens a live stream page for the broadcaster."""
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>LiveAlert — Agora Broadcast</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #111; color: #fff; font-family: sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; padding: 24px; }}
    h2 {{ color: #e53e3e; margin-bottom: 8px; }}
    p {{ color: #aaa; font-size: 13px; margin-bottom: 16px; }}
    video {{ width: 720px; max-width: 100%; border-radius: 8px; background: #000; min-height: 360px; }}
    .badge {{ background: #e53e3e; color: #fff; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; margin-bottom: 12px; display: inline-block; }}
    button {{ margin-top: 12px; padding: 10px 24px; background: #e53e3e; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; }}
    button:hover {{ background: #c53030; }}
    #status {{ margin-top: 8px; font-size: 12px; color: #aaa; }}
  </style>
</head>
<body>
  <span class="badge">⬤ LIVE</span>
  <h2>LiveAlert — Broadcasting</h2>
  <p>Channel: <strong>{channel_id}</strong></p>
  <video id="local-video" autoplay playsinline muted></video>
  <button onclick="startBroadcast()">Start Broadcasting</button>
  <div id="status">Waiting to start...</div>
  <script src="https://download.agora.io/sdk/release/AgoraRTC_N-4.23.2.js"></script>
  <script>
    const channelName = "{channel_id}";
    const appId = "{AGORA_APP_ID}";
    let client;
    let localTracks = {{ audioTrack: null, videoTrack: null }};

    function status(message) {{
      document.getElementById("status").innerText = message;
    }}

    async function startBroadcast() {{
      if (!window.AgoraRTC) {{
        status("Agora SDK failed to load.");
        return;
      }}

      status("Accessing camera and microphone...");
      
      try {{
        // The login token arrives in the URL fragment (#token=...), which the
        // browser never sends to the server. Read it, then wipe it from the address bar.
        const authToken = new URLSearchParams(location.hash.slice(1)).get("token");
        history.replaceState(null, "", location.pathname + location.search);
        const tokenRes = await fetch(`/api/v1/stream/token/${{channelName}}`, {{
          headers: authToken ? {{ Authorization: `Bearer ${{authToken}}` }} : {{}},
        }});
        if (!tokenRes.ok) throw new Error("Failed to get token");
        const tokenData = await tokenRes.json();
        const token = tokenData.token;

        client = AgoraRTC.createClient({{ mode: "rtc", codec: "vp8" }});

        client.on("user-published", async (user, mediaType) => {{
          await client.subscribe(user, mediaType);
          if (mediaType === "video") {{
            user.videoTrack.play("remote-video");
          }}
        }});

        const uid = Math.floor(Math.random() * 100000);
        await client.join(appId, channelName, token, uid);
        [localTracks.audioTrack, localTracks.videoTrack] = await AgoraRTC.createMicrophoneAndCameraTracks();
        await client.publish([localTracks.audioTrack, localTracks.videoTrack]);
        localTracks.videoTrack.play("local-video");
        status("Broadcasting live with Agora.");
      }} catch (err) {{
        status("Error: " + err.message);
      }}
    }}
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


# ── Viewer Page ───────────────────────────────────────────────────────────────

@router.get("/view/{channel_id}", response_class=HTMLResponse)
def view_stream(channel_id: str):
    """View-only live stream page."""
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>LiveAlert — Agora Viewing</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #111; color: #fff; font-family: sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; padding: 24px; }}
    h2 {{ color: #e53e3e; margin-bottom: 8px; }}
    p {{ color: #aaa; font-size: 13px; margin-bottom: 16px; }}
    video {{ width: 720px; max-width: 100%; border-radius: 8px; background: #000; min-height: 360px; }}
    .badge {{ background: #e53e3e; color: #fff; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; margin-bottom: 12px; display: inline-block; }}
    #status {{ margin-top: 8px; font-size: 12px; color: #aaa; }}
  </style>
</head>
<body>
  <span class="badge">⬤ LIVE</span>
  <h2>LiveAlert — Viewing</h2>
  <p>Channel: <strong>{channel_id}</strong></p>
  <video id="remote-video" autoplay playsinline></video>
  <div id="status">Connecting to stream...</div>
  <script src="https://download.agora.io/sdk/release/AgoraRTC_N-4.23.2.js"></script>
  <script>
    const channelName = "{channel_id}";
    const appId = "{AGORA_APP_ID}";
    let client;

    function status(message) {{
      document.getElementById("status").innerText = message;
    }}

    async function watchStream() {{
      if (!window.AgoraRTC) {{
        status("Agora SDK failed to load.");
        return;
      }}

      try {{
        // The login token arrives in the URL fragment (#token=...), which the
        // browser never sends to the server. Read it, then wipe it from the address bar.
        const authToken = new URLSearchParams(location.hash.slice(1)).get("token");
        history.replaceState(null, "", location.pathname + location.search);
        const tokenRes = await fetch(`/api/v1/stream/token/${{channelName}}`, {{
          headers: authToken ? {{ Authorization: `Bearer ${{authToken}}` }} : {{}},
        }});
        if (!tokenRes.ok) throw new Error("Failed to get token");
        const tokenData = await tokenRes.json();
        const token = tokenData.token;

        client = AgoraRTC.createClient({{ mode: "rtc", codec: "vp8" }});
        client.on("user-published", async (user, mediaType) => {{
          await client.subscribe(user, mediaType);
          if (mediaType === "video") {{
            user.videoTrack.play("remote-video");
          }}
          status("Watching live with Agora.");
        }});

        const uid = Math.floor(Math.random() * 100000);
        await client.join(appId, channelName, token, uid);
        status("Connected. Waiting for broadcaster...");
      }} catch (err) {{
        status("Stream not available yet: " + err.message);
      }}
    }}

    watchStream();
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


# ── Report Summary ────────────────────────────────────────────────────────────

@router.get("/{emergency_id}/summary")
def get_report_summary(emergency_id: str, station_id: str | None = Depends(get_station_scope)):
    """
    Full Report Summary for 'View Report' page:
      - Description + severity level (as stored on the call)
      - Reported by (user name)
      - Location
      - Nearest stations with distances
      - Stream URLs
    """
    db = get_db()

    doc = db.collection("ActiveCalls").document(emergency_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Active call not found")

    doc = route_unassigned(db, [doc])[0]
    data = doc.to_dict()

    if station_id:
        assigned = data.get("assignedStation") or {}
        if str(data.get("nearestStationId") or assigned.get("stationId") or "").strip() != station_id:
            raise HTTPException(status_code=404, detail="Active call not found")

    channel_id = data.get("channelName") or data.get("streamChannelId") or emergency_id

    if data.get("streamChannelId") != channel_id:
        db.collection("ActiveCalls").document(emergency_id).update({"streamChannelId": channel_id})

    assigned_station = data.get("assignedStation") or {}
    incident_type = data.get("incidentType") or data.get("channelName") or "Unknown"
    location = data.get("location", "")

    location_str = ""
    if isinstance(location, str):
        location_str = location
    elif isinstance(location, dict):
        location_str = location.get("address", "")
    elif hasattr(location, "latitude"):
        location_str = f"{location.latitude}, {location.longitude}"

    reporter_id = data.get("reporterUuid") or data.get("reportedBy", "")
    reporter_name = "Unknown"
    if reporter_id:
        user_doc = db.collection("Users").document(reporter_id).get()
        if user_doc.exists:
            u = user_doc.to_dict()
            first = u.get("firstName", "")
            last = u.get("lastName", "")
            reporter_name = f"{first} {last}".strip() or u.get("name", "Unknown")
        else:
            reporter_name = reporter_id

    e_lat, e_lng = None, None
    if hasattr(location, "latitude"):
        e_lat, e_lng = location.latitude, location.longitude
    elif isinstance(location, dict):
        e_lat = location.get("latitude", location.get("lat"))
        e_lng = location.get("longitude", location.get("lng"))

    station_docs = list(db.collection("Stations").stream())
    stations = []
    current_nearest = data.get("nearestStationId") or assigned_station.get("stationId", "")

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

        distance_km = None
        if e_lat and e_lng and s_lat and s_lng:
            distance_km = round(_haversine(e_lat, e_lng, s_lat, s_lng), 1)

        stations.append({
            "id": s.id,
            "stationName": sd.get("stationName", "Unknown"),
            "chiefName": sd.get("commanderName") or sd.get("chiefName", ""),
            "lat": s_lat,
            "lng": s_lng,
            "distanceKm": distance_km,
            "isNearest": s.id == current_nearest,
        })

    stations.sort(key=lambda x: x["distanceKm"] if x["distanceKm"] is not None else 9999)

    # Same default report.py uses when a call has no threat level yet.
    normalized_threat = _normalize_threat(data.get("threatLevel"))

    analyzed_at = _to_dt(data.get("analyzed_at") or data.get("analyzedAt"))
    unit_ids = data.get("unitIds")
    if not isinstance(unit_ids, list):
        unit_ids = assigned_station.get("unitIds")
    if not isinstance(unit_ids, list):
        unit_ids = [assigned_station.get("unitId") or data.get("unitId")]
    unit_ids = list(dict.fromkeys(unit_id for unit_id in unit_ids if unit_id))
    unit_id = assigned_station.get("unitId") or data.get("unitId") or (unit_ids[0] if unit_ids else None)
    assigned_station = {
        **{key: value for key, value in assigned_station.items() if key != "unitIds"},
        "unitId": unit_id,
    }

    return {
        "emergencyId": emergency_id,
        "incidentType": incident_type,
        "channelName": data.get("channelName"),
        "status": data.get("status", "pending"),
        "threat": normalized_threat,
        "threatLevel": data.get("threatLevel"),
        "description": data.get("description") or data.get("ai_assessment") or f"{incident_type} reported at {location_str or 'an unknown location'}.",
        "aiAssessment": data.get("ai_assessment"),
        "aiReasoning": data.get("ai_reasoning"),
        "aiSummary": data.get("ai_summary"),
        "analyzedAt": analyzed_at.isoformat() if analyzed_at else None,
        "reportedBy": {
            "id": reporter_id,
            "name": reporter_name,
        },
        "location": location_str,
        "locationRaw": data.get("location") if isinstance(data.get("location"), dict) else None,
        # Coordinates were computed above (e_lat/e_lng) but never sent, so the map had nothing to pin.
        "lat": e_lat,
        "lng": e_lng,
        "nearestStations": stations,
        "respondersInvolved": data.get("respondersInvolved", []),
        "assignedStation": assigned_station,
        "stationId": current_nearest,
        "unitId": unit_id,
        "unitName": _get_unit_name(db, unit_id),
        "streamChannelId": channel_id,
        "viewUrl": f"/api/v1/stream/view/{channel_id}" if channel_id else None,
        "joinUrl": f"/api/v1/stream/join/{channel_id}" if channel_id else None,
        "agoraAppId": AGORA_APP_ID,
        "token": _generate_agora_token(channel_id) if channel_id else None,
        "createdAt": _to_dt(data.get("timestamp") or data.get("createdAt")),
    }             


# ── Responder locations ───────────────────────────────────────────────────────
# Each responding unit's phone writes its position to ActiveCalls/{id}.responderLocations
# (a map keyed by unit id). The report map polls this lightweight endpoint to draw
# the responders and their routes.

_NAME_CACHE: dict[str, tuple[float, str | None]] = {}
_NAME_CACHE_TTL_SECONDS = 60


def _cached_name(db, collection: str, doc_id: str | None, build) -> str | None:
    """Name lookups are cached for a minute: the map polls every few seconds."""
    if not doc_id:
        return None
    key = f"{collection}/{doc_id}"
    now = time.time()
    hit = _NAME_CACHE.get(key)
    if hit and now - hit[0] < _NAME_CACHE_TTL_SECONDS:
        return hit[1]
    name = None
    try:
        snap = db.collection(collection).document(str(doc_id)).get()
        if snap.exists:
            name = build(snap.to_dict() or {})
    except Exception:
        pass
    _NAME_CACHE[key] = (now, name)
    return name


def _person_name(data: dict) -> str | None:
    full = f"{data.get('firstName', '')} {data.get('lastName', '')}".strip()
    return data.get("name") or full or None


def _point_from(value):
    if isinstance(value, dict):
        return value.get("latitude", value.get("lat")), value.get("longitude", value.get("lng"))
    if hasattr(value, "latitude"):
        return value.latitude, value.longitude
    return None, None


@router.get("/{emergency_id}/responders")
def get_responder_locations(emergency_id: str, station_id: str | None = Depends(get_station_scope)):
    """Live positions of the units responding to this emergency."""
    db = get_db()
    snap = db.collection("ActiveCalls").document(emergency_id).get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="Active call not found")
    data = snap.to_dict() or {}

    if station_id:
        assigned = data.get("assignedStation") or {}
        if str(data.get("nearestStationId") or assigned.get("stationId") or "").strip() != station_id:
            raise HTTPException(status_code=404, detail="Active call not found")

    raw = data.get("responderLocations")
    responders = []
    if isinstance(raw, dict):
        for unit_id, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            lat, lng = _point_from(entry)
            if lat is None or lng is None:
                lat, lng = _point_from(entry.get("coordinates"))
            if lat is None or lng is None:
                continue
            updated = _to_dt(entry.get("updatedAt"))
            uid = entry.get("updatedByUid")
            responders.append({
                "id": unit_id,
                "unitName": _cached_name(db, "Units", unit_id, lambda d: d.get("unitName")),
                "responderName": _cached_name(db, "Responders", uid, _person_name),
                "lat": lat,
                "lng": lng,
                "heading": entry.get("heading"),
                "speed": entry.get("speed"),
                "updatedAt": updated.isoformat() if updated else None,
            })

    return {"responders": responders, "count": len(responders)}