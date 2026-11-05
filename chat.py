"""
LiveAlert — Chat Router (website / admin side)
===============================================
Chat between the admin on the dashboard and the reporter's phone during a live call.
Both sides use the same Firestore collection, so the reporter's app (mobile backend) sees admin
messages and the dashboard sees the reporter's messages:

    ActiveCalls/{callId}/Messages/{messageId}
        senderRole   'reporter' | 'admin'   (set by the SERVER from the login, never taken from the request)
        senderUid    uid of the sender
        text         up to 500 characters
        quickReply   key of a one-tap reply from the reporter, or null (always null for admin messages)
        createdAt    server timestamp
        pushedAt     set by the push listener once the reporter's phone was nudged

Endpoints (server.py mounts this router under /api/v1):
    POST /api/v1/chat/send                    {channelName, text}
    GET  /api/v1/chat/{call_id}/messages      ?after=<messageId>  -> only messages after that one (oldest first)
                                              no `after`          -> the latest messages (opening the chat panel)
    GET  /api/v1/chat/{call_id}/stream        live new messages (Server-Sent Events, same style as /stream/live-updates)
    POST /api/v1/chat/{call_id}/seen          the admin opened the chat; clears the "new message" flag

Who may use it: super_admin and responder_admin, and only for calls routed to their own station
(the same rule stream.py uses). Anything else answers 404 so call ids can't be probed.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from firebase_admin import firestore
from pydantic import Field

from accounts import admin_only, get_station_scope
from firebase import get_db
from ratelimit import limiter, acquire_stream, release_stream
from security import get_verified_token
from validation import DocIdPath, DocIdStr, StrictModel

log = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

MAX_TEXT_LEN = 500
DEFAULT_PAGE = 50
MAX_PAGE = 100
SSE_KEEPALIVE_SECONDS = 25

# Same rule as the mobile app: the chat stays open on 'cancelled_fake' on purpose (an admin may have marked a
# real emergency fake by mistake and the reporter must still be able to say so). It is read-only once the
# call is truly over.
CLOSED_STATUSES = {"resolved", "cancelled", "rejected", "fake"}


class ChatSend(StrictModel):
    # `channelName` is the ActiveCalls document id (same name the mobile app uses).
    channelName: DocIdStr
    # Hard cap well above 500 so a huge string is rejected early; the exact limit is checked after cleaning.
    text: str = Field(max_length=2000)


def _clean_text(value: str) -> str:
    """Trimmed text without control characters (newlines and tabs are kept)."""
    return "".join(ch for ch in value if ch >= " " or ch in "\n\t").strip()


def _iso(value):
    return value.isoformat() if hasattr(value, "isoformat") else None


def _status(call: dict) -> str:
    return str(call.get("status") or "").lower()


def _load_call(call_id: str, station_id: str):
    """The call's ref and data, or 404 when it doesn't exist or isn't routed to the caller's station."""
    ref = get_db().collection("ActiveCalls").document(call_id)
    snap = ref.get()
    data = snap.to_dict() if snap.exists else None
    if not data:
        raise HTTPException(status_code=404, detail="Emergency call record not found.")
    assigned = data.get("assignedStation") or {}
    owner = str(data.get("nearestStationId") or assigned.get("stationId") or "").strip()
    if owner != station_id:   # also true for NO_STATION_SCOPE
        raise HTTPException(status_code=404, detail="Emergency call record not found.")
    return ref, data


def _serialize(doc) -> dict:
    m = doc.to_dict() or {}
    return {
        "id": doc.id,
        "senderRole": m.get("senderRole"),
        "senderUid": m.get("senderUid"),
        "text": m.get("text"),
        "quickReply": m.get("quickReply"),
        "createdAt": _iso(m.get("createdAt")),
    }


# ---------------------------------------------------------------------------
# Admin sends a message
# ---------------------------------------------------------------------------
@router.post("/send", status_code=201)
@limiter.limit("30/minute")
def chat_send(
    request: Request,
    payload: ChatSend,
    account: dict = Depends(admin_only),
    station_id: str = Depends(get_station_scope),
):
    text = _clean_text(payload.text)
    if not text:
        raise HTTPException(status_code=400, detail="The message can't be empty.")
    if len(text) > MAX_TEXT_LEN:
        raise HTTPException(status_code=400, detail=f"The message is too long (max {MAX_TEXT_LEN} characters).")

    call_ref, call = _load_call(payload.channelName, station_id)
    if _status(call) in CLOSED_STATUSES:
        raise HTTPException(status_code=409, detail="This call is closed. The chat is read-only now.")

    try:
        db = get_db()
        msg_ref = call_ref.collection("Messages").document()
        batch = db.batch()
        batch.set(msg_ref, {
            "senderRole": "admin",        # decided here, from the login: a client can't pose as the reporter
            "senderUid": account["id"],
            "text": text,
            "quickReply": None,
            "createdAt": firestore.SERVER_TIMESTAMP,
        })
        # Lets the call list show "new message" (the dashboard's /stream/live-updates already carries this).
        batch.update(call_ref, {"lastMessageAt": firestore.SERVER_TIMESTAMP, "lastMessageRole": "admin"})
        batch.commit()
        return {"success": True, "messageId": msg_ref.id}
    except Exception:
        log.exception("Failed to send chat message")
        raise HTTPException(status_code=500, detail="Could not send the message. Please try again.")


# ---------------------------------------------------------------------------
# Catch-up / history
# ---------------------------------------------------------------------------
@router.get("/{call_id}/messages")
def chat_messages(
    call_id: DocIdPath,
    limit: int = Query(DEFAULT_PAGE, ge=1, le=MAX_PAGE),
    after: str | None = Query(None, max_length=128),
    _: dict = Depends(admin_only),
    station_id: str = Depends(get_station_scope),
):
    call_ref, call = _load_call(call_id, station_id)

    try:
        messages = call_ref.collection("Messages")
        cursor = None
        after = (after or "").strip()
        if after and "/" not in after:
            cursor_snap = messages.document(after).get()
            if cursor_snap.exists:
                cursor = cursor_snap

        if cursor is not None:
            docs = list(messages.order_by("createdAt").start_after(cursor).limit(limit).stream())
        else:
            docs = list(messages.order_by("createdAt", direction=firestore.Query.DESCENDING).limit(limit).stream())
            docs.reverse()

        items = [_serialize(d) for d in docs]
        return {
            "success": True,
            "messages": items,
            "count": len(items),
            "lastId": items[-1]["id"] if items else None,   # pass back as ?after= to get what comes next
            "chatOpen": _status(call) not in CLOSED_STATUSES,
        }
    except Exception:
        log.exception("Failed to load chat messages")
        raise HTTPException(status_code=500, detail="Could not load the messages.")


# ---------------------------------------------------------------------------
# Live messages (Server-Sent Events)
#
# Works like /stream/live-updates: the browser opens it with fetch() + the Authorization header and reads the
# stream. On connect it first sends the latest messages, then every new one. The same message can arrive twice
# if the page also called /messages, so the front end should de-duplicate by `id`.
# ---------------------------------------------------------------------------
@router.get("/{call_id}/stream")
async def chat_stream(
    call_id: DocIdPath,
    _: dict = Depends(admin_only),
    station_id: str = Depends(get_station_scope),
    token: dict = Depends(get_verified_token),
):
    call_ref, _call = await asyncio.get_running_loop().run_in_executor(None, _load_call, call_id, station_id)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def snapshot_callback(col_snapshot, changes, read_time):
        # Firestore calls this from its own thread, not the event loop.
        for change in changes:
            if change.type.name != "ADDED":
                continue
            loop.call_soon_threadsafe(queue.put_nowait, _serialize(change.document))

    query = call_ref.collection("Messages").order_by("createdAt", direction=firestore.Query.DESCENDING).limit(MAX_PAGE)
    slot = acquire_stream(token["uid"], "chat-stream")
    watch = query.on_snapshot(snapshot_callback)

    async def event_generator():
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_SECONDS)
                    yield f"data: {json.dumps(item, default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"   # stops proxies from closing an idle connection
        except asyncio.CancelledError:
            pass
        finally:
            watch.unsubscribe()
            release_stream(token["uid"], "chat-stream", slot)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Admin opened the chat
# ---------------------------------------------------------------------------
@router.post("/{call_id}/seen")
def chat_seen(
    call_id: DocIdPath,
    _: dict = Depends(admin_only),
    station_id: str = Depends(get_station_scope),
):
    call_ref, call = _load_call(call_id, station_id)
    try:
        # Only matters when the last message came from the reporter.
        if call.get("lastMessageRole") == "reporter":
            call_ref.update({"lastMessageSeenAt": firestore.SERVER_TIMESTAMP})
        return {"success": True}
    except Exception:
        log.exception("Failed to mark chat as seen")
        raise HTTPException(status_code=500, detail="Could not update the chat.")