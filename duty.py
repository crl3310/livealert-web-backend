from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from accounts import get_current_account
from firebase import get_db

router = APIRouter(prefix="/duty", tags=["Duty"])

RESPONDERS = "Responders"  # the duty field lives on the Responders doc
ON_DUTY = "on_duty"
OFF_DUTY = "off_duty"


class DutyUpdate(BaseModel):
    duty: Literal["on_duty", "off_duty"]


def _responder_ref(account: dict):
    # The caller is the verified account from the Bearer token, so a client
    # can no longer read or change someone else's duty by sending their id.
    return get_db().collection(RESPONDERS).document(account["id"])


def _payload(data: dict) -> dict:
    duty = data.get("duty")
    updated = data.get("dutyUpdatedAt")
    if isinstance(updated, datetime):
        updated = updated if updated.tzinfo else updated.replace(tzinfo=timezone.utc)
        updated = updated.isoformat()
    else:
        updated = None
    # A missing/unknown value counts as off duty, so existing accounts need no migration.
    return {"duty": duty if duty in (ON_DUTY, OFF_DUTY) else OFF_DUTY, "updatedAt": updated}


@router.get("/me")
def get_my_duty(account: dict = Depends(get_current_account)):
    snap = _responder_ref(account).get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="Account not found")
    return _payload(snap.to_dict() or {})


@router.patch("/me")
def set_my_duty(
    body: DutyUpdate,
    account: dict = Depends(get_current_account),
):
    ref = _responder_ref(account)
    snap = ref.get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="Account not found")

    current = (snap.to_dict() or {}).get("duty")
    # Only stamp the time when the state really changes, so "on duty since"
    # stays accurate if the same value is sent twice.
    if current != body.duty:
        ref.update({"duty": body.duty, "dutyUpdatedAt": datetime.now(timezone.utc)})
        snap = ref.get()
    return _payload(snap.to_dict() or {})