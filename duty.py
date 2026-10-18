from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from firebase import get_db

router = APIRouter(prefix="/duty", tags=["Duty"])

RESPONDERS = "Responders"  # the duty field lives on the Responders doc
ON_DUTY = "on_duty"
OFF_DUTY = "off_duty"


class DutyUpdate(BaseModel):
    duty: Literal["on_duty", "off_duty"]


def _responder_ref(account_id: str | None):
    # Same identity the rest of the site uses: the doc id sent as X-Account-Id.
    # If accounts.py has a get_current_account() dependency, swap it in here.
    if not account_id or not account_id.strip():
        raise HTTPException(status_code=401, detail="Missing X-Account-Id")
    return get_db().collection(RESPONDERS).document(account_id.strip())


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
def get_my_duty(x_account_id: str | None = Header(default=None, alias="X-Account-Id")):
    snap = _responder_ref(x_account_id).get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="Account not found")
    return _payload(snap.to_dict() or {})


@router.patch("/me")
def set_my_duty(
    body: DutyUpdate,
    x_account_id: str | None = Header(default=None, alias="X-Account-Id"),
):
    ref = _responder_ref(x_account_id)
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