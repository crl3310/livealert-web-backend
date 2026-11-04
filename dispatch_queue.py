"""
LiveAlert — Report queue
========================
Hands every live report to a station admin, queues it when all admins are busy, and
forwards it to another station when its own station has no free units.

Forwarding to another station is always done by an admin (the Forward button in report.py).
This module never moves a report between stations. For each live report that has a station
but no admin yet, a pass does this:

  1. ASSIGN    A free admin exists -> the report goes to the admin who has waited longest since
               their last report (round robin). Free = on duty with a live session (see
               accounts.is_admin_available) and not holding another open report.
  2. QUEUE     Every admin is busy -> the report waits (queueState = "queued"). Waiting reports
               are served highest threat level first, then oldest first.

Separately, the pass raises a CAPACITY ALERT: when a station is low on units (Forward would be
offered) but no other station can take the report either, the report gets
`capacityAlert = {at, reason}`. The alert is removed again as soon as forwarding is possible or
the report gets units / moves / ends.

An admin is busy from the moment a report is assigned until that report ends. When it ends,
the next queued report goes to the first admin who becomes free. If an admin logs out (or
their session goes stale) while holding a report, the report returns to the queue.

A pass runs every DISPATCH_QUEUE_INTERVAL seconds from server.py, and right after a report
ends (report.update_report). Fields kept on the ActiveCalls document:
  assignedStation.assignedAdminId / assignedAdminName / assignedAt   (inside assignedStation, so
      they are replaced together with the station when a report is forwarded)
  queueState ("assigned" | "queued"), queuedAt,
  capacityAlert {at, reason: "no_station_available" | "no_location"}
and on the admin's Responders document: lastReportAssignedAt.

Settings (environment):
  DISPATCH_QUEUE_ENABLED=0     turn the whole thing off
  DISPATCH_QUEUE_INTERVAL=10   seconds between passes
"""

import asyncio
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from accounts import _is_admin_role, has_station, is_admin_available, route_unassigned
from firebase import get_db
from report import (
    assigned_admin_id, admin_clear_fields,
    _busy_unit_ids, _extract_unit_ids, _forward_options, _free_units_by_station, _has_ended,
    _is_archived, _report_station_id, _to_dt, firestore, queue_sort_key,
    FORWARD_WHEN_AT_MOST_UNITS,
)

ENABLED = os.getenv("DISPATCH_QUEUE_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
INTERVAL_SECONDS = float(os.getenv("DISPATCH_QUEUE_INTERVAL", "10"))

# How often the "can this report be forwarded anywhere?" check is repeated for one report.
FORWARD_CHECK_SECONDS = 30
LEASE_SECONDS = INTERVAL_SECONDS + 20

_EPOCH = datetime.min.replace(tzinfo=timezone.utc)
_PASS_LOCK = threading.Lock()                 # one pass at a time inside this process
_OWNER = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
_last_forward_check: dict[str, float] = {}    # report id -> time.monotonic() of the last check


# ---------------------------------------------------------------------------
# Pure helpers (no database access)
# ---------------------------------------------------------------------------
def _station_key(station_id: Any) -> str:
    return str(station_id or "").strip().upper()


def _admin_name(data: dict[str, Any]) -> str:
    name = f"{data.get('firstName', '')} {data.get('lastName', '')}".strip()
    return name or str(data.get("email") or "").strip() or "Admin"


def order_free_admins(admins: list[dict[str, Any]], busy_ids: set[str]) -> list[dict[str, Any]]:
    """Free admins, the one who has waited longest since their last report first."""
    free = [a for a in admins if a["available"] and a["id"] not in busy_ids]
    return sorted(free, key=lambda a: (a["lastAssigned"] or _EPOCH, a["id"]))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_admins(db: Any) -> dict[str, dict[str, Any]]:
    admins: dict[str, dict[str, Any]] = {}
    for doc in db.collection("Responders").stream():
        data = doc.to_dict() or {}
        if not _is_admin_role(data.get("role") or data.get("position")):
            continue
        admins[doc.id] = {
            "id": doc.id,
            "name": _admin_name(data),
            "station": _station_key(data.get("stationId")),
            "available": is_admin_available(data),
            "lastAssigned": _to_dt(data.get("lastReportAssignedAt")),
        }
    return admins


def _load_reports(db: Any) -> dict[str, dict[str, Any]]:
    """Live reports that have a station (routing the station-less ones first)."""
    live = [d for d in db.collection("ActiveCalls").stream() if not _is_archived(d.to_dict() or {})]

    unrouted = [d for d in live if not has_station(d.to_dict() or {})]
    routed = {d.id: d for d in route_unassigned(db, unrouted)} if unrouted else {}

    reports: dict[str, dict[str, Any]] = {}
    for doc in live:
        data = (routed.get(doc.id) or doc).to_dict() or {}
        station = _report_station_id(data)
        if station:
            reports[doc.id] = {
                "id": doc.id,
                "ref": db.collection("ActiveCalls").document(doc.id),
                "data": data,
                "station": station,
            }
    return reports


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------
def _claim(db: Any, ref: Any, station_key: str, admin: dict[str, Any], now: datetime) -> bool:
    """Gives the report to the admin, unless somebody else got there first (or it ended / moved)."""

    @firestore.transactional
    def txn(tx):
        snap = ref.get(transaction=tx)
        if not snap.exists:
            return False
        current = snap.to_dict() or {}
        if assigned_admin_id(current) or _has_ended(current):
            return False
        if _station_key(_report_station_id(current)) != station_key:
            return False
        tx.update(ref, {
            "assignedStation.assignedAdminId": admin["id"],
            "assignedStation.assignedAdminName": admin["name"],
            "assignedStation.assignedAt": now,
            "queueState": "assigned",
            "queuedAt": current.get("queuedAt") or now,
        })
        return True

    return txn(db.transaction())


def _set_alert(r: dict[str, Any], alert: dict[str, Any] | None, stats: dict[str, int]) -> None:
    if alert is None:
        r["ref"].update({"capacityAlert": firestore.DELETE_FIELD})
        r["data"].pop("capacityAlert", None)
        stats["alerts_cleared"] += 1
    else:
        r["ref"].update({"capacityAlert": alert})
        r["data"]["capacityAlert"] = alert
        stats["alerts"] += 1


def _acquire_lease(db: Any) -> bool:
    """So that several server workers do not run a pass at the same moment."""
    ref = db.collection("DispatchLocks").document("queue")
    now = datetime.now(timezone.utc)

    @firestore.transactional
    def txn(tx):
        held = ref.get(transaction=tx).to_dict() or {}
        until = _to_dt(held.get("until"))
        if until and until > now and held.get("owner") != _OWNER:
            return False
        tx.set(ref, {"owner": _OWNER, "until": now + timedelta(seconds=LEASE_SECONDS)})
        return True

    return txn(db.transaction())


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------
def _run(db: Any) -> dict[str, int]:
    now = datetime.now(timezone.utc)
    stats = {"released": 0, "assigned": 0, "queued": 0, "alerts": 0, "alerts_cleared": 0}

    reports = _load_reports(db)
    admins = _load_admins(db)

    # 0. An admin who is no longer available (logged out, session lost, moved station) gives
    #    their report back to the queue.
    for r in reports.values():
        admin_id = assigned_admin_id(r["data"])
        if not admin_id:
            continue
        admin = admins.get(admin_id)
        if admin is not None and admin["available"] and admin["station"] == _station_key(r["station"]):
            continue
        r["ref"].update({
            **admin_clear_fields(),
            "queueState": "queued",
            "queuedAt": r["data"].get("queuedAt") or now,
        })
        for name in ("assignedAdminId", "assignedAdminName", "assignedAt"):
            (r["data"].get("assignedStation") or {}).pop(name, None)
        r["data"]["queueState"] = "queued"
        stats["released"] += 1

    # Reports that still need an admin. One that already has units is being dispatched
    # (or predates this feature), so it is left alone.
    waiting = [r for r in reports.values()
               if not assigned_admin_id(r["data"]) and not _extract_unit_ids(r["data"])]
    busy_admins = {assigned_admin_id(r["data"]) for r in reports.values() if assigned_admin_id(r["data"])}

    # 1. Capacity alert. Only reports an admin could still forward are looked at: no units assigned,
    #    and the station is low enough on free units that the Forward button would be offered.
    #    If even then no other station can take the report, flag it; otherwise make sure no flag stays.
    free_units = None
    for r in reports.values():
        has_alert = bool(r["data"].get("capacityAlert"))
        if _extract_unit_ids(r["data"]):
            if has_alert:
                _set_alert(r, None, stats)
            continue
        if time.monotonic() - _last_forward_check.get(r["id"], -1e9) < FORWARD_CHECK_SECONDS:
            continue
        if free_units is None:
            free_units = _free_units_by_station(db, _busy_unit_ids(db, ""))
        if len(free_units.get(r["station"], [])) > FORWARD_WHEN_AT_MOST_UNITS:
            if has_alert:
                _set_alert(r, None, stats)       # the station has units again
            continue
        _last_forward_check[r["id"]] = time.monotonic()

        options = _forward_options(db, r["id"], r["data"])
        stuck = not options["canForward"] and options["reason"] in ("no_station_available", "no_location")
        if stuck and not has_alert:
            _set_alert(r, {"at": now, "reason": options["reason"]}, stats)
        elif not stuck and has_alert:
            _set_alert(r, None, stats)

    # 2 + 3. Per station: hand reports to free admins, queue the rest.
    by_station: dict[str, list[dict[str, Any]]] = {}
    for r in waiting:
        by_station.setdefault(_station_key(r["station"]), []).append(r)

    for station_key, station_reports in by_station.items():
        station_admins = [a for a in admins.values() if a["station"] == station_key]
        free = order_free_admins(station_admins, busy_admins)
        for r in sorted(station_reports, key=lambda r: queue_sort_key(r["id"], r["data"])):
            if not free:   # every admin is busy: the report waits its turn
                if r["data"].get("queueState") != "queued":
                    r["ref"].update({"queueState": "queued", "queuedAt": r["data"].get("queuedAt") or now})
                    stats["queued"] += 1
                continue
            admin = free[0]
            # False = somebody else got there first (another worker, or the report ended / moved);
            # the admin stays free for the next report.
            if _claim(db, r["ref"], station_key, admin, now):
                free.pop(0)
                busy_admins.add(admin["id"])
                db.collection("Responders").document(admin["id"]).update({"lastReportAssignedAt": now})
                r["data"].setdefault("assignedStation", {})["assignedAdminId"] = admin["id"]
                stats["assigned"] += 1

    return stats


def run_dispatch_pass() -> dict[str, int] | None:
    """One pass over every live report. Returns what it did, or None if it did not run
    (switched off, another pass is already running, or another worker holds the lease)."""
    if not ENABLED:
        return None
    if not _PASS_LOCK.acquire(blocking=False):
        return None
    try:
        db = get_db()
        if not _acquire_lease(db):
            return None
        stats = _run(db)
        if any(stats.values()):
            print(f"[dispatch_queue] {stats}")
        return stats
    except Exception as exc:  # a failed pass must never take the server down; the next one retries
        print(f"[dispatch_queue] pass failed: {exc.__class__.__name__}: {exc}")
        return None
    finally:
        _PASS_LOCK.release()


async def dispatch_loop() -> None:
    """Started by server.py when the app starts."""
    if ENABLED:
        print(f"[dispatch_queue] running, one pass every {INTERVAL_SECONDS:g}s")
    else:
        print("[dispatch_queue] switched off (DISPATCH_QUEUE_ENABLED=0)")
    while True:
        await asyncio.to_thread(run_dispatch_pass)
        await asyncio.sleep(INTERVAL_SECONDS)