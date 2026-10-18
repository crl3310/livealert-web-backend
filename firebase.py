"""
LiveAlert — Firebase Admin
===========================
Connects to Firestore using the service account key.
Imports project config from config.py.
"""

import threading
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore
from config import PROJECT_ID

# Firebase Storage bucket (used for hazard resolution photos/videos).
# Use the name without the "gs://" prefix.
STORAGE_BUCKET = "livealert-d12cc.firebasestorage.app"

_db = None
# get_db() is called from FastAPI sync endpoints, which FastAPI runs in a
# threadpool — so concurrent requests (e.g. the dashboard's polling firing
# off several endpoints close together) can call this from multiple threads
# at once. Without a lock, two threads can both pass "if _db is None" and
# "if not firebase_admin._apps" before either finishes calling
# initialize_app(), and the second call then raises
# "The default Firebase app already exists." This lock makes the
# check-then-initialize sequence atomic so that can't happen.
_db_lock = threading.Lock()


def get_db() -> firestore.Client:
    global _db
    if _db is not None:
        return _db

    with _db_lock:
        # Re-check inside the lock: another thread may have already
        # finished initializing while we were waiting for the lock.
        if _db is not None:
            return _db

        if not firebase_admin._apps:
            candidates = [
                Path(__file__).resolve().parent / "serviceAccountKey.json",
                Path(__file__).resolve().parent / "serviceAccountKey.json.json",
            ]
            credential_path = next((p for p in candidates if p.exists()), None)
            if credential_path is None:
                raise FileNotFoundError("Firebase service account key not found")
            cred = credentials.Certificate(str(credential_path))
            try:
                firebase_admin.initialize_app(
                    cred,
                    {"projectId": PROJECT_ID, "storageBucket": STORAGE_BUCKET},
                )
            except ValueError:
                # Defensive fallback: something else initialized the default
                # app between our "if not firebase_admin._apps" check and
                # this call (e.g. a different import path racing us despite
                # the lock, such as module re-import under a different
                # module name). Since an app already exists, just proceed to
                # use it rather than crashing the request.
                pass

        _db = firestore.client()

    return _db