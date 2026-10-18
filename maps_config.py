import os

from fastapi import APIRouter, HTTPException

# Read GOOGLE_MAPS_API_KEY from a .env file next to this one, if python-dotenv
# is installed (pip install python-dotenv). Real environment variables win.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

router = APIRouter(prefix="/config", tags=["Config"])


@router.get("/maps")
def get_maps_config():
    """Hands the Google Maps browser key to the website at runtime, so it lives
    in the server's environment (GOOGLE_MAPS_API_KEY) and never in the
    front-end source or the git repository.

    Note: a Maps JavaScript key is, by design, visible to the browser once the
    map loads. What protects it is the restrictions set on the key in Google
    Cloud (allowed websites + allowed APIs + a quota cap), not secrecy."""
    key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=503, detail="Google Maps is not configured on the server.")
    return {
        "apiKey": key,
        "mapId": os.getenv("GOOGLE_MAPS_MAP_ID", "").strip() or None,
    }