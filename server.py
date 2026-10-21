"""
LiveAlert — Server
===================
Run this to start everything:
    python server.py

API Docs: http://localhost:8000/docs
"""

import os
import socket
import subprocess
import sys
import time
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dashboard import router as dashboard_router
from stream import router as stream_router
from accounts import router as accounts_router
from auth import router as auth_router
from history import router as history_router
from report import router as report_router
from stations import router as stations_router
from units import router as units_router
from hazard import router as hazard_router
from maps_config import router as maps_config_router


IS_PROD = os.getenv("ENVIRONMENT") == "production"

app = FastAPI(
    title="LiveAlert API",
    version="1.0.0",
    docs_url=None if IS_PROD else "/docs",
    redoc_url=None if IS_PROD else "/redoc",
    openapi_url=None if IS_PROD else "/openapi.json",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Incoming request: {request.method} {request.url.path}")
    try:
        response = await call_next(request)
    except Exception as exc:
        print(f"Request error: {request.method} {request.url.path} -> {exc}")
        raise
    print(f"Response status: {request.method} {request.url.path} -> {response.status_code}")
    return response

#ALLOWED_ORIGINS = ["https://livealert.tech","https://www.livealert.tech",]

# Uncomment while developing locally:
ALLOWED_ORIGINS = ["http://localhost:5500", "http://127.0.0.1:5500"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
app.include_router(accounts_router, prefix="/api/v1")
app.include_router(history_router, prefix="/api/v1")
app.include_router(report_router, prefix="/api/v1")
app.include_router(stream_router,    prefix="/api/v1")
app.include_router(stations_router,  prefix="/api/v1")
app.include_router(units_router,     prefix="/api/v1")
app.include_router(hazard_router, prefix="/api/v1")  # Hazards router
app.include_router(maps_config_router, prefix="/api/v1")


@app.get("/")
def root():
    return {"status": "ok", "service": "LiveAlert API v1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "LiveAlert API"}

@app.get("/api/v1/health")
def frontend_health():
    return {"status": "healthy", "service": "LiveAlert API"}


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def _find_pid_on_port(port: int) -> int | None:
    """Best-effort lookup of the PID currently bound to `port`."""
    try:
        if sys.platform.startswith("win"):
            out = subprocess.check_output(
                f'netstat -ano | findstr :{port}', shell=True, text=True, stderr=subprocess.DEVNULL
            )
            for line in out.strip().splitlines():
                parts = line.split()
                if parts and parts[-1].isdigit():
                    return int(parts[-1])
        else:
            out = subprocess.check_output(
                f'lsof -t -i :{port}', shell=True, text=True, stderr=subprocess.DEVNULL
            )
            pids = [int(p) for p in out.strip().splitlines() if p.strip().isdigit()]
            if pids:
                return pids[0]
    except Exception:
        return None
    return None


def _kill_pid(pid: int) -> bool:
    try:
        if sys.platform.startswith("win"):
            subprocess.run(f"taskkill /PID {pid} /F", shell=True, check=True, capture_output=True)
        else:
            os.kill(pid, 9)  # SIGKILL
        return True
    except Exception:
        return False


def _ensure_port_free(port: int) -> int:
    """
    Makes sure `port` is actually free before starting, instead of either
    silently rebinding to a random port (the old behavior — which the
    frontend's config.js has no way to discover, since it only ever
    health-checks a fixed set of candidate URLs) or requiring you to
    manually find and kill a stale process every time.

    Most commonly, a busy port 8000 just means a previous, still-running
    instance of this exact server was never stopped. So: detect what's
    holding the port, and kill it automatically. Only if that fails (e.g.
    it's some unrelated process, or we don't have permission to kill it)
    does this fall back to a loud, clear error instead of silently doing
    something the frontend won't be able to find.
    """
    if _port_is_free(port):
        return port

    pid = _find_pid_on_port(port)
    if pid and pid != os.getpid():
        print(f"Port {port} is already in use by PID {pid} — stopping it automatically...")
        if _kill_pid(pid):
            time.sleep(1)  # give the OS a moment to release the socket
            if _port_is_free(port):
                print(f"Freed port {port}. Continuing startup...\n")
                return port

    print("\n" + "=" * 60)
    print(f"ERROR: Port {port} is already in use and could not be freed automatically.")
    print("Find and stop whatever's using it, then re-run this script:")
    print(f"  Windows:      netstat -ano | findstr :{port}")
    print(f"                taskkill /PID <pid_from_above> /F")
    print(f"  macOS/Linux:  lsof -i :{port}")
    print(f"                kill -9 <pid_from_above>")
    print()
    print("(Deliberately not falling back to a random port here — the")
    print("frontend's config.js only checks a fixed set of candidate URLs")
    print("and has no way to discover one, which is what caused the")
    print("'Cannot GET /api/v1/...' errors landing on the wrong origin.)")
    print("=" * 60 + "\n")
    raise SystemExit(1)


def start_server() -> None:
    host = os.getenv("LIVEALERT_HOST", "0.0.0.0")
    requested_port = int(os.getenv("LIVEALERT_PORT", "8000"))
    port = _ensure_port_free(requested_port)
    reload_enabled = os.getenv("LIVEALERT_RELOAD", "0").lower() in {"1", "true", "yes", "on"}

    print("\n" + "=" * 60)
    print("LiveAlert API is starting...")
    print(f"Host: {host}:{port}")
    print(f"Docs: http://localhost:{port}/docs")
    print(f"Health: http://localhost:{port}/health")
    print(f"API Base: http://localhost:{port}/api/v1")
    print("=" * 60 + "\n")

    uvicorn.run("server:app", host=host, port=port, reload=reload_enabled, log_level="info")


if __name__ == "__main__":
    start_server()