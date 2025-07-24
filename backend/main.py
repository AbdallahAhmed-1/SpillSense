from __future__ import annotations

import os
import shutil
import tempfile
import uuid
import traceback
import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------
# Artifact store
# ---------------------------------------------------------------------
from state.artifact_store import load_artifacts, register_artifact

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------
try:
    # Prefer local relative import if command_router.py sits next to this file
    from command_router import route_command
except ModuleNotFoundError:
    from backend.command_router import route_command  # type: ignore

from scripts.predict_new_batch import prepare_features_for_prediction
from state.state_utils import load_state
from agents.llm_reasoner import LLMReasoner
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("spillsense.backend")

# ---------------------------------------------------------------------
# Constants / Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_ROOT = BASE_DIR / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

STATE_PATH = Path("state/rca_state_after_modeling.joblib")
STATIC_DIR = Path("static")
REPORT_DIR = STATIC_DIR / "reports"

# ---------------------------------------------------------------------
# LLM Reasoner
# ---------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=GOOGLE_API_KEY)
reasoner = LLMReasoner(llm=llm)

# ---------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------
app = FastAPI(title="SpillSense Backend", version="0.4")

app.mount("/files", StaticFiles(directory=STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    new_reports: List[str] = []

# ---------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------
def get_session_dir(sid: str) -> Path:
    d = UPLOAD_ROOT / sid
    d.mkdir(parents=True, exist_ok=True)
    return d


def clear_session_dir(sid: str) -> None:
    d = get_session_dir(sid)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


SESSION_FILES: Dict[str, Dict[str, datetime]] = {}
SESSION_EXPIRY = timedelta(hours=1)


def register_session_file(session_id: str, filename: str) -> None:
    """Store just the basename."""
    SESSION_FILES.setdefault(session_id, {})[Path(filename).name] = datetime.utcnow()


def get_session_files(session_id: str) -> List[str]:
    now = datetime.utcnow()
    files = SESSION_FILES.get(session_id, {})
    # prune expired
    files = {k: v for k, v in files.items() if now - v < SESSION_EXPIRY}
    SESSION_FILES[session_id] = files
    return list(files.keys())


def latest_csv_in_session(sid: str) -> Optional[Path]:
    csvs = sorted(get_session_dir(sid).glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def save_uploaded_file(file: UploadFile, session_id: str) -> Path:
    sdir = get_session_dir(session_id)
    dst = sdir / file.filename
    with dst.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)
    register_session_file(session_id, dst.name)
    return dst

# ---------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------
@lru_cache(maxsize=1)
def _cached_state():
    log.info("[CACHE] Loading state once…")
    if not STATE_PATH.exists():
        raise FileNotFoundError(f"State file not found: {STATE_PATH}")
    return load_state(str(STATE_PATH))


@lru_cache(maxsize=1)
def _cached_model():
    log.info("[CACHE] Loading best model once…")
    state = _cached_state()
    model_path = state["model_artifacts"]["best_model"]
    blob = joblib.load(model_path)
    return blob.get("model", blob) if isinstance(blob, dict) else blob


def _predict_and_append(csv_path: str, session_dir: Path) -> str:
    """Same logic, but ensure output goes to session_dir and is registered."""
    state = _cached_state()
    model = _cached_model()

    df_in = pd.read_csv(csv_path)
    X_input = prepare_features_for_prediction(df_in, state)
    preds = model.predict(X_input)

    try:
        probs = model.predict_proba(X_input).max(axis=1)
        df_in["prediction_confidence"] = probs
    except Exception:
        pass

    df_in["predicted_severity"] = preds

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"predicted_{ts}.csv"
    out_path = session_dir / out_name
    df_in.to_csv(out_path, index=False)

    # artifact
    register_artifact(
        session_dir,
        f"CSV Predictions - {datetime.now():%Y-%m-%d %H:%M}",
        "prediction",
        out_path,
        "text/csv",
        {"total_records": len(df_in), "predictions_made": len(preds)},
    )

    # LLM insight
    reasoner.add_insight(
        "csv_analysis",
        {
            "file": Path(csv_path).name,
            "total_records": len(df_in),
            "severity_distribution": df_in["predicted_severity"].value_counts().to_dict(),
            "predictions_made": int(len(preds)),
            "timestamp": datetime.now().isoformat(),
        },
    )

    register_session_file(session_dir.name, out_name)
    return out_name

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = Header(None),
    sid: Optional[str] = Query(None),
):
    session_id = sid or session_id or str(uuid.uuid4())
    dst = save_uploaded_file(file, session_id)

    # infer mime/kind
    lname = dst.name.lower()
    if lname.endswith(".csv"):
        mime, kind = "text/csv", "other"
    elif lname.endswith(".pdf"):
        mime, kind = "application/pdf", "report"
    elif lname.endswith((".png", ".jpg", ".jpeg")):
        mime, kind = "image/png", "image_result"
    elif lname.endswith(".json"):
        mime, kind = "application/json", "other"
    else:
        mime, kind = "application/octet-stream", "other"

    register_artifact(get_session_dir(session_id), f"Uploaded: {dst.name}", kind, dst, mime)

    return {"filename": dst.name, "session_id": session_id, "status": "uploaded successfully"}


@app.get("/files")
async def list_files(session_id: Optional[str] = Header(None), sid: Optional[str] = Query(None)):
    sid = sid or session_id
    if not sid:
        return {"files": []}
    files = get_session_files(sid)
    if not files:
        # fallback to disk scan (legacy)
        sdir = get_session_dir(sid)
        files = [p.name for p in sdir.iterdir() if p.is_file() and not p.name.startswith(".")]
        for f in files:
            register_session_file(sid, f)
    return {"files": files}


@app.get("/artifacts")
async def list_artifacts(session_id: Optional[str] = Header(None), sid: Optional[str] = Query(None)):
    sid = sid or session_id
    if not sid:
        raise HTTPException(403, "No session ID provided")

    sdir = get_session_dir(sid)
    artifacts = load_artifacts(sdir)

    # auto-register legacy files if none
    if not artifacts:
        for file_path in sdir.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                lname = file_path.name.lower()
                if lname.endswith(".csv"):
                    mime, kind = "text/csv", "prediction"
                elif lname.endswith(".pdf"):
                    mime, kind = "application/pdf", "report"
                elif lname.endswith((".png", ".jpg", ".jpeg")):
                    mime, kind = "image/png", "figure"
                elif lname.endswith(".json"):
                    mime, kind = "application/json", "other"
                else:
                    mime, kind = "application/octet-stream", "other"
                register_artifact(sdir, file_path.name, kind, file_path, mime)
        artifacts = load_artifacts(sdir)

    return {"artifacts": [asdict(art) for art in artifacts]}


@app.get("/download/{filename:path}")
async def download_file(
    filename: str,
    session_id: Optional[str] = Header(None),
    sid: Optional[str] = Query(None),
):
    sid = sid or session_id
    if not sid:
        raise HTTPException(403, "No session ID provided")

    sdir = get_session_dir(sid)
    # try artifact id first
    arts = load_artifacts(sdir)
    for art in arts:
        if art.id == filename:
            full = sdir / art.file_path
            if full.exists():
                return FileResponse(full, filename=Path(art.file_path).name, headers={"Cache-Control": "no-store"})

    full = sdir / filename
    if not full.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(full, filename=filename, headers={"Cache-Control": "no-store"})


@app.post("/reset_files")
async def reset_files(session_id: Optional[str] = Header(None), sid: Optional[str] = Query(None)):
    sid = sid or session_id
    try:
        if sid:
            clear_session_dir(sid)
            SESSION_FILES.pop(sid, None)
        return {"status": "cleared", "session_id": sid, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/predict_csv")
async def predict_csv(
    file: UploadFile | None = File(None),
    session_id: Optional[str] = Header(None),
    sid: Optional[str] = Query(None),
    filename: Optional[str] = Query(None),
):
    sid = sid or session_id
    if not sid:
        raise HTTPException(403, "No session ID provided")

    sdir = get_session_dir(sid)

    if file is not None:
        dst = save_uploaded_file(file, sid)
        csv_path = dst
    elif filename:
        csv_path = sdir / filename
        if not csv_path.exists():
            raise HTTPException(404, "CSV not found in your session")
    else:
        csv_path = latest_csv_in_session(sid)
        if not csv_path:
            return JSONResponse(status_code=400, content={"error": "No CSV found in this session."})

    out_name = _predict_and_append(str(csv_path), sdir)
    return {"filename": out_name}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    req: ChatRequest,
    session_id: Optional[str] = Header(None),
    sid: Optional[str] = Query(None),
):
    sid = sid or session_id or "default"
    sdir = get_session_dir(sid)

    reply, reports = route_command(req.message, reasoner, session_dir=sdir)

    moved: List[str] = []
    for rpt in reports:
        dst = sdir / rpt
        if not dst.exists():
            src = REPORT_DIR / rpt
            if src.exists():
                shutil.copy(src, dst)
                # register copied report
                register_artifact(
                    sdir,
                    f"Report: {rpt}",
                    "report",
                    dst,
                    "application/pdf" if rpt.endswith(".pdf") else "application/octet-stream",
                )
        if dst.exists():
            register_session_file(sid, dst.name)
            moved.append(dst.name)

    return {"response": reply, "new_reports": moved}
