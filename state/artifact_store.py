# state/artifact_store.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import uuid
from typing import List, Literal, Optional

ArtifactKind = Literal["plot", "report", "prediction", "scrape", "image_result", "other"]


@dataclass
class Artifact:
    id: str
    title: str
    kind: ArtifactKind
    file_path: str
    mime: str
    created_at: str  # ISO8601
    extra: Optional[dict] = None


def _store_path(session_dir: Path) -> Path:
    return session_dir / "artifacts.json"


def load_artifacts(session_dir: Path) -> List[Artifact]:
    p = _store_path(session_dir)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return [Artifact(**a) for a in data]


def save_artifacts(session_dir: Path, arts: List[Artifact]) -> None:
    p = _store_path(session_dir)
    p.write_text(json.dumps([asdict(a) for a in arts], indent=2))


def register_artifact(session_dir: Path, title: str, kind: ArtifactKind,
                      file_path: Path, mime: str, extra: dict | None = None) -> Artifact:
    arts = load_artifacts(session_dir)

    try:
        relative_path = file_path.relative_to(session_dir).as_posix()
    except ValueError:
        # Fallback if file is not within session_dir (e.g. temp files)
        relative_path = file_path.name

    art = Artifact(
        id=str(uuid.uuid4()),
        title=title,
        kind=kind,
        file_path=relative_path,
        mime=mime,
        created_at=__import__("datetime").datetime.utcnow().isoformat() + "Z",
        extra=extra,
    )
    print(f"📦 Registering: {file_path} (exists: {file_path.exists()})")

    arts.append(art)
    save_artifacts(session_dir, arts)
    return art

