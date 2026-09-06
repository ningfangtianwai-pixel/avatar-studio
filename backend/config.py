"""Local-only settings. Personal paths belong in ignored studio.local.json."""
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
_file = PROJECT / "studio.local.json"
_local = json.loads(_file.read_text(encoding="utf-8")) if _file.exists() else {}

def setting(name: str, default: str) -> str:
    return os.environ.get(name, str(_local.get(name, default)))

RUNTIME = Path(setting("AVATAR_STUDIO_RUNTIME", str(PROJECT / "runtime"))).expanduser().resolve()
MEDIA_ROOT = Path(setting("AVATAR_STUDIO_MEDIA_ROOT", str(RUNTIME / "media"))).expanduser().resolve()
AI_PROJECT = Path(setting("AVATAR_STUDIO_AI_PROJECT", str(PROJECT.parent / "ai-avatar-local"))).expanduser().resolve()
TTS_PYTHON = Path(setting("AVATAR_STUDIO_TTS_PYTHON", str(AI_PROJECT / "tts-qwen3/.venv/bin/python")))
MUSETALK = Path(setting("AVATAR_STUDIO_MUSETALK", str(AI_PROJECT / "MuseTalk")))
MUSETALK_PYTHON = Path(setting("AVATAR_STUDIO_MUSETALK_PYTHON", str(MUSETALK / ".venv/bin/python")))
MAX_UPLOAD_BYTES = 256 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024
ALLOWED_ORIGINS = {"http://localhost:3000", "http://127.0.0.1:3000"}
