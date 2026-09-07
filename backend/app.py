from __future__ import annotations

import os
import fcntl
import json
import logging
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from backend.security import LocalRequestGuard
from backend.config import PROJECT, RUNTIME, MEDIA_ROOT, AI_PROJECT, MAX_UPLOAD_BYTES, ALLOWED_ORIGINS
from pydantic import BaseModel, Field, ConfigDict

from backend.pipeline import run_pipeline


DB_PATH = RUNTIME / "studio.db"
AVATAR_DIR = MEDIA_ROOT / "数字人形象"
VOICE_DIR = MEDIA_ROOT / "声音库"
VIDEO_DIR = MEDIA_ROOT / "生成视频"
TRASH_DIR = MEDIA_ROOT / "回收站"
STOP_EVENT = threading.Event()
WORKER_THREAD: threading.Thread | None = None


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@contextmanager
def connect():
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
    finally:
        connection.close()


def init_storage() -> None:
    for directory in (RUNTIME, RUNTIME / "jobs", RUNTIME / "logs", AVATAR_DIR, VOICE_DIR, VIDEO_DIR, TRASH_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS avatars (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                thumbnail_path TEXT,
                managed INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                deleted_at TEXT
            );
            CREATE TABLE IF NOT EXISTS voices (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                audio_path TEXT NOT NULL,
                source_audio_path TEXT,
                ref_text TEXT NOT NULL,
                managed INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                deleted_at TEXT
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                script_text TEXT NOT NULL,
                avatar_id TEXT NOT NULL,
                voice_id TEXT NOT NULL,
                subtitles INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                output_path TEXT,
                output_managed INTEGER NOT NULL DEFAULT 1,
                duration_seconds REAL,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                deleted_at TEXT,
                FOREIGN KEY (avatar_id) REFERENCES avatars(id),
                FOREIGN KEY (voice_id) REFERENCES voices(id)
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);
            """
        )
        voice_columns = {row["name"] for row in db.execute("PRAGMA table_info(voices)")}
        if "source_audio_path" not in voice_columns:
            db.execute("ALTER TABLE voices ADD COLUMN source_audio_path TEXT")
        db.execute("UPDATE voices SET source_audio_path=audio_path WHERE source_audio_path IS NULL")
        db.execute(
            "UPDATE jobs SET status='queued',stage='上次运行中断，已重新排队',progress=0,started_at=NULL WHERE status='running'"
        )


def validate_media(path: Path, kind: str, max_seconds: float) -> None:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=20, check=False,
    )
    try:
        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
        streams = [stream for stream in data["streams"] if stream.get("codec_type") == kind]
        if result.returncode or not streams or not 0.5 <= duration <= max_seconds:
            raise ValueError()
        if kind == "video" and any(int(stream.get("width", 0)) * int(stream.get("height", 0)) > 3840 * 2160 for stream in streams):
            raise ValueError()
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"素材需包含有效{kind}轨道，时长 0.5–{max_seconds:g} 秒；视频不超过 4K。") from error


def make_thumbnail(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", "1", "-i", str(source), "-frames:v", "1", "-vf", "scale=640:-2", str(destination)],
        check=False,
        timeout=60,
    )


def normalize_avatar_video(source: Path, destination: Path) -> None:
    """Normalize avatar footage to a deterministic 25 fps H.264 input."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".normalizing.mp4")
    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-i", str(source),
                "-map", "0:v:0", "-an", "-vf", "scale=720:1280:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1,fps=25",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp),
            ],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        if completed.returncode != 0 or not temp.exists() or temp.stat().st_size < 1024:
            detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "无法读取视频"
            raise RuntimeError(detail)
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def normalize_voice_audio(source: Path, destination: Path) -> None:
    """Convert any FFmpeg-readable audio to Qwen3-TTS-compatible PCM WAV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".normalizing.wav")
    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-i", str(source),
                "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(temp),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if completed.returncode != 0 or not temp.exists() or temp.stat().st_size < 1024:
            detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "无法读取音频"
            raise RuntimeError(detail)
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def gpu_status() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=4,
            check=True,
        )
        name, memory, utilization = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        return {"available": True, "name": name, "memory_mb": int(memory), "utilization": int(utilization)}
    except Exception:
        return {"available": False, "name": "未检测到", "memory_mb": 0, "utilization": 0}


def row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def state_payload() -> dict[str, Any]:
    with connect() as db:
        avatars = [row_dict(row) for row in db.execute("SELECT id,name,created_at FROM avatars WHERE deleted_at IS NULL ORDER BY created_at DESC")]
        voice_rows = db.execute(
            "SELECT id,name,ref_text,created_at,audio_path FROM voices WHERE deleted_at IS NULL ORDER BY created_at DESC"
        )
        voices = []
        for row in voice_rows:
            item = row_dict(row)
            audio_path = Path(item.pop("audio_path"))
            item["audio_ready"] = audio_path.suffix.lower() == ".wav" and audio_path.exists()
            voices.append(item)
        jobs = [
            row_dict(row)
            for row in db.execute(
                """SELECT j.id,j.title,j.subtitles,j.status,j.stage,j.progress,j.duration_seconds,j.error,j.created_at,j.completed_at,
                          a.name AS avatar_name,v.name AS voice_name
                   FROM jobs j JOIN avatars a ON a.id=j.avatar_id JOIN voices v ON v.id=j.voice_id
                   WHERE j.deleted_at IS NULL ORDER BY j.created_at DESC LIMIT 100"""
            )
        ]
    return {
        "avatars": avatars,
        "voices": voices,
        "jobs": jobs,
        "engine": gpu_status(),
        "media_root": str(MEDIA_ROOT),
    }


def safe_suffix(filename: str, allowed: set[str]) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"不支持的文件格式：{suffix or '无扩展名'}")
    return suffix


def save_upload(upload: UploadFile, destination: Path) -> None:
    temp = destination.with_suffix(destination.suffix + ".uploading")
    try:
        total = 0
        with temp.open("wb") as target:
            while chunk := upload.file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "文件超过 256 MB 限制。")
                target.write(chunk)
        if total == 0:
            raise HTTPException(400, "上传文件为空。")
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def managed_path(path: Path) -> bool:
    try:
        path.resolve().relative_to(MEDIA_ROOT.resolve())
        return True
    except ValueError:
        return False


def trash_file(path_value: str | None, entity_id: str) -> None:
    if not path_value:
        return
    path = Path(path_value)
    if not path.exists() or not managed_path(path):
        return
    destination = TRASH_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{entity_id}_{path.name}"
    shutil.move(str(path), str(destination))


def update_job(job_id: str, stage: str, progress: int) -> None:
    with connect() as db:
        db.execute("UPDATE jobs SET stage=?, progress=? WHERE id=?", (stage, progress, job_id))


def worker_loop() -> None:
    while not STOP_EVENT.is_set():
        try:
            _worker_loop()
        except Exception:
            logging.exception("队列线程发生异常，将在五秒后重试")
            STOP_EVENT.wait(5)


def _worker_loop() -> None:
    while not STOP_EVENT.is_set():
        with connect() as db:
            job = db.execute("SELECT * FROM jobs WHERE status='queued' AND deleted_at IS NULL ORDER BY created_at LIMIT 1").fetchone()
            if job:
                db.execute("UPDATE jobs SET status='running',stage='准备生成',progress=2,started_at=? WHERE id=?", (now(), job["id"]))
        if not job:
            STOP_EVENT.wait(1.5)
            continue
        try:
            with connect() as db:
                avatar = db.execute("SELECT * FROM avatars WHERE id=?", (job["avatar_id"],)).fetchone()
                voice = db.execute("SELECT * FROM voices WHERE id=?", (job["voice_id"],)).fetchone()
            if not avatar or not voice:
                raise RuntimeError("数字人或声音素材已不存在。")
            output_path = VIDEO_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{job['id'][:8]}.mp4"
            result = run_pipeline(
                job_id=job["id"],
                script_text=job["script_text"],
                avatar_path=Path(avatar["path"]),
                voice_path=Path(voice["audio_path"]),
                voice_ref_text=voice["ref_text"],
                work_dir=RUNTIME / "jobs" / job["id"] / ("attempt-" + uuid.uuid4().hex),
                stop_event=STOP_EVENT,
                output_path=output_path,
                subtitles=bool(job["subtitles"]),
                update=lambda stage, progress: update_job(job["id"], stage, progress),
            )
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(result)],
                capture_output=True,
                text=True,
                check=False,
            )
            duration = float(probe.stdout.strip()) if probe.stdout.strip() else None
            with connect() as db:
                db.execute(
                    "UPDATE jobs SET status='completed',stage='已生成，请预览验收',progress=100,output_path=?,duration_seconds=?,completed_at=? WHERE id=?",
                    (str(result), duration, now(), job["id"]),
                )
        except Exception as error:
            with connect() as db:
                db.execute(
                    "UPDATE jobs SET status='failed',stage='生成失败',error=?,completed_at=? WHERE id=?",
                    (str(error)[:1000], now(), job["id"]),
                )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global WORKER_THREAD
    RUNTIME.mkdir(parents=True, exist_ok=True)
    # Acquire before recovering queued jobs: a second process must not reset live jobs.
    with (RUNTIME / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("工作台已有后台进程，请勿同时启动多个后端。") from error
        init_storage()
        STOP_EVENT.clear()
        WORKER_THREAD = threading.Thread(target=worker_loop, name="avatar-studio-worker", daemon=True)
        WORKER_THREAD.start()
        try:
            yield
        finally:
            STOP_EVENT.set()
            WORKER_THREAD.join(timeout=15)


app = FastAPI(title="数字人口播工作台", version="0.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LocalRequestGuard)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"])


class JobCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=120)
    script_text: str = Field(min_length=1, max_length=10_000)
    avatar_id: str
    voice_id: str
    subtitles: bool = True


@app.get("/api/health")
def health() -> dict[str, Any]:
    if WORKER_THREAD is None or not WORKER_THREAD.is_alive():
        raise HTTPException(503, "生成队列线程未运行。")
    return {"ok": True, "service": "数字人口播工作台", "worker_alive": True}


@app.get("/api/state")
def get_state() -> dict[str, Any]:
    return state_payload()


@app.post("/api/avatars", status_code=201)
def create_avatar(name: str = Form(...), file: UploadFile = File(...)) -> dict[str, Any]:
    name = name.strip()
    if not name or len(name) > 120:
        raise HTTPException(400, "请输入数字人名称。")
    suffix = safe_suffix(file.filename or "", {".mp4", ".mov", ".mkv", ".webm"})
    entity_id = uuid.uuid4().hex
    source = AVATAR_DIR / f"{entity_id}.source{suffix}"
    destination = AVATAR_DIR / f"{entity_id}.mp4"
    thumbnail = AVATAR_DIR / f"{entity_id}.jpg"
    save_upload(file, source)
    try:
        validate_media(source, "video", 120)
        normalize_avatar_video(source, destination)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        if destination.exists():
            destination.unlink()
        raise HTTPException(400, f"视频适配失败：{error}") from error
    finally:
        if source.exists():
            source.unlink()
    make_thumbnail(destination, thumbnail)
    with connect() as db:
        db.execute(
            "INSERT INTO avatars(id,name,path,thumbnail_path,managed,created_at) VALUES(?,?,?,?,1,?)",
            (entity_id, name, str(destination), str(thumbnail) if thumbnail.exists() else None, now()),
        )
    return {"id": entity_id, "name": name}


@app.delete("/api/avatars/{avatar_id}")
def delete_avatar(avatar_id: str) -> dict[str, bool]:
    with connect() as db:
        row = db.execute("SELECT * FROM avatars WHERE id=? AND deleted_at IS NULL", (avatar_id,)).fetchone()
        if not row:
            raise HTTPException(404, "数字人不存在。")
        used = db.execute("SELECT COUNT(*) FROM jobs WHERE avatar_id=? AND status IN ('queued','running') AND deleted_at IS NULL", (avatar_id,)).fetchone()[0]
        if used:
            raise HTTPException(409, "这个数字人正在被任务使用。")
        db.execute("UPDATE avatars SET deleted_at=? WHERE id=?", (now(), avatar_id))
    if row["managed"]:
        trash_file(row["path"], avatar_id)
        trash_file(row["thumbnail_path"], avatar_id)
    return {"ok": True}


@app.post("/api/voices", status_code=201)
def create_voice(name: str = Form(...), ref_text: str = Form(...), file: UploadFile = File(...)) -> dict[str, Any]:
    name, ref_text = name.strip(), ref_text.strip()
    if not name or not ref_text or len(name) > 120 or len(ref_text) > 2000:
        raise HTTPException(400, "名称和参考音频原文不能为空。")
    suffix = safe_suffix(file.filename or "", {".wav", ".mp3", ".m4a", ".flac", ".ogg"})
    entity_id = uuid.uuid4().hex
    source = VOICE_DIR / f"{entity_id}-source{suffix}"
    destination = VOICE_DIR / f"{entity_id}.wav"
    save_upload(file, source)
    try:
        validate_media(source, "audio", 60)
        normalize_voice_audio(source, destination)
    except Exception as error:
        source.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise HTTPException(400, f"音频转码失败：{error}") from error
    with connect() as db:
        db.execute(
            "INSERT INTO voices(id,name,audio_path,source_audio_path,ref_text,managed,created_at) VALUES(?,?,?,?,?,1,?)",
            (entity_id, name, str(destination), str(source), ref_text, now()),
        )
    return {"id": entity_id, "name": name, "audio_ready": True}


@app.post("/api/voices/{voice_id}/normalize")
def normalize_voice(voice_id: str) -> dict[str, Any]:
    with connect() as db:
        row = db.execute("SELECT * FROM voices WHERE id=? AND deleted_at IS NULL", (voice_id,)).fetchone()
        if not row:
            raise HTTPException(404, "声音不存在。")
        used = db.execute("SELECT COUNT(*) FROM jobs WHERE voice_id=? AND status IN ('queued','running') AND deleted_at IS NULL", (voice_id,)).fetchone()[0]
        if used:
            raise HTTPException(409, "声音正在被任务使用，请完成后再转码。")
        source = Path(row["source_audio_path"] or row["audio_path"])
        if not source.exists():
            source = Path(row["audio_path"])
        if not source.is_file():
            raise HTTPException(404, "找不到原始音频文件。")
        destination = VOICE_DIR / f"{voice_id}.wav"
        try:
            validate_media(source, "audio", 60)
            normalize_voice_audio(source, destination)
        except Exception as error:
            raise HTTPException(400, f"音频转码失败：{error}") from error
        db.execute("UPDATE voices SET audio_path=?,source_audio_path=? WHERE id=?", (str(destination), str(source), voice_id))
        db.execute("UPDATE jobs SET stage='声音已适配，可重新生成',error=NULL WHERE voice_id=? AND status='failed'", (voice_id,))
    return {"ok": True}


@app.delete("/api/voices/{voice_id}")
def delete_voice(voice_id: str) -> dict[str, bool]:
    with connect() as db:
        row = db.execute("SELECT * FROM voices WHERE id=? AND deleted_at IS NULL", (voice_id,)).fetchone()
        if not row:
            raise HTTPException(404, "声音不存在。")
        used = db.execute("SELECT COUNT(*) FROM jobs WHERE voice_id=? AND status IN ('queued','running') AND deleted_at IS NULL", (voice_id,)).fetchone()[0]
        if used:
            raise HTTPException(409, "这个声音正在被任务使用。")
        db.execute("UPDATE voices SET deleted_at=? WHERE id=?", (now(), voice_id))
    if row["managed"]:
        trash_file(row["audio_path"], voice_id)
        if row["source_audio_path"] != row["audio_path"]:
            trash_file(row["source_audio_path"], voice_id)
    return {"ok": True}


@app.post("/api/jobs", status_code=201)
def create_job(payload: JobCreate) -> dict[str, Any]:
    with connect() as db:
        if db.execute("SELECT COUNT(*) FROM jobs WHERE status IN (\'queued\',\'running\') AND deleted_at IS NULL").fetchone()[0] >= 32:
            raise HTTPException(409, "队列已满，请等待现有任务完成。")
        avatar = db.execute("SELECT id,path FROM avatars WHERE id=? AND deleted_at IS NULL", (payload.avatar_id,)).fetchone()
        voice = db.execute("SELECT id,audio_path FROM voices WHERE id=? AND deleted_at IS NULL", (payload.voice_id,)).fetchone()
        if not avatar or not voice:
            raise HTTPException(400, "请选择可用的数字人和声音。")
        if not Path(avatar["path"]).is_file() or not Path(voice["audio_path"]).is_file():
            raise HTTPException(400, "素材文件已丢失，请重新上传。")
        job_id = uuid.uuid4().hex
        db.execute(
            """INSERT INTO jobs(id,title,script_text,avatar_id,voice_id,subtitles,status,stage,progress,created_at)
               VALUES(?,?,?,?,?,?,'queued','排队中',0,?)""",
            (job_id, payload.title.strip(), payload.script_text.strip(), payload.avatar_id, payload.voice_id, int(payload.subtitles), now()),
        )
    return {"id": job_id, "status": "queued"}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, bool]:
    with connect() as db:
        row = db.execute("SELECT status FROM jobs WHERE id=? AND deleted_at IS NULL", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "任务不存在。")
        if row["status"] != "queued":
            raise HTTPException(409, "只能取消仍在排队的任务。")
        db.execute("UPDATE jobs SET status='cancelled',stage='已取消',completed_at=? WHERE id=?", (now(), job_id))
    return {"ok": True}


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str) -> dict[str, Any]:
    with connect() as db:
        row = db.execute("SELECT status FROM jobs WHERE id=? AND deleted_at IS NULL", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "任务不存在。")
        if row["status"] not in {"failed", "cancelled"}:
            raise HTTPException(409, "只有失败或已取消的任务可以重新生成。")
        active_count = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running') AND deleted_at IS NULL").fetchone()[0]
        if active_count >= 32:
            raise HTTPException(429, "队列已满，请等待已有任务完成。")
        assets = db.execute(
            "SELECT a.path,v.audio_path FROM jobs j JOIN avatars a ON a.id=j.avatar_id JOIN voices v ON v.id=j.voice_id WHERE j.id=? AND a.deleted_at IS NULL AND v.deleted_at IS NULL",
            (job_id,),
        ).fetchone()
        if not assets or not Path(assets["path"]).is_file() or not Path(assets["audio_path"]).is_file():
            raise HTTPException(409, "原任务的形象或声音已删除，请新建任务。")
        db.execute(
            """UPDATE jobs SET status='queued',stage='重新排队',progress=0,error=NULL,
               started_at=NULL,completed_at=NULL,output_path=NULL,duration_seconds=NULL WHERE id=?""",
            (job_id,),
        )
    return {"id": job_id, "status": "queued"}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, bool]:
    with connect() as db:
        row = db.execute("SELECT * FROM jobs WHERE id=? AND deleted_at IS NULL", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "成片不存在。")
        if row["status"] in {"queued", "running"}:
            raise HTTPException(409, "任务运行期间不能删除。")
        db.execute("UPDATE jobs SET deleted_at=? WHERE id=?", (now(), job_id))
    if row["output_managed"]:
        trash_file(row["output_path"], job_id)
    return {"ok": True}


@app.get("/api/media/avatar/{avatar_id}")
def avatar_media(avatar_id: str) -> FileResponse:
    with connect() as db:
        row = db.execute("SELECT path,thumbnail_path FROM avatars WHERE id=? AND deleted_at IS NULL", (avatar_id,)).fetchone()
    if not row:
        raise HTTPException(404)
    path = Path(row["thumbnail_path"] or row["path"])
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/api/media/voice/{voice_id}")
def voice_media(voice_id: str) -> FileResponse:
    with connect() as db:
        row = db.execute("SELECT audio_path FROM voices WHERE id=? AND deleted_at IS NULL", (voice_id,)).fetchone()
    if not row or not Path(row["audio_path"]).exists():
        raise HTTPException(404)
    return FileResponse(row["audio_path"])


@app.get("/api/media/video/{job_id}")
def video_media(job_id: str) -> FileResponse:
    with connect() as db:
        row = db.execute("SELECT output_path FROM jobs WHERE id=? AND deleted_at IS NULL", (job_id,)).fetchone()
    if not row or not row["output_path"] or not Path(row["output_path"]).exists():
        raise HTTPException(404)
    return FileResponse(row["output_path"], media_type="video/mp4")


@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def job_log(job_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise HTTPException(404)
    with connect() as db:
        if not db.execute("SELECT id FROM jobs WHERE id=? AND deleted_at IS NULL", (job_id,)).fetchone():
            raise HTTPException(404)
    job_dir = RUNTIME / "jobs" / job_id
    attempts = sorted(job_dir.glob("attempt-*/job.log"), key=lambda p: p.stat().st_mtime)
    path = attempts[-1] if attempts else job_dir / "job.log"
    if not path.exists():
        return "暂无日志。"
    with path.open("rb") as log:
        log.seek(max(0, path.stat().st_size - 30_000))
        return log.read().decode("utf-8", errors="replace")


@app.post("/api/open-media-folder")
def open_media_folder() -> dict[str, bool]:
    windows_path = subprocess.run(["wslpath", "-w", str(MEDIA_ROOT)], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.Popen(["/mnt/c/Windows/explorer.exe", windows_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"ok": True}
