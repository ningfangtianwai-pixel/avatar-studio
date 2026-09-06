from __future__ import annotations

import json
import os
import shutil
import subprocess
import signal
import threading
import time
from functools import partial
from pathlib import Path
from typing import Callable


from backend.config import MUSETALK, TTS_PYTHON, MUSETALK_PYTHON
PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"


def _run(command: list[str], log_path: Path, *, cwd: Path | None = None, env: dict[str, str] | None = None, stop_event: threading.Event | None = None, timeout: float = 21600) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=cwd, env=merged_env, stdout=log, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        started = time.monotonic()
        try:
            while process.poll() is None:
                if (stop_event and stop_event.is_set()) or time.monotonic() - started > timeout:
                    raise RuntimeError("生成已中断或阶段超时，可重新生成。")
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        if process.returncode != 0:
            raise RuntimeError(f"命令执行失败（退出码 {process.returncode}），请查看任务日志。")


def run_pipeline(
    *,
    job_id: str,
    script_text: str,
    avatar_path: Path,
    voice_path: Path,
    voice_ref_text: str,
    work_dir: Path,
    output_path: Path,
    subtitles: bool,
    update: Callable[[str, int], None],
    stop_event: threading.Event | None = None,
) -> Path:
    run = partial(_run, stop_event=stop_event)
    for dependency in (TTS_PYTHON, MUSETALK_PYTHON, avatar_path, voice_path):
        if not dependency.is_file():
            raise RuntimeError("生成环境或素材文件不存在，请检查设置。")
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work_dir).free < 5 * 1024**3 or shutil.disk_usage(output_path.parent).free < 1024**3:
        raise RuntimeError("磁盘空间不足：任务盘至少需 5 GB，成片盘至少需 1 GB。")
    log_path = work_dir / "job.log"
    script_path = work_dir / "script.txt"
    script_path.write_text(script_text.strip(), encoding="utf-8")

    update("正在克隆声音并生成配音", 8)
    run(
        [
            str(TTS_PYTHON),
            str(PIPELINE_DIR / "generate_voice.py"),
            "--script",
            str(script_path),
            "--ref-audio",
            str(voice_path),
            "--ref-text",
            voice_ref_text,
            "--output-dir",
            str(work_dir / "narration"),
        ],
        log_path,
        env={"HF_HUB_DISABLE_XET": "1", "TORCH_DISABLE_NATIVE_JIT": "1"},
    )

    update("正在拆分章节", 32)
    run(
        [
            str(TTS_PYTHON),
            str(PIPELINE_DIR / "prepare_chapters.py"),
            "--audio",
            str(work_dir / "narration/voice.wav"),
            "--timeline",
            str(work_dir / "narration/timeline.json"),
            "--srt",
            str(work_dir / "narration/subtitles.srt"),
            "--avatar",
            str(avatar_path),
            "--work-dir",
            str(work_dir),
            "--job-id",
            job_id,
        ],
        log_path,
    )

    manifest_path = work_dir / "chapters_manifest.json"
    chapters = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not chapters:
        raise RuntimeError("没有生成可用章节。")

    for index, chapter in enumerate(chapters):
        progress = 36 + round((index / max(1, len(chapters))) * 48)
        update(f"正在驱动数字人口型（{index + 1}/{len(chapters)}）", progress)
        raw_video = Path(chapter["raw_video"])
        if not raw_video.exists() or raw_video.stat().st_size < 1024:
            run(
                [
                    str(MUSETALK_PYTHON),
                    "-m",
                    "scripts.inference",
                    "--inference_config",
                    chapter["config"],
                    "--result_dir",
                    chapter["result_dir"],
                    "--unet_model_path",
                    "models/musetalkV15/unet.pth",
                    "--unet_config",
                    "models/musetalkV15/musetalk.json",
                    "--whisper_dir",
                    "models/whisper",
                    "--version",
                    "v15",
                    "--batch_size",
                    "4",
                    "--use_float16",
                    "--use_saved_coord",
                    "--saved_coord",
                ],
                log_path,
                cwd=MUSETALK,
            )

    update("正在平滑合成画面、声音与字幕", 88)
    raw_videos = [Path(chapter["raw_video"]) for chapter in chapters]
    audio_input_index = len(raw_videos)
    video_filters = [f"[{index}:v]setpts=PTS-STARTPTS,fps=25[v{index}]" for index in range(len(raw_videos))]
    concat_inputs = "".join(f"[v{index}]" for index in range(len(raw_videos)))
    video_filters.append(f"{concat_inputs}concat=n={len(raw_videos)}:v=1:a=0[vcat]")

    if subtitles:
        subtitle_path = work_dir / "narration/subtitles.srt"
        escaped_subtitle_path = str(subtitle_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        video_filters.append(
            "[vcat]"
            f"subtitles='{escaped_subtitle_path}':original_size=720x1280:force_style="
            "'FontName=Noto Sans CJK SC,FontSize=11.5,Bold=-1,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&HCC000000,BackColour=&H50000000,BorderStyle=1,"
            "Outline=0.9,Shadow=0.35,Alignment=2,MarginL=12,MarginR=12,MarginV=34,Spacing=0'"
            "[vstyled]"
        )
    else:
        video_filters.append("[vcat]null[vstyled]")

    video_filters.extend(
        [
            "[vstyled]tpad=stop_mode=clone:stop_duration=2,format=yuv420p[vout]",
            (
                f"[{audio_input_index}:a]aresample=48000:async=1:first_pts=0,"
                "highpass=f=65,lowpass=f=14500,"
                "loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
            ),
        ]
    )

    local_final = work_dir / "final.mp4"
    command = ["ffmpeg", "-y", "-v", "warning"]
    for raw_video in raw_videos:
        command.extend(["-i", str(raw_video)])
    command.extend(["-i", str(work_dir / "narration/voice.wav")])
    command.extend(
        [
            "-filter_complex",
            ";".join(video_filters),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-r",
            "25",
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            "-shortest",
            str(local_final),
        ]
    )
    run(
        command,
        log_path,
    )
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(local_final)], capture_output=True, text=True, timeout=30, check=True)
    if float(json.loads(probe.stdout)["format"]["duration"]) <= 0:
        raise RuntimeError("最终视频校验失败。")
    staging = output_path.with_suffix(".mp4.partial")
    try:
        shutil.copy2(local_final, staging)
        os.replace(staging, output_path)
    finally:
        staging.unlink(missing_ok=True)
    return output_path
