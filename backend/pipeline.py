from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import signal
import threading
import time
from functools import partial
from pathlib import Path
from typing import Callable


from backend.config import MUSETALK, TTS_PYTHON, MUSETALK_PYTHON, WHISPER_BIN, WHISPER_MODEL, SYNCNET_DIR
PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"


def validate_video(path: Path, expected_frames: int, *, require_audio: bool = True) -> dict:
    probe = subprocess.run(['ffprobe','-v','error','-show_streams','-of','json',str(path)],
                           capture_output=True,text=True,timeout=30,check=True)
    streams = json.loads(probe.stdout)['streams']
    video = next((s for s in streams if s['codec_type']=='video'),{})
    audio = next((s for s in streams if s['codec_type']=='audio'),{})
    if video.get('avg_frame_rate') != '25/1' or int(video.get('nb_frames',0)) != expected_frames:
        raise RuntimeError('视频帧数或帧率与驱动音频不一致，已停止发布。')
    audio_duration=float(audio.get('duration',0))
    if require_audio and (not audio or not math.isfinite(audio_duration) or abs(audio_duration-expected_frames/25) > .12):
        raise RuntimeError('音画轨道长度不匹配，已停止发布。')
    return {'frames':expected_frames,'fps':25,'duration':expected_frames/25,'audio_duration':audio.get('duration')}


def validate_sync(report: dict) -> None:
    windows=report.get('windows',[])
    if not report.get('global') or not windows:
        raise RuntimeError('口型样本太短，无法完成同步评估，请人工复核。')
    if any(not math.isfinite(w['offset_frames']) or not math.isfinite(w['confidence']) or abs(w['offset_frames'])>3 or w['confidence']<2.0 for w in windows):
        raise RuntimeError('口型同步估计偏移较大或置信度不足，成片已保留在任务目录，需复核后再发布。')


def _run(command: list[str], log_path: Path, *, cwd: Path | None = None, env: dict[str, str] | None = None, stop_event: threading.Event | None = None, timeout: float = 21600, on_tick: Callable[[], None] | None = None, failure_report: Path | None = None) -> None:
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
                if on_tick:
                    on_tick()
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
            if failure_report and failure_report.is_file():
                try:
                    message = json.loads(failure_report.read_text()).get('error')
                except (OSError,ValueError):
                    message = None
                if message:
                    raise RuntimeError(str(message)[:600])
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
    if not WHISPER_BIN.is_file() or not WHISPER_MODEL.is_file():
        raise RuntimeError('请配置离线语音质检：AVATAR_STUDIO_WHISPER_BIN / AVATAR_STUDIO_WHISPER_MODEL；不能跳过。')
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work_dir).free < 5 * 1024**3 or shutil.disk_usage(output_path.parent).free < 1024**3:
        raise RuntimeError("磁盘空间不足：任务盘至少需 5 GB，成片盘至少需 1 GB。")
    log_path = work_dir / "job.log"
    script_path = work_dir / "script.txt"
    script_path.write_text(script_text.strip(), encoding="utf-8")

    last_voice_progress = (-1,-1)
    def voice_progress():
        nonlocal last_voice_progress
        try:
            attempts = json.loads((work_dir/'narration/quality.json').read_text())['attempts']
        except (OSError,ValueError,KeyError):
            return
        passed = [a for a in attempts if a.get('accepted')]
        failed = sum(a.get('accepted') is False for a in attempts)
        key = (len(passed),failed)
        if key != last_voice_progress:
            last_voice_progress = key
            fraction = sum(len(a['text']) for a in passed)/max(1,len(script_text))
            update(f'配音质检：已通过 {len(passed)} 段，失败重试 {failed} 次',min(30,8+round(22*fraction)))

    update("正在生成配音并逐段质检（失败自动重试）", 8)
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
            '--whisper-bin',str(WHISPER_BIN),
            '--whisper-model',str(WHISPER_MODEL),
        ],
        log_path,
        env={"HF_HUB_DISABLE_XET": "1", "TORCH_DISABLE_NATIVE_JIT": "1"},
        on_tick=voice_progress,
        failure_report=work_dir/'narration/quality.json',
    )

    voice_report = json.loads((work_dir / 'narration/quality.json').read_text())
    if voice_report.get('status') != 'passed':
        raise RuntimeError('配音质检未通过，禁止进入口型渲染。')
    update("配音质检通过，正在按统一帧时钟拆分章节", 32)
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
        validate_video(raw_video,chapter['frame_count'])

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
            "[vstyled]format=yuv420p[vout]",
            (
                f"[{audio_input_index}:a]aresample=48000:async=1:first_pts=0,"
                "highpass=f=65,lowpass=f=14500,"
                "loudnorm=I=-16:TP=-1.5:LRA=11,apad[aout]"
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
            '-t',str(sum(c['frame_count'] for c in chapters)/25),
            str(local_final),
        ]
    )
    run(
        command,
        log_path,
    )
    update('正在核验完整成片解码、帧数和音画时长',96)
    check = validate_video(local_final,sum(c['frame_count'] for c in chapters))
    run(['ffmpeg','-v','error','-xerror','-i',str(local_final),'-f','null','-'],log_path)
    sync_status='not configured; manual review required'
    if (SYNCNET_DIR/'data/syncnet_v2.model').is_file():
        update('正在逐段评估嘴型与声音同步',98)
        sync_path=work_dir/'lipsync-quality.json'
        run([str(MUSETALK_PYTHON),str(PIPELINE_DIR.parent/'scripts/evaluate-lipsync.py'),
             '--video',str(local_final),'--coords',str(work_dir/'avatar_chapters'/(avatar_path.name.split('.')[0]+'.pkl.json')),
             '--syncnet-dir',str(SYNCNET_DIR),'--output',str(sync_path)],log_path)
        validate_sync(json.loads(sync_path.read_text()))
        sync_status='passed auxiliary SyncNet thresholds; visual review still required'
    (work_dir/'quality.json').write_text(json.dumps({'status':'technical_checks_passed','narration':'passed',
        'video':check,'lip_sync':sync_status},indent=2))
    staging = output_path.with_suffix(".mp4.partial")
    try:
        shutil.copy2(local_final, staging)
        os.replace(staging, output_path)
    finally:
        staging.unlink(missing_ok=True)
    return output_path
