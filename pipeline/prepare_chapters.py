from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import soundfile as sf
import numpy as np


def chapter_frames(timeline: list[dict], samples: int, sample_rate: int, max_seconds: float, fps: int = 25) -> list[tuple[int,int]]:
    """Use a shared integer frame clock; never accumulate per-chapter flooring."""
    if samples <= 0 or sample_rate % fps or not math.isfinite(max_seconds) or max_seconds < 1:
        raise ValueError('无效音频或章节时钟参数')
    total = (samples * fps + sample_rate - 1) // sample_rate
    maximum = int(max_seconds * fps)
    ends = sorted({min(total,max(1,round(float(s['end'])*fps))) for s in timeline} | {total})
    spans = []; start = 0
    while start < total:
        limit = min(total,start+maximum)
        end = max((e for e in ends if start < e <= limit),default=limit)
        spans.append((start,end));start=end
    return spans


def parse_time(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, milliseconds = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000


def format_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def parse_srt(path: Path) -> list[dict]:
    cues: list[dict] = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip()):
        lines = block.splitlines()
        if len(lines) >= 3 and " --> " in lines[1]:
            start, end = lines[1].split(" --> ", 1)
            cues.append({"start": parse_time(start), "end": parse_time(end), "text": "\n".join(lines[2:])})
    return cues


def write_srt(path: Path, cues: list[dict]) -> None:
    lines: list[str] = []
    for index, cue in enumerate(cues, start=1):
        lines.extend([str(index), f"{format_time(cue['start'])} --> {format_time(cue['end'])}", cue["text"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--srt", type=Path, required=True)
    parser.add_argument("--avatar", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--max-chapter-seconds", type=float, default=120.0)
    args = parser.parse_args()

    timeline = json.loads(args.timeline.read_text(encoding="utf-8"))
    cues = parse_srt(args.srt)
    audio, sample_rate = sf.read(args.audio, dtype="float32")
    boundaries = chapter_frames(timeline,len(audio),sample_rate,args.max_chapter_seconds)
    samples_per_frame = sample_rate // 25
    audio = np.pad(audio,(0,boundaries[-1][1]*samples_per_frame-len(audio)))

    chapter_dir = args.work_dir / "chapters"
    config_dir = args.work_dir / "configs"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for index, (first_frame, last_frame) in enumerate(boundaries, start=1):
        start, end = first_frame / 25, last_frame / 25
        frame_offset = first_frame
        chapter_id = f"chapter_{index:03d}"
        audio_path = chapter_dir / f"{chapter_id}.wav"
        srt_path = chapter_dir / f"{chapter_id}.srt"
        config_path = config_dir / f"{chapter_id}.yaml"
        result_name = f"{args.job_id}_{index:03d}.mp4"
        result_dir = args.work_dir / "avatar_chapters" / chapter_id
        raw_video = result_dir / "v15" / result_name
        sf.write(audio_path, audio[first_frame*samples_per_frame:last_frame*samples_per_frame], sample_rate)
        local_cues = [
            {"start": max(0.0, cue["start"] - start), "end": min(end - start, cue["end"] - start), "text": cue["text"]}
            for cue in cues
            if cue["end"] > start and cue["start"] < end
        ]
        write_srt(srt_path, local_cues)
        config_path.write_text(
            f"{chapter_id}:\n  video_path: {json.dumps(str(args.avatar), ensure_ascii=False)}\n  audio_path: {json.dumps(str(audio_path), ensure_ascii=False)}\n  result_name: {json.dumps(result_name)}\n  start_frame_offset: {frame_offset}\n",
            encoding="utf-8",
        )
        manifest.append({"id": chapter_id, "audio": str(audio_path), "srt": str(srt_path), "config": str(config_path), "result_dir": str(result_dir), "raw_video": str(raw_video), "start_frame_offset": frame_offset,
                         "frame_count":last_frame-first_frame,"start_seconds":start,"end_seconds":end})
    (args.work_dir / "chapters_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
