from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import soundfile as sf


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
    boundaries: list[tuple[float, float]] = []
    chapter_start = 0.0
    last_end = 0.0
    for segment in timeline:
        segment_end = float(segment["end"])
        if last_end > chapter_start and segment_end - chapter_start > args.max_chapter_seconds:
            boundaries.append((chapter_start, last_end))
            chapter_start = last_end
        last_end = segment_end
    if last_end > chapter_start:
        boundaries.append((chapter_start, last_end))

    chapter_dir = args.work_dir / "chapters"
    config_dir = args.work_dir / "configs"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    frame_offset = 0
    for index, (start, end) in enumerate(boundaries, start=1):
        chapter_id = f"chapter_{index:03d}"
        audio_path = chapter_dir / f"{chapter_id}.wav"
        srt_path = chapter_dir / f"{chapter_id}.srt"
        config_path = config_dir / f"{chapter_id}.yaml"
        result_name = f"{args.job_id}_{index:03d}.mp4"
        result_dir = args.work_dir / "avatar_chapters" / chapter_id
        raw_video = result_dir / "v15" / result_name
        sf.write(audio_path, audio[round(start * sample_rate) : min(len(audio), round(end * sample_rate))], sample_rate)
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
        manifest.append({"id": chapter_id, "audio": str(audio_path), "srt": str(srt_path), "config": str(config_path), "result_dir": str(result_dir), "raw_video": str(raw_video), "start_frame_offset": frame_offset})
        frame_offset += math.floor((end - start) * 25)
    (args.work_dir / "chapters_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
