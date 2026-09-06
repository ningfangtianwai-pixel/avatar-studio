"""Small real-GPU check, invoked explicitly with locally supplied references."""
import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import RUNTIME
from backend.pipeline import run_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--avatar", required=True, type=Path)
parser.add_argument("--voice", required=True, type=Path)
parser.add_argument("--ref-text", required=True)
args = parser.parse_args()
work = RUNTIME / "smoke" / uuid.uuid4().hex
result = run_pipeline(
    job_id=work.name, script_text="你好，欢迎使用数字人工作台。",
    avatar_path=args.avatar, voice_path=args.voice, voice_ref_text=args.ref_text,
    work_dir=work, output_path=work / "deliverable.mp4", subtitles=True,
    update=lambda stage, progress: print(f"{progress}% {stage}", flush=True),
)
print(result, flush=True)
