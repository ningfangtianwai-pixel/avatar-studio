from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import jieba
from qwen_tts import Qwen3TTSModel


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
CAPTION_TOKEN_RE = re.compile(
    r"\d+(?:[.,]\d+)?(?:万亿|[万亿倍条年月日%％])*|[A-Za-z]+(?:[./+_-][A-Za-z]+)*|[\u4e00-\u9fff]|[^\s]"
)
CAPTION_PROTECTED_RE = re.compile(r"\d+(?:[.,]\d+)?(?:万亿|[万亿倍条年月日%％])*|[A-Za-z]+(?:[./+_-][A-Za-z]+)*")
CAPTION_PUNCTUATION = set("，。！？!?、：；;,.…’”」』】）》")
CAPTION_OPENING_PUNCTUATION = set("‘“「『【《（(")
PREFERRED_LINE_STARTS = set("的了是在向从为把将与和及并而但也都就")

for domain_word in (
    "人工智能",
    "农业农村部",
    "微信小程序",
    "数字乡村",
    "智慧农业",
    "农业大模型",
    "智能农机",
    "数字人工作台",
):
    jieba.add_word(domain_word, freq=100_000)


def normalize_for_speech(text: str) -> str:
    replacements = {
        "ChatGPT": "Chat G P T",
        "TCP/IP": "T C P I P",
        "GPU": "G P U",
        "APP": "A P P",
        "GB": "G B",
        "WSL": "W S L",
        "AI": "A I",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def hard_split(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    current = ""
    for part in re.split(r"(?<=[，、：；])", text):
        if not part:
            continue
        if current and len(current) + len(part) > limit:
            pieces.append(current)
            current = ""
        while len(part) > limit:
            room = limit - len(current)
            current += part[:room]
            part = part[room:]
            pieces.append(current)
            current = ""
        current += part
    if current:
        pieces.append(current)
    return [piece.strip() for piece in pieces if piece.strip()]


def build_segments(script: str, target: int = 82, limit: int = 110) -> list[dict]:
    """Merge short written paragraphs into longer, more natural TTS turns."""
    units: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", script) if p.strip()]
    for paragraph in paragraphs:
        paragraph = re.sub(r"\s+", "", paragraph)
        sentences = [s.strip() for s in re.split(r"(?<=[。！？!?])", paragraph) if s.strip()]
        for sentence in sentences:
            units.extend(hard_split(sentence, limit))

    merged: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) > limit:
            merged.append(current)
            current = ""
        current += unit
        if len(current) >= target and re.search(r"[。！？!?]$", current):
            merged.append(current)
            current = ""
    if current:
        merged.append(current)

    segments: list[dict] = []
    for index, text in enumerate(merged):
        if index == len(merged) - 1:
            pause_after = 0.24
        elif re.search(r"[！？!?]$", text):
            pause_after = 0.22
        elif text.endswith("。"):
            pause_after = 0.18
        else:
            pause_after = 0.12
        segments.append(
            {
                "display_text": text,
                "speech_text": normalize_for_speech(text),
                "pause_after": pause_after,
            }
        )
    return segments


def trim_silence(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak <= 1e-7:
        return wav.astype(np.float32)
    threshold = max(peak * 0.008, 1e-4)
    active = np.flatnonzero(np.abs(wav) >= threshold)
    if not active.size:
        return wav.astype(np.float32)
    pad = int(0.03 * sample_rate)
    return wav[max(0, int(active[0]) - pad) : min(len(wav), int(active[-1]) + pad + 1)].astype(np.float32)


def active_rms(wav: np.ndarray) -> float:
    if not wav.size:
        return 0.0
    peak = float(np.max(np.abs(wav)))
    active = wav[np.abs(wav) >= max(peak * 0.015, 1e-4)]
    return float(np.sqrt(np.mean(np.square(active)))) if active.size else 0.0


def fade_edges(wav: np.ndarray, sample_rate: int, milliseconds: float = 10.0) -> np.ndarray:
    fade_samples = min(int(sample_rate * milliseconds / 1000), len(wav) // 2)
    if fade_samples <= 1:
        return wav
    result = wav.copy()
    curve = np.sin(np.linspace(0.0, np.pi / 2, fade_samples, dtype=np.float32)) ** 2
    result[:fade_samples] *= curve
    result[-fade_samples:] *= curve[::-1]
    return result


def token_width(token: str) -> float:
    if token in CAPTION_PUNCTUATION:
        return 0.55
    if token.isascii():
        return max(1.0, len(token) * 0.56)
    return float(len(token))


def caption_tokens(text: str) -> list[str]:
    """Keep Chinese words, numbers with units, and Latin terms intact."""
    tokens: list[str] = []

    def append_words(fragment: str) -> None:
        for word in jieba.lcut(fragment, HMM=False):
            if not word or word.isspace():
                continue
            if re.fullmatch(r"[\u4e00-\u9fff]+", word):
                tokens.append(word)
            else:
                tokens.extend(CAPTION_TOKEN_RE.findall(word))

    cursor = 0
    for match in CAPTION_PROTECTED_RE.finditer(text):
        append_words(text[cursor : match.start()])
        tokens.append(match.group())
        cursor = match.end()
    append_words(text[cursor:])
    return tokens


def caption_width(text: str) -> float:
    return sum(token_width(token) for token in caption_tokens(text.replace("\n", "")))


def wrap_caption(text: str, line_width: float = 13.0) -> str:
    tokens = caption_tokens(text)
    if sum(token_width(token) for token in tokens) <= line_width:
        return "".join(tokens)
    best_index: int | None = None
    best_score = float("inf")
    for index in range(1, len(tokens)):
        if tokens[index] in CAPTION_PUNCTUATION:
            continue
        if tokens[index - 1] in CAPTION_OPENING_PUNCTUATION:
            continue
        left_width = sum(token_width(token) for token in tokens[:index])
        right_width = sum(token_width(token) for token in tokens[index:])
        if left_width <= line_width + 1.0 and right_width <= line_width + 1.0:
            score = abs(left_width - right_width)
            if tokens[index] in PREFERRED_LINE_STARTS:
                score -= 3.0
            if score < best_score:
                best_index = index
                best_score = score
    if best_index is None:
        best_index = max(1, len(tokens) // 2)
    return "".join(tokens[:best_index]) + "\n" + "".join(tokens[best_index:])


def balanced_token_chunks(tokens: list[str], max_width: float) -> list[list[str]]:
    chunks: list[list[str]] = []
    remaining = tokens[:]
    while sum(token_width(token) for token in remaining) > max_width:
        remaining_width = sum(token_width(token) for token in remaining)
        target_width = remaining_width / math.ceil(remaining_width / max_width)
        best_index = 1
        best_score = float("inf")
        for index in range(1, len(remaining)):
            if remaining[index] in CAPTION_PUNCTUATION:
                continue
            if remaining[index - 1] in CAPTION_OPENING_PUNCTUATION:
                continue
            left_width = sum(token_width(token) for token in remaining[:index])
            if left_width > max_width:
                break
            score = abs(left_width - target_width)
            if remaining[index] in PREFERRED_LINE_STARTS:
                score -= 2.5
            if score < best_score:
                best_index = index
                best_score = score
        chunks.append(remaining[:best_index])
        remaining = remaining[best_index:]
    if remaining:
        chunks.append(remaining)
    return chunks


def split_caption(text: str, max_width: float = 24.0) -> list[str]:
    tokens = caption_tokens(text.strip())
    phrases: list[list[str]] = []
    current_phrase: list[str] = []
    for token in tokens:
        current_phrase.append(token)
        if token in CAPTION_PUNCTUATION:
            phrases.append(current_phrase)
            current_phrase = []
    if current_phrase:
        phrases.append(current_phrase)

    chunks: list[list[str]] = []
    for phrase in phrases:
        chunks.extend(balanced_token_chunks(phrase, max_width))

    groups: list[list[str]] = []
    for chunk in chunks:
        if groups and sum(token_width(token) for token in groups[-1] + chunk) <= max_width:
            groups[-1].extend(chunk)
        else:
            groups.append(chunk)
    return [wrap_caption("".join(group)) for group in groups if group]


def srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_subtitles(path: Path, timeline: list[dict]) -> None:
    cues: list[dict] = []
    for segment in timeline:
        groups = split_caption(segment["display_text"])
        weights = [max(1.0, caption_width(group)) for group in groups]
        weight_total = sum(weights)
        duration = segment["speech_end"] - segment["start"]
        consumed = 0.0
        for group, weight in zip(groups, weights):
            start = segment["start"] + duration * consumed / weight_total
            consumed += weight
            end = segment["start"] + duration * consumed / weight_total
            cues.append({"start": start, "end": max(start + 0.01, end), "text": group})
    lines: list[str] = []
    for index, cue in enumerate(cues, start=1):
        lines.extend([str(index), f"{srt_time(cue['start'])} --> {srt_time(cue['end'])}", cue["text"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--ref-audio", type=Path, required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    segment_dir = args.output_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    segments = build_segments(args.script.read_text(encoding="utf-8"))
    if not segments:
        raise RuntimeError("文案为空。")

    torch.manual_seed(20260902)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cuda:0", dtype=torch.float32, attn_implementation="sdpa")
    prompt = model.create_voice_clone_prompt(ref_audio=str(args.ref_audio), ref_text=args.ref_text, x_vector_only_mode=False)

    generated: list[tuple[dict, np.ndarray]] = []
    sample_rate: int | None = None
    for index, segment in enumerate(segments):
        wavs, current_rate = model.generate_voice_clone(
            text=segment["speech_text"], language="Chinese", voice_clone_prompt=prompt, non_streaming_mode=True, max_new_tokens=2048
        )
        wav = trim_silence(np.asarray(wavs[0]), current_rate)
        if sample_rate is None:
            sample_rate = int(current_rate)
        if int(current_rate) != sample_rate:
            raise RuntimeError("采样率发生变化。")
        generated.append((segment, wav))
        print(f"[{index + 1}/{len(segments)}] {segment['display_text']}", flush=True)

    if sample_rate is None:
        raise RuntimeError("没有生成音频。")

    rms_values = [value for _, wav in generated if (value := active_rms(wav)) > 0]
    target_rms = float(np.median(rms_values)) if rms_values else 0.0
    parts: list[np.ndarray] = []
    timeline: list[dict] = []
    cursor = 0.0
    for index, (segment, wav) in enumerate(generated):
        current_rms = active_rms(wav)
        if target_rms > 0 and current_rms > 0:
            wav = wav * float(np.clip(target_rms / current_rms, 0.85, 1.18))
        wav = fade_edges(np.clip(wav, -0.98, 0.98).astype(np.float32), sample_rate)
        sf.write(segment_dir / f"{index:03d}.wav", wav, sample_rate)
        speech_duration = len(wav) / sample_rate
        pause = np.zeros(int(round(segment["pause_after"] * sample_rate)), dtype=np.float32)
        timeline.append(
            {
                **segment,
                "index": index,
                "start": cursor,
                "speech_end": cursor + speech_duration,
                "end": cursor + speech_duration + segment["pause_after"],
            }
        )
        parts.extend([wav, pause])
        cursor += speech_duration + segment["pause_after"]

    full_audio = np.concatenate(parts)
    peak = float(np.max(np.abs(full_audio)))
    if peak > 0:
        full_audio *= min(1.0, 0.92 / peak)
    sf.write(args.output_dir / "voice.wav", full_audio, sample_rate)
    (args.output_dir / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
    write_subtitles(args.output_dir / "subtitles.srt", timeline)


if __name__ == "__main__":
    main()
