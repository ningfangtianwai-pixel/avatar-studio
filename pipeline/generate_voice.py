from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import jieba
from voice_quality import QualityError, acoustic_check, align_text, clean_script, normalized, speech_budget, transcribe


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
CAPTION_TOKEN_RE = re.compile(
    r"\d+(?:[.,]\d+)?(?:万亿|[万亿倍条年月日%％])*|[A-Za-z]+(?:[./+_-][A-Za-z]+)*|[\u4e00-\u9fff]|[^\s]"
)
CAPTION_PROTECTED_RE = re.compile(r"\d+(?:[.,]\d+)?(?:万亿|[万亿倍条年月日%％])*|[A-Za-z]+(?:[./+_-][A-Za-z]+)*")
CAPTION_PUNCTUATION = set("，。！？!?、：；;,.…’”」』】）》")
CAPTION_OPENING_PUNCTUATION = set("‘“「『【《（(")
PREFERRED_LINE_STARTS = set("是在向从为把将与和及并而但也都就")
CLOSING_PUNCTUATION = set("’”」』】）》）)")

for domain_word in (
    "人工智能",
    "农业农村部",
    "微信小程序",
    "数字乡村",
    "智慧农业",
    "农业大模型",
    "智能农机",
    "数字人工作台",
    "绕不过去",
):
    jieba.add_word(domain_word, freq=100_000)


def normalize_for_speech(text: str) -> str:
    text = clean_script(text)
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


def build_segments(script: str, target: int = 60, limit: int = 88) -> list[dict]:
    """Merge short written paragraphs into longer, more natural TTS turns."""
    units: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", clean_script(script)) if p.strip()]
    for paragraph in paragraphs:
        paragraph = re.sub(r"\s+", " ", paragraph)
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
            if tokens[index-1] in CAPTION_PUNCTUATION:
                score -= 8.0
            if tokens[index-1] in {'谁','这','那','第','一个'}:
                score += 6.0
            if tokens[index] in PREFERRED_LINE_STARTS:
                score -= 3.0
            if tokens[index] in {'的','了','着','过'}:
                score += 6.0
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
        if token in CLOSING_PUNCTUATION and not current_phrase and phrases:
            phrases[-1].append(token)
            continue
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
        previous_end = ''.join(groups[-1]).rstrip(''.join(CLOSING_PUNCTUATION))[-1:] if groups else ''
        if groups and previous_end not in {'。','！','？','!','?'} and sum(token_width(token) for token in groups[-1] + chunk) <= max_width:
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
    # Sentence splitting may leave a closing quote at the next TTS boundary.
    # Move only punctuation back; audio and normalized character clocks stay intact.
    caption_segments = [dict(segment) for segment in timeline]
    for index in range(1,len(caption_segments)):
        text = caption_segments[index]['display_text']
        count = len(text)-len(text.lstrip(''.join(CLOSING_PUNCTUATION)))
        if count:
            caption_segments[index-1]['display_text'] += text[:count]
            caption_segments[index]['display_text'] = text[count:]
    for segment in caption_segments:
        groups = split_caption(segment["display_text"])
        char_times = segment.get("char_times")
        if not char_times or len(char_times) != len(normalized(segment['display_text'])):
            raise QualityError('字幕缺少有效语音对齐结果，停止生成')
        consumed = 0
        for group in groups:
            start = segment["start"] + char_times[min(consumed,len(char_times)-1)]
            consumed += len(normalized(group))
            end = segment["start"] + char_times[consumed] if consumed < len(char_times) else segment['speech_end']
            if end <= start or (cues and start < cues[-1]['end'] - .001):
                raise QualityError('字幕识别时间重叠或无有效时长，请复核配音')
            cues.append({"start": start, "end": end, "text": group})
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
    parser.add_argument("--whisper-bin", type=Path, required=True)
    parser.add_argument("--whisper-model", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    segment_dir = args.output_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    segments = build_segments(args.script.read_text(encoding="utf-8"))
    if not segments:
        raise RuntimeError("文案为空。")
    if not args.whisper_bin.is_file() or not args.whisper_model.is_file():
        raise QualityError('离线语音质检未配置，不能跳过质检生成')

    import torch
    from qwen_tts import Qwen3TTSModel

    torch.manual_seed(20260902)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cuda:0", dtype=torch.float32, attn_implementation="sdpa")
    prompt = model.create_voice_clone_prompt(ref_audio=str(args.ref_audio), ref_text=args.ref_text, x_vector_only_mode=False)

    generated: list[tuple[dict, np.ndarray]] = []
    sample_rate: int | None = None
    report = {'version':1,'status':'checking','attempts':[]}
    report_path = args.output_dir / 'quality.json'
    def save_report():
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    # Capture actual code count without patching the installed upstream package.
    code_counts = []
    original_generate = model.model.generate
    def counted_generate(*values, **options):
        result = original_generate(*values, **options)
        code_counts[:] = [int(code.shape[0]) for code in result[0]]
        return result
    model.model.generate = counted_generate

    def generate_checked(segment, depth=0):
        nonlocal sample_rate
        for attempt in range(2):
            _, budget = speech_budget(segment['speech_text'])
            torch.manual_seed(20260907 + len(report['attempts']))
            code_counts.clear()
            wavs, current_rate = model.generate_voice_clone(
                text=segment['speech_text'],language='Chinese',voice_clone_prompt=prompt,
                non_streaming_mode=True,max_new_tokens=budget,
            )
            raw = np.asarray(wavs[0])
            check = acoustic_check(raw,current_rate,segment['speech_text'],code_counts[0] if code_counts else None,budget)
            wav = trim_silence(raw,current_rate)
            if sample_rate is None: sample_rate=int(current_rate)
            if int(current_rate)!=sample_rate: raise QualityError('采样率发生变化')
            entry = {'text':segment['display_text'],'attempt':attempt+1,'split_depth':depth,**check}
            report['attempts'].append(entry)
            if not check['reasons']:
                candidate = segment_dir / f"candidate-{len(report['attempts']):03d}.wav"
                sf.write(candidate,wav,sample_rate)
                data = transcribe(candidate,args.whisper_bin,args.whisper_model)
                alignment = align_text(segment['display_text'],data,len(wav)/sample_rate)
                entry.update(alignment)
                if not alignment['reasons']:
                    entry['accepted']=True
                    generated.append(({**segment,'char_times':alignment['char_times']},wav))
                    save_report()
                    print(f"[配音质检通过 {len(generated)}] {len(wav)/sample_rate:.2f}s，转写误差 {alignment['error_rate']:.1%}",flush=True)
                    return
            entry['accepted']=False
            save_report()
            print(f"[配音质检重试] {'；'.join(entry['reasons'])}",flush=True)
        if depth < 1 and len(segment['display_text']) > 28:
            pieces = hard_split(segment['display_text'], max(18,len(segment['display_text'])//2))
            if len(pieces)>1:
                for i,piece in enumerate(pieces):
                    generate_checked({'display_text':piece,'speech_text':normalize_for_speech(piece),
                                      'pause_after':segment['pause_after'] if i==len(pieces)-1 else .12},depth+1)
                return
        raise QualityError('配音多次质检未通过，已停止口型渲染：'+'；'.join(entry['reasons']))
    try:
        for segment in segments: generate_checked(segment)
    except Exception as error:
        report.update(status='failed',error=str(error));save_report()
        raise
    finally:
        model.model.generate = original_generate

    if sample_rate is None:
        raise RuntimeError("没有生成音频。")

    rms_values = [value for _, wav in generated if (value := active_rms(wav)) > 0]
    target_rms = float(np.median(rms_values)) if rms_values else 0.0
    parts: list[np.ndarray] = []
    timeline: list[dict] = []
    cursor_samples = 0
    for index, (segment, wav) in enumerate(generated):
        current_rms = active_rms(wav)
        if target_rms > 0 and current_rms > 0:
            wav = wav * float(np.clip(target_rms / current_rms, 0.85, 1.18))
        wav = fade_edges(np.clip(wav, -0.98, 0.98).astype(np.float32), sample_rate)
        sf.write(segment_dir / f"{index:03d}.wav", wav, sample_rate)
        speech_duration = len(wav) / sample_rate
        pause = np.zeros(int(round(segment["pause_after"] * sample_rate)), dtype=np.float32)
        cursor = cursor_samples / sample_rate
        timeline.append(
            {
                **segment,
                "index": index,
                "start": cursor,
                "speech_end": cursor + speech_duration,
                "end": (cursor_samples + len(wav) + len(pause)) / sample_rate,
                "start_sample": cursor_samples,
                "end_sample": cursor_samples + len(wav) + len(pause),
            }
        )
        parts.extend([wav, pause])
        cursor_samples += len(wav) + len(pause)

    full_audio = np.concatenate(parts)
    peak = float(np.max(np.abs(full_audio)))
    if peak > 0:
        full_audio *= min(1.0, 0.92 / peak)
    sf.write(args.output_dir / "voice.wav", full_audio, sample_rate)
    (args.output_dir / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        write_subtitles(args.output_dir / "subtitles.srt", timeline)
    except Exception as error:
        report.update(status='failed',error=str(error));save_report()
        raise
    report.update(status='passed',accepted_segments=len(generated),duration=len(full_audio)/sample_rate)
    save_report()


if __name__ == "__main__":
    main()
