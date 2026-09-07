"""Offline, fail-closed narration checks and ASR-anchored caption timing.

ASR alignment is approximate, not phoneme-level forced alignment. Reports are
private task artifacts and must not be included in source releases.
"""
from __future__ import annotations

import difflib
import json
import re
import subprocess
import unicodedata
from pathlib import Path
from pypinyin import lazy_pinyin


class QualityError(RuntimeError):
    pass


def clean_script(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"!\[([^\]]*)\]\([^\n)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^\n)]*\)", r"\1", text)
    text = re.sub(r"(?m)^\s*(?:#{1,6}\s+|>\s*|[-+*]\s+|\d+[.)]\s+)", "", text)
    text = re.sub(r"(?m)^\s*(?:[-*_]\s*){3,}$", "", text)
    text = re.sub(r"[*`_~]+", "", text)
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    return text.strip()


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    # Match frequent Arabic-number ASR output to Chinese script, without
    # rewriting the actual narration or subtitles.
    digits = "零一二三四五六七八九"
    def number(match):
        s = match.group()
        n = int(s)
        if n > 9999 or (len(s) > 1 and s[0] == '0'):
            return ''.join(digits[int(c)] for c in s)
        if n == 0:
            return '零'
        result = ''; pending_zero = False
        for value, unit in [(1000,'千'), (100,'百'), (10,'十'), (1,'')]:
            d,n = divmod(n,value)
            if d:
                if pending_zero: result += '零'
                result += ('' if d == 1 and value == 10 and not result else digits[d]) + unit
                pending_zero = False
            elif result and n: pending_zero = True
        return result
    text = re.sub(r"\d+", number, text)
    return ''.join(c for c in text if c.isalnum())


def speech_budget(text: str) -> tuple[float, int]:
    seconds = min(48.0, max(8.0, len(normalized(text)) / 2.2 + 2.0))
    return seconds, int(seconds * 12.5) + 2


def acoustic_check(wav, sample_rate: int, text: str, code_count: int | None, budget: int) -> dict:
    import numpy as np
    a = np.asarray(wav, dtype=np.float32)
    if a.ndim != 1 or not a.size or not np.all(np.isfinite(a)):
        raise QualityError('配音为空、声道异常或包含无效样本')
    duration = len(a) / sample_rate
    maximum, _ = speech_budget(text)
    reasons = []
    if duration > maximum or duration < max(.35, len(normalized(text)) / 14):
        reasons.append('文本与配音时长不匹配')
    if code_count is not None and code_count >= budget - 1:
        reasons.append('语音生成达到长度预算，可能未自然结束')
    block = max(1, sample_rate // 20)
    padded = np.pad(a, (0, (-len(a)) % block))
    rms = np.sqrt(np.mean(padded.reshape(-1,block)**2,axis=1))
    threshold = max(.002, float(np.percentile(rms,90)) * .12)
    active = rms > threshold
    longest = current = 0
    for value in active:
        current = 0 if value else current + 1
        longest = max(longest,current)
    if not np.any(active) or float(np.mean(active)) < .28:
        reasons.append('有效声音比例过低')
    if longest * .05 > 2.5:
        reasons.append('配音含异常长低能量间隔')
    report = {'duration':duration,'max_seconds':maximum,'code_count':code_count,'token_budget':budget,
              'active_fraction':float(np.mean(active)), 'longest_quiet_seconds':longest*.05,'reasons':reasons}
    return report


def transcribe(wav: Path, binary: Path, model: Path) -> dict:
    if not binary.is_file() or not model.is_file():
        raise QualityError('离线语音质检未配置：请设置 Whisper 可执行文件和模型路径')
    prefix = wav.with_suffix('.asr')
    actual = Path(str(prefix) + '.json')
    if actual.exists():
        raise QualityError('识别缓存已存在，请使用新的尝试目录，禁止混用旧结果')
    result = subprocess.run([str(binary),'-m',str(model),'-f',str(wav),'-l','zh','-t','6',
                             '-ng','-np','-bo','1','-bs','1','-mc','0','-ojf','-of',str(prefix)],
                            capture_output=True,text=True,timeout=180)
    # whisper-cli can return zero for invalid arguments; require actual output.
    if result.returncode or not actual.is_file():
        raise QualityError('离线语音识别失败，未通过质检')
    data = json.loads(actual.read_text(encoding='utf-8'))
    if not data.get('transcription'):
        raise QualityError('未识别出有效配音')
    return data


def align_text(text: str, data: dict, duration: float) -> dict:
    expected = normalized(text)
    recognized = ''; times = []
    for segment in data['transcription']:
        # Token timestamps may split a Chinese character into invalid byte
        # fragments. Prefer them only when they reconstruct the segment text.
        tokens = [t for t in segment.get('tokens',[]) if not t['text'].startswith('[_')]
        token_text = ''.join(normalized(t['text']) for t in tokens)
        segment_text = normalized(segment['text'])
        if token_text == segment_text:
            for token in tokens:
                value = normalized(token['text']); start=token['offsets']['from']/1000; end=token['offsets']['to']/1000
                times.extend([start+(end-start)*i/max(1,len(value)) for i in range(len(value))])
        else:
            start=segment['offsets']['from']/1000; end=segment['offsets']['to']/1000
            times.extend([start+(end-start)*i/max(1,len(segment_text)) for i in range(len(segment_text))])
        recognized += segment_text
    matcher = difflib.SequenceMatcher(None,expected,recognized,autojunk=False)
    anchors = {}; edit_count=0; inserted=0; trailing=0; max_gap=0
    for tag,a,b,c,d in matcher.get_opcodes():
        if tag == 'equal':
            anchors.update({i: times[c+i-a] for i in range(a,b)})
        else:
            edit_count += max(b-a,d-c); max_gap=max(max_gap,b-a,d-c)
            if tag == 'insert':
                inserted += d-c
                if a in (0,len(expected)): trailing += d-c
    error_rate = edit_count / max(1,len(expected))
    expected_sounds=lazy_pinyin(expected,errors=lambda value:list(value))
    actual_sounds=lazy_pinyin(recognized,errors=lambda value:list(value))
    sound_matcher=difflib.SequenceMatcher(None,expected_sounds,actual_sounds,autojunk=False)
    sound_edits=sum(max(b-a,d-c) for tag,a,b,c,d in sound_matcher.get_opcodes() if tag!='equal')
    # Exact pronunciation matches also anchor homophones and Traditional Chinese
    # ASR output. Character-level CER alone must not reject identical speech.
    for tag,a,b,c,d in sound_matcher.get_opcodes():
        if tag=='equal': anchors.update({i:times[c+i-a] for i in range(a,b)})
    # A small paragraph-level CER can hide a whole corrupted phrase. Inspect
    # local edits as well, while allowing exact homophone transcription errors.
    phrase_errors=[]
    for tag,a,b,c,d in matcher.get_opcodes():
        if tag=='replace' and max(b-a,d-c)>=2:
            left=lazy_pinyin(expected[a:b],errors=lambda value:list(value))
            right=lazy_pinyin(recognized[c:d],errors=lambda value:list(value))
            local_edits=sum(max(j-i,l-k) for t,i,j,k,l in difflib.SequenceMatcher(None,left,right,autojunk=False).get_opcodes() if t!='equal')
            if local_edits>=1: phrase_errors.append([a,b,c,d])
        elif tag=='delete' and b-a>=3:
            phrase_errors.append([a,b,c,d])
    reasons = []
    if not anchors or (sound_edits and (error_rate > .22 or max_gap > 8)):
        reasons.append('朗读与文案匹配不足')
    if inserted > max(2,len(expected)*.06) or trailing >= 2:
        reasons.append('检测到疑似额外朗读或重复')
    if phrase_errors or sound_edits/max(1,len(expected_sounds))>.14:
        reasons.append('检测到局部词句发音偏离或漏读')
    if times and (min(times)<0 or max(times)>duration+.15):
        reasons.append('识别时间超出配音范围')
    # Interpolate only unmatched characters between nearby ASR anchors, never
    # spread every caption proportionally over an entire TTS segment.
    char_times=[]
    keys=sorted(anchors)
    for i in range(len(expected)):
        if i in anchors: value=anchors[i]
        else:
            left=max((k for k in keys if k<i),default=-1)
            right=min((k for k in keys if k>i),default=len(expected))
            lo=anchors.get(left,0.0); hi=anchors.get(right,duration)
            value=lo+(hi-lo)*(i-left)/(right-left)
        value=max(char_times[-1] if char_times else 0.0,min(duration,max(0.,value)))
        char_times.append(value)
    return {'recognized_text':''.join(s['text'] for s in data['transcription']),
            'error_rate':error_rate,'phonetic_error_rate':sound_edits/max(1,len(expected_sounds)),
            'phrase_error_spans':phrase_errors,'inserted_chars':inserted,'trailing_extra_chars':trailing,
            'char_times':char_times,'reasons':reasons,'alignment':'ASR anchors (approximate)'}
