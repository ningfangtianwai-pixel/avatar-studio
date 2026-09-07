"""Synthetic quality regression; run with the isolated TTS Python environment."""
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline'))
try:
    import numpy as np
    import jieba
    import soundfile
    import pypinyin
except ImportError as error:
    raise unittest.SkipTest('Voice quality tests require the isolated TTS environment') from error
from voice_quality import clean_script, acoustic_check, align_text, normalized, transcribe, QualityError
from generate_voice import build_segments, write_subtitles, split_caption, wrap_caption
from prepare_chapters import chapter_frames
import generate_voice


def asr(text, start=0., end=2.):
    return {'transcription':[{'text':text,'offsets':{'from':start*1000,'to':end*1000}}]}


class QualityTests(unittest.TestCase):
    def test_format_cleaning(self):
        self.assertEqual(clean_script('# 标题\n\n---\n**你好**，`AI`。\n[说明](https://example.com)'), '标题\n\n你好，AI。\n说明')
        self.assertFalse(any('*' in x['speech_text'] for x in build_segments('**欢迎使用人工智能。**'*20)))
        self.assertIn('Chat G P T', build_segments('ChatGPT 是一种工具。')[0]['speech_text'])

    def test_original_failure_signature_rejected(self):
        # Synthetic 164-second low-energy tail, never use private media in tests.
        a=np.ones(3930240,dtype=np.float32)*.003
        a[:24000*5]=np.sin(np.arange(24000*5)*.04)*.12
        report=acoustic_check(a,24000,'欢迎使用新的数字人工作台。',2047,2048)
        self.assertTrue(report['reasons'])

    def test_nonfinite_silence_and_budget(self):
        with self.assertRaises(QualityError): acoustic_check(np.array([np.nan]),24000,'你好',None,100)
        self.assertTrue(acoustic_check(np.zeros(24000),24000,'你好',None,100)['reasons'])
        self.assertTrue(acoustic_check(np.ones(48000)*.1,24000,'欢迎使用人工智能',99,100)['reasons'])

    def test_long_internal_pause_rejected(self):
        a=np.concatenate([np.ones(24000)*.1,np.zeros(72000),np.ones(24000)*.1])
        self.assertIn('配音含异常长低能量间隔',acoustic_check(a,24000,'欢迎使用人工智能。',None,500)['reasons'])

    def test_extra_ending_rejected(self):
        result=align_text('欢迎使用人工智能',asr('欢迎使用人工智能然后'),2)
        self.assertIn('检测到疑似额外朗读或重复',result['reasons'])

    def test_small_homophone_error_tolerated(self):
        result=align_text('欢迎使用人工智能工作台',asr('欢迎使用人工只能工作台'),2)
        self.assertEqual(result['reasons'],[])
        self.assertEqual(len(result['char_times']),11)

    def test_traditional_transcription_with_same_sounds(self):
        result=align_text('欢迎使用农业智能系统',asr('歡迎使用農業智能系統'),3)
        self.assertEqual(result['reasons'],[])
        self.assertEqual(len(result['char_times']),10)
        self.assertEqual(result['phonetic_error_rate'],0)

    def test_whole_homophone_phrase_tolerated(self):
        result=align_text('我们今天讨论一下种植指导和基本情况',asr('我们今天讨论一下众志知道和基本情况'),4)
        self.assertEqual(result['phrase_error_spans'],[])

    def test_local_phrase_corruption_rejected(self):
        # Generic synthetic phrase, not private user narration.
        result=align_text('今天我们一起讨论设备维护的方法最后需要检查电源',asr('今天我们一起讨论设备维护的方法最后需要检讨演员'),4)
        self.assertIn('检测到局部词句发音偏离或漏读',result['reasons'])

    def test_two_character_word_with_one_wrong_sound_rejected(self):
        result=align_text('我们现在需要认真检查工厂周围的环境',asr('我们现在需要认真检查工厂周围的幻觉'),4)
        self.assertIn('检测到局部词句发音偏离或漏读',result['reasons'])

    def test_number_normalization(self):
        self.assertEqual(normalized('100个客户和200个客户'),normalized('一百个客户和两百个客户').replace('两','二'))

    def test_subtitles_require_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(QualityError):
                write_subtitles(Path(tmp)/'x.srt',[{'display_text':'你好','start':0,'speech_end':4}])
            write_subtitles(Path(tmp)/'x.srt',[{'display_text':'你好','start':0,'speech_end':4,'char_times':[1,2]}])
            self.assertIn('00:00:01,000', (Path(tmp)/'x.srt').read_text())

    def test_caption_closing_quotes_stay_with_sentence(self):
        groups=split_caption('他说：“现在可以了吗？”然后继续介绍。')
        self.assertFalse(any(g.startswith('”') for g in groups))
        self.assertTrue(any('？”' in g for g in groups))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'quotes.srt'
            timeline=[{'display_text':'他说：“好。','start':0,'speech_end':2,'char_times':[0,.5,1]},
                      {'display_text':'”随后继续。','start':2.2,'speech_end':4,'char_times':[0,.4,.8,1.2]}]
            write_subtitles(path,timeline)
            self.assertIn('好。”',path.read_text())
            self.assertNotIn('\n”随后',path.read_text())
            self.assertEqual(timeline[1]['display_text'],'”随后继续。')

    def test_caption_does_not_prefer_orphan_particle(self):
        wrapped=wrap_caption('也可能只是原先的方案，换了一种更可靠的方法。')
        self.assertNotIn('\n了',wrapped)
        self.assertNotIn('绕不\n过去',wrap_caption('有一个问题绕不过去：谁来回答？'))

    def test_missing_asr_fails_closed(self):
        with self.assertRaises(QualityError): transcribe(Path('/missing.wav'),Path('/missing-bin'),Path('/missing-model'))

    def test_generation_retry_then_accepts(self):
        self.run_fake_generation(always_bad=False)

    def test_generation_exhaustion_never_publishes_audio(self):
        self.run_fake_generation(always_bad=True)

    def run_fake_generation(self, always_bad):
        text='欢迎使用人工智能工作台。'
        calls=[]
        class Fake:
            def __init__(self): self.model=SimpleNamespace(generate=lambda *a,**k: ([np.zeros((20,16))],None))
            def create_voice_clone_prompt(self,**kwargs): return None
            def generate_voice_clone(self,**kwargs):
                calls.append(kwargs);self.model.generate()
                return [np.zeros(96000) if always_bad or len(calls)==1 else np.ones(96000)*.1],24000
        fake=Fake()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'script.txt').write_text(text);(root/'binary').touch();(root/'model').touch()
            argv=['generate_voice','--script',str(root/'script.txt'),'--ref-audio',str(root/'ref.wav'),
                  '--ref-text',text,'--output-dir',str(root/'out'),'--whisper-bin',str(root/'binary'),
                  '--whisper-model',str(root/'model')]
            module=SimpleNamespace(Qwen3TTSModel=SimpleNamespace(from_pretrained=lambda *a,**k:fake))
            torch_stub=SimpleNamespace(manual_seed=lambda _:None,float32=np.float32,
                backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))))
            with patch.dict(sys.modules,{'qwen_tts':module,'torch':torch_stub}),patch.object(sys,'argv',argv),patch.object(generate_voice,'transcribe',return_value=asr(text,end=4)):
                if always_bad:
                    with self.assertRaises(QualityError): generate_voice.main()
                else: generate_voice.main()
            report=json.loads((root/'out/quality.json').read_text())
            self.assertEqual(len(calls),2)
            self.assertEqual(report['status'],'failed' if always_bad else 'passed')
            self.assertEqual((root/'out/voice.wav').exists(),not always_bad)

    def test_frame_clock_no_drift_and_hard_limit(self):
        spans=chapter_frames([{'end':163.94},{'end':327.88},{'end':551.648}],13239549,24000,120)
        self.assertTrue(all(b-a <= 3000 for a,b in spans))
        self.assertEqual(spans[0][0],0)
        self.assertEqual(spans[-1][1],13792)
        self.assertTrue(all(a[1]==b[0] for a,b in zip(spans,spans[1:])))
        self.assertEqual(sum(b-a for a,b in spans),13792)


if __name__ == '__main__': unittest.main()
