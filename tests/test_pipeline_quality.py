"""Fail-closed pipeline acceptance tests; no model or private media required."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from backend.pipeline import validate_video, validate_sync, _run
import sys


class PipelineQualityTests(unittest.TestCase):
    def test_missing_frames_rejected(self):
        streams={'streams':[{'codec_type':'video','avg_frame_rate':'25/1','nb_frames':'99'}]}
        with patch('backend.pipeline.subprocess.run',return_value=SimpleNamespace(stdout=json.dumps(streams))):
            with self.assertRaisesRegex(RuntimeError,'帧数'): validate_video(Path('synthetic.mp4'),100)

    def test_wrong_audio_duration_rejected(self):
        streams={'streams':[{'codec_type':'video','avg_frame_rate':'25/1','nb_frames':'100'},
                            {'codec_type':'audio','duration':'5'}]}
        with patch('backend.pipeline.subprocess.run',return_value=SimpleNamespace(stdout=json.dumps(streams))):
            with self.assertRaisesRegex(RuntimeError,'长度'): validate_video(Path('synthetic.mp4'),100)

    def test_sync_rejects_uncertain_or_shifted_windows(self):
        for report in [{}, {'global':{'offset_frames':0},'windows':[{'offset_frames':5,'confidence':6}]},
                       {'global':{'offset_frames':0},'windows':[{'offset_frames':0,'confidence':float('nan')}]},
                       {'global':{'offset_frames':0},'windows':[{'offset_frames':0,'confidence':1}]}]:
            with self.assertRaises(RuntimeError): validate_sync(report)
        validate_sync({'global':{'offset_frames':1},'windows':[{'offset_frames':1,'confidence':6}]})

    def test_failure_report_reaches_user(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'quality.json').write_text(json.dumps({'error':'配音质检未通过'}))
            with self.assertRaisesRegex(RuntimeError,'配音质检未通过'):
                _run([sys.executable,'-c','raise SystemExit(1)'],p/'log',failure_report=p/'quality.json')


if __name__ == '__main__': unittest.main()
