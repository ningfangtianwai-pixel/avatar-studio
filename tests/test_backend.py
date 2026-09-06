"""Regression tests use synthetic media and a private temporary database only."""
import io
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

_temp = tempfile.TemporaryDirectory(prefix="avatar-studio-tests-")
os.environ["AVATAR_STUDIO_RUNTIME"] = str(Path(_temp.name) / "runtime")
os.environ["AVATAR_STUDIO_MEDIA_ROOT"] = str(Path(_temp.name) / "media")
from backend import app as studio
from backend.pipeline import _run
from fastapi.testclient import TestClient


def wav_bytes():
    target = io.BytesIO()
    with wave.open(target, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\0\0" * 24000)
    return target.getvalue()


class StudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = patch.object(studio, "worker_loop", return_value=None)
        cls.worker.start()
        cls.client = TestClient(studio.app, base_url="http://127.0.0.1:8765", headers={"X-Avatar-Studio": "1", "Origin": "http://localhost:3000"})
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.worker.stop()
        _temp.cleanup()

    def test_cross_origin_write_rejected(self):
        response = self.client.post("/api/jobs", json={}, headers={"Origin": "https://untrusted.example"})
        self.assertEqual(response.status_code, 403)

    def test_plain_form_without_header_rejected(self):
        response = self.client.post("/api/voices", data={"name": "test"}, headers={"X-Avatar-Studio": ""})
        self.assertEqual(response.status_code, 403)

    def test_rebinding_host_rejected(self):
        self.assertEqual(self.client.get("/api/health", headers={"Host": "untrusted.example"}).status_code, 400)

    def test_oversize_before_upload_parse(self):
        response = self.client.post("/api/voices", content=b"x", headers={"Content-Length": str(300 * 1024 * 1024)})
        self.assertEqual(response.status_code, 413)

    def test_blank_script_rejected(self):
        response = self.client.post("/api/jobs", json={"title": "  ", "script_text": "  ", "avatar_id": "x", "voice_id": "x"})
        self.assertEqual(response.status_code, 422)

    def test_invalid_media_rejected_without_orphan(self):
        before = set(studio.VOICE_DIR.iterdir())
        response = self.client.post("/api/voices", data={"name": "test", "ref_text": "test"}, files={"file": ("test.wav", b"invalid", "audio/wav")})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(set(studio.VOICE_DIR.iterdir()), before)

    def test_media_and_job_lifecycle(self):
        response = self.client.post("/api/voices", data={"name": "Test voice", "ref_text": "A synthetic test."}, files={"file": ("test.wav", wav_bytes(), "audio/wav")})
        self.assertEqual(response.status_code, 201, response.text)
        voice_id = response.json()["id"]
        video = Path(_temp.name) / "test.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=128x128:r=25:d=1", "-c:v", "libx264", str(video)], check=True)
        response = self.client.post("/api/avatars", data={"name": "Test avatar"}, files={"file": ("test.mp4", video.read_bytes(), "video/mp4")})
        self.assertEqual(response.status_code, 201, response.text)
        avatar_id = response.json()["id"]
        response = self.client.post("/api/jobs", json={"title": "Test", "script_text": "Synthetic test.", "avatar_id": avatar_id, "voice_id": voice_id})
        self.assertEqual(response.status_code, 201, response.text)
        job_id = response.json()["id"]
        self.assertEqual(self.client.post(f"/api/voices/{voice_id}/normalize").status_code, 409)
        self.assertEqual(self.client.delete(f"/api/avatars/{avatar_id}").status_code, 409)
        self.assertEqual(self.client.post(f"/api/jobs/{job_id}/cancel").status_code, 200)
        self.assertEqual(self.client.post(f"/api/jobs/{job_id}/retry").status_code, 200)
        self.assertEqual(self.client.post(f"/api/jobs/{job_id}/cancel").status_code, 200)
        voice_response = self.client.get(f"/api/media/voice/{voice_id}", headers={"Range": "bytes=0-43"})
        self.assertEqual(voice_response.status_code, 206)
        self.assertEqual(len(voice_response.content), 44)
        self.assertEqual(self.client.delete(f"/api/avatars/{avatar_id}").status_code, 200)
        self.assertEqual(self.client.post(f"/api/jobs/{job_id}/retry").status_code, 409)
        self.assertTrue(list(studio.TRASH_DIR.glob(f"*{avatar_id}*")))

    def test_second_worker_rejected_before_recovery(self):
        with self.assertRaisesRegex(RuntimeError, "已有后台进程"):
            with TestClient(studio.app, base_url="http://127.0.0.1:8765"):
                pass

    def test_process_timeout_and_shutdown(self):
        log = Path(_temp.name) / "process.log"
        for stop, timeout in [(None, 0.1), (threading.Event(), 10)]:
            if stop is not None:
                stop.set()
            with self.assertRaisesRegex(RuntimeError, "中断或阶段超时"):
                _run([sys.executable, "-c", "import time;time.sleep(60)"], log, timeout=timeout, stop_event=stop)

    def test_missing_log_does_not_read_other_files(self):
        self.assertEqual(self.client.get("/api/jobs/nonexistent/log").status_code, 404)

    def test_retry_uses_a_fresh_attempt_directory(self):
        from types import SimpleNamespace
        paths = []
        with studio.connect() as db:
            db.execute("INSERT INTO avatars VALUES(?,?,?,?,?,?,?)", ("worker-avatar", "Test", "/synthetic.mp4", None, 0, studio.now(), None))
            db.execute("INSERT INTO voices VALUES(?,?,?,?,?,?,?,?)", ("worker-voice", "Test", "/synthetic.wav", "/synthetic.wav", "Test", 0, studio.now(), None))
            db.execute("INSERT INTO jobs(id,title,script_text,avatar_id,voice_id,status,stage,created_at) VALUES(?,?,?,?,?,?,?,?)", ("worker-test", "Test", "Test", "worker-avatar", "worker-voice", "queued", "queued", studio.now()))
        def run(**kwargs):
            paths.append(kwargs["work_dir"])
            studio.STOP_EVENT.set()
            return Path(_temp.name) / "result.mp4"
        # setUpClass patches the loop; call its original implementation explicitly.
        for _ in range(2):
            studio.STOP_EVENT.clear()
            with studio.connect() as db:
                db.execute("UPDATE jobs SET status='queued' WHERE id='worker-test'")
            with patch.object(studio, "run_pipeline", side_effect=run), patch.object(studio.subprocess, "run", return_value=SimpleNamespace(stdout="1.0", returncode=0)):
                _real_worker_loop()
        self.assertEqual(len(paths), 2)
        self.assertNotEqual(paths[0], paths[1])
        self.assertEqual(paths[0].parent, paths[1].parent)
        studio.STOP_EVENT.clear()


_real_worker_loop = studio.worker_loop

if __name__ == "__main__":
    unittest.main()
