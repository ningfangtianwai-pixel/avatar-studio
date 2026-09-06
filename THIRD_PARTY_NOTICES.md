# Third-party software and models

The original workstation UI, backend and orchestration code are Copyright (c) 2026 Manny, MIT licensed. This does not replace upstream licenses or transfer rights to personal faces, voices, datasets, model weights, or trademarks.

This source release includes no model weights, original media, generated videos, voice embeddings, task databases, logs, credentials, or installed dependencies. Dependencies are obtained separately; retain their notices when distributing them.

| Component | Upstream terms / release boundary |
| --- | --- |
| MuseTalk code | MIT, Copyright (c) 2024 Tencent Music Entertainment Group. Only a small compatibility patch is included; its full upstream license is in `licenses/MuseTalk.txt`. |
| MuseTalk weights and dependencies | Upstream permits commercial use of its model, but separately requires compliance with Whisper, VAE, DWPose, S3FD and other model terms. Upstream test media is described as non-commercial research only and is excluded. |
| Qwen3-TTS / 0.6B Base | Apache-2.0. Independently installed, not sublicensed as MIT. |
| React / Vinext / Vite / Shadcn UI / Tailwind | See respective package licenses; installed through the committed npm lockfile. Preserve their copyright notices in redistributed builds. |
| FastAPI / Starlette / Uvicorn / python-multipart | MIT / BSD-family licenses as supplied by upstream packages. |
| jieba | MIT. Installed in the separate TTS environment. |
| FFmpeg | License depends on build options; commonly LGPL with GPL components in libx264-enabled builds. Called as an external executable; no FFmpeg binary is bundled. |

Sources verified on 2026-09-06:

- https://github.com/TMElyralab/MuseTalk#disclaimerlicense
- https://github.com/TMElyralab/MuseTalk/blob/main/LICENSE
- https://github.com/QwenLM/Qwen3-TTS
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base
- https://ffmpeg.org/legal.html

The MuseTalk patch targets commit `0a89dec45a0192b824e3cf4daf96c239440c5ed8`. It changes ping-pong endpoints and carries animation phase across chapters. Attribution to Tencent Music Entertainment is retained. Manny is the author of the workstation, not the author of its underlying models.
