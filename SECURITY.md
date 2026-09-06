# Local security boundary

This release is a single-user Linux/WSL desktop application. Bind both services to loopback and use one backend process. It has no account system, tenant isolation, public authentication, or protection against other programs already running as the same OS user. Origin/Host checks and the `X-Avatar-Studio: 1` header prevent ordinary cross-site browser writes; the header is not a password or an authentication token.

Do not expose this service through a public tunnel or bind it to a LAN address. Deploying it as a shared web service requires a separate authentication, authorization, quota and storage design.

Files in `runtime/`, the configured media folder, and `studio.local.json` are private. Logs can contain scripts, reference transcripts and paths. Never attach these folders to an issue or publish the live project directory wholesale. Use `python3 scripts/export-source.py` to produce the allowlisted source-only archive.

Uploads are limited to 256 MB per request/file. Video references are at most 120 seconds and 4K before conversion; audio references at most 60 seconds. Jobs accept at most 10,000 characters and the create queue is limited to 32 active jobs. Generation stages time out after six hours. These are operational limits, not a sandbox for hostile media; upload trusted references and keep FFmpeg and model runtimes maintained.

Never download arbitrary pickle/checkpoint files from unknown sources. Upstream inference loads trusted model files outside this repository.

When reporting an issue, provide a minimal synthetic reproduction and dependency versions. Do not include a person's original face, voice, voice embeddings, private script, or environment credentials. Public dependency advisories can be filed with their advisory URLs; coordinate unpatched vulnerabilities privately with the project maintainer.
