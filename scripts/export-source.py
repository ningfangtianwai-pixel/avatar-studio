"""Export an allowlisted, scanned source archive; never package a live workspace."""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("app", "components", "hooks", "lib", "public", "backend", "pipeline",
               "scripts", "tests", "patches", "licenses", "docs")
TOP_FILES = ("README.md", "LICENSE", "SECURITY.md", "THIRD_PARTY_NOTICES.md",
             "package.json", "package-lock.json", "components.json", "tsconfig.json",
             "next.config.ts", "vite.config.ts", "next-env.d.ts", ".gitignore",
             ".oxfmtrc.json", ".oxlintrc.json", "studio.example.json", ".openai/hosting.json")
EXTENSIONS = {".py", ".ts", ".tsx", ".json", ".css", ".md", ".txt", ".sh", ".svg", ".patch"}
EXCLUDED = {".venv", "__pycache__", "node_modules", "runtime", ".git", "release", "dist"}
PATTERNS = (
    (re.compile(r"/home/[a-zA-Z0-9_.-]+/"), "personal Linux home path"),
    (re.compile(r"/mnt/[a-z]/(?!path(?:/|$))[^\s\"'<>]+"), "mounted drive path"),
    (re.compile(r"[A-Z]:[\\/](?:Users|人像)[\\/及]"), "personal Windows path"),
    (re.compile(r"\b(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"), "credential"),
    (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"), "private key"),
)


def collect_sources(root: Path) -> dict[str, bytes]:
    paths = [root / name for name in TOP_FILES if (root / name).is_file()]
    for directory in DIRECTORIES:
        base = root / directory
        if not base.exists():
            continue
        if base.is_symlink():
            raise RuntimeError(f"Refusing symlink directory: {directory}")
        for path in base.rglob("*"):
            relative = path.relative_to(root)
            if any(part in EXCLUDED for part in relative.parts):
                continue
            if path.is_symlink():
                raise RuntimeError(f"Refusing symlink: {relative}")
            if path.is_file() and path.suffix in EXTENSIONS:
                paths.append(path)
    result = {}
    for path in sorted(set(paths)):
        name = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise RuntimeError(f"Unsafe source path: {name}")
        if path.stat().st_size > 2_000_000:
            raise RuntimeError(f"Source file too large: {name}")
        data = path.read_bytes()
        text = data.decode("utf-8")
        if "\0" in text:
            raise RuntimeError(f"Binary file in source list: {name}")
        # This public OS executable is not a user's media/home path.
        scanned_text = text.replace("/mnt/c/Windows/explorer.exe", "<system-explorer>")
        for pattern, reason in PATTERNS:
            if pattern.search(scanned_text):
                # Report only the filename/reason, never matched sensitive content.
                raise RuntimeError(f"Review {name}: {reason}")
        result[name] = data
    if not {"LICENSE", "README.md", "package.json"}.issubset(result):
        raise RuntimeError("Required release files missing")
    return result


def main():
    files = collect_sources(ROOT)
    package = json.loads(files["package.json"])
    if package.get("author") != "Manny" or package.get("license") != "MIT":
        raise RuntimeError("Expected Manny / MIT metadata")
    if not re.fullmatch(r"\d+\.\d+\.\d+", package["version"]):
        raise RuntimeError("Unsafe release version")
    stem = f"avatar-studio-{package['version']}-source"
    output = ROOT / "release"
    output.mkdir(exist_ok=True)
    target = output / f"{stem}.zip"
    staging = target.with_suffix(".zip.partial")
    manifest = {
        "author": "Manny", "license": "MIT (original code only)",
        "version": package["version"],
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "excludes": ["personal media", "models", "databases", "logs", "local configuration", "Git history"],
    }
    try:
        with zipfile.ZipFile(staging, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in files.items():
                info = zipfile.ZipInfo(f"{stem}/{name}")
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
                archive.writestr(info, data)
            archive.writestr(f"{stem}/SOURCE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        staging.replace(target)
    finally:
        staging.unlink(missing_ok=True)
    print(f"Exported {len(files)} source files: {target}")
    print(f"SHA256 {hashlib.sha256(target.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
