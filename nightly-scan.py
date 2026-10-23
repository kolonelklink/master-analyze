#!/usr/bin/env python3
"""
nightly-scan.py — walk ~/Music and do two independent passes:

  1. LYRICS   — write a .txt of the words next to any track that has no lyrics
                file yet (transcribe-lyrics.py, local Whisper, no LLM needed).
  2. MASTERING — run master-analyze.py on any track with no report next to it,
                skipping folders that aren't this user's own productions.

The lyrics pass runs first and is deliberately independent of vLLM: it needs
only the GPU, so a night where gemma-heretic is down still gets lyrics.

Intended to run unattended via a systemd timer; logs everything so a failed
night is diagnosable after the fact instead of silently missing tracks.
"""

import json
import subprocess
import sys
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

MUSIC_ROOT = Path("/home/klink/Music")
SCRIPT_DIR = Path(__file__).resolve().parent
MASTER_ANALYZE = SCRIPT_DIR / "master-analyze.py"
TRANSCRIBE_LYRICS = SCRIPT_DIR / "transcribe-lyrics.py"
PYTHON = SCRIPT_DIR / ".venv" / "bin" / "python"
VLLM_HOST = "http://10.43.89.157:8000"
SEEN_MANIFEST = SCRIPT_DIR / ".nightly-scan-seen.json"

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif", ".m4a"}

# Top-level /home/klink/Music folders that aren't this user's own releases
# (commercial tracks / other bands' bootlegs) — never send these to the LLM.
EXCLUDE_DIRS = {"Beastie Boys", "This End Up"}

# Per-artist genre context passed to master-analyze.py. Anything not listed
# here falls back to DEFAULT_GENRE — add a line when a new persona shows up.
GENRE_MAP = {
    "Daisy Kink": "metalcore / heavy",
    "Sydney Scott": "Hardcore country rock",
    "KolonelKlink": "alternative rock / singer-songwriter ballads",
}
DEFAULT_GENRE = "metalcore / heavy"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def vllm_reachable() -> bool:
    try:
        urllib.request.urlopen(f"{VLLM_HOST}/v1/models", timeout=10)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        log(f"vLLM health check failed: {e}")
        return False


def already_analyzed(audio_path: Path) -> bool:
    return any(audio_path.parent.glob(f"{audio_path.stem}_master_report_*.md"))


def load_seen() -> set[str]:
    if not SEEN_MANIFEST.exists():
        return set()
    return set(json.loads(SEEN_MANIFEST.read_text()))


def save_seen(seen: set[str]) -> None:
    SEEN_MANIFEST.write_text(json.dumps(sorted(seen), indent=2))


def genre_for(audio_path: Path) -> str:
    top_level = audio_path.relative_to(MUSIC_ROOT).parts[0]
    return GENRE_MAP.get(top_level, DEFAULT_GENRE)


def run_lyrics_pass() -> int:
    """Transcribe any track that still has no lyrics file. Returns failure count.

    Whisper-only, so this runs before (and independently of) the vLLM check.
    transcribe-lyrics.py surveys first and only loads the model when there is
    actually something to do, so a quiet night costs a directory walk.
    """
    log("Lyrics pass starting")
    result = subprocess.run(
        [str(PYTHON), str(TRANSCRIBE_LYRICS)],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.strip():
            log(f"  lyrics | {line}")
    if result.returncode != 0:
        log("Lyrics pass FAILED (continuing to mastering pass anyway)")
        log(result.stderr[-2000:])
        return 1
    log("Lyrics pass complete")
    return 0


def main() -> int:
    log("Nightly scan starting")

    lyrics_failures = run_lyrics_pass()

    if not vllm_reachable():
        log("Aborting mastering pass: gemma-heretic vLLM endpoint unreachable at "
            f"{VLLM_HOST} — check `kubectl get pods -l app=gembrain-prompt-vllm`")
        return 1

    seen = load_seen()

    candidates = []
    for path in sorted(MUSIC_ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        if path.name.startswith("._"):
            continue  # AppleDouble sidecar junk from exFAT sync, not real audio
        top_level = path.relative_to(MUSIC_ROOT).parts[0]
        if top_level in EXCLUDE_DIRS:
            continue
        key = str(path)
        if key in seen or already_analyzed(path):
            continue
        candidates.append(path)

    if not candidates:
        log("No new tracks to analyze")
        return 1 if lyrics_failures else 0

    log(f"Found {len(candidates)} new track(s) to analyze")

    failures = 0
    for path in candidates:
        genre = genre_for(path)
        log(f"Analyzing: {path}  [genre: {genre}]")
        result = subprocess.run(
            [str(PYTHON), str(MASTER_ANALYZE), str(path), "--genre", genre],
            env={"MASTER_ANALYZE_HOST": VLLM_HOST, "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            failures += 1
            log(f"FAILED: {path}  (will retry next run)")
            log(result.stdout[-2000:])
            log(result.stderr[-2000:])
        else:
            log(f"OK: {path}")
            seen.add(str(path))
            save_seen(seen)

    log(f"Nightly scan complete — {len(candidates) - failures} ok, {failures} failed")
    return 1 if (failures or lyrics_failures) else 0


if __name__ == "__main__":
    sys.exit(main())
