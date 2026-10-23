#!/usr/bin/env python3
"""Write a lyrics .txt next to every track in ~/Music that doesn't already have one.

Two strategies, cheapest first:

  1. COPY  - if another take of the same song in the same folder already has
             lyrics, reuse them. Suno takes ("Foo", "Foo-2", "Foo-3") share the
             same lyric input, so this is exact, not a guess.
  2. WHISPER - otherwise transcribe with faster-whisper on the GPU.

Matching is by NORMALIZED song name within a single folder, never by folder
alone: a folder like "Adoration" holds several distinct songs, so a folder-wide
copy would staple the wrong lyrics onto most of them.

macOS AppleDouble stubs ("._foo.m4a") are skipped - they carry an audio
extension but contain no audio, and there are ~950 of them in this library.

Usage:
  transcribe-lyrics.py --dry-run          # show the plan, write nothing
  transcribe-lyrics.py                    # do it
  transcribe-lyrics.py --only "Daisy Kink"  # limit to a subtree
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def _bootstrap_cuda12() -> None:
    """Put the venv's bundled CUDA 12 libs on LD_LIBRARY_PATH, then re-exec.

    ctranslate2 links against libcublas.so.12, but this box ships CUDA 13
    (/opt/cuda/lib64/libcublas.so.13) - so the system copy will never satisfy
    it and the GPU path dies with "Library libcublas.so.12 is not found".
    The nvidia-cublas-cu12 / nvidia-cudnn-cu12 wheels in this venv provide the
    right ones, but the loader only reads LD_LIBRARY_PATH at process start,
    hence the re-exec. Doing it here rather than in a wrapper keeps the nightly
    systemd job working without an EnvironmentFile.
    """
    if os.environ.get("_LYRICS_CUDA_BOOTSTRAPPED"):
        return
    site = Path(__file__).resolve().parent / ".venv/lib"
    libs = [str(p) for p in sorted(site.glob("python3.*/site-packages/nvidia/*/lib")) if p.is_dir()]
    if not libs:
        return  # nothing bundled; fall through and let ctranslate2 try the system libs
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(libs + [os.environ.get("LD_LIBRARY_PATH", "")])
    os.environ["_LYRICS_CUDA_BOOTSTRAPPED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_bootstrap_cuda12()

MUSIC_ROOT = Path.home() / "Music"
# Tracks Whisper judged instrumental. Without this memo the nightly job would
# re-decode them every night forever (the two band jams here are 45 min of
# audio between them) and reach the same verdict each time. --retry-instrumental
# clears it.
INSTRUMENTAL_MANIFEST = Path(__file__).resolve().parent / ".transcribe-instrumental.json"
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif", ".m4a"}
LYRIC_EXTS = {".txt", ".lrc"}

# Commercial releases that happen to live in the library. Not our songs, and a
# machine transcription of someone else's record isn't something we want to
# generate - skip the whole artist folder.
SKIP_DIRS = {"Beastie Boys"}

MODEL_SIZE = "large-v3"

# A take suffix ("-2") or a key suffix (".G minor", ".Am", ".C#") distinguishes
# variants of one song, not different songs.
_TAKE = re.compile(r"-\d+$")
_KEY = re.compile(r"\.(?:[A-G](?:#|b|\s?(?:flat|sharp))?\s?(?:major|minor|maj|min|m)?)$", re.I)

# Whisper was trained on subtitle tracks, so on music it emits caption furniture
# that is not a lyric: "🎶 Music Outro 🎶", "[Music]", "Subtitles by ...". These
# score as confident speech (the one observed here had no_speech_prob 0.22), so
# the probability gate below will not catch them - match the text instead.
_JUNK = re.compile(
    # Repeated group, not a single keyword: the real output was "🎶 Music Outro 🎶",
    # which stacks two of them.
    r"^(?:(?:[\W_\s]*(?:music|applause|laughter|singing|instrumental|outro|intro))+[\W_\s]*"
    # The YouTube-caption stock phrases. Whisper emits these over intro/outro
    # instrumental with high confidence, so only a text match catches them.
    r"|thank(?:s|\s+you)?\s+(?:you\s+)?(?:so\s+much\s+|very\s+much\s+)?for\s+watch(?:ing)?.*"
    r"|thank\s+you(?:\s+very\s+much|\s+so\s+much)?[.!]?"
    r"|thanks?\s+for\s+(?:listening|joining).*"
    r"|we'?ll\s+be\s+right\s+back[.!]?"
    r"|(?:please\s+)?(?:subscribe|like\s+and\s+subscribe).*|subtitles?\s+by.*|.*amara\.org.*"
    r"|see\s+you\s+(?:next\s+time|in\s+the\s+next).*"
    r"|[\W_\s]+)$",
    re.I,
)

# A real song runs roughly 8-15 transcribed lines per minute. An 18.8-minute
# band jam that yields 8 lines is instrumental with a few hallucinations in it,
# not a song with sparse lyrics - don't write a lyrics file for it.
MIN_LINES_PER_MIN = 2.0


def song_key(stem: str) -> str:
    s = _TAKE.sub("", stem)
    s = _KEY.sub("", s)
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def has_lyrics(track: Path) -> bool:
    return any(track.with_suffix(e).exists() for e in LYRIC_EXTS)


def load_instrumental() -> set[str]:
    if not INSTRUMENTAL_MANIFEST.exists():
        return set()
    try:
        return set(json.loads(INSTRUMENTAL_MANIFEST.read_text()))
    except (OSError, ValueError):
        return set()


def mark_instrumental(track: Path) -> None:
    known = load_instrumental()
    known.add(str(track))
    INSTRUMENTAL_MANIFEST.write_text(json.dumps(sorted(known), indent=2))


def survey(root: Path, only: str | None):
    """Return (to_copy, to_transcribe) for tracks with no lyrics file."""
    instrumental = load_instrumental()
    lyrics_by_song: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    tracks: list[Path] = []

    for p in root.rglob("*"):
        if not p.is_file() or p.name.startswith("._"):
            continue
        rel = p.relative_to(root)
        if rel.parts and rel.parts[0] in SKIP_DIRS:
            continue
        if only and only.lower() not in str(rel).lower():
            continue
        ext = p.suffix.lower()
        if ext in AUDIO_EXTS:
            tracks.append(p)
        elif ext in LYRIC_EXTS:
            lyrics_by_song[(p.parent, song_key(p.stem))].append(p)

    to_copy, to_transcribe = [], []
    # Two encodings of one track ("Chamber.flac" and "Chamber.mp3") differ only
    # by extension, so both resolve to the same "Chamber.txt". Without this the
    # second transcription silently overwrites the first - same song, so the
    # result is harmless, but it burns a GPU pass for nothing.
    claimed: set[Path] = set()
    for t in sorted(tracks):
        if has_lyrics(t) or str(t) in instrumental:
            continue
        out = t.with_suffix(".txt")
        if out in claimed:
            continue
        siblings = lyrics_by_song.get((t.parent, song_key(t.stem)))
        if siblings:
            to_copy.append((t, sorted(siblings)[0]))
        else:
            to_transcribe.append(t)
        claimed.add(out)
    return len(tracks), to_copy, to_transcribe


def transcribe_all(paths: list[Path], dry_run: bool) -> None:
    from faster_whisper import WhisperModel

    print(f"\nloading {MODEL_SIZE} on cuda ...", flush=True)
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

    for i, track in enumerate(paths, 1):
        print(f"\n[{i}/{len(paths)}] {track.relative_to(MUSIC_ROOT)}", flush=True)
        segments, info = model.transcribe(
            str(track),
            language="en",
            beam_size=5,
            # NO vad_filter. Silero treats dense metalcore instrumentation as
            # non-speech and threw away 100% of "Make Me-2.m4a" (3.7 min of
            # sung vocal). Whisper's own per-segment no_speech_prob below is a
            # far better judge on music.
            vad_filter=False,
            # Stops a hallucinated line from seeding an endless repeat of
            # itself over instrumental passages.
            condition_on_previous_text=False,
        )

        lines: list[str] = []
        dropped_silence = 0
        dropped_repeat = 0
        dropped_junk = 0
        for s in segments:
            text = s.text.strip()
            if not text:
                continue
            if _JUNK.match(text):
                dropped_junk += 1
                continue
            # Confident silence: whisper says "no speech here" AND the decode
            # was poor. Both conditions, so a quiet sung line still survives.
            if s.no_speech_prob > 0.7 and s.avg_logprob < -0.7:
                dropped_silence += 1
                continue
            # Collapse the classic whisper stutter loop.
            if lines and text == lines[-1]:
                dropped_repeat += 1
                continue
            lines.append(text)

        mins = info.duration / 60
        drops = [f"-{n} {label}" for n, label in
                 ((dropped_silence, "silent"), (dropped_repeat, "repeats"), (dropped_junk, "junk")) if n]
        note = f"({', '.join(drops)})" if drops else ""
        density = len(lines) / mins if mins else 0
        if len(lines) < 3 or density < MIN_LINES_PER_MIN:
            print(f"      {mins:.1f} min, {len(lines)} usable lines {note} "
                  f"= {density:.1f}/min - instrumental, recording so we don't retry")
            if not dry_run:
                mark_instrumental(track)
            continue
        out = track.with_suffix(".txt")
        print(f"      {mins:.1f} min, {len(lines)} lines {note} -> {out.name}")
        print(f"      first line: {lines[0][:70]}")
        if not dry_run:
            out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="show the plan, write nothing")
    ap.add_argument("--only", help="limit to paths containing this substring")
    ap.add_argument("--copy-only", action="store_true", help="skip the GPU step")
    ap.add_argument("--retry-instrumental", action="store_true",
                    help="forget the instrumental memo and try those tracks again")
    args = ap.parse_args()

    if args.retry_instrumental:
        INSTRUMENTAL_MANIFEST.unlink(missing_ok=True)
        print("cleared the instrumental memo")

    total, to_copy, to_transcribe = survey(MUSIC_ROOT, args.only)
    print(f"{total} tracks scanned")
    print(f"  {total - len(to_copy) - len(to_transcribe)} already have lyrics")
    print(f"  {len(to_copy)} can copy from another take")
    print(f"  {len(to_transcribe)} need transcription")

    for track, src in to_copy:
        print(f"  copy  {src.name}  ->  {track.with_suffix('.txt').name}")
        if not args.dry_run:
            track.with_suffix(".txt").write_bytes(src.read_bytes())

    if to_transcribe and not args.copy_only:
        transcribe_all(to_transcribe, args.dry_run)

    if args.dry_run:
        print("\n(dry run - nothing written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
