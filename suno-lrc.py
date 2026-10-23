#!/usr/bin/env python3
"""Write a synced .lrc next to every Suno .wav, using Suno's OWN lyrics text.

Suno exports carry the exact lyrics (SunoExplorer: <name>.txt; Suno
Downloads: json metadata.prompt) but no timing. Whisper supplies only the
timing: its word timestamps are aligned to Suno's words with difflib, and each
lyric line takes the time of its first matched word. Unmatched lines are
interpolated between neighbours. Whisper's own (often wrong) words never reach
the .lrc.

Lines that align too poorly (< MIN_MATCH of lyric words found in the audio,
e.g. Suno ignored or rewrote the lyrics) get an UNSYNCED .lrc - plain lines,
no timestamps - rather than confidently wrong ones. Every result goes in
REPORT so low-match songs can be reviewed.

  suno-lrc.py --limit 5          # try a few
  suno-lrc.py --only "Backroom"  # one song
  suno-lrc.py                    # everything without an .lrc yet (resumable)
"""
import argparse
import ast
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOTS = [Path.home() / "Downloads/SunoExplorer", Path.home() / "Downloads/Suno Downloads"]
REPORT = Path(__file__).resolve().parent / "suno-lrc-report.tsv"
MODEL_SIZE = "large-v3"
MIN_MATCH = 0.35


def _bootstrap_cuda12() -> None:
    # Same as transcribe-lyrics.py: ctranslate2 needs the venv's CUDA 12 libs.
    if os.environ.get("_LYRICS_CUDA_BOOTSTRAPPED"):
        return
    site = Path(__file__).resolve().parent / ".venv/lib"
    libs = [str(p) for p in sorted(site.glob("python3.*/site-packages/nvidia/*/lib")) if p.is_dir()]
    if not libs:
        return
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(libs + [os.environ.get("LD_LIBRARY_PATH", "")])
    os.environ["_LYRICS_CUDA_BOOTSTRAPPED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_bootstrap_cuda12()

SECTION = re.compile(
    r"^\((?:verse|pre-?chorus|chorus|post-?chorus|bridge|intro|outro|hook|refrain|"
    r"interlude|instrumental|break|breakdown|drop|solo|end|fade)\b[^)]*\)$", re.I)


def lyrics_for(wav: Path) -> str | None:
    txt = wav.with_suffix(".txt")
    if txt.exists():
        return txt.read_text(encoding="utf-8", errors="replace")
    js = wav.with_suffix(".json")
    if js.exists():
        d = json.loads(js.read_text(encoding="utf-8"))
        if d.get("prompt"):
            return d["prompt"]
        m = d.get("metadata")
        if isinstance(m, str):
            try:
                m = ast.literal_eval(m)
            except (ValueError, SyntaxError):
                m = {}
        if isinstance(m, dict) and m.get("prompt"):
            return m["prompt"]
    return None


def sung_lines(raw: str) -> list[str]:
    """Drop Suno's directions ([Intro], [BPM: 85], (Verse 1)...); keep sung text."""
    out = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or re.fullmatch(r"\[[^\]]*\]", s) or SECTION.match(s):
            continue
        s = re.sub(r"\[[^\]]*\]", "", s).strip()  # inline tags; "[Chorus]:" leaves ":"
        if re.search(r"[A-Za-z0-9]", s):
            out.append(s)
    return out


def is_parenthetical(line: str) -> bool:
    return bool(re.fullmatch(r"\(.*\)", line.strip()))


def norm(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower().replace("’", "'").replace("'", ""))


def align(lines: list[str], words: list[tuple[str, float]]):
    lyric_tok, owner = [], []
    for i, line in enumerate(lines):
        for w in line.split():
            n = norm(w)
            if n:
                lyric_tok.append(n)
                owner.append(i)
    heard = [norm(w) for w, _ in words]
    sm = difflib.SequenceMatcher(None, lyric_tok, heard, autojunk=False)
    tok_time = [None] * len(lyric_tok)
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            tok_time[a + k] = words[b + k][1]
    # A fully parenthesized line is either a sung backing vocal or a stage
    # direction like "(heavy erotic female moaning)". Text can't tell them
    # apart, the audio can: drop it unless Whisper heard any of its words.
    heard_line = {owner[i] for i, t in enumerate(tok_time) if t is not None}
    keep = [i for i, l in enumerate(lines) if not is_parenthetical(l) or i in heard_line]
    if len(keep) < len(lines):
        lines = [lines[i] for i in keep]
        return align(lines, words)
    matched = sum(t is not None for t in tok_time) / max(1, len(lyric_tok))

    line_time = [None] * len(lines)
    for idx, t in enumerate(tok_time):
        if t is not None and line_time[owner[idx]] is None:
            line_time[owner[idx]] = t
    known = [i for i, t in enumerate(line_time) if t is not None]
    if not known:
        return lines, None, matched
    for i in range(len(lines)):  # fill gaps by linear interpolation, clamp the ends
        if line_time[i] is not None:
            continue
        prev = max((k for k in known if k < i), default=None)
        nxt = min((k for k in known if k > i), default=None)
        if prev is None:
            line_time[i] = line_time[nxt]
        elif nxt is None:
            line_time[i] = line_time[prev]
        else:
            line_time[i] = line_time[prev] + (line_time[nxt] - line_time[prev]) * (i - prev) / (nxt - prev)
    for i in range(1, len(lines)):  # never go backwards
        line_time[i] = max(line_time[i], line_time[i - 1])
    return lines, line_time, matched


def stamp(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"[{int(m):02d}:{s:05.2f}]"


def write_lrc(wav: Path, lines, times, duration):
    head = [f"[ti:{re.sub(r' \([0-9a-f]{8}\)$', '', wav.stem)}]",
            "[by:Suno lyrics, Whisper-aligned]"]
    if duration:
        head.append(f"[length:{int(duration // 60):02d}:{int(duration % 60):02d}]")
    body = [f"{stamp(t)}{l}" for t, l in zip(times, lines)] if times else lines
    tmp = wav.with_suffix(".lrc.tmp")
    tmp.write_text("\n".join(head + body) + "\n", encoding="utf-8")
    tmp.replace(wav.with_suffix(".lrc"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    wavs = sorted(p for r in ROOTS for p in r.rglob("*.wav") if not p.name.startswith("._"))
    if a.only:
        wavs = [p for p in wavs if a.only.lower() in str(p).lower()]
    todo = [p for p in wavs if a.force or not p.with_suffix(".lrc").exists()]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(wavs)} wavs, {len(todo)} to do", flush=True)

    from faster_whisper import WhisperModel
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    new_report = not REPORT.exists()
    rep = REPORT.open("a", encoding="utf-8")
    if new_report:
        rep.write("status\tmatch\tlines\tseconds\tpath\n")

    for n, wav in enumerate(todo, 1):
        raw = lyrics_for(wav)
        lines = sung_lines(raw) if raw else []
        if not lines:
            rep.write(f"no-lyrics\t\t0\t\t{wav}\n"); rep.flush()
            print(f"[{n}/{len(todo)}] no lyrics: {wav.name}", flush=True)
            continue
        t0 = time.time()
        try:
            segs, info = model.transcribe(
                str(wav), language="en", word_timestamps=True, vad_filter=False,
                condition_on_previous_text=False, beam_size=5,
                initial_prompt=" ".join(lines)[:600])
            words = [(w.word, w.start) for s in segs for w in (s.words or [])]
        except Exception as e:  # noqa: BLE001 - one bad file must not stop a 2,700-song batch
            rep.write(f"error\t\t{len(lines)}\t\t{wav}\t{e}\n"); rep.flush()
            print(f"[{n}/{len(todo)}] ERROR {wav.name}: {e}", flush=True)
            continue
        lines, times, matched = align(lines, words)
        synced = times is not None and matched >= MIN_MATCH
        write_lrc(wav, lines, times if synced else None, info.duration)
        status = "synced" if synced else "unsynced"
        rep.write(f"{status}\t{matched:.2f}\t{len(lines)}\t{time.time() - t0:.0f}\t{wav}\n"); rep.flush()
        print(f"[{n}/{len(todo)}] {status} {matched:.0%} {time.time() - t0:.0f}s {wav.name}", flush=True)


if __name__ == "__main__":
    main()
