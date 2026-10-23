#!/usr/bin/env python3
"""
master-analyze.py — AI Mastering Analysis Tool
Analyzes audio files and generates mastering reports using Gemma 4 31B via vLLM.

Usage:
    python3 master-analyze.py <audio_file> [options]

Options:
    --model MODEL       vLLM model name (default: gemma-heretic)
    --host HOST         vLLM host (default: $MASTER_ANALYZE_HOST or http://localhost:8000)
    --genre GENRE       Genre override (default: metalcore)
    --output DIR        Directory to save report (default: same as input file)
    --no-save           Print to terminal only, don't save .md file
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile
import textwrap
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

import numpy as np
import librosa
import pyloudnorm as pyln


# ─── ANSI colors ─────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    CYAN   = "\033[36m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    RED    = "\033[31m"
    MAGENTA= "\033[35m"

def banner():
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════╗
║       master-analyze  //  AI Mastering Tool  ║
║       Powered by Gemma 4 31B + vLLM          ║
╚══════════════════════════════════════════════╝{C.RESET}
""")

def section(label):
    print(f"\n{C.BOLD}{C.MAGENTA}▸ {label}{C.RESET}")

def ok(msg):
    print(f"  {C.GREEN}✓{C.RESET}  {msg}")

def info(msg):
    print(f"  {C.DIM}→{C.RESET}  {msg}")

def warn(msg):
    print(f"  {C.YELLOW}⚠{C.RESET}  {msg}")

def err(msg):
    print(f"  {C.RED}✗{C.RESET}  {msg}", file=sys.stderr)


# ─── Audio loading ────────────────────────────────────────────────────────────
def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load audio file, converting via ffmpeg if needed."""
    suffix = path.suffix.lower()
    supported = {".wav", ".flac", ".ogg", ".aiff", ".aif"}

    if suffix not in supported:
        info(f"Converting {suffix} → WAV via ffmpeg...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-ar", "44100",
             "-ac", "2", tmp_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            err("ffmpeg conversion failed:")
            print(result.stderr, file=sys.stderr)
            sys.exit(1)
        y, sr = librosa.load(tmp_path, sr=None, mono=False)
        os.unlink(tmp_path)
    else:
        y, sr = librosa.load(str(path), sr=None, mono=False)

    return y, sr


# ─── Feature extraction ───────────────────────────────────────────────────────
def extract_features(y: np.ndarray, sr: int) -> dict:
    """Extract comprehensive mastering-relevant audio features."""

    # Ensure stereo shape (2, N) and mono (N,)
    if y.ndim == 1:
        y_mono = y
        y_stereo = np.stack([y, y])
    elif y.shape[0] == 2:
        y_stereo = y
        y_mono = librosa.to_mono(y)
    else:
        y_mono = y[0]
        y_stereo = np.stack([y[0], y[0]])

    duration = librosa.get_duration(y=y_mono, sr=sr)

    # ── Loudness (ITU-R BS.1770) ──────────────────────────────────────────────
    meter = pyln.Meter(sr)
    audio_for_lufs = y_stereo.T  # pyloudnorm wants (samples, channels)
    try:
        integrated_lufs = meter.integrated_loudness(audio_for_lufs)
    except Exception:
        integrated_lufs = -99.0

    # ── True Peak ────────────────────────────────────────────────────────────
    true_peak_l = float(20 * np.log10(np.max(np.abs(y_stereo[0])) + 1e-10))
    true_peak_r = float(20 * np.log10(np.max(np.abs(y_stereo[1])) + 1e-10))
    true_peak = max(true_peak_l, true_peak_r)

    # ── RMS ──────────────────────────────────────────────────────────────────
    rms = float(20 * np.log10(np.sqrt(np.mean(y_mono**2)) + 1e-10))

    # ── Dynamic Range (simplified crest factor) ───────────────────────────────
    peak_db = float(20 * np.log10(np.max(np.abs(y_mono)) + 1e-10))
    dynamic_range = peak_db - rms

    # ── Frequency band energy (dBFS) ─────────────────────────────────────────
    def band_rms(signal, lo, hi, sr):
        from scipy.signal import butter, sosfilt
        nyq = sr / 2.0
        lo_n = max(lo / nyq, 0.001)
        hi_n = min(hi / nyq, 0.999)
        sos = butter(4, [lo_n, hi_n], btype="band", output="sos")
        filtered = sosfilt(sos, signal)
        rms_val = np.sqrt(np.mean(filtered**2))
        return float(20 * np.log10(rms_val + 1e-10))

    bands = {
        "sub_bass_20_60hz":    band_rms(y_mono, 20,   60,   sr),
        "bass_60_250hz":       band_rms(y_mono, 60,   250,  sr),
        "low_mid_250_500hz":   band_rms(y_mono, 250,  500,  sr),
        "mid_500_2khz":        band_rms(y_mono, 500,  2000, sr),
        "upper_mid_2k_6khz":   band_rms(y_mono, 2000, 6000, sr),
        "presence_6k_10khz":   band_rms(y_mono, 6000, 10000,sr),
        "air_10k_20khz":       band_rms(y_mono, 10000,20000,sr),
    }

    # ── Spectral features ─────────────────────────────────────────────────────
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_mono, sr=sr)))
    spectral_rolloff  = float(np.mean(librosa.feature.spectral_rolloff(y=y_mono, sr=sr, roll_percent=0.85)))
    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y_mono)))
    zcr               = float(np.mean(librosa.feature.zero_crossing_rate(y_mono)))

    # ── Stereo width (mid/side correlation) ──────────────────────────────────
    mid  = (y_stereo[0] + y_stereo[1]) / 2.0
    side = (y_stereo[0] - y_stereo[1]) / 2.0
    mid_rms  = float(np.sqrt(np.mean(mid**2)))
    side_rms = float(np.sqrt(np.mean(side**2)))
    stereo_width_ratio = float(side_rms / (mid_rms + 1e-10))

    # correlation coefficient (-1 = inverted, 0 = wide, 1 = mono)
    corr = float(np.corrcoef(y_stereo[0], y_stereo[1])[0, 1])

    # ── Tempo / rhythm ────────────────────────────────────────────────────────
    try:
        tempo, _ = librosa.beat.beat_track(y=y_mono, sr=sr)
        tempo = float(tempo[0] if hasattr(tempo, '__len__') else tempo)
    except Exception:
        tempo = 0.0

    # ── Clipping detection ────────────────────────────────────────────────────
    clip_threshold = 0.99
    clipped_samples = int(np.sum(np.abs(y_mono) >= clip_threshold))
    clipping_pct = 100.0 * clipped_samples / len(y_mono)

    return {
        "duration_sec":         round(duration, 2),
        "sample_rate":          sr,
        "integrated_lufs":      round(integrated_lufs, 2),
        "true_peak_dbfs":       round(true_peak, 2),
        "rms_dbfs":             round(rms, 2),
        "dynamic_range_db":     round(dynamic_range, 2),
        "spectral_centroid_hz": round(spectral_centroid, 1),
        "spectral_rolloff_hz":  round(spectral_rolloff, 1),
        "spectral_flatness":    round(spectral_flatness, 4),
        "zero_crossing_rate":   round(zcr, 4),
        "stereo_width_ratio":   round(stereo_width_ratio, 3),
        "stereo_correlation":   round(corr, 3),
        "estimated_bpm":        round(tempo, 1),
        "clipped_samples":      clipped_samples,
        "clipping_pct":         round(clipping_pct, 4),
        "frequency_bands_dbfs": {k: round(v, 2) for k, v in bands.items()},
    }


# ─── vLLM call (OpenAI-compatible) ─────────────────────────────────────────────
def call_vllm(prompt: str, model: str, host: str) -> str:
    """Send prompt to the vLLM OpenAI-compatible endpoint and return the response text."""
    url = f"{host}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 2048,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except urllib.error.URLError as e:
        err(f"Cannot reach vLLM at {host}: {e}")
        err("Is the pod running? Try: kubectl get pods -l app=gemma-vllm")
        raise RuntimeError(f"Cannot reach vLLM at {host}: {e}") from e
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        err(f"vLLM returned HTTP {e.code}: {body}")
        raise RuntimeError(f"vLLM returned HTTP {e.code}: {body}") from e


# ─── Prompt builder ───────────────────────────────────────────────────────────
def system_context(genre: str) -> str:
    return f"""You are an expert mastering engineer with 20+ years of experience across genres,
currently focused on **{genre}**. You have worked on records distributed via Spotify and Apple
Music where the target is -14 LUFS integrated loudness.

Your job is to analyze the provided audio feature data and deliver a professional mastering report
calibrated to what's appropriate for {genre} — don't apply heavy-music loudness/EQ conventions to
a genre that calls for more dynamic range or a different tonal balance, and vice versa.
Be specific, technical, and actionable. Reference the actual numbers in your analysis.
Flag any serious problems clearly. Provide concrete EQ frequency ranges (e.g. "cut 2-3dB at 400Hz"),
specific compressor settings where relevant, and realistic LUFS headroom math.

Output your report in clean Markdown with these sections:
1. **Track Overview** — brief summary of what the data suggests about the mix
2. **Loudness & Dynamics** — LUFS, true peak, dynamic range, streaming compliance
3. **Frequency Balance** — per-band analysis and EQ recommendations
4. **Stereo Field** — width, correlation, any mono compatibility concerns
5. **Clipping & Distortion** — any hard limiting or clipping artifacts detected
6. **Mastering Chain Recommendations** — ordered signal chain with settings
7. **Priority Actions** — top 3 things to fix before mastering, ranked by impact
8. **Streaming Readiness** — pass/warn/fail for Spotify/Apple Music at -14 LUFS
"""

def build_prompt(features: dict, filename: str, genre: str) -> str:
    bands = features["frequency_bands_dbfs"]
    band_table = "\n".join(
        f"  - {k.replace('_', ' ').title()}: {v} dBFS"
        for k, v in bands.items()
    )

    prompt = f"""{system_context(genre)}

---

## Audio File: `{filename}`
**Genre context:** {genre}
**Distribution target:** Spotify/Apple Music — -14 LUFS integrated, true peak ≤ -1 dBTP

## Extracted Audio Features

**Duration:** {features['duration_sec']} seconds
**Sample Rate:** {features['sample_rate']} Hz
**Estimated BPM:** {features['estimated_bpm']}

### Loudness
- Integrated LUFS: {features['integrated_lufs']} LUFS
- True Peak: {features['true_peak_dbfs']} dBTP
- RMS: {features['rms_dbfs']} dBFS
- Dynamic Range (crest factor): {features['dynamic_range_db']} dB

### Clipping
- Clipped samples (≥ 0.99 amplitude): {features['clipped_samples']} ({features['clipping_pct']}%)

### Frequency Bands (energy in dBFS)
{band_table}

### Spectral Character
- Spectral Centroid: {features['spectral_centroid_hz']} Hz (brightness center of mass)
- Spectral Rolloff (85%): {features['spectral_rolloff_hz']} Hz
- Spectral Flatness: {features['spectral_flatness']} (0=tonal, 1=noise-like)
- Zero Crossing Rate: {features['zero_crossing_rate']}

### Stereo Field
- Width Ratio (side/mid RMS): {features['stereo_width_ratio']}
- L/R Correlation: {features['stereo_correlation']} (-1=inverted, 0=wide, 1=mono)

---

Please provide your full mastering report now.
"""
    return prompt


# ─── Report saving ────────────────────────────────────────────────────────────
def save_report(report: str, source_path: Path, output_dir: Path | None) -> Path:
    """Save the markdown report alongside the audio file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = source_path.stem
    report_name = f"{stem}_master_report_{timestamp}.md"

    if output_dir:
        out_path = output_dir / report_name
    else:
        out_path = source_path.parent / report_name

    header = f"""# Mastering Report: {source_path.name}
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Tool:** master-analyze.py / Gemma 4 31B (vLLM)
**Target:** Spotify/Apple Music (-14 LUFS, -1 dBTP)

---

"""
    out_path.write_text(header + report, encoding="utf-8")
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AI mastering analysis using Gemma 4 31B via vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or "")
    )
    parser.add_argument("audio_file", help="Path to audio file (wav, flac, mp3, aiff, ogg, etc.)")
    parser.add_argument("--model",  default="gemma-heretic", help="vLLM model name")
    parser.add_argument("--host",   default=os.environ.get("MASTER_ANALYZE_HOST", "http://localhost:8000"), help="vLLM host URL (OpenAI-compatible)")
    parser.add_argument("--genre",  default="metalcore / heavy", help="Genre context for the model")
    parser.add_argument("--output", default=None, help="Directory to save .md report")
    parser.add_argument("--no-save", action="store_true", help="Don't save report to file")
    args = parser.parse_args()

    banner()

    audio_path = Path(args.audio_file).resolve()
    if not audio_path.exists():
        err(f"File not found: {audio_path}")
        sys.exit(1)

    output_dir = Path(args.output).resolve() if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    section("Loading audio")
    info(f"{audio_path.name}  ({audio_path.stat().st_size / 1024 / 1024:.1f} MB)")
    y, sr = load_audio(audio_path)
    ok(f"Loaded  —  {sr} Hz  /  {'stereo' if y.ndim == 2 else 'mono'}")

    # ── Step 2: Analyze ───────────────────────────────────────────────────────
    section("Extracting features")
    features = extract_features(y, sr)

    ok(f"Integrated LUFS:  {features['integrated_lufs']} LUFS  (target: -14.0)")
    ok(f"True Peak:        {features['true_peak_dbfs']} dBTP  (ceiling: -1.0)")
    ok(f"Dynamic Range:    {features['dynamic_range_db']} dB")
    ok(f"Stereo Corr:      {features['stereo_correlation']}")
    if features['clipping_pct'] > 0.01:
        warn(f"Clipping detected: {features['clipped_samples']} samples ({features['clipping_pct']}%)")
    else:
        ok(f"No significant clipping detected")

    # ── Step 3: Call Gemma ────────────────────────────────────────────────────
    section(f"Sending to {args.model} @ {args.host}")
    info("This may take 30–90 seconds depending on your hardware...")

    prompt = build_prompt(features, audio_path.name, args.genre)
    try:
        report = call_vllm(prompt, args.model, args.host)
    except RuntimeError:
        sys.exit(1)

    if not report:
        err("vLLM returned an empty response. Try increasing --max_tokens or check model availability.")
        sys.exit(1)

    ok("Report received")

    # ── Step 4: Output ────────────────────────────────────────────────────────
    section("Mastering Report")
    print("\n" + "─" * 60)
    print(report)
    print("─" * 60)

    if not args.no_save:
        report_path = save_report(report, audio_path, output_dir)
        print(f"\n{C.GREEN}{C.BOLD}Report saved:{C.RESET} {report_path}")

    print(f"\n{C.DIM}Done.{C.RESET}\n")


if __name__ == "__main__":
    main()
