"""
title: Mastering Analyzer
author: klink
description: Measures real mastering metrics (integrated LUFS per ITU-R BS.1770, true peak, dynamics, frequency balance, stereo field) from an audio file attached to the chat. Uses pyloudnorm + scipy — the numbers are computed, never guessed.
required_open_webui_version: 0.5.0
version: 1.1.0
requirements: pyloudnorm
"""

import glob
import json
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

UPLOAD_DIR = os.path.join(os.environ.get("DATA_DIR", "/app/backend/data"), "uploads")

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif", ".m4a", ".opus", ".wma", ".aac"}


def _resolve_upload_path(file_entry: dict) -> str | None:
    """Find the on-disk path for an Open WebUI uploaded file entry."""
    f = file_entry.get("file", file_entry)
    # Newer versions may store an explicit path
    for candidate in (f.get("path"), (f.get("meta") or {}).get("path")):
        if candidate and os.path.isfile(candidate):
            return candidate
    file_id = f.get("id") or file_entry.get("id")
    if file_id:
        matches = glob.glob(os.path.join(UPLOAD_DIR, f"{file_id}_*"))
        if matches:
            return matches[0]
    return None


def _entry_name(file_entry: dict) -> str:
    f = file_entry.get("file", file_entry)
    return f.get("filename") or (f.get("meta") or {}).get("name") or file_entry.get("name") or ""


def _decode(path: str) -> tuple[np.ndarray, int]:
    """Decode any audio format to float32 samples via ffmpeg (keeps native sample rate)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-map", "0:a:0", "-c:a", "pcm_f32le", tmp_path],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-400:]}")
        data, sr = sf.read(tmp_path, dtype="float32", always_2d=True)  # (samples, channels)
        return data, sr
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _analyze(data: np.ndarray, sr: int) -> dict:
    from scipy.signal import butter, resample_poly, sosfilt, welch
    import pyloudnorm as pyln

    n_samples, n_ch = data.shape
    duration = n_samples / sr
    mono = data.mean(axis=1)

    # ── Loudness (ITU-R BS.1770-4) ──
    meter = pyln.Meter(sr)
    try:
        integrated_lufs = float(meter.integrated_loudness(data))
    except Exception:
        integrated_lufs = -99.0

    # ── Short-term loudness → approximate LRA (EBU R128 style) ──
    lra = None
    st_max = None
    block = int(3 * sr)
    hop = sr
    if n_samples > block:
        st = []
        for start in range(0, n_samples - block, hop):
            seg = data[start:start + block]
            try:
                v = meter.integrated_loudness(seg)
            except Exception:
                continue
            if v > -70:
                st.append(v)
        if st:
            st = np.array(st)
            st_max = float(np.max(st))
            gated = st[st > max(-70.0, integrated_lufs - 20.0)]
            if len(gated) >= 2:
                lra = float(np.percentile(gated, 95) - np.percentile(gated, 10))

    # ── True peak (4x oversampled) ──
    tp = 0.0
    for ch in range(n_ch):
        up = resample_poly(data[:, ch], 4, 1)
        tp = max(tp, float(np.max(np.abs(up))))
    true_peak_dbtp = float(20 * np.log10(tp + 1e-10))

    # ── Sample peak / RMS / crest ──
    # RMS over the full stereo signal (mean channel power), not the mono
    # mixdown — side energy cancels in mono, which overstates crest on
    # wide masters.
    sample_peak_db = float(20 * np.log10(np.max(np.abs(data)) + 1e-10))
    rms_db = float(20 * np.log10(np.sqrt(np.mean(data**2)) + 1e-10))
    crest_db = sample_peak_db - rms_db

    # ── Clipping ──
    clipped = int(np.sum(np.abs(data) >= 0.99))
    clipping_pct = 100.0 * clipped / data.size

    # ── Frequency band energy ──
    def band_rms_db(lo, hi):
        nyq = sr / 2.0
        sos = butter(4, [max(lo / nyq, 0.001), min(hi / nyq, 0.999)], btype="band", output="sos")
        filtered = sosfilt(sos, mono)
        return round(float(20 * np.log10(np.sqrt(np.mean(filtered**2)) + 1e-10)), 2)

    bands = {
        "sub_bass_20_60hz": band_rms_db(20, 60),
        "bass_60_250hz": band_rms_db(60, 250),
        "low_mid_250_500hz": band_rms_db(250, 500),
        "mid_500_2khz": band_rms_db(500, 2000),
        "upper_mid_2k_6khz": band_rms_db(2000, 6000),
        "presence_6k_10khz": band_rms_db(6000, 10000),
        "air_10k_20khz": band_rms_db(10000, min(20000, sr / 2 - 1)),
    }

    # ── Spectral centroid / rolloff (Welch PSD) ──
    freqs, psd = welch(mono, fs=sr, nperseg=4096)
    psd_sum = np.sum(psd) + 1e-20
    centroid = float(np.sum(freqs * psd) / psd_sum)
    cumsum = np.cumsum(psd)
    rolloff = float(freqs[int(np.searchsorted(cumsum, 0.85 * cumsum[-1]))])

    # ── Stereo field ──
    if n_ch >= 2:
        l, r = data[:, 0], data[:, 1]
        mid = (l + r) / 2.0
        side = (l - r) / 2.0
        width = float(np.sqrt(np.mean(side**2)) / (np.sqrt(np.mean(mid**2)) + 1e-10))
        corr = float(np.corrcoef(l, r)[0, 1])
    else:
        width, corr = 0.0, 1.0

    gain_to_target = round(-14.0 - integrated_lufs, 2)

    return {
        "duration_sec": round(duration, 2),
        "sample_rate_hz": sr,
        "channels": n_ch,
        "integrated_lufs": round(integrated_lufs, 2),
        "short_term_lufs_max": round(st_max, 2) if st_max is not None else None,
        "loudness_range_lu_approx": round(lra, 2) if lra is not None else None,
        "true_peak_dbtp": round(true_peak_dbtp, 2),
        "sample_peak_dbfs": round(sample_peak_db, 2),
        "rms_dbfs": round(rms_db, 2),
        "crest_factor_db": round(crest_db, 2),
        "clipped_samples": clipped,
        "clipping_pct": round(clipping_pct, 4),
        "frequency_bands_dbfs": bands,
        "spectral_centroid_hz": round(centroid, 1),
        "spectral_rolloff_85pct_hz": round(rolloff, 1),
        "stereo_width_ratio": round(width, 3),
        "stereo_correlation": round(corr, 3),
        "streaming_target": {
            "platform": "Spotify / Apple Music",
            "target_integrated_lufs": -14.0,
            "true_peak_ceiling_dbtp": -1.0,
            "gain_change_to_reach_target_db": gain_to_target,
            "estimated_true_peak_after_gain_dbtp": round(true_peak_dbtp + gain_to_target, 2),
        },
    }


class Tools:
    async def analyze_track(
        self,
        file_name: str = "",
        __files__: list = [],
        __event_emitter__=None,
    ) -> str:
        """
        Measure the mastering metrics of an audio file attached to this chat:
        integrated LUFS (ITU-R BS.1770), true peak dBTP, loudness range, RMS,
        crest factor, clipping, frequency band balance, spectral character,
        and stereo field. Call this whenever the user attaches a song or asks
        about loudness, LUFS, mastering, or streaming readiness. Never estimate
        these values yourself — always call this tool.

        :param file_name: Optional file name to select when multiple audio files are attached. Leave empty to use the most recently attached audio file.
        :return: JSON string with the measured metrics.
        """

        async def status(msg, done=False):
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": msg, "done": done},
                })

        candidates = [
            f for f in (__files__ or [])
            if os.path.splitext(_entry_name(f))[1].lower() in AUDIO_EXTS
        ]
        if file_name:
            named = [f for f in candidates if file_name.lower() in _entry_name(f).lower()]
            if named:
                candidates = named

        if not candidates:
            await status("No audio file found", done=True)
            return json.dumps({
                "error": "No audio file is attached to this chat. Ask the user to "
                         "attach a track (wav, flac, mp3, aiff, ogg...) using the + "
                         "button, then call this tool again."
            })

        entry = candidates[-1]
        name = _entry_name(entry)
        path = _resolve_upload_path(entry)
        if not path:
            await status("File not found on disk", done=True)
            return json.dumps({"error": f"Could not locate uploaded file '{name}' on disk."})

        try:
            await status(f"Decoding {name}...")
            data, sr = _decode(path)
            await status(f"Measuring loudness of {name} (this can take ~10-30s)...")
            metrics = _analyze(data, sr)
            await status(f"Analysis of {name} complete", done=True)
        except Exception as e:
            await status(f"Analysis failed: {e}", done=True)
            return json.dumps({"error": f"Analysis failed: {e}"})

        return json.dumps({
            "file": name,
            "metrics": metrics,
            "instructions_for_assistant": (
                "These values were measured with pyloudnorm (ITU-R BS.1770-4) and scipy. "
                "Quote them exactly as given — do not round further, alter, or invent any "
                "numbers. Base your mastering feedback on these measurements."
            ),
        })
