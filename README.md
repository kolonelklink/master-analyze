# master-analyze

A CLI tool that measures real mastering metrics from an audio file — integrated loudness (ITU-R BS.1770), true peak, dynamic range, per-band frequency balance, stereo field — and sends those measurements to an LLM to generate a structured, actionable mastering report.

The core idea: **LLMs can't hear audio.** Ask one to critique a mix and it will confidently invent LUFS numbers, EQ curves, and stereo-width claims that have no connection to the actual file. This tool inverts that — it measures everything with signal-processing libraries first, then hands the LLM real numbers and asks it to reason about *those*, never to guess.

## What it measures

- **Loudness** — integrated LUFS (ITU-R BS.1770-4 via `pyloudnorm`), true peak, RMS, crest factor / dynamic range
- **Frequency balance** — energy across 7 bands from sub-bass (20–60Hz) to air (10–20kHz), via `scipy` bandpass filters
- **Stereo field** — mid/side width ratio and L/R correlation
- **Spectral character** — centroid, rolloff, flatness, zero-crossing rate
- **Clipping** — sample-level clip detection
- **Tempo** — estimated BPM

All of it computed, none of it guessed — see `extract_features()` in [`master-analyze.py`](master-analyze.py).

## How it works

```
audio file → ffmpeg (format normalize) → librosa/pyloudnorm/scipy (feature extraction)
           → structured prompt with real measurements → LLM (any OpenAI-compatible endpoint)
           → markdown mastering report (terminal + saved file)
```

The LLM only ever sees numbers this tool already computed — its job is analysis and recommendations (EQ moves, compression, streaming-readiness verdict), not measurement.

## Two ways to use it

**1. Standalone CLI**
```bash
pip install librosa pyloudnorm numpy scipy
# ffmpeg must be on PATH

python3 master-analyze.py track.wav --genre "metalcore / heavy"
```
Points at any OpenAI-compatible chat completions endpoint (vLLM, Ollama, etc.) via `--host` / `--model`, or the `MASTER_ANALYZE_HOST` env var. Defaults to `http://localhost:8000`.

**2. Open WebUI tool** ([`openwebui/mastering_analyzer_tool.py`](openwebui/mastering_analyzer_tool.py))
Drop into Open WebUI's Tools admin panel and an attached-audio-file gets analyzed inline in chat — lets a persona/assistant call `analyze_track` as a tool mid-conversation instead of running the CLI separately. See [`openwebui/persona_prompt_addendum.md`](openwebui/persona_prompt_addendum.md) for the system-prompt rules that keep the model honest about only reporting measured numbers.

## Example report sections

Reports are structured Markdown with:
1. Track Overview
2. Loudness & Dynamics (streaming compliance vs. Spotify/Apple Music's -14 LUFS target)
3. Frequency Balance (with specific EQ moves, e.g. "cut 2-3dB at 400Hz")
4. Stereo Field / mono compatibility
5. Clipping & Distortion
6. Mastering Chain Recommendations
7. Priority Actions (top 3, ranked by impact)
8. Streaming Readiness verdict (pass/warn/fail)

## Stack

Python · librosa · pyloudnorm · scipy · numpy · ffmpeg · any OpenAI-compatible LLM API (vLLM, Ollama, etc.)
