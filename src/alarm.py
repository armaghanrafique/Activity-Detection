"""Siren generation + muxing into the output video (for activity alerts).

Generates a two-tone siren that sounds only during alert time-windows and
embeds it into the (silent) annotated video with ffmpeg. Uses ffmpeg from the
``imageio-ffmpeg`` pip package if no system ffmpeg is on PATH.
"""

import shutil
import subprocess
import wave
from typing import List, Tuple

import numpy as np


def ffmpeg_exe() -> str:
    """Locate an ffmpeg binary (system PATH, else the imageio-ffmpeg one)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return ""


def has_ffmpeg() -> bool:
    return bool(ffmpeg_exe())


def build_siren_wav(
    wav_path: str,
    total_duration: float,
    intervals: List[Tuple[float, float]],
    sample_rate: int = 22050,
    freq: float = 1000.0,
) -> None:
    """Write a mono WAV that sounds a two-tone siren during each interval."""
    n_total = max(1, int(total_duration * sample_rate))
    audio = np.zeros(n_total, dtype=np.float32)
    for start, end in intervals:
        s = int(start * sample_rate)
        e = min(int(end * sample_rate), n_total)
        if s >= n_total or e <= s:
            continue
        tt = np.arange(e - s) / sample_rate
        warble = np.sin(2 * np.pi * 3.0 * tt)          # 3 Hz warble
        inst_freq = freq * (1.0 + 0.5 * (warble > 0))   # alternate hi/lo tone
        phase = 2 * np.pi * np.cumsum(inst_freq) / sample_rate
        env = np.minimum(1.0, np.minimum(tt, (e - s) / sample_rate - tt) * 30)
        audio[s:e] += 0.7 * env * np.sin(phase)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(wav_path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def mux_audio(video_in: str, wav_in: str, video_out: str) -> None:
    """Attach the siren WAV as the audio track of the video (ffmpeg)."""
    exe = ffmpeg_exe()
    if not exe:
        raise RuntimeError("ffmpeg not available")
    cmd = [
        exe, "-y",
        "-i", video_in,
        "-i", wav_in,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        video_out,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def flags_to_intervals(flags: List[bool], fps: float) -> List[Tuple[float, float]]:
    """Convert a per-frame alert boolean list into (start, end) time windows."""
    intervals, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            intervals.append((start / fps, i / fps))
            start = None
    if start is not None:
        intervals.append((start / fps, len(flags) / fps))
    return intervals
