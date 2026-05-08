from __future__ import annotations

import io
import os
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    import winsound  # Windows only
except Exception:  # pragma: no cover
    winsound = None

from .tape import TapeImage
from .defects import AudioPlaybackDefects
from .audio import apply_audio_playback_defects


@dataclass
class AudioState:
    playing: bool = False
    sample_rate: int = 44100
    chunk_sec: float = 0.25


class AudioPlayer:
    """Chunked audio playback.

    Why chunked?
    - winsound can't stream PCM; chunking lets us "follow" seeks/FF/REW and apply modulation
      based on current lock/tracking state without external deps.

    Notes:
    - On non-Windows OS, available=False (silent).
    - For screen-capture recording, audio capture is not implemented; file recording uses ffmpeg.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._gen = 0
        self._thread: Optional[threading.Thread] = None
        self._tmp_paths: list[str] = []
        self.state = AudioState()
        self.last_error: str | None = None

    @property
    def available(self) -> bool:
        return winsound is not None

    def stop(self) -> None:
        with self._lock:
            self._gen += 1
            self.state.playing = False
            self.last_error = None
        if winsound is not None:
            try:
                winsound.PlaySound(None, winsound.SND_ASYNC)
            except Exception:
                pass

    def _ensure_tmp_files(self) -> None:
        if self._tmp_paths:
            return
        for _ in range(2):
            fd, path = tempfile.mkstemp(prefix="digital_vcr_audio_", suffix=".wav")
            os.close(fd)
            self._tmp_paths.append(path)

    def _write_wav(self, path: str, pcm16: np.ndarray, sr: int) -> None:
        """Disk wav (fallback)."""
        pcm16 = np.asarray(pcm16, dtype=np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sr))
            wf.writeframes(pcm16.tobytes())

    def _wav_bytes(self, pcm16: np.ndarray, sr: int) -> bytes:
        """In-memory WAV bytes for winsound SND_MEMORY (fast, no temp files)."""
        pcm16 = np.asarray(pcm16, dtype=np.int16)
        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sr))
            wf.writeframes(pcm16.tobytes())
        return bio.getvalue()

    def play_from_seconds(self, pcm16: np.ndarray, sample_rate: int, start_sec: float) -> None:
        """Simple one-shot playback (best-effort)."""
        if winsound is None:
            self.last_error = "Windows audio backend is unavailable."
            return
        if pcm16 is None or pcm16.size == 0:
            self.last_error = "No audio samples to play."
            return
        self._ensure_tmp_files()
        sr = int(sample_rate)
        start = int(max(0.0, float(start_sec)) * sr)
        start = max(0, min(start, int(pcm16.size)))
        seg = pcm16[start:]
        self._write_wav(self._tmp_paths[0], seg, sr)
        try:
            winsound.PlaySound(self._tmp_paths[0], winsound.SND_FILENAME | winsound.SND_ASYNC)
            self.last_error = None
        except Exception:
            self.last_error = "Could not start audio playback."

    def start_stream(self,
                     tape: TapeImage,
                     get_pos_sec: Callable[[], float],
                     get_lock: Callable[[], float],
                     ap_def: AudioPlaybackDefects,
                     chunk_sec: float = 0.25) -> None:
        """Start chunked playback that follows the current playhead position."""
        if winsound is None:
            self.last_error = "Windows audio backend is unavailable."
            return
        if tape.audio.pcm16 is None or tape.audio.pcm16.size == 0:
            self.last_error = "No audio is stored on this tape."
            return

        with self._lock:
            self._gen += 1
            gen = self._gen
            self.last_error = None
            self.state = AudioState(playing=True, sample_rate=int(tape.audio.sample_rate or 44100), chunk_sec=float(chunk_sec))
            self._ensure_tmp_files()

        def worker():
            sr = int(getattr(tape.audio, "sample_rate", 44100) or 44100)
            pcm = tape.audio.pcm16
            cs = float(max(0.12, min(1.0, float(chunk_sec))))
            n = int(cs * sr)
            tmp = self._tmp_paths[0]  # fallback only

            # purge any previous
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

            # Smooth cursor so tiny tracking jitters don't cause "garbage audio"
            try:
                cursor = float(get_pos_sec())
            except Exception:
                cursor = 0.0

            while True:
                with self._lock:
                    if gen != self._gen:
                        break
                try:
                    pos = float(get_pos_sec())
                    lock = float(get_lock())
                except Exception:
                    pos = cursor
                    lock = 1.0

                # Only snap if user seeks or drift is big
                # Follow video playhead tightly (avoid accumulating delay).
                if abs(pos - cursor) > 0.10:
                    cursor = pos
                else:
                    cursor = (0.85*cursor + 0.15*pos)

                start = int(max(0.0, cursor) * sr)
                if start >= pcm.size:
                    break
                seg = pcm[start:start+n]
                if seg.size < 16:
                    break

                # Lock-modulated but subtle (consumer VCR: audio degrades a bit during unlock)
                l = float(np.clip(lock, 0.0, 1.0))
                dip = float(0.85 + 0.15*l)  # small gain dip only
                out = (seg.astype(np.float32) * dip)
                # Subtle wow/flutter during poor lock (modulation, not delay)
                if (1.0 - l) > 0.02 and out.size > 32:
                    n0 = out.size
                    t = np.linspace(0.0, 1.0, n0, dtype=np.float32)
                    wob = (1.0 - l)
                    drift = (0.0008*wob) * (np.sin(2*np.pi*(0.35+0.6*wob)*t) + 0.25*np.sin(2*np.pi*(6.0+8.0*wob)*t))
                    # random micro-jitter
                    drift += (np.random.randn(n0).astype(np.float32) * (0.00005*wob))
                    idx = (np.arange(n0, dtype=np.float32) + drift*sr).clip(0, n0-1)
                    out = np.interp(idx, np.arange(n0, dtype=np.float32), out).astype(np.float32)

                hiss = 0.0
                pops = 0.0
                try:
                    if ap_def is not None:
                        hiss = float(getattr(ap_def, "hiss", 0.0))
                        pops = float(getattr(ap_def, "pops", 0.0))
                except Exception:
                    pass

                # Add a *little* extra hiss during poor lock
                hiss += (1.0 - l) * 0.05
                pops += (1.0 - l) * 0.03

                if hiss > 1e-4:
                    out += (np.random.randn(out.size).astype(np.float32) * (hiss * 600.0))
                if pops > 1e-4 and np.random.rand() < pops:
                    k = int(np.random.randint(0, out.size))
                    out[k:k+min(12, out.size-k)] += np.random.randn(min(12, out.size-k)).astype(np.float32) * 4500.0

                out16 = np.clip(out, -32768, 32767).astype(np.int16)
                try:
                    snd = self._wav_bytes(out16, sr)
                    winsound.PlaySound(snd, winsound.SND_MEMORY | winsound.SND_SYNC)
                except Exception:
                    # Fallback to file playback if SND_MEMORY fails on some systems
                    try:
                        self._write_wav(tmp, out16, sr)
                        winsound.PlaySound(tmp, winsound.SND_FILENAME | winsound.SND_SYNC)
                    except Exception as e:
                        # store last error for debugging
                        try:
                            self.state.last_error = str(e)
                            self.last_error = str(e)
                        except Exception:
                            pass
                        break
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        with self._lock:
            self._thread = t
