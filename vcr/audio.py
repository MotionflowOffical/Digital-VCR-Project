from __future__ import annotations
import subprocess
import shutil
import wave
from pathlib import Path
import numpy as np



def simulate_vhs_linear_audio(pcm16: np.ndarray, sample_rate: int) -> np.ndarray:
    """Approximate VHS *linear track* audio bandwidth + mild compression.

    Kept intentionally gentle: real VHS linear audio is bandwidth-limited but not 'destroyed'.
    """
    if pcm16 is None or pcm16.size == 0:
        return pcm16
    sr = int(sample_rate) if sample_rate else 44100
    x = pcm16.astype(np.float32) / 32768.0

    # Band-limit ~10 kHz
    cutoff = 10000.0
    dt = 1.0 / float(sr)
    rc = 1.0 / (2.0 * np.pi * cutoff)
    alpha = dt / (rc + dt)
    y = np.empty_like(x)
    prev = 0.0
    for i in range(x.size):
        prev = prev + alpha * (x[i] - prev)
        y[i] = prev

    # Mild soft compression
    y = np.tanh(y * 1.2) / np.tanh(1.2)

    # Quantize lightly (~12-bit) to emulate consumer-grade linear track
    q = np.round(y * 2047.0) / 2047.0

    out = np.clip(q, -1.0, 1.0)
    return (out * 32767.0).astype(np.int16)


def get_ffmpeg_exe() -> str | None:
    """Return an ffmpeg executable path.

    Uses PATH if available; otherwise tries imageio-ffmpeg (pip install imageio-ffmpeg).
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None

def ffmpeg_available() -> bool:
    return get_ffmpeg_exe() is not None

def extract_audio_mono_pcm16(video_path: str, sample_rate: int = 44100) -> tuple[np.ndarray | None, str | None]:
    if not ffmpeg_available():
        return None, "ffmpeg not found on PATH."
    exe = get_ffmpeg_exe()
    if not exe:
        return None, "ffmpeg not found (install ffmpeg or imageio-ffmpeg)."
    cmd = [exe,"-y","-i",video_path,"-vn","-ac","1","-ar",str(sample_rate),"-f","s16le","pipe:1"]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if p.returncode != 0:
            return None, p.stderr.decode("utf-8", errors="ignore")[-4000:]
        pcm = np.frombuffer(p.stdout, dtype=np.int16).copy()
        return pcm, None
    except Exception as e:
        return None, str(e)

def write_wav_mono_pcm16(path: str, pcm16: np.ndarray, sample_rate: int = 44100) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.astype(np.int16).tobytes())

def read_wav_mono_pcm16(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    pcm = np.frombuffer(raw, dtype=np.int16).copy()
    return pcm, int(sr)

def apply_audio_record_defects(pcm16: np.ndarray, sr: int, wow: float, hiss: float, dropouts: float, compression: float = 0.0) -> np.ndarray:
    """Record-time audio effects.

    compression: 0..1 blends in a gentle VHS linear-track bandwidth + mild comp/quantize stage.
    """
    x = pcm16.astype(np.float32) / 32768.0

    # Wow/flutter (timebase wobble)
    if wow > 0:
        n = x.size
        t = np.linspace(0, 1, n, dtype=np.float32)
        f1 = 0.3 + 0.7*wow
        f2 = 6.0 + 10.0*wow
        drift = (0.0015*wow) * (np.sin(2*np.pi*f1*t) + 0.35*np.sin(2*np.pi*f2*t))
        idx = (np.arange(n, dtype=np.float32) + drift*sr).clip(0, n-1)
        x = np.interp(idx, np.arange(n, dtype=np.float32), x).astype(np.float32)

    # Hiss
    if hiss > 0:
        x += np.random.randn(x.size).astype(np.float32) * (0.02*hiss)

    # Dropouts (brief level dips)
    if dropouts > 0:
        events = int(1 + dropouts * 30)
        n = x.size
        for _ in range(events):
            start = np.random.randint(0, max(1, n-1000))
            length = np.random.randint(200, 2400)
            end = min(n, start + length)
            x[start:end] *= 0.05

    out16 = (np.clip(x, -1, 1) * 32767.0).astype(np.int16)

    # Compression / bandwidth (linear audio feel) — blended
    c = float(np.clip(compression, 0.0, 1.0))
    if c > 1e-4:
        proc = simulate_vhs_linear_audio(out16, sr).astype(np.int16)
        out = ( (1.0 - c) * out16.astype(np.float32) + c * proc.astype(np.float32) )
        out16 = np.clip(out, -32768, 32767).astype(np.int16)

    return out16


def apply_audio_playback_defects(pcm16: np.ndarray, sr: int, hiss: float, pops: float) -> np.ndarray:
    x = pcm16.astype(np.float32) / 32768.0
    if hiss > 0:
        x += np.random.randn(x.size).astype(np.float32) * (0.015*hiss)
    if pops > 0:
        n = x.size
        num = int(pops * 50)
        for _ in range(num):
            i = np.random.randint(0, max(1, n-50))
            x[i:i+8] += (np.random.randn(8).astype(np.float32) * 0.4)
    return (np.clip(x, -1, 1) * 32767.0).astype(np.int16)

def mux_audio_into_mp4(video_mp4: str, audio_wav: str, out_mp4: str) -> tuple[bool, str | None]:
    if not ffmpeg_available():
        return False, "ffmpeg not found."
    exe = get_ffmpeg_exe()
    if not exe:
        return False, "ffmpeg not found."
    cmd = [exe,"-y","-i",video_mp4,"-i",audio_wav,"-c:v","copy","-c:a","aac","-shortest",out_mp4]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode != 0:
        return False, p.stderr.decode("utf-8", errors="ignore")[-4000:]
    return True, None


# --- Simple μ-law companding (8-bit) to simulate "encoded" linear audio track ---
def pcm16_to_ulaw(pcm16: np.ndarray) -> np.ndarray:
    # ITU-T G.711 μ-law (approx) for 16-bit PCM mono
    x = pcm16.astype(np.float32) / 32768.0
    mu = 255.0
    s = np.sign(x)
    y = s * (np.log1p(mu*np.abs(x)) / np.log1p(mu))
    u = ((y + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    return u

def ulaw_to_pcm16(ulaw: np.ndarray) -> np.ndarray:
    u = ulaw.astype(np.float32) / 255.0
    y = (u * 2.0) - 1.0
    mu = 255.0
    s = np.sign(y)
    x = s * (1.0/mu) * ((1.0 + mu) ** np.abs(y) - 1.0)
    return (x.clip(-1, 1) * 32767.0).astype(np.int16)
