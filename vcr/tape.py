from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np

@dataclass
class TapeTrack:
    """One helical-scan track (prototype: one FIELD)."""
    y_dphi8: np.ndarray             # uint8 quantized phase increments for FM luma (bandwidth-limited)
    c_u8: np.ndarray                # uint8 packed chroma (Cb,Cr interleaved)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TapeAudio:
    sample_rate: int = 44100
    pcm16: Optional[np.ndarray] = None  # int16 mono

@dataclass
class TapeCartridge:
    """Fixed-length tape with sparse storage (blank tracks not stored)."""
    length_tracks: int
    tracks: Dict[int, TapeTrack] = field(default_factory=dict)

    # Optional: when a tape is loaded from a bundle, we keep the original backing arrays alive
    # so TapeTrack slices are views (fast load, low memory).
    bundle_backing: Any = None

    def get(self, idx: int) -> Optional[TapeTrack]:
        return self.tracks.get(idx)

    def set(self, idx: int, tr: TapeTrack) -> None:
        if 0 <= idx < self.length_tracks:
            self.tracks[idx] = tr

    def clear_range(self, start: int, end: int) -> None:
        for i in range(start, end):
            if i in self.tracks:
                del self.tracks[i]

    def recorded_count(self) -> int:
        return len(self.tracks)

@dataclass
class TapeImage:
    cart: TapeCartridge
    audio: TapeAudio = field(default_factory=TapeAudio)

    def duration_seconds(self) -> float:
        dt = 1/60
        if self.cart.tracks:
            any_tr = next(iter(self.cart.tracks.values()))
            dt = float(any_tr.meta.get("dt", dt))
        return float(self.cart.length_tracks) * float(dt)
