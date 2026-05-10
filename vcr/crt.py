from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np

CRT_PRESETS = ("Consumer TV", "Pro Monitor")
CRT_QUALITIES = ("Draft", "Balanced", "High", "Ultra")
CRT_MASK_TYPES = ("aperture", "slot", "shadow")

QUALITY_RENDER_WIDTHS = {
    "Draft": 960,
    "Balanced": 1440,
    "High": 1920,
    "Ultra": 2880,
}


def _clamp(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = float(default)
    if not np.isfinite(v):
        v = float(default)
    return float(np.clip(v, lo, hi))


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(round(float(value)))
    except Exception:
        v = int(default)
    return int(max(lo, min(hi, v)))


@dataclass
class CRTSettings:
    enabled: bool = False
    preview_enabled: bool = True
    live_enabled: bool = True
    export_enabled: bool = False
    direct_player: bool = False
    direct_live: bool = False

    preset: str = "Consumer TV"
    quality: str = "Balanced"
    render_width: int = QUALITY_RENDER_WIDTHS["Balanced"]
    mask_type: str = "slot"

    mask_strength: float = 0.62
    scanline_strength: float = 0.58
    beam_sharpness: float = 0.45
    bloom: float = 0.28
    halation: float = 0.18
    glass_diffusion: float = 0.12
    curvature: float = 0.14
    overscan: float = 0.035
    vignette: float = 0.22
    edge_focus: float = 0.18
    phosphor_decay: float = 0.18
    convergence_x: float = 0.45
    convergence_y: float = 0.20
    brightness: float = 0.02
    contrast: float = 0.08
    saturation: float = 0.04

    def validated(self) -> "CRTSettings":
        preset = self.preset if self.preset in CRT_PRESETS else "Consumer TV"
        quality = self.quality if self.quality in CRT_QUALITIES else "Balanced"
        mask = self.mask_type if self.mask_type in CRT_MASK_TYPES else (
            "aperture" if preset == "Pro Monitor" else "slot"
        )
        return CRTSettings(
            enabled=bool(self.enabled),
            preview_enabled=bool(self.preview_enabled),
            live_enabled=bool(self.live_enabled),
            export_enabled=bool(self.export_enabled),
            direct_player=bool(self.direct_player),
            direct_live=bool(self.direct_live),
            preset=preset,
            quality=quality,
            render_width=_clamp_int(self.render_width, 320, 4096, QUALITY_RENDER_WIDTHS[quality]),
            mask_type=mask,
            mask_strength=_clamp(self.mask_strength, 0.0, 1.0, 0.62),
            scanline_strength=_clamp(self.scanline_strength, 0.0, 1.0, 0.58),
            beam_sharpness=_clamp(self.beam_sharpness, 0.0, 1.0, 0.45),
            bloom=_clamp(self.bloom, 0.0, 1.0, 0.28),
            halation=_clamp(self.halation, 0.0, 1.0, 0.18),
            glass_diffusion=_clamp(self.glass_diffusion, 0.0, 1.0, 0.12),
            curvature=_clamp(self.curvature, 0.0, 1.0, 0.14),
            overscan=_clamp(self.overscan, 0.0, 0.18, 0.035),
            vignette=_clamp(self.vignette, 0.0, 1.0, 0.22),
            edge_focus=_clamp(self.edge_focus, 0.0, 1.0, 0.18),
            phosphor_decay=_clamp(self.phosphor_decay, 0.0, 1.0, 0.18),
            convergence_x=_clamp(self.convergence_x, -4.0, 4.0, 0.45),
            convergence_y=_clamp(self.convergence_y, -4.0, 4.0, 0.20),
            brightness=_clamp(self.brightness, -1.0, 1.0, 0.02),
            contrast=_clamp(self.contrast, -0.5, 1.0, 0.08),
            saturation=_clamp(self.saturation, -1.0, 1.0, 0.04),
        )

    def render_size_for(self, frame_shape: tuple[int, int, int] | tuple[int, int]) -> tuple[int, int]:
        s = self.validated()
        h = int(frame_shape[0])
        w = int(frame_shape[1])
        if w <= 0 or h <= 0:
            return int(s.render_width), int(max(1, round(s.render_width * 3 / 4)))
        rw = int(s.render_width)
        rh = int(max(1, round(h * (rw / float(w)))))
        return rw, rh


def consumer_tv_preset() -> CRTSettings:
    return CRTSettings(
        preset="Consumer TV",
        quality="Balanced",
        render_width=QUALITY_RENDER_WIDTHS["Balanced"],
        mask_type="slot",
        mask_strength=0.64,
        scanline_strength=0.60,
        beam_sharpness=0.42,
        bloom=0.32,
        halation=0.22,
        glass_diffusion=0.16,
        curvature=0.16,
        overscan=0.045,
        vignette=0.25,
        edge_focus=0.24,
        phosphor_decay=0.20,
        convergence_x=0.55,
        convergence_y=0.25,
        brightness=0.03,
        contrast=0.06,
        saturation=0.02,
    )


def pro_monitor_preset() -> CRTSettings:
    return CRTSettings(
        preset="Pro Monitor",
        quality="High",
        render_width=QUALITY_RENDER_WIDTHS["High"],
        mask_type="aperture",
        mask_strength=0.52,
        scanline_strength=0.48,
        beam_sharpness=0.72,
        bloom=0.14,
        halation=0.08,
        glass_diffusion=0.06,
        curvature=0.035,
        overscan=0.018,
        vignette=0.12,
        edge_focus=0.08,
        phosphor_decay=0.10,
        convergence_x=0.22,
        convergence_y=0.08,
        brightness=0.01,
        contrast=0.10,
        saturation=0.06,
    )


def preset_by_name(name: str) -> CRTSettings:
    if name == "Pro Monitor":
        return pro_monitor_preset()
    return consumer_tv_preset()


def crt_settings_to_dict(settings: CRTSettings) -> Dict[str, Any]:
    return asdict(settings.validated())


def crt_settings_from_dict(data: Dict[str, Any] | None) -> CRTSettings:
    if not isinstance(data, dict):
        return consumer_tv_preset().validated()

    raw = data.get("crt", data)
    if not isinstance(raw, dict):
        return consumer_tv_preset().validated()

    preset = raw.get("preset", "Consumer TV")
    base = preset_by_name(str(preset))
    values = asdict(base)
    for key in values:
        if key in raw:
            values[key] = raw[key]
    return CRTSettings(**values).validated()
