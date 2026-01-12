from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any
import numpy as np
import cv2

@dataclass
class RecordDefects:
    tape_mode: str = "SP"
    record_blur: float = 0.20
    record_jitter: float = 0.18
    record_rf_noise: float = 0.030
    record_dropouts: float = 0.018
    luma_bw: float = 0.66

@dataclass
class PlaybackDefects:
    aspect_display: str = "4:3"

    # Tracking as a knob (0..1), like a real VCR
    tracking_knob: float = 0.50
    tracking_sensitivity: float = 0.70
    tracking_artifacts: float = 1.0  # scales visible tracking errors

    # Auto-tracking (like a consumer VCR trying to lock): 0=off, 1=on
    auto_tracking: float = 0.0
    auto_tracking_strength: float = 0.70  # how aggressively it hunts/recenters

    # Sync/servo tuning
    servo_recovery: float = 0.55  # higher = faster lock reacquire
    sync_bias: float = 0.50       # 0..1; 0.5 = neutral

    playback_timebase: float = 0.35

    # Whole-frame wobble clamp/control (caps displacement; 0 disables extra shake)
    wobble_px: float = 20.0      # max displacement in pixels
    wobble_speed: float = 1.00   # speed multiplier for wobble phases
    timebase_freq: float = 0.70
    playback_rf_noise: float = 0.08
    playback_dropouts: float = 0.030
    dropout_freq: float = 0.65
    interference: float = 0.14
    interference_freq: float = 0.75
    snow: float = 0.18
    variance: float = 0.55  # randomness/instability amount
    snow_freq: float = 0.85  # 0..1, how often snow is active
    composite_view: bool = False

    chroma_shift_x: float = 0.14
    chroma_shift_y: float = 0.07
    chroma_phase: float = 0.10
    chroma_noise: float = 0.12
    chroma_noise_freq: float = 0.55  # how often chroma noise bursts
    chroma_wobble: float = 0.10      # slow chroma phase wobble amount
    chroma_wobble_freq: float = 0.55 # how often wobble is active


    brightness: float = 0.0
    contrast: float = 0.10
    saturation: float = 0.15
    bloom: float = 0.10
    sharpen: float = 0.10

    # interlaced scanline look (0=off)
    scanline_strength: float = 0.0

    # reduce visible scanlines (vertical blend)
    scanline_soften: float = 1.00

    head_switch_strength: float = 0.22  # bottom-of-frame head switch band/noise
    head_switch_freq: float = 0.70
    servo_hunt: float = 0.22            # servo hunting oscillation amount
    servo_hunt_freq: float = 0.55


    playback_blur: float = 0.10
    playback_blur_freq: float = 0.65

    frame_jitter: float = 0.12
    frame_jitter_freq: float = 0.65

@dataclass
class AudioRecordDefects:
    wow: float = 0.15
    hiss: float = 0.20
    dropouts: float = 0.05
    compression: float = 0.55  # 0..1, VHS-ish band-limit + compand

@dataclass
class AudioPlaybackDefects:
    hiss: float = 0.15
    pops: float = 0.10

def mode_scale(mode: str) -> float:
    return {"SP": 1.0, "LP": 1.25, "EP": 1.6}.get(mode.upper(), 1.0)

def apply_record_defects_to_field(field_bgr: np.ndarray, rec: RecordDefects) -> np.ndarray:
    out = field_bgr
    s = mode_scale(rec.tape_mode)
    blur_amt = min(1.0, rec.record_blur * s)
    if blur_amt > 0:
        k = int(1 + blur_amt * 10)
        if k % 2 == 0: k += 1
        out = cv2.GaussianBlur(out, (k, k), 0)

    jit = min(1.0, rec.record_jitter * s)
    if jit > 0:
        h, w = out.shape[:2]
        max_shift = int(1 + jit * 6)
        shifts = (np.random.randn(h).astype(np.float32) * max_shift).astype(np.int32)
        tmp = np.zeros_like(out)
        for y in range(h):
            sh = shifts[y]
            if sh >= 0:
                tmp[y, sh:] = out[y, :w-sh]
            else:
                tmp[y, :w+sh] = out[y, -sh:]
        out = tmp
    return out

def apply_rf_defects_y_dphi_u8(y_dphi8: np.ndarray, noise: float, dropouts: float, mode: str = "SP", lock: float = 1.0) -> np.ndarray:
    x = y_dphi8.astype(np.int16).copy()

    # Low-frequency baseline drift (depends on noise/dropouts; zero-mean to avoid full-frame flashes)
    s = mode_scale(mode)
    drift_strength = float((0.18*noise + 0.10*dropouts) * s)
    # Tie baseline shading drift to poor sync/lock; near-zero when locked.
    drift_strength *= float(np.clip(1.0 - float(lock), 0.0, 1.0)) ** 1.35
    if drift_strength > 1e-6:
        L = x.size
        k = max(16, int(L // 900))
        coarse = np.random.randn(k).astype(np.float32)
        coarse -= coarse.mean()
        xp = np.linspace(0, L-1, k, dtype=np.float32)
        drift = np.interp(np.arange(L, dtype=np.float32), xp, coarse)
        # scale conservatively; keep it subtle
        drift = drift * (12.0 * drift_strength)
        drift = np.clip(drift, -28, 28).astype(np.int16)
        x = x + drift

    n = noise * s
    if n > 0:
        x += (np.random.randn(x.size).astype(np.float32) * (30*n)).astype(np.int16)

    d = dropouts * s
    if d > 0:
        events = int(1 + 12 * d)
        L = x.size
        for _ in range(events):
            start = np.random.randint(0, max(1, L-200))
            length = np.random.randint(100, 1400)
            end = min(L, start + length)
            burst = (np.random.randn(end-start).astype(np.float32) * (22 + 26*d*s)).astype(np.int16)
            burst += (np.random.rand(end-start) < (0.02 + 0.06*d)).astype(np.int16) * 110
            x[start:end] = np.clip(128 + burst, 0, 255).astype(np.int16)

    return np.clip(x, 0, 255).astype(np.uint8)

def apply_rf_defects_chroma_u8(c_u8: np.ndarray, noise: float, dropouts: float, mode: str = "SP") -> np.ndarray:
    s = mode_scale(mode)
    x = c_u8.astype(np.int16).copy()

    n = noise * s
    if n > 0:
        x += (np.random.randn(x.size).astype(np.float32) * (25*n)).astype(np.int16)

    d = dropouts * s
    if d > 0:
        events = int(1 + 10*d)
        L = x.size
        for _ in range(events):
            start = np.random.randint(0, max(1, L-80))
            length = np.random.randint(80, 420)
            end = min(L, start + length)
            x[start:end] = 128 + (np.random.randn(end-start).astype(np.float32)*18).astype(np.int16)

    return np.clip(x, 0, 255).astype(np.uint8)

def apply_timebase_wobble(frame_bgr: np.ndarray, strength: float, lock: float, global_dx: int = 0, global_dy: int = 0) -> np.ndarray:
    if strength <= 0 and (global_dx == 0 and global_dy == 0):
        return frame_bgr
    img = frame_bgr
    if global_dx != 0 or global_dy != 0:
        M = np.float32([[1, 0, global_dx], [0, 1, global_dy]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    if strength <= 0:
        return img

    h, w = img.shape[:2]
    out = np.zeros_like(img)
    t = np.linspace(0, 1, h, dtype=np.float32)
    # Avoid 'baked-in' jitter when the servo is locked: scale wobble by (1-lock).
    stress = float(np.clip(1.0 - float(lock), 0.0, 1.0))
    # Low-frequency line wander (flagging) + random line jitter, both driven by servo stress.
    lf = (np.sin(2*np.pi*(1.0+2.0*strength)*t + np.random.rand()*6.28)
          * (2 + 10*strength) * (stress ** 0.85))
    jitter = (np.random.randn(h).astype(np.float32)
              * (1.5 + 10.0*strength) * (stress ** 1.20))
    shifts = (lf + jitter).astype(np.int32)
    for y in range(h):
        sft = shifts[y]
        if sft >= 0:
            out[y, sft:] = img[y, :w-sft]
        else:
            out[y, :w+sft] = img[y, -sft:]

    if lock < 0.7 and strength > 0.2 and np.random.rand() < (0.04 + 0.18*(1.0-lock)):
        roll = int((1.0-lock) * strength * 30)
        out = np.roll(out, roll if np.random.rand()<0.5 else -roll, axis=0)
    return out

def apply_chroma_shift(frame_bgr: np.ndarray, shift_x: float, shift_y: float, phase: float, chroma_noise: float) -> np.ndarray:
    if shift_x <= 0 and shift_y <= 0 and abs(phase) <= 1e-3 and chroma_noise <= 0:
        return frame_bgr
    bgr = frame_bgr.astype(np.float32) / 255.0
    rgb = bgr[..., ::-1]
    r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
    y = 0.299*r + 0.587*g + 0.114*b
    cb = -0.168736*r - 0.331264*g + 0.5*b + 0.5
    cr = 0.5*r - 0.418688*g - 0.081312*b + 0.5
    h, w = y.shape
    dx = int(shift_x * 18)
    dy = int(shift_y * 14)
    cb2 = np.roll(cb, (dy, dx), axis=(0,1))
    cr2 = np.roll(cr, (-dy, -dx), axis=(0,1))
    ang = (phase * 28.0) * np.pi/180.0
    cbn = cb2 - 0.5
    crn = cr2 - 0.5
    cbp = cbn*np.cos(ang) - crn*np.sin(ang)
    crp = cbn*np.sin(ang) + crn*np.cos(ang)
    cb2 = cbp + 0.5
    cr2 = crp + 0.5
    # Chroma smear/delay: directional blur in chroma channels (looks like color-under decoding struggle)
    smear = float(np.clip(1.15*abs(shift_x) + 0.90*abs(phase) + 0.60*chroma_noise, 0.0, 1.0))
    if smear > 1e-4:
        kx = int(3 + smear*23)
        if kx % 2 == 0: kx += 1
        ky = 1 if smear < 0.35 else 3
        cb_blur = cv2.GaussianBlur(cb2, (kx, ky), sigmaX=0)
        cr_blur = cv2.GaussianBlur(cr2, (kx, ky), sigmaX=0)
        cb2 = (1.0-smear)*cb2 + smear*cb_blur
        cr2 = (1.0-smear)*cr2 + smear*cr_blur
    if chroma_noise > 0:
        cb2 += np.random.randn(h,w).astype(np.float32) * (0.090*chroma_noise)
        cr2 += np.random.randn(h,w).astype(np.float32) * (0.090*chroma_noise)
    cbn = cb2 - 0.5
    crn = cr2 - 0.5
    r2 = y + 1.402*crn
    g2 = y - 0.344136*cbn - 0.714136*crn
    b2 = y + 1.772*cbn
    rgb2 = np.stack([r2,g2,b2], axis=-1)
    rgb2 = np.clip(rgb2, 0, 1)
    return (rgb2[..., ::-1] * 255.0).astype(np.uint8)

def apply_interference(frame_bgr: np.ndarray, amount: float, variance: float = 0.5) -> np.ndarray:
    if amount <= 0:
        return frame_bgr
    v = float(np.clip(variance, 0.0, 1.0))
    out = frame_bgr.astype(np.float32)
    h, w = out.shape[:2]

    # A couple of moving brightness bars + subtle "buzz"
    t = np.random.rand() * 2*np.pi
    ys = np.arange(h, dtype=np.float32)
    f1 = (0.035 + 0.07*np.random.rand()) * (1.0 + 0.9*v)
    f2 = (0.090 + 0.10*np.random.rand()) * (1.0 + 0.7*v)

    bar1 = (np.sin(ys*f1 + t) * (22 + 55*amount)).reshape(h,1,1)
    bar2 = (np.sin(ys*f2 + t*0.7) * (10 + 35*amount)).reshape(h,1,1)

    buzz = (np.random.randn(h,1,1).astype(np.float32) * (2.0 + 10.0*amount*v))

    out += (0.55 + 0.75*v) * (bar1 + 0.6*bar2) + buzz
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_composite_view(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img
    x = img.astype(np.float32)
    h, w = x.shape[:2]
    shifted = x.copy()
    shift = int(1 + amount * 3)
    shifted[0::2, shift:] = x[0::2, :-shift]
    shifted[1::2, :-shift] = x[1::2, shift:]
    yy, xx = np.mgrid[0:h, 0:w]
    crawl = (np.sin(xx*0.25 + yy*0.15 + np.random.rand()*6.28) * (12*amount)).astype(np.float32)
    shifted[...,1] += crawl
    return np.clip(shifted, 0, 255).astype(np.uint8)

def enforce_aspect(img: np.ndarray, aspect_display: str) -> np.ndarray:
    h, w = img.shape[:2]
    target = 4/3 if aspect_display == "4:3" else 16/9
    cur = w / h
    if abs(cur - target) < 1e-3:
        return img
    if cur > target:
        new_h = int(w / target)
        pad = max(0, new_h - h)
        top = pad//2
        bot = pad - top
        return cv2.copyMakeBorder(img, top, bot, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    else:
        new_w = int(h * target)
        pad = max(0, new_w - w)
        left = pad//2
        right = pad - left
        return cv2.copyMakeBorder(img, 0, 0, left, right, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

def apply_image_controls(frame_bgr: np.ndarray, brightness: float, contrast: float, saturation: float, bloom: float, sharpen: float) -> np.ndarray:
    img = frame_bgr.astype(np.float32) / 255.0
    if abs(brightness) > 1e-4 or abs(contrast) > 1e-4:
        img = (img - 0.5) * (1.0 + 2.0*contrast) + 0.5 + 0.25*brightness
        img = np.clip(img, 0, 1)

    if abs(saturation) > 1e-4:
        rgb = img[..., ::-1]
        r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
        y = 0.299*r + 0.587*g + 0.114*b
        cb = -0.168736*r - 0.331264*g + 0.5*b
        cr = 0.5*r - 0.418688*g - 0.081312*b
        cb *= (1.0 + 2.0*saturation)
        cr *= (1.0 + 2.0*saturation)
        r2 = y + 1.402*cr
        g2 = y - 0.344136*cb - 0.714136*cr
        b2 = y + 1.772*cb
        rgb2 = np.stack([r2,g2,b2], axis=-1)
        rgb2 = np.clip(rgb2, 0, 1)
        img = rgb2[..., ::-1]

    if bloom > 0:
        k = int(3 + bloom*18)
        if k % 2 == 0: k += 1
        blur = cv2.GaussianBlur((img*255).astype(np.uint8), (k,k), 0).astype(np.float32)/255.0
        img = np.clip(img + blur*(0.35*bloom), 0, 1)

    if sharpen > 0:
        k = int(3 + sharpen*8)
        if k % 2 == 0: k += 1
        blur = cv2.GaussianBlur((img*255).astype(np.uint8), (k,k), 0).astype(np.float32)/255.0
        img = np.clip(img + (img - blur)*(0.8*sharpen), 0, 1)

    return (img*255.0).astype(np.uint8)


def apply_scanlines(frame_bgr: np.ndarray, strength: float) -> np.ndarray:
    """Optional scanline overlay (subtle). strength=0 disables."""
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 1e-4:
        return frame_bgr
    x = frame_bgr.astype(np.float32)
    h, w = x.shape[:2]
    mask = np.ones((h, 1, 1), dtype=np.float32)
    # Darken every other line slightly
    dark = 1.0 - 0.28*s
    mask[1::2] = dark
    out = x * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_scanline_soften(frame_bgr: np.ndarray, amount: float) -> np.ndarray:
    """Reduce visible interlaced scanline look by blending vertically.
    VHS is inherently bandwidth-limited, so this is a mild vertical low-pass rather than 'drawn scanlines'.
    amount=0 -> no change, amount=1 -> strongest soften.
    """
    a = float(np.clip(amount, 0.0, 1.0))
    if a <= 1e-4:
        return frame_bgr
    x = frame_bgr.astype(np.float32)

    # 1D vertical blur (3-tap), then blend
    up = np.vstack([x[0:1], x[:-1]])
    dn = np.vstack([x[1:], x[-1:]])
    blur = (up + x + dn) / 3.0

    # Extra tiny gaussian at high soften to kill "hard lines"
    if a > 0.75:
        k = int(3 + (a-0.75)*12)
        if k % 2 == 0:
            k += 1
        blur2 = cv2.GaussianBlur(blur.astype(np.uint8), (1, k), 0).astype(np.float32)
        blur = 0.65*blur + 0.35*blur2

    out = (1.0 - a) * x + a * blur
    return np.clip(out, 0, 255).astype(np.uint8)


def settings_to_dict(rec: RecordDefects, pb: PlaybackDefects, ar: AudioRecordDefects, ap: AudioPlaybackDefects) -> Dict[str, Any]:
    return {"record": asdict(rec), "playback": asdict(pb), "audio_record": asdict(ar), "audio_playback": asdict(ap)}

def settings_from_dict(d: Dict[str, Any]):
    r = d.get("record", {})
    p = d.get("playback", {})
    ar = d.get("audio_record", {})
    ap = d.get("audio_playback", {})
    return RecordDefects(**r), PlaybackDefects(**p), AudioRecordDefects(**ar), AudioPlaybackDefects(**ap)
