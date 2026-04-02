from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import cv2
from collections import OrderedDict

from .tape import TapeImage
from .modulation import decode_field_bgr
from .rf_model import rf_roundtrip_luma_dphi_u8, rf_roundtrip_chroma_u8
from .defects import (
    PlaybackDefects,
    apply_rf_defects_y_dphi_u8, apply_rf_defects_chroma_u8,
    apply_timebase_wobble, apply_chroma_shift, apply_interference, apply_composite_view, enforce_aspect,
    apply_image_controls, apply_scanlines, apply_scanline_soften
)

@dataclass
class ServoState:
    inserted: bool = False
    inserting_timer: float = 0.0
    playing: bool = False
    speed: float = 0.0
    pos_tracks: float = 0.0
    _last_pos_tracks: float = 0.0

    # Servo-related
    lock: float = 0.0
    tracking_opt: float = 0.50    # ideal knob point
    tracking_drift: float = 0.0
    _last_seg: int = -1
    snow_gate: float = 1.0
    drop_gate: float = 1.0
    intf_gate: float = 1.0
    blur_gate: float = 1.0
    chroma_gate: float = 1.0
    wobble_gate: float = 1.0
    hs_gate: float = 1.0
    hunt_gate: float = 1.0
    hunt_phase: float = 0.0
    hunt_amp: float = 0.0


    # Edit-point behaviour / switching confusion
    cut_black_timer: float = 0.0
    switch_confuse_timer: float = 0.0

    # Auto tracking internal state
    auto_knob: float = 0.50

    _jit_dx: float = 0.0
    _jit_dy: float = 0.0


class SnowField:
    def __init__(self):
        # kept for potential future temporal persistence, but current model is per-frame
        self.prev: np.ndarray | None = None

    def apply(self, frame: np.ndarray, strength: float, dropout_boost: float = 0.0, bleed: float = 0.55) -> np.ndarray:
        """Classic RF snow: sparse, crisp 1px-ish sparkles with rare dotted horizontal trails (within-frame) + glow bleed.

        This is intentionally *not* temporally persistent (so it doesn't 'drift' slowly frame-to-frame).
        """
        s = float(np.clip(strength, 0.0, 1.0))
        d = float(np.clip(dropout_boost, 0.0, 1.0))
        if s < 1e-4 and d < 1e-4:
            return frame

        h, w = frame.shape[:2]

        # Sparse, punchy impulses (core sparkles)
        dens = float(np.clip(0.00035 + 0.0021*s + 0.0030*d, 0.0, 0.010))
        core = (np.random.rand(h, w).astype(np.float32) < dens).astype(np.float32)
        core *= (0.95 + 1.55*np.random.rand(h, w).astype(np.float32))

        spark = core

        # Rare dotted trails: pick a much sparser seed, then copy it forward in X in steps (gaps).
        trail_seed_dens = float(np.clip(0.00005 + 0.00030*s + 0.00035*d, 0.0, 0.0020))
        if np.random.rand() < (0.15 + 0.35*s + 0.25*d):
            seed = (np.random.rand(h, w).astype(np.float32) < trail_seed_dens).astype(np.float32)
            dx = int(np.random.choice([2, 3, 4]))
            n = int(2 + np.random.randint(2, 6))  # 4..7 dots max
            for k in range(1, n + 1):
                spark += np.roll(seed, dx*k, axis=1) * (0.65 ** k)

        # Bleed/glow: controlled separately; keep crisp core visible
        b = float(np.clip(bleed, 0.0, 1.0))
        kx = int(max(3, (3 + 10*b) // 2 * 2 + 1))
        ky = 3

        glow = cv2.GaussianBlur(spark, (kx, ky), sigmaX=0)
        glow *= (75.0 + 240.0*b) * (0.75 + 1.15*s + 0.9*d)

        core_amp = spark * (185.0 + 430.0*s + 260.0*d)
        add = core_amp + glow

        # Slight chroma fringe like composite sparkle
        out = frame.astype(np.float32)
        r = np.roll(add, 1, axis=1)
        bch = np.roll(add, -1, axis=1)
        g = add
        out[..., 0] += bch
        out[..., 1] += g
        out[..., 2] += r

        return np.clip(out, 0, 255).astype(np.uint8)

class VCRPlayer:


    def __init__(self):
        self.state = ServoState()
        self._t_last = time.perf_counter()
        self._cache = OrderedDict()
        self._cache_cap = 120
        self._snow = SnowField()
        self._phase = 0.0

    def _pair_base(self, tape: TapeImage, idx: int) -> int:
        """Align a track index to the first field of its recorded frame.

        The recorder writes two consecutive fields per source frame.
        If playback starts on the second field, weaving will combine fields from
        adjacent frames, which looks like digital tearing/combing and can make
        small text unreadable.

        Each recorded track stores `frame_base_track` in its meta so we can pair
        correctly even if recording started on an odd track index.
        """
        L = int(tape.cart.length_tracks)
        if L < 2:
            return 0
        i = int(np.clip(int(idx), 0, L - 2))
        tr = tape.cart.get(i)
        if tr is not None:
            try:
                fb = int(tr.meta.get("frame_base_track", i - (i % 2)))
                fb = int(np.clip(fb, 0, L - 2))
                return fb
            except Exception:
                pass
        # Fallback for legacy bundles: assume even/odd pairing.
        return int(i - (i % 2))

    def _cache_get(self, idx: int, token: int):
        """Return cached decoded field if it matches the current track object.

        In Live mode we overwrite the same track indices in a ring buffer;
        without a token check we'd show stale (ghost) frames.
"""
        if idx in self._cache:
            tok, val = self._cache.pop(idx)
            if tok == token:
                self._cache[idx] = (tok, val)
                return val
            # track was overwritten -> drop stale entry
        return None

    def _cache_put(self, idx: int, token: int, val):
        self._cache[idx] = (token, val)
        while len(self._cache) > self._cache_cap:
            self._cache.popitem(last=False)

    def insert(self):
        s = self.state
        # decay edit-point effects
        s.cut_black_timer = 0.0
        s.switch_confuse_timer = 0.0
        s.inserted = True
        s.inserting_timer = 0.0
        s.playing = False
        s.speed = 0.0
        s.pos_tracks = 0.0
        s.lock = 0.0
        s.tracking_opt = float(np.clip(0.42 + np.random.rand()*0.16, 0.0, 1.0))
        s.tracking_drift = 0.0
        self._cache.clear()

    def eject(self):
        self.state = ServoState()
        self._cache.clear()

    def play(self):
        if not self.state.inserted:
            return
        self.state.playing = True
        self.state.speed = 1.0

    def stop(self):
        self.state.playing = False
        self.state.speed = 0.0

    def ff(self, mult: float = 10.0):
        if not self.state.inserted:
            return
        self.state.playing = True
        self.state.speed = float(mult)

    def rew(self, mult: float = 10.0):
        if not self.state.inserted:
            return
        self.state.playing = True
        self.state.speed = -float(mult)

    def update(self, tape: TapeImage, pb: PlaybackDefects):
        now = time.perf_counter()
        dt = max(1e-4, now - self._t_last)
        self._t_last = now
        wob_spd = float(getattr(pb, 'wobble_speed', 1.0))
        self._phase += dt * max(0.05, wob_spd)

        s = self.state

        # decay edit-point effects (timers used by get_frame)
        s.cut_black_timer = max(0.0, float(getattr(s,'cut_black_timer',0.0)) - dt)
        s.switch_confuse_timer = max(0.0, float(getattr(s,'switch_confuse_timer',0.0)) - dt)

        if not s.inserted:
            return

        if s.inserting_timer < 1.2:
            s.inserting_timer += dt
            # ramp lock up slowly while "inserting"
            s.lock = min(0.25, s.lock + 0.10*dt)
            return

        # advance tape
        if s.playing and tape.cart.length_tracks > 2:
            s.pos_tracks += s.speed * (dt * 60.0)  # ~60 fields/sec
            s.pos_tracks = max(0.0, min(float(tape.cart.length_tracks-2), s.pos_tracks))

            # Seek/jump destabilizes lock (like a real VCR needing time to re-lock)
            seek_jump = abs(float(s.pos_tracks) - float(getattr(s, "_last_pos_tracks", s.pos_tracks)))
            s._last_pos_tracks = float(s.pos_tracks)
            if seek_jump > 90.0:  # ~1.5s or more
                s.lock = min(s.lock, 0.10)
                s.inserting_timer = min(s.inserting_timer, 1.0)

        # Pair fields to their recorded frame boundary to avoid weaving mismatched fields.
        base = self._pair_base(tape, int(s.pos_tracks))
        t0 = tape.cart.get(base)
        t1 = tape.cart.get(base+1)
        # Segment boundary: entering a newly-recorded region should destabilize lock briefly.
        try:
            seg = int((t0.meta.get('seg_id', -1) if t0 else -1))
        except Exception:
            seg = -1
        if seg != -1 and seg != getattr(s, '_last_seg', -1):
            prev_seg = getattr(s, '_last_seg', -1)
            s._last_seg = seg
            # entering a new recorded region: brief black + unstable lock like a real edit point
            s.cut_black_timer = max(float(getattr(s,'cut_black_timer',0.0)), 0.12 + 0.18*np.random.rand())
            s.switch_confuse_timer = max(float(getattr(s,'switch_confuse_timer',0.0)), 0.70 + 0.90*np.random.rand())
            s.lock = min(s.lock, 0.10)
            s.tracking_drift *= 0.20
            s.inserting_timer = min(s.inserting_timer, 1.2)

        # entering blank/unrecorded area (cut to black)
        if seg == -1 and getattr(s, '_last_seg', -1) != -1:
            s._last_seg = -1
            s.cut_black_timer = max(float(getattr(s,'cut_black_timer',0.0)), 0.18 + 0.22*np.random.rand())
            s.switch_confuse_timer = max(float(getattr(s,'switch_confuse_timer',0.0)), 0.45 + 0.60*np.random.rand())
            s.lock = min(s.lock, 0.08)

        # control track / sync "pulse strength"
        sync0 = int(t0.meta.get("ctl_sync_u8", 90)) if t0 else 30
        sync1 = int(t1.meta.get("ctl_sync_u8", 90)) if t1 else 30
        vjit0 = int(t0.meta.get("ctl_vjit_u8", 15)) if t0 else 60
        vjit1 = int(t1.meta.get("ctl_vjit_u8", 15)) if t1 else 60
        sync = ((sync0 + sync1) * 0.5) / 255.0
        # User-adjustable sync bias (0..1, 0.5 = neutral)
        try:
            bias = float(getattr(pb, 'sync_bias', 0.5))
        except Exception:
            bias = 0.5
        sync = float(np.clip(sync + (bias - 0.5) * 0.6, 0.0, 1.0))
        vjit = ((vjit0 + vjit1) * 0.5) / 255.0

        # tracking drift (tape/mechanics)
        s.tracking_drift += (np.random.randn()*0.003) * dt
        s.tracking_drift = float(np.clip(s.tracking_drift, -0.15, 0.15))
        tracking_opt = float(np.clip(s.tracking_opt + s.tracking_drift, 0.0, 1.0))

        # speed mismatch worsens lock
        speed_err = min(1.0, abs(s.speed - 1.0) / 2.0)

        # Auto tracking: a consumer VCR will hunt around the best tracking point when lock is poor.
        eff_knob = float(pb.tracking_knob)
        if float(getattr(pb, 'auto_tracking', 0.0)) > 0.5:
            strength = float(np.clip(getattr(pb, 'auto_tracking_strength', 0.70), 0.0, 1.0))
            # Move internal knob toward the apparent optimum, but with some hunting/overshoot.
            drive = float(np.clip((1.0 - s.lock) * 1.2 + (1.0 - sync) * 0.8, 0.0, 1.0))
            target = tracking_opt
            step = (target - float(getattr(s,'auto_knob',0.50)))
            s.auto_knob = float(np.clip(float(getattr(s,'auto_knob',0.50)) + step * dt * (0.8 + 3.0*strength) * (0.35 + drive), 0.0, 1.0))
            # hunting dither
            s.auto_knob = float(np.clip(s.auto_knob + (np.random.randn()*0.10) * dt * strength * drive, 0.0, 1.0))
            # User knob remains a bias/override.
            eff_knob = float(np.clip(eff_knob + (s.auto_knob - 0.50) * strength, 0.0, 1.0))

        # "real knob" effect:
        knob_err = abs(eff_knob - tracking_opt)
        tracking_err = float(np.clip(knob_err * pb.tracking_sensitivity + 0.55*speed_err, 0.0, 1.0))

        # servo lock update driven by sync pulses & tracking
        if s.playing and abs(s.speed) < 1.6:
            srv = float(getattr(pb, 'servo_recovery', 0.55))
            # servo_recovery: higher = faster lock reacquire, but can 'hunt'
            acquire = (0.40 + 1.45*sync) * (1.0 - 0.85*tracking_err) * (0.55 + 1.05*srv)
            s.lock = float(np.clip(s.lock + acquire*dt, 0.0, 1.0))
        else:
            s.lock = float(np.clip(s.lock - (0.8+0.9*speed_err)*dt, 0.0, 1.0))

        # occasional loss of lock on bad sync / bad tracking
        if sync < 0.35 and np.random.rand() < (0.05 + 0.25*(0.35-sync)) * dt:
            s.lock = max(0.0, s.lock - 0.25)

        # Servo hunting: lock recovery can overshoot/oscillate (VHS-like "hunting")
        hunt_amt = float(getattr(pb, "servo_hunt", 0.22)) * float(getattr(s, "hunt_gate", 1.0))
        drive = float(np.clip((1.0 - s.lock) * 1.2 + tracking_err * 0.8, 0.0, 1.0))
        s.hunt_amp = float(np.clip(0.92 * getattr(s, "hunt_amp", 0.0) + 0.25 * hunt_amt * drive, 0.0, 1.0))
        s.hunt_phase = float(getattr(s, "hunt_phase", 0.0) + dt * (6.5 + 9.0*drive))

        # store for rendering (vjit influences global jitter)
        s._last_tracking_err = tracking_err

        # Update effect gates (amount vs frequency).
        v = float(getattr(pb, "variance", 0.55))

        def upd_gate(old: float, freq: float) -> float:
            freq = float(np.clip(freq, 0.0, 1.0))
            target = 1.0 if (np.random.rand() < freq) else 0.0
            a = 0.08 + 0.28*v
            return float(np.clip((1.0-a)*old + a*target, 0.0, 1.0))

        s.snow_gate = upd_gate(getattr(s, "snow_gate", 1.0), getattr(pb, "snow_freq", 0.85))
        s.drop_gate = upd_gate(getattr(s, "drop_gate", 1.0), getattr(pb, "dropout_freq", 0.65))
        s.intf_gate = upd_gate(getattr(s, "intf_gate", 1.0), getattr(pb, "interference_freq", 0.75))
        s.tbase_gate = upd_gate(getattr(s, "tbase_gate", 1.0), getattr(pb, "timebase_freq", 0.70))
        s.jit_gate  = upd_gate(getattr(s, "jit_gate", 1.0), getattr(pb, "frame_jitter_freq", 0.65))

        s.blur_gate = upd_gate(getattr(s, "blur_gate", 1.0), getattr(pb, "playback_blur_freq", 0.65))
        s.chroma_gate = upd_gate(getattr(s, "chroma_gate", 1.0), getattr(pb, "chroma_noise_freq", 0.55))
        s.wobble_gate = upd_gate(getattr(s, "wobble_gate", 1.0), getattr(pb, "chroma_wobble_freq", 0.55))
        s.hs_gate = upd_gate(getattr(s, "hs_gate", 1.0), getattr(pb, "head_switch_freq", 0.70))
        s.hunt_gate = upd_gate(getattr(s, "hunt_gate", 1.0), getattr(pb, "servo_hunt_freq", 0.55))



        # Vertical sync roll / drift (looping vertical drift) — driven by tracking/lock
        vroll_gate = float(np.clip(1.25*tracking_err + 0.90*(1.0 - s.lock), 0.0, 1.0))
        auto = bool(getattr(pb, 'auto_sync', True))
        if vroll_gate > 0.12 and ((not auto) or (s.lock < 0.86) or (tracking_err > 0.22)):
            # Direction follows tracking knob sign (like manual tracking pulling the lock)
            knob = float(getattr(pb, 'tracking_knob', 0.0))
            if abs(knob) > 0.02:
                s._vroll_dir = 1.0 if knob > 0.0 else -1.0
            else:
                s._vroll_dir = float(getattr(s, '_vroll_dir', 1.0))
            # Speed in pixels/sec (slow loop), faster when lock is poor
            sp = (6.0 + 48.0*vroll_gate) * (1.0 + 1.8*(1.0 - s.lock))
            s._vroll_off = float(getattr(s, '_vroll_off', 0.0) + s._vroll_dir * sp * dt)
        else:
            # Decay toward stable (no roll)
            s._vroll_off = float(getattr(s, '_vroll_off', 0.0) * (0.92 ** (dt*30.0)))
        s._vroll_gate = vroll_gate

        s._last_vjit = vjit
        s._last_sync = sync

    def _decode_track_with_rf(self, tape: TapeImage, idx: int, pb: PlaybackDefects):
        tr = tape.cart.get(idx)
        if tr is None:
            return None

        token = id(tr)
        cached = self._cache_get(idx, token)
        if cached is not None:
            return cached

        s = self.state
        lock = float(getattr(s, 'lock', 1.0))

        # Defects should largely emerge from *read difficulty* (lock/tracking/switching),
        # not from fixed overlays. Sliders still control overall sensitivity.
        tracking_err = float(getattr(s, "_last_tracking_err", 0.0))
        conf = float(getattr(s, "switch_confuse_timer", 0.0))
        stress = float(np.clip(0.55*(1.0-lock) + 0.35*tracking_err + 0.25*conf, 0.0, 1.0))
        drop_gate = float(getattr(s, "drop_gate", 1.0))
        snow_gate = float(getattr(s, "snow_gate", 1.0))
        rf_noise = float(getattr(pb, "playback_rf_noise", 0.0)) * (0.55 + 0.85*stress) * (0.25 + 0.75*snow_gate)
        dropouts = float(getattr(pb, "playback_dropouts", 0.0)) * (0.35 + 1.25*stress) * drop_gate


        mode = str(tr.meta.get("tape_mode","SP"))

        # If the tape was recorded with the RF round-trip, or the user explicitly enabled RF playback,
        # apply carrier-level model here before decode.
        use_rf = bool(tr.meta.get('real_rf_modulation', False)) or bool(getattr(pb, 'rf_playback_model', False))
        if use_rf:
            try:
                y = rf_roundtrip_luma_dphi_u8(
                    tr.y_dphi8,
                    tr.meta,
                    noise=float(rf_noise),
                    dropouts=float(dropouts),
                    mode=mode,
                    fm_depth=float(getattr(pb, 'rf_playback_fm_depth', float(tr.meta.get('rf_fm_depth', 1.0)))),
                    am_depth=float(getattr(pb, 'rf_playback_am_depth', 0.18)),
                    nonlinearity=float(getattr(pb, 'rf_playback_nonlinearity', 0.20)),
                    carrier_noise=float(getattr(pb, 'rf_playback_carrier_noise', 0.20)),
                    phase_noise=float(getattr(pb, 'rf_playback_phase_noise', 0.12)),
                )
                c = rf_roundtrip_chroma_u8(
                    tr.c_u8,
                    tr.meta,
                    noise=float(rf_noise) * (0.8 + float(getattr(pb, 'chroma_noise', 0.12))),
                    dropouts=float(dropouts),
                    mode=mode,
                    fc_frac=float(tr.meta.get('rf_chroma_fc_frac', 0.12)),
                    lpf_strength=float(tr.meta.get('rf_chroma_lpf', 0.35)),
                    am_depth=float(getattr(pb, 'rf_playback_am_depth', 0.18)),
                    nonlinearity=float(getattr(pb, 'rf_playback_nonlinearity', 0.20)),
                    carrier_noise=float(getattr(pb, 'rf_playback_carrier_noise', 0.20)),
                    phase_noise=float(getattr(pb, 'rf_playback_phase_noise', 0.12)),
                )
            except Exception:
                # Fallback to simpler byte-space RF defects
                y = apply_rf_defects_y_dphi_u8(tr.y_dphi8, rf_noise, dropouts, mode, lock=lock)
                c = apply_rf_defects_chroma_u8(tr.c_u8, rf_noise * (0.8 + pb.chroma_noise), dropouts, mode)
        else:
            y = apply_rf_defects_y_dphi_u8(tr.y_dphi8, rf_noise, dropouts, mode, lock=lock)
            c = apply_rf_defects_chroma_u8(tr.c_u8, rf_noise * (0.8 + pb.chroma_noise), dropouts, mode)

        img = decode_field_bgr(y, c, tr.meta, bleed=float(getattr(pb, 'luma_chroma_bleed', 0.0)))
        self._cache_put(idx, token, img)
        return img

    def get_frame(self, tape: TapeImage, pb: PlaybackDefects) -> np.ndarray:
        s = self.state
        if (not s.inserted) or tape.cart.length_tracks < 2:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        if s.inserting_timer < 1.2:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        if float(getattr(s,'cut_black_timer',0.0)) > 1e-3:
            # show black with a bit of snow/interference while lock is reacquiring
            base = np.zeros((480, 640, 3), dtype=np.uint8)
            base = self._snow.apply(base, 0.55 + 0.45*float(getattr(pb,'snow',0.18)), dropout_boost=0.10)
            return base

        # Pair fields to their recorded frame boundary to avoid weaving mismatched fields.
        base = self._pair_base(tape, int(s.pos_tracks))
        f0 = self._decode_track_with_rf(tape, base, pb)
        f1 = self._decode_track_with_rf(tape, base+1, pb)
        if f0 is None or f1 is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # defensive crop
        hh = min(f0.shape[0], f1.shape[0])
        ww = min(f0.shape[1], f1.shape[1])
        f0 = f0[:hh, :ww]
        f1 = f1[:hh, :ww]

        # tracking error drives crosstalk and line tearing
        tracking_err = float(getattr(s, "_last_tracking_err", 0.3))
        ta = float(getattr(pb, 'tracking_artifacts', 1.0))
        tracking_err = float(np.clip(tracking_err * (2.2*ta), 0.0, 1.0))
        if tracking_err > 0.05:
            # mix adjacent tracks slightly (crosstalk)
            adj0 = self._decode_track_with_rf(tape, min(tape.cart.length_tracks-1, base+2), pb)
            adj1 = self._decode_track_with_rf(tape, min(tape.cart.length_tracks-1, base+3), pb)
            if adj0 is not None:
                adj0 = adj0[:hh, :ww]
                f0 = np.clip((1.0-tracking_err)*f0.astype(np.float32) + tracking_err*adj0.astype(np.float32), 0, 255).astype(np.uint8)
            if adj1 is not None:
                adj1 = adj1[:hh, :ww]
                f1 = np.clip((1.0-tracking_err)*f1.astype(np.float32) + tracking_err*adj1.astype(np.float32), 0, 255).astype(np.uint8)

        frame = np.zeros((hh*2, ww, 3), dtype=np.uint8)
        frame[0::2] = f0
        frame[1::2] = f1

        if tracking_err > 0.15:
            h2 = frame.shape[0]
            blocks = int(1 + tracking_err*7)
            for _ in range(blocks):
                y0 = np.random.randint(0, h2)
                ln = np.random.randint(2, int(8 + tracking_err*26))
                y1 = min(h2, y0+ln)
                frame[y0:y1] = np.roll(frame[y0:y1], np.random.randint(-14,15), axis=1)

        # Head switching band near bottom (gated by head_switch controls)
        hs = float(getattr(pb, "head_switch_strength", 0.0)) * float(getattr(s, "hs_gate", 1.0))
        if hs > 1e-4:
            gate = float(np.clip((1.0 - s.lock) + 0.8*tracking_err, 0.0, 1.0))
            freq = float(np.clip(float(getattr(pb, "head_switch_freq", 0.70)), 0.0, 1.0))
            p = (0.05 + 0.85*freq) * (0.10 + 0.55*hs) * (0.25 + 0.75*gate)
            if np.random.rand() < p:
                band = int(6 + (10 + 28*hs) * (0.35 + 0.65*gate))
                band = min(frame.shape[0]//4, max(6, band))
                y0 = frame.shape[0] - band
                noise = np.random.randint(0, 256, size=(band, frame.shape[1], 3), dtype=np.uint8)
                blend = hs * (0.12 + 0.55*gate)
                frame[y0:] = np.clip((1.0-blend)*frame[y0:].astype(np.float32) + blend*noise.astype(np.float32), 0, 255).astype(np.uint8)

        lock = float(s.lock)
        vjit = float(getattr(s, "_last_vjit", 0.15))
        sync = float(getattr(s, "_last_sync", 0.7))

        # global sync shake (only when timebase/edit confusion is enabled)
        conf = float(np.clip(float(getattr(s,'switch_confuse_timer',0.0)) / 1.35, 0.0, 1.0))
        ta = float(getattr(pb, "tracking_artifacts", 1.0))
        conf_vis = conf * ta
        tb_user = float(getattr(pb, "playback_timebase", 0.0))
        amp_tb = tb_user * (0.35 + 0.65*(1.0-lock)) * (0.7 + 0.6*vjit)
        amp_edit = conf_vis * (0.6 + 1.0*(1.0-lock))
        amp_x = 14.0*amp_tb + 10.0*amp_edit
        amp_y = 7.0*amp_tb + 6.0*amp_edit
        gdx = float(np.sin(self._phase*(1.0+3.0*vjit)) * amp_x)
        gdy = float(np.sin(self._phase*(0.8+2.5*vjit)) * amp_y)

        # Servo hunting adds low-frequency whole-frame wobble
        ha = float(getattr(s,'hunt_amp',0.0))
        if ha > 0.001:
            gdx += (np.sin(getattr(s,'hunt_phase',0.0)*1.3) * (0.8 + 6.5*ha))
            gdy += (np.cos(getattr(s,'hunt_phase',0.0)*1.1) * (0.6 + 4.5*ha))

        # User wobble clamp/control (caps displacement; preserves existing behaviour)
        wob_px = float(getattr(pb, 'wobble_px', 20.0))
        if wob_px >= 0.0:
            if wob_px < 0.5:
                gdx = 0.0
                gdy = 0.0
            else:
                gdx = float(np.clip(gdx, -wob_px, wob_px))
                gdy = float(np.clip(gdy, -wob_px, wob_px))

        # timebase wobble grows when lock is poor
        tb = float(np.clip(float(getattr(pb,'playback_timebase',0.0)) + 0.55*conf_vis, 0.0, 1.0))
        frame = apply_timebase_wobble(frame, tb, lock=lock, global_dx=gdx, global_dy=gdy)

        # Vertical drift/roll: whole image slowly drifts and wraps when off-sync
        vroll = float(getattr(s, '_vroll_off', 0.0))
        if abs(vroll) >= 0.5:
            sh = int(vroll) % frame.shape[0]
            if sh != 0:
                frame = np.roll(frame, sh, axis=0)

        # Chroma: make phase error/wobble feel more like unstable color-under decoding
        conf = float(np.clip(float(getattr(s,'switch_confuse_timer',0.0)) / 1.35, 0.0, 1.0))
        ch_noise = pb.chroma_noise * float(getattr(s,'chroma_gate',1.0)) * (1.0 + 3.2*conf)
        wob = float(getattr(pb, 'chroma_wobble', 0.10)) * float(getattr(s,'wobble_gate',1.0))
        wob_s = float(np.sin(getattr(s,'hunt_phase',0.0)*0.9 + 1.7))
        wob_c = float(np.cos(getattr(s,'hunt_phase',0.0)*0.65 + 0.9))
        # stronger phase wobble + a bit of chroma positional wobble
        wob_phase = pb.chroma_phase + (2.8*wob) * wob_s
        csx = (pb.chroma_shift_x * (1.0 + 2.1*conf)) + (0.35*wob) * wob_s
        csy = (pb.chroma_shift_y * (1.0 + 1.7*conf)) + (0.25*wob) * wob_c
        frame = apply_chroma_shift(frame, csx, csy, wob_phase, ch_noise + 0.55*conf)
        stress2 = float(np.clip(0.55*(1.0-lock) + 0.35*tracking_err + 0.25*conf, 0.0, 1.0))
        intf_strength = float(np.clip(pb.interference*(0.55 + 0.85*stress2) + 0.25*(1.0-sync), 0.0, 1.0)) * float(getattr(s, "intf_gate", 1.0))

        frame = apply_interference(frame, intf_strength, variance=getattr(pb, 'variance', 0.55))

        snow_strength = float(np.clip(float(getattr(pb, "snow", 0.0))*(0.4 + 0.9*stress2) + 0.85*float(getattr(pb, "playback_rf_noise", 0.0))*(0.55 + 0.85*stress2), 0.0, 1.0))
        snow_strength *= float(getattr(s, "snow_gate", 1.0))
        dropout_boost = min(1.0, float(getattr(pb, "playback_dropouts", 0.0)) * (0.55 + 1.20*stress2) * float(getattr(s, "drop_gate", 1.0)))

        frame = self._snow.apply(frame, snow_strength, dropout_boost=dropout_boost, bleed=float(getattr(pb,'snow_bleed',0.55)))

        if pb.composite_view:
            frame = apply_composite_view(frame, 0.30)

        # optional scanline overlay + soften control
        frame = apply_scanlines(frame, float(getattr(pb,'scanline_strength',0.0)))
        frame = apply_scanline_soften(frame, pb.scanline_soften)

        frame = enforce_aspect(frame, pb.aspect_display)
        frame = apply_image_controls(frame, pb.brightness, pb.contrast, pb.saturation, pb.bloom, pb.sharpen)
                # Post: playback blur + whole-frame transport jitter
        blur = float(getattr(pb, 'playback_blur', 0.0))
        if blur > 1e-3:
            k = int(1 + 2*int(blur*4))
            k = max(1, min(15, k))
            if k % 2 == 0:
                k += 1
            frame = cv2.GaussianBlur(frame, (k, k), sigmaX=0)

        jamp = float(getattr(pb, 'frame_jitter', 0.0)) * float(getattr(s, 'jit_gate', 1.0))
        if jamp > 1e-3:
            tgt_dx = (np.random.randn()*2.0) * (0.5 + 4.0*jamp)
            tgt_dy = (np.random.randn()*1.2) * (0.5 + 3.0*jamp)
            s._jit_dx = 0.82*getattr(s, '_jit_dx', 0.0) + 0.18*tgt_dx
            s._jit_dy = 0.82*getattr(s, '_jit_dy', 0.0) + 0.18*tgt_dy
            M = np.float32([[1,0,s._jit_dx],[0,1,s._jit_dy]])
            frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        return frame
