from __future__ import annotations

"""RF-level (carrier) simulation helpers.

This codebase's "standard" luma path stores a quantized FM phase-increment stream (y_dphi8)
and chroma stores packed Cb/Cr samples (c_u8).

When "real_rf_modulation" is enabled, we run a lightweight RF round-trip:
  - luma: dphi -> complex carrier IQ (FM) -> AM envelope + noise + dropouts + nonlinearity -> demod -> dphi
  - chroma: Cb/Cr -> complex subcarrier (color-under style) -> channel effects -> demod -> Cb/Cr

We still store results in the same tape format (dphi8 + c_u8), so playback stays fast.
"""

import numpy as np

from .defects import mode_scale


def _smooth_noise_1d(n: int, coarse: int = 32) -> np.ndarray:
    """Return smooth-ish noise in [-1,1] (linear-interp of coarse random knots)."""
    n = int(max(1, n))
    coarse = int(max(4, min(coarse, n)))
    knots = np.random.randn(coarse).astype(np.float32)
    knots -= knots.mean()
    knots /= (knots.std() + 1e-6)
    xk = np.linspace(0, n - 1, coarse, dtype=np.float32)
    x = np.arange(n, dtype=np.float32)
    y = np.interp(x, xk, knots).astype(np.float32)
    y = np.clip(y / (np.max(np.abs(y)) + 1e-6), -1.0, 1.0)
    return y


def _u8_to_dphi(y_dphi8: np.ndarray, dphi_min: float, dphi_max: float) -> np.ndarray:
    q = y_dphi8.astype(np.float32) / 255.0
    return (dphi_min + q * (dphi_max - dphi_min)).astype(np.float32)


def _dphi_to_u8(dphi: np.ndarray, dphi_min: float, dphi_max: float) -> np.ndarray:
    q = (dphi.astype(np.float32) - float(dphi_min)) / (float(dphi_max) - float(dphi_min) + 1e-12)
    q = np.clip(q, 0.0, 1.0)
    return np.round(q * 255.0).astype(np.uint8)


def rf_roundtrip_luma_dphi_u8(
    y_dphi8: np.ndarray,
    meta: dict,
    *,
    noise: float,
    dropouts: float,
    mode: str = "SP",
    fm_depth: float = 1.0,
    am_depth: float = 0.25,
    nonlinearity: float = 0.25,
    carrier_noise: float = 0.20,
    phase_noise: float = 0.10,
) -> np.ndarray:
    """FM + AM carrier round-trip for luma.

    Inputs/outputs are dphi8 so the tape format stays the same.
    """

    dphi_min = float(meta.get("dphi_min"))
    dphi_max = float(meta.get("dphi_max"))
    y_h, y_w = meta.get("y_shape", [0, 0])
    y_mod_w = int(meta.get("y_mod_w", y_w))
    if y_h <= 0 or y_mod_w <= 0:
        return y_dphi8

    s = mode_scale(mode)
    dphi = _u8_to_dphi(y_dphi8.reshape(int(y_h), int(y_mod_w)), dphi_min, dphi_max)

    # FM deviation scaling ("FM depth").
    # Implemented as a reversible scale around the middle of the dphi range so:
    #   - a perfect channel still round-trips back to the original dphi
    #   - noise/nonlinearity sensitivity changes with deviation
    fm = float(np.clip(fm_depth, 0.35, 2.25))
    dmid = 0.5 * (dphi_min + dphi_max)
    if abs(fm - 1.0) > 1e-5:
        dphi = (dmid + (dphi - dmid) * fm).astype(np.float32)

    # Integrate phase per line (reset every line like sync)
    phase = np.cumsum(dphi, axis=1).astype(np.float32)

    # Phase jitter (slow-ish) -> mimics timebase/servo micro-jitter in the RF domain
    pn = float(np.clip(phase_noise, 0.0, 1.0))
    # Ease-in so tiny slider values don't instantly create obvious artifacts.
    pn_eff = pn * pn * (3.0 - 2.0 * pn)  # smoothstep
    if pn_eff > 1e-6:
        # smooth noise per-line
        jit = np.stack([_smooth_noise_1d(y_mod_w, coarse=max(8, y_mod_w // 20)) for _ in range(int(y_h))], axis=0)
        phase = phase + (jit * (0.10 * pn_eff) * (1.0 + 0.8 * s)).astype(np.float32)

    i = np.cos(phase)
    q = np.sin(phase)

    # AM envelope ripple
    a = float(np.clip(am_depth, 0.0, 1.0))
    a_eff = a * a * (3.0 - 2.0 * a)  # smoothstep
    if a_eff > 1e-6:
        env = np.stack([_smooth_noise_1d(y_mod_w, coarse=max(8, y_mod_w // 24)) for _ in range(int(y_h))], axis=0)
        # No "always-on" baseline: effect strength is proportional to slider value.
        env = 1.0 + (env * (0.55 * a_eff) * (1.0 + 0.6 * s))
        i = (i * env).astype(np.float32)
        q = (q * env).astype(np.float32)

    # Additive carrier noise
    # Combine existing RF noise slider with the RF-model-specific one
    cn = float(np.clip(carrier_noise, 0.0, 1.0))
    n = float(np.clip(noise, 0.0, 1.0))
    sigma = (0.004 + 0.020 * cn + 0.030 * n) * (1.0 + 0.7 * s)
    if sigma > 1e-7:
        i = (i + np.random.randn(*i.shape).astype(np.float32) * sigma).astype(np.float32)
        q = (q + np.random.randn(*q.shape).astype(np.float32) * sigma).astype(np.float32)

    # Dropouts: short segments where RF collapses (or turns into noisy burst)
    d = float(np.clip(dropouts, 0.0, 1.0)) * s
    if d > 1e-7:
        events = int(1 + 10 * d)
        for _ in range(events):
            yy = np.random.randint(0, int(y_h))
            start = np.random.randint(0, max(1, y_mod_w - 40))
            length = int(np.random.randint(20, int(40 + 220 * d)))
            end = min(y_mod_w, start + length)
            burst = np.random.randn(end - start).astype(np.float32) * (0.10 + 0.35 * d)
            i[yy, start:end] = burst
            q[yy, start:end] = burst

    # Nonlinearity / saturation
    nl = float(np.clip(nonlinearity, 0.0, 1.0))
    nl_eff = nl * nl * (3.0 - 2.0 * nl)  # smoothstep
    if nl_eff > 1e-6:
        # Blend between linear and normalized soft-clip.
        # This avoids the "instant on" harmonic/banding jump as soon as nl > 0.
        drive = 1.0 + 6.0 * nl_eff
        norm = float(np.tanh(drive) + 1e-6)
        ni = (np.tanh(i * drive) / norm).astype(np.float32)
        nq = (np.tanh(q * drive) / norm).astype(np.float32)
        i = ((1.0 - nl_eff) * i + nl_eff * ni).astype(np.float32)
        q = ((1.0 - nl_eff) * q + nl_eff * nq).astype(np.float32)

    # Demod: dphi = angle(z[n] * conj(z[n-1]))
    # (Do per line; output length = y_mod_w)
    i0 = i[:, :-1]
    q0 = q[:, :-1]
    i1 = i[:, 1:]
    q1 = q[:, 1:]
    re = (i1 * i0 + q1 * q0).astype(np.float32)
    im = (q1 * i0 - i1 * q0).astype(np.float32)
    dphi_est = np.arctan2(im, re).astype(np.float32)
    # pad last sample
    last = dphi_est[:, -1:].copy()
    dphi_est = np.concatenate([dphi_est, last], axis=1)

    # Undo FM depth scaling so the mean mapping stays consistent.
    if abs(fm - 1.0) > 1e-5:
        dphi_est = (dmid + (dphi_est - dmid) / fm).astype(np.float32)

    return _dphi_to_u8(dphi_est.reshape(-1), dphi_min, dphi_max)


def rf_roundtrip_chroma_u8(
    c_u8: np.ndarray,
    meta: dict,
    *,
    noise: float,
    dropouts: float,
    mode: str = "SP",
    fc_frac: float = 0.12,
    lpf_strength: float = 0.35,
    am_depth: float = 0.20,
    nonlinearity: float = 0.25,
    carrier_noise: float = 0.20,
    phase_noise: float = 0.10,
) -> np.ndarray:
    """Color-under-ish chroma channel round-trip (complex subcarrier + channel effects)."""

    c_h, c_w = meta.get("c_shape", [0, 0])
    c_h = int(c_h)
    c_w = int(c_w)
    need = int(c_h * c_w * 2)
    if need <= 0:
        return c_u8

    cu = c_u8.reshape(-1)
    if cu.size != need:
        # If sizes mismatch, don't do RF round-trip; let existing resampler handle it elsewhere.
        return c_u8

    s = mode_scale(mode)
    cb = (cu[0::2].astype(np.float32) / 255.0) - 0.5
    cr = (cu[1::2].astype(np.float32) / 255.0) - 0.5
    bb = cb + 1j * cr

    n = bb.size
    fc = float(np.clip(fc_frac, 0.01, 0.49))
    ph = (2.0 * np.pi * fc * np.arange(n, dtype=np.float32)).astype(np.float32)
    carrier = np.cos(ph).astype(np.float32) + 1j * np.sin(ph).astype(np.complex64)
    z = (bb.astype(np.complex64) * carrier)

    # Phase noise
    pn = float(np.clip(phase_noise, 0.0, 1.0))
    if pn > 1e-6:
        jit = _smooth_noise_1d(n, coarse=max(16, n // 40))
        z *= (np.cos(jit * (0.10 * pn) * (1.0 + 0.8 * s)) + 1j * np.sin(jit * (0.10 * pn) * (1.0 + 0.8 * s))).astype(np.complex64)

    # AM ripple
    a = float(np.clip(am_depth, 0.0, 1.0))
    a_eff = a * a * (3.0 - 2.0 * a)
    if a_eff > 1e-6:
        env = 1.0 + _smooth_noise_1d(n, coarse=max(16, n // 36)) * (0.60 * a_eff) * (1.0 + 0.6 * s)
        z *= env.astype(np.float32)

    # Add noise (complex)
    cn = float(np.clip(carrier_noise, 0.0, 1.0))
    nn = float(np.clip(noise, 0.0, 1.0))
    sigma = (0.004 + 0.030 * cn + 0.028 * nn) * (1.0 + 0.7 * s)
    if sigma > 1e-7:
        z = (z + (np.random.randn(n).astype(np.float32) + 1j * np.random.randn(n).astype(np.float32)) * sigma).astype(np.complex64)

    # Dropouts
    d = float(np.clip(dropouts, 0.0, 1.0)) * s
    if d > 1e-7:
        events = int(1 + 8 * d)
        for _ in range(events):
            start = np.random.randint(0, max(1, n - 40))
            length = int(np.random.randint(20, int(40 + 180 * d)))
            end = min(n, start + length)
            burst = (np.random.randn(end - start).astype(np.float32) + 1j * np.random.randn(end - start).astype(np.float32))
            burst = burst.astype(np.complex64) * (0.06 + 0.28 * d)
            z[start:end] = burst

    # Nonlinearity
    nl = float(np.clip(nonlinearity, 0.0, 1.0))
    nl_eff = nl * nl * (3.0 - 2.0 * nl)
    if nl_eff > 1e-6:
        drive = 1.0 + 6.0 * nl_eff
        norm = float(np.tanh(drive) + 1e-6)
        zr_nl = np.tanh(z.real.astype(np.float32) * drive) / norm
        zi_nl = np.tanh(z.imag.astype(np.float32) * drive) / norm
        z_lin = z.astype(np.complex64)
        z_nl = (zr_nl.astype(np.float32) + 1j * zi_nl.astype(np.float32)).astype(np.complex64)
        z = ((1.0 - nl_eff) * z_lin + nl_eff * z_nl).astype(np.complex64)

    # Demod back to baseband
    zbb = (z * np.conj(carrier)).astype(np.complex64)

    # Low-pass (moving average) to suppress residual carrier
    lp = float(np.clip(lpf_strength, 0.0, 1.0))
    if lp > 1e-6:
        k = int(3 + lp * 19)
        if k % 2 == 0:
            k += 1
        ker = (np.ones((k,), dtype=np.float32) / float(k))
        zr = np.convolve(zbb.real.astype(np.float32), ker, mode="same")
        zi = np.convolve(zbb.imag.astype(np.float32), ker, mode="same")
        zbb = (zr + 1j * zi).astype(np.complex64)

    cb2 = np.clip(zbb.real + 0.5, 0.0, 1.0)
    cr2 = np.clip(zbb.imag + 0.5, 0.0, 1.0)
    cb_u8 = np.round(cb2 * 255.0).astype(np.uint8)
    cr_u8 = np.round(cr2 * 255.0).astype(np.uint8)

    out = np.empty((need,), dtype=np.uint8)
    out[0::2] = cb_u8
    out[1::2] = cr_u8
    return out
