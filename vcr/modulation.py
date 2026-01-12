from __future__ import annotations
import numpy as np
import cv2

def rgb_to_ycbcr(rgb01: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb01[...,0], rgb01[...,1], rgb01[...,2]
    y  = 0.299*r + 0.587*g + 0.114*b
    cb = -0.168736*r - 0.331264*g + 0.5*b + 0.5
    cr = 0.5*r - 0.418688*g - 0.081312*b + 0.5
    return (np.clip(y,0,1).astype(np.float32),
            np.clip(cb,0,1).astype(np.float32),
            np.clip(cr,0,1).astype(np.float32))

def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    cbn = cb - 0.5
    crn = cr - 0.5
    r = y + 1.402*crn
    g = y - 0.344136*cbn - 0.714136*crn
    b = y + 1.772*cbn
    rgb = np.stack([r,g,b], axis=-1)
    return np.clip(rgb, 0, 1).astype(np.float32)

def _dphi_range(sample_rate: int, f0: float, fdev: float) -> tuple[float, float]:
    fmin = f0 - fdev
    fmax = f0 + fdev
    dmin = 2*np.pi * fmin / float(sample_rate)
    dmax = 2*np.pi * fmax / float(sample_rate)
    return float(dmin), float(dmax)

def fm_to_dphi_u8(signal01: np.ndarray, sample_rate: int, f0: float, fdev: float) -> tuple[np.ndarray, dict]:
    s = np.clip(signal01.astype(np.float32), 0.0, 1.0).reshape(-1)
    inst_f = f0 + (s - 0.5) * 2.0 * fdev
    dphi = (2*np.pi * inst_f / float(sample_rate)).astype(np.float32)

    dmin, dmax = _dphi_range(sample_rate, f0, fdev)
    q = np.clip((dphi - dmin) / (dmax - dmin), 0.0, 1.0)
    u8 = np.round(q * 255.0).astype(np.uint8)
    meta = {"dphi_min": dmin, "dphi_max": dmax}
    return u8, meta

def dphi_u8_to_luma(u8: np.ndarray, out_len: int, sample_rate: int, f0: float, fdev: float, dphi_min: float, dphi_max: float) -> np.ndarray:
    q = u8.astype(np.float32) / 255.0
    dphi = dphi_min + q * (dphi_max - dphi_min)
    inst_f = (dphi * float(sample_rate)) / (2*np.pi)
    s = (inst_f - f0) / (2.0 * fdev) + 0.5
    if s.size != out_len:
        s = cv2.resize(s.reshape(1,-1), (out_len,1), interpolation=cv2.INTER_LINEAR).reshape(-1)
    return np.clip(s, 0.0, 1.0).astype(np.float32)

def encode_field_bgr(field_bgr: np.ndarray, sample_rate: int = 2_000_000,
                     f0: float = 350_000.0, fdev: float = 120_000.0,
                     chroma_subsample: int = 2,
                     luma_bw: float = 0.66) -> tuple[np.ndarray, np.ndarray, dict]:
    rgb = field_bgr[..., ::-1].astype(np.float32) / 255.0
    y, cb, cr = rgb_to_ycbcr(rgb)
    h, w = y.shape

    y_mod_w = max(80, int(w * float(luma_bw)))
    y_mod = cv2.resize(y, (y_mod_w, h), interpolation=cv2.INTER_AREA)

    ch = max(1, h // chroma_subsample)
    cw = max(1, w // chroma_subsample)
    cb_s = cv2.resize(cb, (cw, ch), interpolation=cv2.INTER_AREA)
    cr_s = cv2.resize(cr, (cw, ch), interpolation=cv2.INTER_AREA)

    y_dphi8, dmeta = fm_to_dphi_u8(y_mod, sample_rate=sample_rate, f0=f0, fdev=fdev)

    cb_u8 = np.round(np.clip(cb_s, 0, 1) * 255.0).astype(np.uint8).reshape(-1)
    cr_u8 = np.round(np.clip(cr_s, 0, 1) * 255.0).astype(np.uint8).reshape(-1)
    c_u8 = np.empty((cb_u8.size * 2,), dtype=np.uint8)
    c_u8[0::2] = cb_u8
    c_u8[1::2] = cr_u8

    meta = {
        "y_shape": [int(h), int(w)],
        "y_mod_w": int(y_mod_w),
        "c_shape": [int(ch), int(cw)],
        "sample_rate": int(sample_rate),
        "f0": float(f0),
        "fdev": float(fdev),
        "chroma_subsample": int(chroma_subsample),
        "luma_bw": float(luma_bw),
        **dmeta
    }
    return y_dphi8, c_u8, meta

def decode_field_bgr(y_dphi8: np.ndarray, c_u8: np.ndarray, meta: dict) -> np.ndarray:
    y_h, y_w = meta["y_shape"]
    y_mod_w = int(meta.get("y_mod_w", y_w))
    c_h, c_w = meta["c_shape"]
    sr = int(meta["sample_rate"])
    f0 = float(meta["f0"])
    fdev = float(meta["fdev"])
    dmin = float(meta["dphi_min"])
    dmax = float(meta["dphi_max"])

    y_mod = dphi_u8_to_luma(y_dphi8, out_len=y_h*y_mod_w, sample_rate=sr, f0=f0, fdev=fdev, dphi_min=dmin, dphi_max=dmax)
    y_mod = y_mod.reshape(y_h, y_mod_w)
    y = cv2.resize(y_mod, (y_w, y_h), interpolation=cv2.INTER_LINEAR)

    need = c_h * c_w * 2
    cu = c_u8
    if cu.size != need:
        cu = cv2.resize(cu.reshape(1,-1), (need,1), interpolation=cv2.INTER_LINEAR).reshape(-1).astype(np.uint8)

    cb_u8 = cu[0::2].reshape(c_h, c_w).astype(np.float32) / 255.0
    cr_u8 = cu[1::2].reshape(c_h, c_w).astype(np.float32) / 255.0
    cb = cv2.resize(cb_u8, (y_w, y_h), interpolation=cv2.INTER_LINEAR)
    cr = cv2.resize(cr_u8, (y_w, y_h), interpolation=cv2.INTER_LINEAR)

    rgb01 = ycbcr_to_rgb(y, cb, cr)
    bgr = (rgb01[..., ::-1] * 255.0).astype(np.uint8)
    return bgr
