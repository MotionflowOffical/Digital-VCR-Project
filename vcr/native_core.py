from __future__ import annotations

"""Optional Rust acceleration for Digital VCR hot loops.

The native library is intentionally narrow. It accelerates operations that were
previously executed as Python loops while leaving the signal equations, OpenCV
filters and NumPy math in Python.

If the library is absent, incompatible, or fails its startup self-check, every
function transparently falls back to the original Python implementation.
"""

from ctypes import CDLL, POINTER, c_float, c_int32, c_size_t, c_uint8
from pathlib import Path
import platform
from typing import Optional

import numpy as np

_LIB = None
_NATIVE_ERROR: Optional[str] = None
_SELF_TESTED = False
_INTERP_TESTED = False
_INTERP_ENABLED = True


def _candidate_names() -> tuple[str, ...]:
    system = platform.system().lower()
    if system == "windows":
        return ("digital_vcr_core.dll",)
    if system == "darwin":
        return ("libdigital_vcr_core.dylib", "digital_vcr_core.dylib")
    return ("libdigital_vcr_core.so", "digital_vcr_core.so")


def _load_native():
    global _LIB, _NATIVE_ERROR
    if _LIB is not None or _NATIVE_ERROR is not None:
        return _LIB

    here = Path(__file__).resolve().parent
    search_dirs = (here / "native", here, Path.cwd())
    for directory in search_dirs:
        for name in _candidate_names():
            path = directory / name
            if not path.exists():
                continue
            try:
                lib = CDLL(str(path))
                lib.dvcr_shift_rows_zero_u8.argtypes = [
                    POINTER(c_uint8), POINTER(c_uint8), c_size_t, c_size_t,
                    c_size_t, POINTER(c_int32),
                ]
                lib.dvcr_shift_rows_zero_u8.restype = c_int32
                lib.dvcr_interp_rows_f32.argtypes = [
                    POINTER(c_float), c_size_t, c_size_t,
                    POINTER(c_float), POINTER(c_float), c_size_t,
                    POINTER(c_float),
                ]
                lib.dvcr_interp_rows_f32.restype = c_int32
                _LIB = lib
                return _LIB
            except Exception as exc:  # native acceleration must never break playback
                _NATIVE_ERROR = f"Failed to load {path.name}: {exc}"
                return None

    _NATIVE_ERROR = "Rust native core not built; using exact Python fallback."
    return None


def _python_shift_rows_zero_u8(img: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Reference implementation matching the pre-native Python row loop."""
    src = np.ascontiguousarray(img, dtype=np.uint8)
    sh = np.asarray(shifts, dtype=np.int32).reshape(-1)
    h, w = src.shape[:2]
    out = np.zeros_like(src)
    for y in range(h):
        sft = int(sh[y])
        # Preserve the original slicing semantics for shifts outside the frame.
        if sft >= w or sft <= -w:
            continue
        if sft >= 0:
            out[y, sft:] = src[y, :w-sft]
        else:
            out[y, :w+sft] = src[y, -sft:]
    return out


def shift_rows_zero_u8(img: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Shift each image row horizontally with zero fill.

    This is mathematically the same operation as the old Python loop. The Rust
    path is used only after a byte-exact startup self-test succeeds.
    """
    global _SELF_TESTED, _LIB, _NATIVE_ERROR
    src = np.ascontiguousarray(img, dtype=np.uint8)
    if src.ndim != 3:
        return _python_shift_rows_zero_u8(src, shifts)
    sh = np.ascontiguousarray(shifts, dtype=np.int32).reshape(-1)
    h, w, channels = src.shape
    if sh.size != h:
        raise ValueError(f"shift count {sh.size} does not match image height {h}")

    lib = _load_native()
    if lib is None:
        return _python_shift_rows_zero_u8(src, sh)

    def _call(arr: np.ndarray, row_shifts: np.ndarray) -> np.ndarray:
        dst = np.zeros_like(arr)
        rc = lib.dvcr_shift_rows_zero_u8(
            arr.ctypes.data_as(POINTER(c_uint8)),
            dst.ctypes.data_as(POINTER(c_uint8)),
            c_size_t(arr.shape[0]), c_size_t(arr.shape[1]), c_size_t(arr.shape[2]),
            row_shifts.ctypes.data_as(POINTER(c_int32)),
        )
        if rc != 0:
            raise RuntimeError(f"dvcr_shift_rows_zero_u8 returned {rc}")
        return dst

    if not _SELF_TESTED:
        try:
            test = np.arange(5 * 9 * 3, dtype=np.uint8).reshape(5, 9, 3)
            tsh = np.asarray([0, 1, -1, 4, -8], dtype=np.int32)
            if not np.array_equal(_call(test, tsh), _python_shift_rows_zero_u8(test, tsh)):
                raise RuntimeError("native row-shift self-test was not byte exact")
            _SELF_TESTED = True
        except Exception as exc:
            _NATIVE_ERROR = f"Rust native core disabled after self-test failure: {exc}"
            _LIB = None
            return _python_shift_rows_zero_u8(src, sh)

    try:
        return _call(src, sh)
    except Exception as exc:
        _NATIVE_ERROR = f"Rust row shift failed; using Python fallback: {exc}"
        return _python_shift_rows_zero_u8(src, sh)


def _python_interp_rows(knots: np.ndarray, xk: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.interp(x, xk, row).astype(np.float32) for row in knots],
        axis=0,
    )


def interp_rows_f32(knots: np.ndarray, xk: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Interpolate multiple rows using the same X coordinates.

    Native results are accepted only when a first-use numerical guard agrees
    with NumPy's reference interpolation to float32 precision.
    """
    global _LIB, _NATIVE_ERROR, _INTERP_TESTED, _INTERP_ENABLED
    k = np.ascontiguousarray(knots, dtype=np.float32)
    xk = np.ascontiguousarray(xk, dtype=np.float32).reshape(-1)
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    if k.ndim != 2 or k.shape[1] != xk.size:
        raise ValueError("knots/xk shape mismatch")
    if k.shape[0] == 0 or x.size == 0:
        return np.empty((k.shape[0], x.size), dtype=np.float32)

    lib = _load_native()
    if lib is None or not _INTERP_ENABLED:
        return _python_interp_rows(k, xk, x)

    out = np.empty((k.shape[0], x.size), dtype=np.float32)
    try:
        rc = lib.dvcr_interp_rows_f32(
            k.ctypes.data_as(POINTER(c_float)), c_size_t(k.shape[0]), c_size_t(k.shape[1]),
            xk.ctypes.data_as(POINTER(c_float)), x.ctypes.data_as(POINTER(c_float)), c_size_t(x.size),
            out.ctypes.data_as(POINTER(c_float)),
        )
        if rc != 0:
            raise RuntimeError(f"dvcr_interp_rows_f32 returned {rc}")

        # First-use guard protects the reference model from compiler/platform
        # floating-point surprises without paying an np.interp cost every field.
        if not _INTERP_TESTED:
            ref0 = np.interp(x, xk, k[0]).astype(np.float32)
            if not np.allclose(out[0], ref0, rtol=0.0, atol=2e-7, equal_nan=True):
                _INTERP_ENABLED = False
                raise RuntimeError("native interpolation exceeded reference tolerance")
            _INTERP_TESTED = True
        return out
    except Exception as exc:
        _NATIVE_ERROR = f"Rust interpolation disabled; using Python fallback: {exc}"
        _INTERP_ENABLED = False
        return _python_interp_rows(k, xk, x)


def native_status() -> tuple[bool, str]:
    lib = _load_native()
    if lib is not None:
        return True, "Rust native core loaded"
    return False, _NATIVE_ERROR or "Rust native core unavailable"
