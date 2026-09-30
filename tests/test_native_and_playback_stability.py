import numpy as np

from vcr.defects import PlaybackDefects
from vcr.modulation import encode_field_bgr
from vcr.native_core import shift_rows_zero_u8
from vcr.player import VCRPlayer
from vcr.rf_model import _smooth_noise_rows
from vcr.tape import TapeCartridge, TapeImage, TapeTrack


def _reference_smooth_noise_rows(rows: int, n: int, coarse: int) -> np.ndarray:
    return np.stack([
        _reference_smooth_noise_1d(n, coarse) for _ in range(rows)
    ], axis=0)


def _reference_smooth_noise_1d(n: int, coarse: int) -> np.ndarray:
    n = int(max(1, n))
    coarse = int(max(4, min(coarse, n)))
    knots = np.random.randn(coarse).astype(np.float32)
    knots -= knots.mean()
    knots /= (knots.std() + 1e-6)
    xk = np.linspace(0, n - 1, coarse, dtype=np.float32)
    x = np.arange(n, dtype=np.float32)
    y = np.interp(x, xk, knots).astype(np.float32)
    return np.clip(y / (np.max(np.abs(y)) + 1e-6), -1.0, 1.0)


def test_batched_smooth_noise_matches_reference_exactly_without_native_requirement():
    np.random.seed(123456)
    expected = _reference_smooth_noise_rows(12, 237, 11)
    np.random.seed(123456)
    actual = _smooth_noise_rows(12, 237, 11)
    assert np.array_equal(actual, expected)


def test_row_shift_matches_original_zero_fill_operation():
    img = np.arange(7 * 13 * 3, dtype=np.uint8).reshape(7, 13, 3)
    shifts = np.asarray([0, 1, -1, 3, -4, 12, -12], dtype=np.int32)
    expected = np.zeros_like(img)
    w = img.shape[1]
    for y, sft in enumerate(shifts):
        if sft >= 0:
            expected[y, sft:] = img[y, :w-sft]
        else:
            expected[y, :w+sft] = img[y, -sft:]
    assert np.array_equal(shift_rows_zero_u8(img, shifts), expected)


def _small_tape() -> TapeImage:
    img = np.zeros((80, 120, 3), dtype=np.uint8)
    img[..., 0] = np.arange(120, dtype=np.uint8)[None, :]
    img[..., 1] = np.arange(80, dtype=np.uint8)[:, None]
    img[..., 2] = 180
    f0 = img[0::2].copy()
    f1 = img[1::2].copy()
    y0, c0, m0 = encode_field_bgr(f0)
    y1, c1, m1 = encode_field_bgr(f1)
    for field_i, meta in enumerate((m0, m1)):
        meta.update({
            "frame_base_track": 0,
            "field_in_frame": field_i,
            "tape_mode": "SP",
            "seg_id": 1,
            "ctl_sync_u8": 240,
            "ctl_vjit_u8": 4,
            "real_rf_modulation": False,
        })
    cart = TapeCartridge(length_tracks=2)
    cart.set(0, TapeTrack(y0, c0, m0))
    cart.set(1, TapeTrack(y1, c1, m1))
    return TapeImage(cart)


def test_decode_cache_invalidates_when_recombination_setting_changes():
    tape = _small_tape()
    player = VCRPlayer()
    player.state.inserted = True
    player.state.inserting_timer = 2.0
    player.state.lock = 1.0
    player.state._last_tracking_err = 0.0

    clean = PlaybackDefects(
        playback_rf_noise=0.0,
        playback_dropouts=0.0,
        chroma_noise=0.0,
        luma_chroma_bleed=0.0,
        rf_playback_model=False,
    )
    bleed = PlaybackDefects(
        playback_rf_noise=0.0,
        playback_dropouts=0.0,
        chroma_noise=0.0,
        luma_chroma_bleed=0.9,
        rf_playback_model=False,
    )

    a = player._decode_track_with_rf(tape, 0, clean)
    b = player._decode_track_with_rf(tape, 0, bleed)
    assert a is not b
    assert not np.array_equal(a, b)
