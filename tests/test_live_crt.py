import numpy as np

from vcr.crt import CRTSettings
from vcr.defects import RecordDefects
from vcr.gui.app import (
    CAMERA_BACKENDS,
    DigitalVCRApp,
    SETTING_HELP,
    _camera_selection_label,
    _camera_values_from_discovery,
    _camera_backend_fallbacks,
    _parse_camera_selection,
    _put_latest,
    _preprocess_live_frame,
    _should_publish_live_signal_loss,
    _update_live_track_meta,
)
from vcr.tape import TapeCartridge, TapeImage


class _BoolVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _FakeCapture:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class _FakeRenderer:
    def __init__(self):
        self.closed = []

    def close_direct(self, key):
        self.closed.append(key)


class _FakePlayer:
    def __init__(self):
        self.ejected = False

    def eject(self):
        self.ejected = True


class _FakeQueue:
    def __init__(self):
        self.items = []

    def put_nowait(self, item):
        self.items.append(item)


def test_live_track_meta_marks_field_pair_base_for_odd_ring_positions():
    meta0 = {}
    meta1 = {}
    rec_def = RecordDefects(tape_mode="SP")

    _update_live_track_meta(
        meta0,
        frame_idx=12,
        base_track=7,
        field_i=0,
        rec_def=rec_def,
        sync_u8=210,
        vjit_u8=9,
        seg_id=33,
    )
    _update_live_track_meta(
        meta1,
        frame_idx=12,
        base_track=7,
        field_i=1,
        rec_def=rec_def,
        sync_u8=210,
        vjit_u8=9,
        seg_id=33,
    )

    assert meta0["frame_base_track"] == 7
    assert meta1["frame_base_track"] == 7
    assert meta0["field_in_frame"] == 0
    assert meta1["field_in_frame"] == 1
    assert meta0["tape_track"] == 7
    assert meta1["tape_track"] == 8
    assert meta0["field_idx"] == 24
    assert meta1["field_idx"] == 25


def test_turning_live_off_clears_stale_frame_without_blocking_on_capture_release():
    app = DigitalVCRApp.__new__(DigitalVCRApp)
    cap = _FakeCapture()
    renderer = _FakeRenderer()
    player = _FakePlayer()

    app.live_toggle_var = _BoolVar(False)
    app.live_cam_var = _BoolVar("0")
    app.live_bufsec_var = _BoolVar("6")
    app._live_on = True
    app._live_cap = cap
    app._latest_live_frame = np.ones((4, 4, 3), dtype=np.uint8)
    app._live_last_good = {"cached": True}
    app._live_tape = TapeImage(TapeCartridge(length_tracks=4))
    app.crt_renderer = renderer
    app.live_player = player

    app._toggle_live()

    assert cap.released is False
    assert app._live_on is False
    assert app._live_cap is cap
    assert app._latest_live_frame is None
    assert app._live_last_good is None
    assert app._live_tape is None
    assert renderer.closed == ["live"]
    assert player.ejected is True


def test_late_live_worker_frame_is_ignored_after_live_turns_off():
    app = DigitalVCRApp.__new__(DigitalVCRApp)
    app._live_on = False
    app._latest_live_frame = None

    def fail_if_called(frame, mode):
        raise AssertionError("CRT should not receive late live frames after stop")

    app._apply_crt_to_frame = fail_if_called

    app._publish_live_frame(np.ones((4, 4, 3), dtype=np.uint8))

    assert app._latest_live_frame is None


def test_live_publish_keeps_raw_frame_when_crt_live_is_enabled():
    app = DigitalVCRApp.__new__(DigitalVCRApp)
    frame = np.ones((4, 4, 3), dtype=np.uint8)
    q = _FakeQueue()

    app._live_on = True
    app._latest_live_frame = None
    app._live_crt_q = q
    app._cached_crt_settings = CRTSettings(enabled=True, live_enabled=True)

    def fail_if_sync_crt_called(frame, mode):
        raise AssertionError("Live publish must not synchronously render CRT")

    app._apply_crt_to_frame = fail_if_sync_crt_called

    app._publish_live_frame(frame)

    assert np.array_equal(app._latest_live_frame, frame)
    assert len(q.items) == 1
    seq, queued = q.items[0]
    assert seq == 1
    assert np.array_equal(queued, frame)


def test_live_crt_worker_does_not_publish_stale_render_over_newer_live_frame():
    app = DigitalVCRApp.__new__(DigitalVCRApp)
    rendered = np.full((4, 4, 3), 255, dtype=np.uint8)

    app._live_on = True
    app._live_publish_seq = 2
    app._latest_live_frame = np.ones((4, 4, 3), dtype=np.uint8)

    app._publish_live_crt_frame(1, rendered)

    assert not np.array_equal(app._latest_live_frame, rendered)


def test_camera_selection_labels_roundtrip_backend_name():
    api = dict(CAMERA_BACKENDS)["Media Foundation"]
    label = _camera_selection_label(3, api)

    assert label == "3"
    assert _parse_camera_selection(label) == (3, dict(CAMERA_BACKENDS)["Auto"])


def test_manual_camera_selection_uses_backend_auto():
    api = dict(CAMERA_BACKENDS)["Auto"]

    assert _parse_camera_selection("12") == (12, api)
    assert _parse_camera_selection("not a camera") == (0, api)


def test_camera_values_keep_manual_indexes_without_probe_results():
    values = _camera_values_from_discovery([], max_index=2)

    assert values == ["0", "1", "2"]


def test_camera_values_put_discovered_entries_first_without_duplicates():
    dshow = dict(CAMERA_BACKENDS)["DirectShow"]

    values = _camera_values_from_discovery([(1, dshow), (1, dict(CAMERA_BACKENDS)["Auto"]), (0, dshow)], max_index=2)

    assert values == ["1", "0"]
    assert values.count("1") == 1


def test_camera_backend_fallbacks_try_selected_then_safer_backends():
    backends = dict(CAMERA_BACKENDS)

    assert _camera_backend_fallbacks(backends["Media Foundation"]) == [
        backends["Media Foundation"],
        backends["DirectShow"],
        backends["Auto"],
    ]
    assert _camera_backend_fallbacks(backends["Auto"]) == [
        backends["Auto"],
        backends["DirectShow"],
    ]


def test_put_latest_drops_stale_queue_item():
    import queue

    q = queue.Queue(maxsize=1)
    _put_latest(q, "old")
    _put_latest(q, "new")

    assert q.get_nowait() == "new"
    assert q.empty()


def test_live_signal_loss_is_not_published_for_transient_read_misses():
    assert _should_publish_live_signal_loss(1) is False
    assert _should_publish_live_signal_loss(7) is False
    assert _should_publish_live_signal_loss(8) is True


def test_preprocess_live_frame_resizes_and_keeps_even_height():
    frame = np.zeros((7, 20, 3), dtype=np.uint8)

    out = _preprocess_live_frame(frame, target_w=10, use_opencl=False)

    assert out.shape == (2, 10, 3)
    assert out.dtype == np.uint8


def test_crt_and_live_tooltips_cover_visible_controls():
    expected = {
        "crt_enabled",
        "crt_preview",
        "crt_live",
        "crt_export",
        "crt_direct",
        "crt_preset",
        "crt_quality",
        "crt_mask",
        "crt_render_width",
        "crt_mask_strength",
        "crt_phosphor_decay",
        "crt_bloom",
        "crt_brightness",
        "live_cam",
        "live_bufsec",
        "live_downscale_width",
        "live_mode",
        "live_overlay",
    }

    assert expected.issubset(SETTING_HELP)
    assert "OpenCL" in SETTING_HELP["live_downscale_width"]
    assert "CRT renderer thread" in SETTING_HELP["crt_direct"]
