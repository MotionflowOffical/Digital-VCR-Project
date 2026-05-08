from __future__ import annotations
from dataclasses import dataclass
import os
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from .tape import TapeImage, TapeTrack
from .modulation import encode_field_bgr, decode_field_bgr
from .defects import RecordDefects, apply_record_defects_to_field, apply_rf_defects_y_dphi_u8, apply_rf_defects_chroma_u8
from .rf_model import rf_roundtrip_luma_dphi_u8, rf_roundtrip_chroma_u8
from .audio import extract_audio_mono_pcm16, apply_audio_record_defects
from .defects import AudioRecordDefects, mode_scale

@dataclass
class RecordOptions:
    downscale_width: int = 360
    enforce_real_time: bool = True
    sample_rate: int = 2_000_000
    chroma_subsample: int = 2
    extract_audio: bool = True

    # How to derive two tape fields from each source frame.
    # - "interlaced": take even/odd lines from the same frame (classic weave source -> 2 fields)
    # - "progressive": vertically downsample then duplicate into both fields (avoids comb/tearing on progressive displays)
    field_sampling: str = "progressive"

    # For variable-frame-rate sources, using container timestamps prevents uneven sampling.
    use_src_timestamps: bool = True

    # Encoding threads for record_from_file.
    # 0 = auto (min(max(2, cpu//2), 8)), 1 = single-thread, >1 = ThreadPoolExecutor workers.
    encode_threads: int = 0


def sample_fields_from_frame(frame_bgr: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a progressive BGR frame into two half-height fields.

    This is a *sampling* choice, not a defect:
    - interlaced: classic even/odd line split (shows comb/scanline artifacts on progressive monitors during scrubbing)
    - progressive: vertical downsample + duplicate (removes combing/tearing that comes purely from weaving)
    """
    m = (mode or "").strip().lower()
    if m in ("progressive", "prog", "dup", "duplicate"):
        # "Progressive" sampling should avoid inter-field *time* mismatch artifacts
        # without throwing away half the vertical detail.
        #
        # Earlier implementation downscaled to half-height and duplicated into both fields.
        # That removes combing, but also halves vertical resolution, making fine detail
        # (especially text) look blocky/digital.
        #
        # Instead: apply a small vertical low-pass (reduces line twitter) and then split
        # even/odd lines from the same *filtered* frame.
        # Keep this subtle; too much vertical blur will kill readability (text, UI).
        # OpenCV requires a positive odd kernel size when sigmaX==0.
        # Use a *vertical* kernel (width=1) to minimize horizontal softening.
        filt = cv2.GaussianBlur(frame_bgr, (1, 5), sigmaX=0.0, sigmaY=0.55)
        f0 = filt[0::2].copy()
        f1 = filt[1::2].copy()
        # If height is odd, ensure both fields have identical height.
        hh = min(f0.shape[0], f1.shape[0])
        return f0[:hh], f1[:hh]
    # default
    f0 = frame_bgr[0::2].copy()
    f1 = frame_bgr[1::2].copy()
    return f0, f1


def _cap_time_sec(cap: cv2.VideoCapture, fallback_idx: int, fallback_fps: float, use_ts: bool) -> float:
    if use_ts:
        try:
            t_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            if t_msec > 0.0:
                return t_msec / 1000.0
        except Exception:
            pass
    # fallback: constant-fps timebase
    fps = float(fallback_fps) if float(fallback_fps) > 1e-6 else 30.0
    return float(fallback_idx) / fps


def _auto_encode_threads(req: int) -> int:
    """Return a sensible thread count for record-side encoding.

    Notes:
      - Most heavy ops here (OpenCV resize/blur, NumPy ops) release the GIL.
      - Too many threads can hurt due to memory bandwidth.
    """
    if int(req) > 0:
        return int(req)
    cpu = os.cpu_count() or 4
    return int(max(2, min(8, cpu // 2)))


def _process_frame_to_tracks(frame_bgr: np.ndarray,
                             frame_idx: int,
                             base_track: int,
                             dt_tape: float,
                             tape_fps: float,
                             opts: RecordOptions,
                             rec_def: RecordDefects,
                             seg_id: int,
                             sync_u8: int,
                             vjit_u8: int) -> tuple[int, int, TapeTrack, TapeTrack]:
    """CPU-heavy per-frame pipeline.

    Returns two TapeTracks (field A/B) for a single source frame.
    """
    field0, field1 = sample_fields_from_frame(frame_bgr, getattr(opts, 'field_sampling', 'interlaced'))

    field0b = apply_record_defects_to_field(field0, rec_def)
    field1b = apply_record_defects_to_field(field1, rec_def)

    y0, c0, meta0 = encode_field_bgr(
        field0b,
        sample_rate=opts.sample_rate,
        chroma_subsample=opts.chroma_subsample,
        luma_bw=rec_def.luma_bw,
        chroma_bw=float(getattr(rec_def, 'chroma_bw', 1.0)),
    )
    y1, c1, meta1 = encode_field_bgr(
        field1b,
        sample_rate=opts.sample_rate,
        chroma_subsample=opts.chroma_subsample,
        luma_bw=rec_def.luma_bw,
        chroma_bw=float(getattr(rec_def, 'chroma_bw', 1.0)),
    )

    y0 = apply_rf_defects_y_dphi_u8(y0, rec_def.record_rf_noise, rec_def.record_dropouts, rec_def.tape_mode)
    c0 = apply_rf_defects_chroma_u8(c0, rec_def.record_rf_noise, rec_def.record_dropouts, rec_def.tape_mode)
    y1 = apply_rf_defects_y_dphi_u8(y1, rec_def.record_rf_noise, rec_def.record_dropouts, rec_def.tape_mode)
    c1 = apply_rf_defects_chroma_u8(c1, rec_def.record_rf_noise, rec_def.record_dropouts, rec_def.tape_mode)

    # Optional: RF carrier simulation round-trip (FM + AM).
    if bool(getattr(rec_def, 'real_rf_modulation', False)):
        try:
            y0 = rf_roundtrip_luma_dphi_u8(
                y0, meta0,
                noise=float(rec_def.record_rf_noise),
                dropouts=float(rec_def.record_dropouts),
                mode=str(rec_def.tape_mode),
                fm_depth=float(getattr(rec_def, 'rf_fm_depth', 1.0)),
                am_depth=float(getattr(rec_def, 'rf_am_depth', 0.25)),
                nonlinearity=float(getattr(rec_def, 'rf_nonlinearity', 0.25)),
                carrier_noise=float(getattr(rec_def, 'rf_carrier_noise', 0.20)),
                phase_noise=float(getattr(rec_def, 'rf_phase_noise', 0.10)),
            )
            c0 = rf_roundtrip_chroma_u8(
                c0, meta0,
                noise=float(rec_def.record_rf_noise),
                dropouts=float(rec_def.record_dropouts),
                mode=str(rec_def.tape_mode),
                fc_frac=float(getattr(rec_def, 'rf_chroma_fc_frac', 0.12)),
                lpf_strength=float(getattr(rec_def, 'rf_chroma_lpf', 0.35)),
                am_depth=float(getattr(rec_def, 'rf_am_depth', 0.25)),
                nonlinearity=float(getattr(rec_def, 'rf_nonlinearity', 0.25)),
                carrier_noise=float(getattr(rec_def, 'rf_carrier_noise', 0.20)),
                phase_noise=float(getattr(rec_def, 'rf_phase_noise', 0.10)),
            )

            y1 = rf_roundtrip_luma_dphi_u8(
                y1, meta1,
                noise=float(rec_def.record_rf_noise),
                dropouts=float(rec_def.record_dropouts),
                mode=str(rec_def.tape_mode),
                fm_depth=float(getattr(rec_def, 'rf_fm_depth', 1.0)),
                am_depth=float(getattr(rec_def, 'rf_am_depth', 0.25)),
                nonlinearity=float(getattr(rec_def, 'rf_nonlinearity', 0.25)),
                carrier_noise=float(getattr(rec_def, 'rf_carrier_noise', 0.20)),
                phase_noise=float(getattr(rec_def, 'rf_phase_noise', 0.10)),
            )
            c1 = rf_roundtrip_chroma_u8(
                c1, meta1,
                noise=float(rec_def.record_rf_noise),
                dropouts=float(rec_def.record_dropouts),
                mode=str(rec_def.tape_mode),
                fc_frac=float(getattr(rec_def, 'rf_chroma_fc_frac', 0.12)),
                lpf_strength=float(getattr(rec_def, 'rf_chroma_lpf', 0.35)),
                am_depth=float(getattr(rec_def, 'rf_am_depth', 0.25)),
                nonlinearity=float(getattr(rec_def, 'rf_nonlinearity', 0.25)),
                carrier_noise=float(getattr(rec_def, 'rf_carrier_noise', 0.20)),
                phase_noise=float(getattr(rec_def, 'rf_phase_noise', 0.10)),
            )
        except Exception:
            pass

    # Meta for each field
    for field_i, meta in enumerate((meta0, meta1)):
        idx = int(base_track + field_i)
        meta.update({
            "dt": float(dt_tape) / 2.0,
            "fps": float(tape_fps),
            "frame_idx": int(frame_idx),
            "field_idx": int(frame_idx * 2 + field_i),
            "frame_base_track": int(base_track),
            "field_in_frame": int(field_i),
            "head": "A" if (frame_idx*2 + field_i) % 2 == 0 else "B",
            "tape_mode": str(rec_def.tape_mode),
            "tape_track": int(idx),
            "ctl_sync_u8": int(sync_u8),
            "ctl_vjit_u8": int(vjit_u8),
            "seg_id": int(seg_id),
            "real_rf_modulation": bool(getattr(rec_def, 'real_rf_modulation', False)),
            "rf_fm_depth": float(getattr(rec_def, 'rf_fm_depth', 1.0)),
            "rf_chroma_fc_frac": float(getattr(rec_def, 'rf_chroma_fc_frac', 0.12)),
            "rf_chroma_lpf": float(getattr(rec_def, 'rf_chroma_lpf', 0.35)),
        })

    tr0 = TapeTrack(y_dphi8=y0, c_u8=c0, meta=meta0)
    tr1 = TapeTrack(y_dphi8=y1, c_u8=c1, meta=meta1)
    return int(frame_idx), int(base_track), tr0, tr1

class Recorder:
    def __init__(self):
        self.last_error: str | None = None
        self.last_audio_error: str | None = None

    def _control_track_values(self, rec_def: RecordDefects) -> tuple[int, int]:
        # u8 sync strength + vsync jitter proxy
        s = mode_scale(rec_def.tape_mode)
        base = 1.0 - (0.9*rec_def.record_rf_noise*s + 1.3*rec_def.record_dropouts*s)
        base = float(np.clip(base + np.random.randn()*0.05, 0.0, 1.0))
        sync_u8 = int(np.clip(base * 255.0, 0, 255))
        vjit = float(np.clip(rec_def.record_jitter*s + np.random.rand()*0.15*s, 0.0, 1.0))
        vjit_u8 = int(np.clip(vjit * 255.0, 0, 255))
        return sync_u8, vjit_u8

    def record_from_file(self, path: str, tape: TapeImage, start_track: int,
                         opts: RecordOptions, rec_def: RecordDefects, aud_rec: AudioRecordDefects,
                         progress_cb=None, preview_cb=None, monitor_mode: str = "tape") -> tuple[bool, int]:
        """Record a source file into the tape.

        Performance notes:
          - The GUI already runs this method on a background thread.
          - This method can additionally parallelize the CPU-heavy encode/RF steps
            across multiple worker threads (opts.encode_threads).
        """
        self.last_error = None
        self.last_audio_error = None

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.last_error = f"Could not open video: {path}"
            return False, start_track

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-3:
            fps = 30.0
        src_fps = float(fps) if fps and fps > 1e-6 else 30.0

        tape_fps = 30.0
        dt_tape = 1.0 / tape_fps

        # ---- Audio capture from source (ffmpeg) and overwrite into tape audio track ----
        src_audio = None
        if opts.extract_audio:
            src_audio, _audio_err = extract_audio_mono_pcm16(path, sample_rate=int(tape.audio.sample_rate or 44100))
            if src_audio is None and _audio_err:
                self.last_audio_error = _audio_err
        sr = int(tape.audio.sample_rate or 44100)
        tape.audio.sample_rate = sr

        # ensure tape audio buffer exists and matches tape length (60 fields/sec)
        tape_len_s = float(tape.cart.length_tracks) / 60.0
        need_n = int(tape_len_s * sr)
        if tape.audio.pcm16 is None or tape.audio.pcm16.size != need_n:
            tape.audio.pcm16 = np.zeros((need_n,), dtype=np.int16)

        # Clamp start position (must have room for 2 tracks)
        base_track = int(max(0, min(tape.cart.length_tracks-2, start_track)))

        # Overwrite behaviour: erase a short region *before* the new recording so playback cuts to black
        wipe_tracks = int(60 * 0.60)  # ~0.60s
        wipe0 = max(0, base_track - wipe_tracks)
        for k in range(wipe0, base_track):
            try:
                tape.cart.tracks.pop(k, None)
            except Exception:
                pass
        if tape.audio.pcm16 is not None:
            wipe_sec0 = float(wipe0) / 60.0
            wipe_sec1 = float(base_track) / 60.0
            a0 = int(wipe_sec0 * sr)
            a1 = int(wipe_sec1 * sr)
            a0 = max(0, min(a0, tape.audio.pcm16.size))
            a1 = max(a0, min(a1, tape.audio.pcm16.size))
            if a1 > a0:
                tape.audio.pcm16[a0:a1] = 0

        # Prime the source reader
        ret, last_frame = cap.read()
        if not ret:
            cap.release()
            return True, int(base_track)
        src_idx = 0
        src_time = _cap_time_sec(cap, src_idx, src_fps, bool(getattr(opts, 'use_src_timestamps', True)))

        # Threading controls
        enc_threads = _auto_encode_threads(int(getattr(opts, 'encode_threads', 0)))
        use_mt = bool(enc_threads > 1)
        max_inflight = int(max(4, enc_threads * 3))

        t0 = time.perf_counter()
        seg_id = int(time.time()*1000) & 0x7fffffff

        submit_frame_idx = 0
        written_frames = 0
        endpos = int(base_track)

        # Helpers for write-back + UI callbacks (runs on the caller thread)
        def _write_one(fi: int, bt: int, tr0: TapeTrack, tr1: TapeTrack):
            nonlocal written_frames, endpos
            tape.cart.set(bt, tr0)
            tape.cart.set(bt + 1, tr1)

            # --- Audio overwrite for this frame ---
            if tape.audio.pcm16 is not None:
                out_sec = float(bt) / 60.0
                out_start = int(out_sec * sr)
                out_len = int(sr / 30.0)
                out_end = min(out_start + out_len, tape.audio.pcm16.size)

                if src_audio is not None and src_audio.size > 0:
                    in_start = int((fi * dt_tape) * sr)
                    in_end = int(((fi + 1) * dt_tape) * sr)
                    in_start = max(0, min(in_start, src_audio.size))
                    in_end = max(in_start, min(in_end, src_audio.size))
                    src_seg = src_audio[in_start:in_end].astype(np.int16, copy=False)

                    if out_end > out_start and out_len > 8:
                        if src_seg.size < 2:
                            res = np.zeros((out_end - out_start,), dtype=np.int16)
                        else:
                            xs = np.linspace(0.0, 1.0, src_seg.size, dtype=np.float32)
                            xo = np.linspace(0.0, 1.0, out_end - out_start, dtype=np.float32)
                            res_f = np.interp(xo, xs, src_seg.astype(np.float32))
                            res = np.clip(res_f, -32768, 32767).astype(np.int16)

                        res = apply_audio_record_defects(
                            res, sr,
                            wow=aud_rec.wow,
                            hiss=aud_rec.hiss,
                            dropouts=aud_rec.dropouts,
                            compression=float(getattr(aud_rec, 'compression', 0.55))
                        )
                        tape.audio.pcm16[out_start:out_end] = res[:(out_end - out_start)]
                else:
                    if out_end > out_start:
                        tape.audio.pcm16[out_start:out_end] = 0

            # --- Record monitor ---
            if preview_cb is not None and monitor_mode == "tape":
                try:
                    tA = tape.cart.get(bt)
                    tB = tape.cart.get(bt+1)
                    if tA and tB:
                        f0 = decode_field_bgr(tA.y_dphi8, tA.c_u8, tA.meta)
                        f1 = decode_field_bgr(tB.y_dphi8, tB.c_u8, tB.meta)
                        hh = min(f0.shape[0], f1.shape[0])
                        ww = min(f0.shape[1], f1.shape[1])
                        f0 = f0[:hh, :ww]
                        f1 = f1[:hh, :ww]
                        out = np.zeros((hh*2, ww, 3), dtype=np.uint8)
                        out[0::2] = f0
                        out[1::2] = f1
                        preview_cb(out)
                except Exception:
                    pass

            written_frames += 1
            endpos = int(bt + 2)

            if progress_cb:
                try:
                    progress_cb(int(fi + 1), int(endpos))
                except Exception:
                    pass

            if opts.enforce_real_time:
                target = t0 + written_frames * dt_tape
                now = time.perf_counter()
                sleep_s = target - now
                if sleep_s > 0:
                    time.sleep(sleep_s)

        # --- Multi-thread bookkeeping ---
        executor = None
        pending = {}           # future -> base_track
        order = []             # base_tracks in submission order
        ready = {}             # base_track -> (fi, bt, tr0, tr1)

        def _collect_done(block: bool = False):
            if not pending:
                return
            timeout = None if block else 0.0
            try:
                done, _ = wait(list(pending.keys()), timeout=timeout, return_when=FIRST_COMPLETED)
            except Exception:
                done = []
            for fut in list(done):
                bt = pending.pop(fut, None)
                if bt is None:
                    continue
                try:
                    ready[bt] = fut.result()
                except Exception as e:
                    raise e

        def _flush_ready():
            while order and order[0] in ready:
                bt = order.pop(0)
                fi, bt2, tr0, tr1 = ready.pop(bt)
                _write_one(fi, bt2, tr0, tr1)

        try:
            if use_mt:
                executor = ThreadPoolExecutor(max_workers=enc_threads)

            while True:
                if base_track + 1 >= tape.cart.length_tracks:
                    break

                # Tape records at fixed 30fps (60 fields/sec). We sample the source at tape time.
                target_time = float(submit_frame_idx) * float(dt_tape)
                while src_time + 1e-6 < target_time:
                    ret, fr = cap.read()
                    if not ret:
                        last_frame = None
                        break
                    last_frame = fr
                    src_idx += 1
                    src_time = _cap_time_sec(cap, src_idx, src_fps, bool(getattr(opts, 'use_src_timestamps', True)))
                if last_frame is None:
                    break

                frame = last_frame

                h, w = frame.shape[:2]
                if w > opts.downscale_width:
                    new_w = int(opts.downscale_width)
                    new_h = int(h * (new_w / w))
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # ensure even height to avoid field size mismatch
                if frame.shape[0] % 2 == 1:
                    frame = frame[:-1, :, :]

                if preview_cb is not None and monitor_mode == "input":
                    try:
                        preview_cb(frame)
                    except Exception:
                        pass

                sync_u8, vjit_u8 = self._control_track_values(rec_def)

                if use_mt and executor is not None:
                    fut = executor.submit(
                        _process_frame_to_tracks,
                        frame,
                        int(submit_frame_idx),
                        int(base_track),
                        float(dt_tape),
                        float(tape_fps),
                        opts,
                        rec_def,
                        int(seg_id),
                        int(sync_u8),
                        int(vjit_u8),
                    )
                    pending[fut] = int(base_track)
                    order.append(int(base_track))

                    submit_frame_idx += 1
                    base_track += 2

                    _collect_done(block=False)
                    _flush_ready()

                    # bound memory
                    if len(pending) >= max_inflight:
                        _collect_done(block=True)
                        _flush_ready()
                else:
                    fi, bt, tr0, tr1 = _process_frame_to_tracks(
                        frame,
                        int(submit_frame_idx),
                        int(base_track),
                        float(dt_tape),
                        float(tape_fps),
                        opts,
                        rec_def,
                        int(seg_id),
                        int(sync_u8),
                        int(vjit_u8),
                    )
                    _write_one(fi, bt, tr0, tr1)
                    submit_frame_idx += 1
                    base_track += 2

            # Drain any remaining pending work
            if use_mt and executor is not None:
                while pending:
                    _collect_done(block=True)
                    _flush_ready()
        except Exception as e:
            self.last_error = f"Recording failed: {e}"
            cap.release()
            try:
                if executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            return False, int(endpos)
        finally:
            cap.release()
            try:
                if executor is not None:
                    executor.shutdown(wait=False)
            except Exception:
                pass

        return True, int(endpos)
