from __future__ import annotations
from dataclasses import dataclass
import time
import cv2
import numpy as np

from .tape import TapeImage, TapeTrack
from .modulation import encode_field_bgr, decode_field_bgr
from .defects import RecordDefects, apply_record_defects_to_field, apply_rf_defects_y_dphi_u8, apply_rf_defects_chroma_u8
from .audio import extract_audio_mono_pcm16, apply_audio_record_defects
from .defects import AudioRecordDefects, mode_scale

@dataclass
class RecordOptions:
    downscale_width: int = 360
    enforce_real_time: bool = True
    sample_rate: int = 2_000_000
    chroma_subsample: int = 2
    extract_audio: bool = True

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
        self.last_error = None
        self.last_audio_error = None

        # Audio extraction is handled below (src_audio) so we can overwrite only the recorded region.
        if opts.extract_audio:
            _pcm, err = extract_audio_mono_pcm16(path, sample_rate=int(tape.audio.sample_rate or 44100))
            if _pcm is None and err:
                self.last_audio_error = err

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.last_error = f"Could not open video: {path}"
            return False, start_track

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-3:
            fps = 30.0
        dt = 1.0 / float(fps)
        src_fps = float(fps) if fps and fps > 1e-6 else 30.0
        tape_fps = 30.0
        dt_tape = 1.0 / tape_fps

        # ---- Audio capture from source (ffmpeg) and overwrite into tape audio track ----
        src_audio, _audio_err = extract_audio_mono_pcm16(path, sample_rate=int(tape.audio.sample_rate or 44100))
        sr = int(tape.audio.sample_rate or 44100)
        tape.audio.sample_rate = sr

        # ensure tape audio buffer exists and matches tape length (60 fields/sec)
        tape_len_s = float(tape.cart.length_tracks) / 60.0
        need_n = int(tape_len_s * sr)
        if tape.audio.pcm16 is None or tape.audio.pcm16.size != need_n:
            tape.audio.pcm16 = np.zeros((need_n,), dtype=np.int16)

        track_pos = int(max(0, min(tape.cart.length_tracks-2, start_track)))

        # Overwrite behaviour: erase a short region *before* the new recording so playback cuts to black
        # (like a real tape edit point) and old audio doesn't "bleed" through.
        wipe_tracks = int(60 * 0.60)  # ~0.60s
        wipe0 = max(0, track_pos - wipe_tracks)
        for k in range(wipe0, track_pos):
            try:
                tape.cart.tracks.pop(k, None)
            except Exception:
                pass
        if tape.audio.pcm16 is not None:
            wipe_sec0 = float(wipe0) / 60.0
            wipe_sec1 = float(track_pos) / 60.0
            a0 = int(wipe_sec0 * sr)
            a1 = int(wipe_sec1 * sr)
            a0 = max(0, min(a0, tape.audio.pcm16.size))
            a1 = max(a0, min(a1, tape.audio.pcm16.size))
            if a1 > a0:
                tape.audio.pcm16[a0:a1] = 0
        t0 = time.perf_counter()
        frame_idx = 0
        seg_id = int(time.time()*1000) & 0x7fffffff

        # Prime the source reader
        ret, last_frame = cap.read()
        if not ret:
            cap.release()
            return True, int(track_pos)
        src_idx = 0
        src_time = 0.0

        while True:
            if track_pos + 1 >= tape.cart.length_tracks:
                break

            # Tape records at fixed 30fps (60 fields/sec). We sample the source at tape time.
            target_time = float(frame_idx) * float(dt_tape)
            while src_time + 1e-6 < target_time:
                ret, fr = cap.read()
                if not ret:
                    last_frame = None
                    break
                last_frame = fr
                src_idx += 1
                src_time = float(src_idx) / float(src_fps if src_fps > 1e-6 else 30.0)
            if last_frame is None:
                break

            frame = last_frame

            h, w = frame.shape[:2]
            if w > opts.downscale_width:
                new_w = int(opts.downscale_width)
                new_h = int(h * (new_w / w))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # IMPORTANT: ensure even height to avoid field size mismatch
            if frame.shape[0] % 2 == 1:
                frame = frame[:-1, :, :]

            field0 = frame[0::2].copy()
            field1 = frame[1::2].copy()

            field0b = apply_record_defects_to_field(field0, rec_def)
            field1b = apply_record_defects_to_field(field1, rec_def)

            y0, c0, meta0 = encode_field_bgr(field0b, sample_rate=opts.sample_rate,
                                             chroma_subsample=opts.chroma_subsample,
                                             luma_bw=rec_def.luma_bw)
            y1, c1, meta1 = encode_field_bgr(field1b, sample_rate=opts.sample_rate,
                                             chroma_subsample=opts.chroma_subsample,
                                             luma_bw=rec_def.luma_bw)

            y0 = apply_rf_defects_y_dphi_u8(y0, rec_def.record_rf_noise, rec_def.record_dropouts, rec_def.tape_mode)
            c0 = apply_rf_defects_chroma_u8(c0, rec_def.record_rf_noise, rec_def.record_dropouts, rec_def.tape_mode)
            y1 = apply_rf_defects_y_dphi_u8(y1, rec_def.record_rf_noise, rec_def.record_dropouts, rec_def.tape_mode)
            c1 = apply_rf_defects_chroma_u8(c1, rec_def.record_rf_noise, rec_def.record_dropouts, rec_def.tape_mode)

            sync_u8, vjit_u8 = self._control_track_values(rec_def)

            for field_i, (yy, cc, meta) in enumerate([(y0,c0,meta0),(y1,c1,meta1)]):
                idx = track_pos + field_i
                meta.update({
                    "dt": dt_tape/2.0,
                    "fps": float(tape_fps),
                    "frame_idx": int(frame_idx),
                    "field_idx": int(frame_idx*2 + field_i),
                    "head": "A" if (frame_idx*2 + field_i) % 2 == 0 else "B",
                    "tape_mode": rec_def.tape_mode,
                    "tape_track": int(idx),
                    # control track fields (lightweight)
                    "ctl_sync_u8": int(sync_u8),
                    "ctl_vjit_u8": int(vjit_u8),
                    "seg_id": int(seg_id),
                })
                tape.cart.set(idx, TapeTrack(y_dphi8=yy, c_u8=cc, meta=meta))

            
# Write audio chunk for this frame into tape audio buffer (overwrite)
            if tape.audio.pcm16 is not None:
                out_sec = float(track_pos) / 60.0
                out_start = int(out_sec * sr)
                out_len = int(sr / 30.0)
                out_end = min(out_start + out_len, tape.audio.pcm16.size)

                if src_audio is not None and src_audio.size > 0:
                    in_start = int((frame_idx * dt_tape) * sr)
                    in_end = int(((frame_idx + 1) * dt_tape) * sr)
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

                        res = apply_audio_record_defects(res, sr, wow=aud_rec.wow, hiss=aud_rec.hiss, dropouts=aud_rec.dropouts, compression=float(getattr(aud_rec,'compression',0.55)))

                        tape.audio.pcm16[out_start:out_end] = res[:(out_end - out_start)]
                else:
                    # If source has no audio (or extract failed), overwrite the recorded region with silence
                    if out_end > out_start:
                        tape.audio.pcm16[out_start:out_end] = 0

            if preview_cb is not None:
                if monitor_mode == "input":
                    preview_cb(frame)
                else:
                    tA = tape.cart.get(track_pos)
                    tB = tape.cart.get(track_pos+1)
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

            frame_idx += 1
            track_pos += 2

            if progress_cb:
                progress_cb(frame_idx, track_pos)

            if opts.enforce_real_time:
                target = t0 + frame_idx * dt_tape
                now = time.perf_counter()
                sleep_s = target - now
                if sleep_s > 0:
                    time.sleep(sleep_s)

        cap.release()
        return True, track_pos
