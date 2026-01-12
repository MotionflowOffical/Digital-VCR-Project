from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np

from .tape import TapeImage, TapeTrack, TapeCartridge
from .modulation import decode_field_bgr, encode_field_bgr
from .defects import (
    RecordDefects, PlaybackDefects,
    apply_record_defects_to_field,
    apply_rf_defects_y_dphi_u8, apply_rf_defects_chroma_u8,
    apply_timebase_wobble, apply_chroma_shift, apply_interference, apply_composite_view, enforce_aspect,
    apply_image_controls, apply_scanline_soften
)

@dataclass
class DubOptions:
    enforce_real_time: bool = True
    sample_rate: int = 2_000_000
    chroma_subsample: int = 2

class Editor:
    def preview_step(self, tape: TapeImage, pos_tracks: int, rec_def: RecordDefects, pb: PlaybackDefects, lock: float = 0.95):
        if tape.cart.length_tracks < 2:
            return None, pos_tracks
        if pos_tracks + 1 >= tape.cart.length_tracks:
            pos_tracks = 0

        t0 = tape.cart.get(pos_tracks)
        t1 = tape.cart.get(pos_tracks+1)

        if t0 is None or t1 is None:
            return np.zeros((480, 640, 3), dtype=np.uint8), pos_tracks + 2

        f0 = decode_field_bgr(t0.y_dphi8, t0.c_u8, t0.meta)
        f1 = decode_field_bgr(t1.y_dphi8, t1.c_u8, t1.meta)

        hh = min(f0.shape[0], f1.shape[0])
        ww = min(f0.shape[1], f1.shape[1])
        f0 = f0[:hh, :ww]
        f1 = f1[:hh, :ww]

        frame = np.zeros((hh*2, ww, 3), dtype=np.uint8)
        frame[0::2] = f0
        frame[1::2] = f1

        frame = apply_record_defects_to_field(frame, rec_def)

        gdx = int(np.sin(time.time()*1.1) * (1+8*pb.playback_timebase*(1.0-lock)))
        gdy = int(np.sin(time.time()*0.9) * (1+4*pb.playback_timebase*(1.0-lock)))
        frame = apply_timebase_wobble(frame, pb.playback_timebase, lock=lock, global_dx=gdx, global_dy=gdy)

        frame = apply_chroma_shift(frame, pb.chroma_shift_x, pb.chroma_shift_y, pb.chroma_phase, pb.chroma_noise)
        frame = apply_interference(frame, pb.interference)
        if pb.composite_view:
            frame = apply_composite_view(frame, 0.30)

        frame = apply_scanline_soften(frame, pb.scanline_soften)
        frame = enforce_aspect(frame, pb.aspect_display)
        frame = apply_image_controls(frame, pb.brightness, pb.contrast, pb.saturation, pb.bloom, pb.sharpen)

        return frame, pos_tracks + 2

    def dub_rerecord(self, tape_in: TapeImage, rec_def: RecordDefects,
                     opts: DubOptions, progress_cb=None, preview_cb=None) -> TapeImage:
        out = TapeImage(cart=TapeCartridge(length_tracks=tape_in.cart.length_tracks))
        out.audio = tape_in.audio

        fps = 30.0
        if tape_in.cart.tracks:
            any_tr = next(iter(tape_in.cart.tracks.values()))
            fps = float(any_tr.meta.get("fps", 30.0))
        dt = 1.0 / fps

        t0 = time.perf_counter()
        frame_idx = 0

        for pos in range(0, tape_in.cart.length_tracks-1, 2):
            a = tape_in.cart.get(pos)
            b = tape_in.cart.get(pos+1)
            if a is None or b is None:
                continue

            f0 = decode_field_bgr(a.y_dphi8, a.c_u8, a.meta)
            f1 = decode_field_bgr(b.y_dphi8, b.c_u8, b.meta)

            hh = min(f0.shape[0], f1.shape[0])
            ww = min(f0.shape[1], f1.shape[1])
            f0 = f0[:hh, :ww]
            f1 = f1[:hh, :ww]

            frame = np.zeros((hh*2, ww, 3), dtype=np.uint8)
            frame[0::2] = f0
            frame[1::2] = f1

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

            # preserve control track if present
            ctl_sync = int(a.meta.get("ctl_sync_u8", 200))
            ctl_vjit = int(a.meta.get("ctl_vjit_u8", 20))

            for field_i, (yy, cc, meta) in enumerate([(y0,c0,meta0),(y1,c1,meta1)]):
                idx = pos + field_i
                meta.update({
                    "dt": dt/2.0,
                    "fps": float(fps),
                    "frame_idx": int(frame_idx),
                    "field_idx": int(frame_idx*2 + field_i),
                    "head": "A" if (frame_idx*2 + field_i) % 2 == 0 else "B",
                    "tape_mode": rec_def.tape_mode,
                    "tape_track": int(idx),
                    "ctl_sync_u8": int(ctl_sync),
                    "ctl_vjit_u8": int(ctl_vjit),
                })
                out.cart.set(idx, TapeTrack(y_dphi8=yy, c_u8=cc, meta=meta))

            if preview_cb:
                preview_cb(frame)

            frame_idx += 1
            if progress_cb:
                progress_cb(frame_idx, tape_in.cart.length_tracks//2)

            if opts.enforce_real_time:
                target = t0 + frame_idx * dt
                now = time.perf_counter()
                sleep_s = target - now
                if sleep_s > 0:
                    time.sleep(sleep_s)

        return out
