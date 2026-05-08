from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np

from .tape import TapeImage, TapeTrack, TapeCartridge, TapeAudio
from .audio import (
    write_wav_mono_pcm16, read_wav_mono_pcm16,
    pcm16_to_ulaw, ulaw_to_pcm16,
)

MODE_TO_U8 = {"SP": 0, "LP": 1, "EP": 2}
U8_TO_MODE = {0: "SP", 1: "LP", 2: "EP"}


def create_blank_bundle(folder: str, *, length_tracks: int, settings: Dict[str, Any],
                        sample_rate: int = 2_000_000, f0: float = 350_000.0, fdev: float = 120_000.0,
                        chroma_subsample: int = 2) -> None:
    """Create a minimal empty bundle on disk quickly.
    This avoids building large arrays and is used for 'New tape' creation.
    """
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)

    fmin = f0 - fdev
    fmax = f0 + fdev
    dphi_min = float(2*np.pi * fmin / float(sample_rate))
    dphi_max = float(2*np.pi * fmax / float(sample_rate))

    tape_info = {
        "length_tracks": int(length_tracks),
        "sample_rate": int(sample_rate),
        "f0": float(f0),
        "fdev": float(fdev),
        "chroma_subsample": int(chroma_subsample),
        "dphi_min": float(dphi_min),
        "dphi_max": float(dphi_max),
    }

    (out / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (out / "tape_info.json").write_text(json.dumps(tape_info, indent=2), encoding="utf-8")

    # Empty compact tape containers
    np.savez(out / "tape_luma.npz",
             track_index=np.zeros((0,), np.int32),
             y_data=np.zeros((0,), np.uint8),
             y_offsets=np.zeros((1,), np.int64),
             y_h=np.zeros((0,), np.uint16),
             y_w=np.zeros((0,), np.uint16),
             y_mod_w=np.zeros((0,), np.uint16),
             c_h=np.zeros((0,), np.uint16),
             c_w=np.zeros((0,), np.uint16),
             luma_bw=np.zeros((0,), np.float32),
             dt=np.zeros((0,), np.float32),
             fps=np.zeros((0,), np.float32),
             frame_idx=np.zeros((0,), np.int32),
             field_idx=np.zeros((0,), np.int32),
             head_u8=np.zeros((0,), np.uint8),
             mode_u8=np.zeros((0,), np.uint8),
             ctl_sync_u8=np.zeros((0,), np.uint8),
             ctl_vjit_u8=np.zeros((0,), np.uint8),
             real_rf_u8=np.zeros((0,), np.uint8),
             rf_chroma_fc_frac=np.zeros((0,), np.float32),
             rf_chroma_lpf=np.zeros((0,), np.float32))

    np.savez(out / "tape_chroma.npz",
             track_index=np.zeros((0,), np.int32),
             c_data=np.zeros((0,), np.uint8),
             c_offsets=np.zeros((1,), np.int64))


def _pack_track_meta(tr: TapeTrack):
    m = tr.meta
    y_h, y_w = m.get("y_shape", [0,0])
    c_h, c_w = m.get("c_shape", [0,0])
    return {
        "y_h": int(y_h), "y_w": int(y_w),
        "y_mod_w": int(m.get("y_mod_w", y_w)),
        "c_h": int(c_h), "c_w": int(c_w),
        "luma_bw": float(m.get("luma_bw", 0.66)),
        "dt": float(m.get("dt", 1/60)),
        "fps": float(m.get("fps", 30.0)),
        "frame_idx": int(m.get("frame_idx", 0)),
        "field_idx": int(m.get("field_idx", 0)),
        "head_u8": 0 if str(m.get("head","A")) == "A" else 1,
        "mode_u8": int(MODE_TO_U8.get(str(m.get("tape_mode","SP")), 0)),
        "ctl_sync_u8": int(m.get("ctl_sync_u8", 255)),
        "ctl_vjit_u8": int(m.get("ctl_vjit_u8", 0)),

        # RF model flags / params (per-track, so they survive save/load)
        "real_rf_u8": 1 if bool(m.get("real_rf_modulation", False)) else 0,
        "rf_chroma_fc_frac": float(m.get("rf_chroma_fc_frac", 0.12)),
        "rf_chroma_lpf": float(m.get("rf_chroma_lpf", 0.35)),
    }

def _base_decode_meta(global_info: Dict[str, Any], packed: Dict[str, Any]) -> Dict[str, Any]:
    # Combine bundle global info + per-track packed fields into the meta dict needed by decode_field_bgr.
    # Required by modulation.decode_field_bgr: y_shape, y_mod_w, c_shape, sample_rate, f0, fdev, dphi_min, dphi_max
    sr = int(global_info.get("sample_rate", 2_000_000))
    f0 = float(global_info.get("f0", 350_000.0))
    fdev = float(global_info.get("fdev", 120_000.0))
    dphi_min = float(global_info.get("dphi_min"))
    dphi_max = float(global_info.get("dphi_max"))
    chroma_sub = int(global_info.get("chroma_subsample", 2))

    meta = {
        "y_shape": [int(packed["y_h"]), int(packed["y_w"])],
        "y_mod_w": int(packed["y_mod_w"]),
        "c_shape": [int(packed["c_h"]), int(packed["c_w"])],
        "sample_rate": sr,
        "f0": f0,
        "fdev": fdev,
        "dphi_min": dphi_min,
        "dphi_max": dphi_max,
        "chroma_subsample": chroma_sub,
        "luma_bw": float(packed["luma_bw"]),
        # playback / indexing
        "dt": float(packed["dt"]),
        "fps": float(packed["fps"]),
        "frame_idx": int(packed["frame_idx"]),
        "field_idx": int(packed["field_idx"]),
        "head": "A" if int(packed["head_u8"]) == 0 else "B",
        "tape_mode": U8_TO_MODE.get(int(packed["mode_u8"]), "SP"),
        "ctl_sync_u8": int(packed["ctl_sync_u8"]),
        "ctl_vjit_u8": int(packed["ctl_vjit_u8"]),

        # RF model flags / params
        "real_rf_modulation": bool(int(packed.get("real_rf_u8", 0)) != 0),
        "rf_chroma_fc_frac": float(packed.get("rf_chroma_fc_frac", 0.12)),
        "rf_chroma_lpf": float(packed.get("rf_chroma_lpf", 0.35)),
    }
    return meta

def save_bundle(folder: str, tape: TapeImage, settings: Dict[str, Any], *, compress: bool = True, embed_audio: bool = True) -> None:
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)

    # Bundle-global info used to reconstruct decode meta quickly
    # dphi range from modulation._dphi_range: we store it explicitly
    # (keeps decode consistent even if code changes later)
    global_info = {
        "length_tracks": int(tape.cart.length_tracks),
        "sample_rate": 2_000_000,
        "f0": 350_000.0,
        "fdev": 120_000.0,
        "chroma_subsample": 2,
    }
    # Try to copy actual encoder params from first track if present
    if tape.cart.tracks:
        any_tr = next(iter(tape.cart.tracks.values()))
        m = any_tr.meta
        global_info["sample_rate"] = int(m.get("sample_rate", global_info["sample_rate"]))
        global_info["f0"] = float(m.get("f0", global_info["f0"]))
        global_info["fdev"] = float(m.get("fdev", global_info["fdev"]))
        global_info["chroma_subsample"] = int(m.get("chroma_subsample", global_info["chroma_subsample"]))
        global_info["dphi_min"] = float(m.get("dphi_min"))
        global_info["dphi_max"] = float(m.get("dphi_max"))
    else:
        # derive dphi range from params if no track exists
        sr = global_info["sample_rate"]; f0 = global_info["f0"]; fdev = global_info["fdev"]
        fmin = f0 - fdev; fmax = f0 + fdev
        global_info["dphi_min"] = float(2*np.pi * fmin / float(sr))
        global_info["dphi_max"] = float(2*np.pi * fmax / float(sr))

    (out / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (out / "tape_info.json").write_text(json.dumps(global_info, indent=2), encoding="utf-8")

    if tape.audio.pcm16 is not None and tape.audio.pcm16.size > 0:
        write_wav_mono_pcm16(str(out / "audio.wav"), tape.audio.pcm16, tape.audio.sample_rate)
        # Separate audio tape file (compact); load_bundle will prefer this.
        if embed_audio:
            try:
                audio_sr = int(tape.audio.sample_rate or 44100)
                aul = pcm16_to_ulaw(tape.audio.pcm16)
                np.savez_compressed(
                    str(out / "audio_tape.npz"),
                    audio_ulaw=aul.astype(np.uint8),
                    audio_sr=np.array([audio_sr], dtype=np.int32),
                )
            except Exception:
                # fall back to raw pcm
                try:
                    audio_sr = int(tape.audio.sample_rate or 44100)
                    np.savez_compressed(
                        str(out / "audio_tape.npz"),
                        audio_pcm16=tape.audio.pcm16.astype(np.int16),
                        audio_sr=np.array([audio_sr], dtype=np.int32),
                    )
                except Exception:
                    pass

    indices = np.array(sorted(tape.cart.tracks.keys()), dtype=np.int32)
    n = int(indices.size)

    y_offsets = np.zeros((n+1,), dtype=np.int64)
    c_offsets = np.zeros((n+1,), dtype=np.int64)
    y_chunks = []
    c_chunks = []

    # Packed per-track meta arrays (fast + small)
    y_h = np.zeros((n,), dtype=np.uint16)
    y_w = np.zeros((n,), dtype=np.uint16)
    y_mod_w = np.zeros((n,), dtype=np.uint16)
    c_h = np.zeros((n,), dtype=np.uint16)
    c_w = np.zeros((n,), dtype=np.uint16)
    luma_bw = np.zeros((n,), dtype=np.float32)
    dt = np.zeros((n,), dtype=np.float32)
    fps = np.zeros((n,), dtype=np.float32)
    frame_idx = np.zeros((n,), dtype=np.int32)
    field_idx = np.zeros((n,), dtype=np.int32)
    head_u8 = np.zeros((n,), dtype=np.uint8)
    mode_u8 = np.zeros((n,), dtype=np.uint8)
    ctl_sync_u8 = np.zeros((n,), dtype=np.uint8)
    ctl_vjit_u8 = np.zeros((n,), dtype=np.uint8)
    real_rf_u8 = np.zeros((n,), dtype=np.uint8)
    rf_chroma_fc_frac = np.zeros((n,), dtype=np.float32)
    rf_chroma_lpf = np.zeros((n,), dtype=np.float32)

    for k, idx in enumerate(indices.tolist()):
        tr = tape.cart.tracks[int(idx)]
        packed = _pack_track_meta(tr)

        y_h[k] = packed["y_h"]; y_w[k] = packed["y_w"]; y_mod_w[k] = packed["y_mod_w"]
        c_h[k] = packed["c_h"]; c_w[k] = packed["c_w"]
        luma_bw[k] = packed["luma_bw"]
        dt[k] = packed["dt"]; fps[k] = packed["fps"]
        frame_idx[k] = packed["frame_idx"]; field_idx[k] = packed["field_idx"]
        head_u8[k] = packed["head_u8"]; mode_u8[k] = packed["mode_u8"]
        ctl_sync_u8[k] = packed["ctl_sync_u8"]; ctl_vjit_u8[k] = packed["ctl_vjit_u8"]
        real_rf_u8[k] = int(packed.get("real_rf_u8", 0))
        rf_chroma_fc_frac[k] = float(packed.get("rf_chroma_fc_frac", 0.12))
        rf_chroma_lpf[k] = float(packed.get("rf_chroma_lpf", 0.35))

        y = tr.y_dphi8.astype(np.uint8).reshape(-1)
        c = tr.c_u8.astype(np.uint8).reshape(-1)
        y_chunks.append(y); c_chunks.append(c)
        y_offsets[k+1] = y_offsets[k] + y.size
        c_offsets[k+1] = c_offsets[k] + c.size

    y_data = np.concatenate(y_chunks, axis=0) if y_chunks else np.zeros((0,), np.uint8)
    c_data = np.concatenate(c_chunks, axis=0) if c_chunks else np.zeros((0,), np.uint8)

    save_fn = np.savez_compressed if compress else np.savez
    save_fn(
        out / "tape_luma.npz",
        track_index=indices,
        y_data=y_data,
        y_offsets=y_offsets,
        y_h=y_h, y_w=y_w, y_mod_w=y_mod_w,
        c_h=c_h, c_w=c_w,
        luma_bw=luma_bw, dt=dt, fps=fps,
        frame_idx=frame_idx, field_idx=field_idx,
        head_u8=head_u8, mode_u8=mode_u8,
        ctl_sync_u8=ctl_sync_u8, ctl_vjit_u8=ctl_vjit_u8,
        real_rf_u8=real_rf_u8,
        rf_chroma_fc_frac=rf_chroma_fc_frac,
        rf_chroma_lpf=rf_chroma_lpf,
    )
    save_fn(
        out / "tape_chroma.npz",
        track_index=indices,
        c_data=c_data,
        c_offsets=c_offsets,
    )

def load_bundle(folder: str) -> Tuple[TapeImage, Dict[str, Any]]:
    src = Path(folder)
    settings = json.loads((src / "settings.json").read_text(encoding="utf-8"))
    tape_info = json.loads((src / "tape_info.json").read_text(encoding="utf-8"))
    length_tracks = int(tape_info.get("length_tracks", 0))
    if length_tracks <= 0:
        raise ValueError("Invalid tape_info.json (length_tracks).")

    # Prefer the current split luma/chroma files.
    luma_path = src / "tape_luma.npz"
    chroma_path = src / "tape_chroma.npz"
    legacy_path = src / "tape.npz"

    if luma_path.exists() and chroma_path.exists():
        zL = np.load(luma_path, allow_pickle=False)
        zC = np.load(chroma_path, allow_pickle=False)

        indices = zL["track_index"].astype(np.int32)
        y_data = zL["y_data"].astype(np.uint8)
        y_offsets = zL["y_offsets"].astype(np.int64)

        idx_c = zC["track_index"].astype(np.int32)
        if idx_c.size != indices.size or (idx_c.size and not np.all(idx_c == indices)):
            raise ValueError("Chroma tape does not match luma tape (track_index mismatch).")
        c_data = zC["c_data"].astype(np.uint8)
        c_offsets = zC["c_offsets"].astype(np.int64)

        # Per-track meta arrays
        y_h = zL["y_h"].astype(np.uint16)
        y_w = zL["y_w"].astype(np.uint16)
        y_mod_w = zL["y_mod_w"].astype(np.uint16)
        c_h = zL["c_h"].astype(np.uint16)
        c_w = zL["c_w"].astype(np.uint16)
        luma_bw = zL["luma_bw"].astype(np.float32)
        dt = zL["dt"].astype(np.float32)
        fps = zL["fps"].astype(np.float32)
        frame_idx = zL["frame_idx"].astype(np.int32)
        field_idx = zL["field_idx"].astype(np.int32)
        head_u8 = zL["head_u8"].astype(np.uint8)
        mode_u8 = zL["mode_u8"].astype(np.uint8)
        ctl_sync_u8 = zL["ctl_sync_u8"].astype(np.uint8)
        ctl_vjit_u8 = zL["ctl_vjit_u8"].astype(np.uint8)
        real_rf_u8 = (zL["real_rf_u8"].astype(np.uint8)
                      if "real_rf_u8" in zL.files else np.zeros_like(head_u8).astype(np.uint8))
        rf_chroma_fc_frac = (zL["rf_chroma_fc_frac"].astype(np.float32)
                             if "rf_chroma_fc_frac" in zL.files else (np.zeros_like(luma_bw).astype(np.float32) + 0.12))
        rf_chroma_lpf = (zL["rf_chroma_lpf"].astype(np.float32)
                         if "rf_chroma_lpf" in zL.files else (np.zeros_like(luma_bw).astype(np.float32) + 0.35))

    else:
        # Legacy single-file format
        z = np.load(legacy_path, allow_pickle=False)
        if "track_index" not in z:
            raise ValueError("Unsupported bundle format (no track_index). Re-save the bundle with a current Digital VCR build.")

        indices = z["track_index"].astype(np.int32)
        y_data = z["y_data"].astype(np.uint8)
        c_data = z["c_data"].astype(np.uint8)
        y_offsets = z["y_offsets"].astype(np.int64)
        c_offsets = z["c_offsets"].astype(np.int64)

        # Per-track meta arrays
        y_h = z["y_h"].astype(np.uint16)
        y_w = z["y_w"].astype(np.uint16)
        y_mod_w = z["y_mod_w"].astype(np.uint16)
        c_h = z["c_h"].astype(np.uint16)
        c_w = z["c_w"].astype(np.uint16)
        luma_bw = z["luma_bw"].astype(np.float32)
        dt = z["dt"].astype(np.float32)
        fps = z["fps"].astype(np.float32)
        frame_idx = z["frame_idx"].astype(np.int32)
        field_idx = z["field_idx"].astype(np.int32)
        head_u8 = z["head_u8"].astype(np.uint8)
        mode_u8 = z["mode_u8"].astype(np.uint8)
        ctl_sync_u8 = z["ctl_sync_u8"].astype(np.uint8)
        ctl_vjit_u8 = z["ctl_vjit_u8"].astype(np.uint8)

        # Defaults (legacy bundles won't have these)
        real_rf_u8 = (z["real_rf_u8"].astype(np.uint8)
                      if "real_rf_u8" in z.files else np.zeros_like(head_u8).astype(np.uint8))
        rf_chroma_fc_frac = (z["rf_chroma_fc_frac"].astype(np.float32)
                             if "rf_chroma_fc_frac" in z.files else (np.zeros_like(luma_bw).astype(np.float32) + 0.12))
        rf_chroma_lpf = (z["rf_chroma_lpf"].astype(np.float32)
                         if "rf_chroma_lpf" in z.files else (np.zeros_like(luma_bw).astype(np.float32) + 0.35))

    cart = TapeCartridge(length_tracks=length_tracks, tracks={})
    tape = TapeImage(cart=cart, audio=TapeAudio())

    wav = src / "audio.wav"
    audio_npz = src / "audio_tape.npz"
    # Prefer audio_tape.npz (compact), fallback to audio.wav
    if audio_npz.exists():
        try:
            az = np.load(audio_npz, allow_pickle=False)
            if "audio_ulaw" in az and "audio_sr" in az:
                aul = az["audio_ulaw"].astype(np.uint8)
                asr = int(az["audio_sr"].astype(np.int32).reshape(-1)[0]) if az["audio_sr"].size else 44100
                if aul.size:
                    tape.audio.pcm16 = ulaw_to_pcm16(aul)
                    tape.audio.sample_rate = asr
            elif "audio_pcm16" in az and "audio_sr" in az:
                apcm = az["audio_pcm16"].astype(np.int16)
                asr = int(az["audio_sr"].astype(np.int32).reshape(-1)[0]) if az["audio_sr"].size else 44100
                if apcm.size:
                    tape.audio.pcm16 = apcm
                    tape.audio.sample_rate = asr
        except Exception:
            pass
    if (tape.audio.pcm16 is None or tape.audio.pcm16.size == 0) and wav.exists():
        pcm, sr = read_wav_mono_pcm16(str(wav))
        tape.audio.pcm16 = pcm
        tape.audio.sample_rate = sr

    # Keep backing arrays alive (views in TapeTrack point into these)
    cart.bundle_backing = {"y_data": y_data, "c_data": c_data}

    n = int(indices.size)
    for k in range(n):
        idx = int(indices[k])
        ys, ye = int(y_offsets[k]), int(y_offsets[k+1])
        cs, ce = int(c_offsets[k]), int(c_offsets[k+1])

        packed = {
            "y_h": int(y_h[k]), "y_w": int(y_w[k]), "y_mod_w": int(y_mod_w[k]),
            "c_h": int(c_h[k]), "c_w": int(c_w[k]),
            "luma_bw": float(luma_bw[k]),
            "dt": float(dt[k]), "fps": float(fps[k]),
            "frame_idx": int(frame_idx[k]), "field_idx": int(field_idx[k]),
            "head_u8": int(head_u8[k]), "mode_u8": int(mode_u8[k]),
            "ctl_sync_u8": int(ctl_sync_u8[k]), "ctl_vjit_u8": int(ctl_vjit_u8[k]),
            "real_rf_u8": int(real_rf_u8[k]),
            "rf_chroma_fc_frac": float(rf_chroma_fc_frac[k]),
            "rf_chroma_lpf": float(rf_chroma_lpf[k]),
        }
        meta = _base_decode_meta(tape_info, packed)
        tr = TapeTrack(y_dphi8=y_data[ys:ye], c_u8=c_data[cs:ce], meta=meta)
        tape.cart.set(idx, tr)

    return tape, settings
