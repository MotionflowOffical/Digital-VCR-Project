from __future__ import annotations
from dataclasses import dataclass
import cv2
import time

from .tape import TapeImage
from .player import VCRPlayer
from .defects import PlaybackDefects, AudioPlaybackDefects
from .audio import write_wav_mono_pcm16, apply_audio_playback_defects, mux_audio_into_mp4
from .crt import CRTSettings

@dataclass
class ExportOptions:
    out_mp4: str
    fps: float = 30.0
    codec: str = "mp4v"
    upscale_width: int = 960
    seconds_leadin_lock: float = 2.0

def export_playback_video_mp4(
    tape: TapeImage,
    pb: PlaybackDefects,
    opts: ExportOptions,
    progress_cb=None,
    crt_settings: CRTSettings | None = None,
) -> bool:
    if tape.cart.length_tracks < 2:
        return False

    player = VCRPlayer()
    player.insert()
    player.play()

    t_end = time.perf_counter() + float(opts.seconds_leadin_lock)
    while time.perf_counter() < t_end:
        player.update(tape, pb)
        time.sleep(0.02)

    frame0 = player.get_frame(tape, pb)
    if frame0 is None:
        return False

    raw_h, raw_w = frame0.shape[:2]
    if raw_w < opts.upscale_width:
        w = int(opts.upscale_width)
        h = int(raw_h * (w / raw_w))
    else:
        w, h = int(raw_w), int(raw_h)

    crt = crt_settings.validated() if crt_settings is not None else None
    use_crt = bool(crt is not None and crt.enabled and crt.export_enabled)
    renderer = None
    if use_crt:
        # Lazy import keeps OpenGL dependencies optional unless CRT export is used.
        from .crt_renderer import CRTFrameRenderer
        renderer = CRTFrameRenderer()

    vw = cv2.VideoWriter(opts.out_mp4, cv2.VideoWriter_fourcc(*opts.codec), float(opts.fps), (w, h))
    if not vw.isOpened():
        if renderer is not None:
            renderer.close()
        return False

    try:
        total_frames = tape.cart.length_tracks // 2
        for i in range(total_frames):
            # deterministic stepping: advance exactly 1/fps per output frame
            try:
                player._t_last = time.perf_counter() - (1.0 / float(opts.fps))
            except Exception:
                pass
            player.update(tape, pb)
            frame = player.get_frame(tape, pb)
            if frame is None:
                continue

            if use_crt:
                # CRT always renders internally at its selected quality width, then
                # downsamples to the final export dimensions.  This preserves mask,
                # scanline and bloom behavior independent of source tape width.
                frame = renderer.render_frame(frame, crt, output_size=(w, h), timeout=10.0)
            elif frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

            vw.write(frame)
            if progress_cb:
                progress_cb(i + 1, total_frames)
    except Exception:
        return False
    finally:
        vw.release()
        if renderer is not None:
            renderer.close()
    return True

def export_audio_and_optional_mux(tape: TapeImage, audio_out_wav: str, video_mp4: str, out_with_audio_mp4: str,
                                 ap: AudioPlaybackDefects) -> tuple[bool, str | None]:
    if tape.audio.pcm16 is None or tape.audio.pcm16.size == 0:
        return False, "No audio in tape bundle."

    sr = int(tape.audio.sample_rate)
    pcm = apply_audio_playback_defects(tape.audio.pcm16, sr, hiss=ap.hiss, pops=ap.pops)
    write_wav_mono_pcm16(audio_out_wav, pcm, sr)

    ok, err = mux_audio_into_mp4(video_mp4, audio_out_wav, out_with_audio_mp4)
    if not ok:
        return False, err
    return True, None
