from __future__ import annotations
from dataclasses import dataclass
import cv2
import time

from .tape import TapeImage
from .player import VCRPlayer
from .defects import PlaybackDefects, AudioPlaybackDefects
from .audio import write_wav_mono_pcm16, apply_audio_playback_defects, mux_audio_into_mp4
from .crt import CRTSettings
from .crt_renderer import CRTFrameRenderer

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
    *,
    crt: CRTSettings | None = None,
    crt_renderer: CRTFrameRenderer | None = None,
) -> bool:
    if tape.cart.length_tracks < 2:
        return False

    player = VCRPlayer()
    player.insert()
    player.play()

    crt_settings = crt.validated() if crt is not None else None
    use_crt = bool(crt_settings and crt_settings.enabled and crt_settings.export_enabled)
    owned_crt_renderer = None

    try:
        if use_crt and crt_renderer is None:
            owned_crt_renderer = CRTFrameRenderer()
            crt_renderer = owned_crt_renderer

        t_end = time.perf_counter() + float(opts.seconds_leadin_lock)
        while time.perf_counter() < t_end:
            player.update(tape, pb)
            time.sleep(0.02)

        frame0 = player.get_frame(tape, pb)
        if frame0.shape[1] < opts.upscale_width:
            base_w = int(opts.upscale_width)
            base_h = int(frame0.shape[0] * (base_w / frame0.shape[1]))
            frame0 = cv2.resize(frame0, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
        base_h, base_w = frame0.shape[:2]
        if use_crt:
            frame0 = crt_renderer.render_frame(frame0, crt_settings, timeout=15.0)
        h, w = frame0.shape[:2]

        vw = cv2.VideoWriter(opts.out_mp4, cv2.VideoWriter_fourcc(*opts.codec), float(opts.fps), (w, h))
        if not vw.isOpened():
            return False

        total_frames = tape.cart.length_tracks // 2
        for i in range(total_frames):
            # deterministic stepping: advance exactly 1/fps per output frame
            try:
                player._t_last = time.perf_counter() - (1.0 / float(opts.fps))
            except Exception:
                pass
            player.update(tape, pb)
            frame = player.get_frame(tape, pb)
            if frame.shape[1] != base_w or frame.shape[0] != base_h:
                frame = cv2.resize(frame, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
            if use_crt:
                frame = crt_renderer.render_frame(frame, crt_settings, timeout=15.0)
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            vw.write(frame)
            if progress_cb:
                progress_cb(i+1, total_frames)

        vw.release()
        return True
    finally:
        if owned_crt_renderer is not None:
            owned_crt_renderer.close()

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
