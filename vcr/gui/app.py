from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import threading
import queue
import cv2
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
import time
import traceback
import datetime
import json

from ..tape import TapeImage, TapeCartridge, TapeTrack
from ..bundle import save_bundle, load_bundle, create_blank_bundle
from ..recorder import Recorder, RecordOptions, sample_fields_from_frame
from ..modulation import encode_field_bgr
from ..defects import apply_record_defects_to_field, apply_rf_defects_y_dphi_u8, apply_rf_defects_chroma_u8
from ..rf_model import rf_roundtrip_luma_dphi_u8, rf_roundtrip_chroma_u8
from ..editor import Editor, DubOptions
from ..player import VCRPlayer
from ..audio_player import AudioPlayer
from ..exporter import export_playback_video_mp4, ExportOptions, export_audio_and_optional_mux
from ..defects import (
    RecordDefects, PlaybackDefects, AudioRecordDefects, AudioPlaybackDefects,
    settings_to_dict, settings_from_dict
)
from ..crt import (
    CRTSettings, CRT_QUALITIES, CRT_MASK_TYPES,
    consumer_tv_preset, pro_monitor_preset, preset_by_name,
    crt_settings_from_dict, crt_settings_to_dict
)
from ..crt_renderer import CRTFrameRenderer, CRTGPUUnavailable
from ..modulation import decode_field_bgr

APP_VERSION = "V6_13_8"

DARK_BG = "#0b0f14"
PANEL_BG = "#111821"
PANEL_BG_2 = "#16202b"
SURFACE_BG = "#0f151d"
BORDER = "#253241"
TEXT = "#e7edf4"
MUTED = "#96a3b3"
ACCENT = "#4ea1ff"
ACCENT_ACTIVE = "#6fb4ff"
SIDEBAR_BG = "#0d131b"
CARD_BG = "#121a24"
CARD_BG_2 = "#182332"
TOOLTIP_BG = "#101923"

def _help(purpose: str, low: str, mid: str, high: str, impact: str, scope: str) -> str:
    return (
        f"{purpose}\n\n"
        f"Low: {low}\n"
        f"Mid: {mid}\n"
        f"High: {high}\n\n"
        f"Impact: {impact}\n"
        f"Scope: {scope}"
    )

SETTING_HELP = {
    "base_dir": _help("Folder used when creating new tape bundles.", "Type a simple local folder.", "Use a project tape library.", "Use a large fast disk for many bundles.", "Affects where bundle files are stored, not image quality.", "Workflow setting."),
    "tape_minutes": _help("Length of a newly created blank tape.", "Short tests and quick previews.", "Normal working tapes.", "Long sessions with more disk/RAM use.", "Longer tapes allocate more track/audio capacity.", "New-tape setting."),
    "recpos": _help("Track where recording will begin overwriting tape.", "Start near the beginning.", "Insert into the current program.", "Record near the end of the tape.", "Controls edit point and where audio/video are written.", "Baked into the tape."),
    "scrub": _help("Preview a track location without playing in real time.", "Start of tape.", "Middle of tape.", "Later tape locations.", "Useful for finding edit points; does not alter media.", "Preview-only setting."),
    "source_file": _help("Input video file to record into the virtual tape.", "Small source files are faster.", "Typical MP4/MOV/MKV sources.", "Large/high-res files need more CPU.", "Source resolution and codec affect recording speed.", "Recording input."),
    "monitor_mode": _help("Chooses what the recorder preview shows while recording.", "Input shows the source before tape encoding.", "Tape shows the recorded result.", "Use tape for final confidence checks.", "Input is faster; tape preview reflects VHS processing.", "Preview-only setting."),
    "downscale_width": _help("Horizontal resolution recorded to tape.", "Softer image, faster recording.", "Balanced VHS-like detail.", "Sharper image, more CPU and memory.", "Higher values improve detail but slow encoding/playback.", "Baked into recording."),
    "field_sampling": _help("How source frames become VHS-style fields.", "Interlaced preserves line split behavior.", "Progressive reduces false combing.", "Use progressive for most digital sources.", "Affects motion/detail character before recording.", "Baked into recording."),
    "encode_threads": _help("CPU worker count used during recording.", "1 is predictable and lower load.", "Auto balances speed and responsiveness.", "More threads can record faster but use more CPU.", "Affects recording speed, not final look.", "Recording performance setting."),
    "real_rf_modulation": _help("Runs the advanced FM/AM RF round-trip when recording.", "Off uses the faster byte-domain path.", "On adds analog-like carrier behavior.", "Heavy RF settings can become unstable.", "More realistic but slower recording.", "Baked into recording."),
    "tape_mode": _help("Tape speed/quality model.", "SP is cleaner and most stable.", "LP adds moderate degradation.", "EP is softer/noisier and more unstable.", "Changes how strongly defects are applied.", "Baked into recording/live buffer."),
    "luma_bw": _help("Brightness-detail bandwidth recorded to tape.", "Softer luminance/detail.", "Balanced VHS softness.", "More fine brightness detail.", "Affects sharpness and text readability.", "Baked into recording."),
    "chroma_bw": _help("Color-detail bandwidth recorded to tape.", "Bleedier, softer color.", "Moderate VHS color softness.", "Cleaner color edges.", "Lower values look more VHS-like; higher is cleaner.", "Baked into recording."),
    "record_blur": _help("Optical/tape softness applied while recording.", "Cleaner, sharper source.", "Mild VHS softness.", "Heavy blur and detail loss.", "Softens all recorded frames.", "Baked into recording."),
    "record_jitter": _help("Horizontal line jitter written into tape fields.", "Stable image.", "Small analog wiggle.", "Strong line wobble/tearing.", "Adds unstable tape transport character.", "Baked into recording."),
    "record_rf_noise": _help("RF noise applied at record time.", "Clean signal.", "Light analog speckle/noise.", "Noisy, unstable signal.", "Can reduce clarity and increase snow-like artifacts.", "Baked into recording."),
    "record_dropouts": _help("Signal loss events while recording.", "Few or no dropouts.", "Occasional tape damage.", "Frequent visible dropouts.", "Creates lost/garbled parts of the tape signal.", "Baked into recording."),
    "ar_wow": _help("Record-time audio pitch wobble.", "Stable pitch.", "Subtle VHS wow/flutter.", "Obvious pitch warble.", "Affects recorded audio tone and timing.", "Baked into tape audio."),
    "ar_hiss": _help("Record-time audio noise floor.", "Cleaner audio.", "Light tape hiss.", "Very noisy audio.", "Adds hiss before audio is stored.", "Baked into tape audio."),
    "ar_dropouts": _help("Record-time audio level dropouts.", "Few dropouts.", "Occasional dips.", "Frequent audio gaps.", "Simulates tape audio loss.", "Baked into tape audio."),
    "ar_compression": _help("VHS linear-track audio bandwidth/compression amount.", "More source-like audio.", "Balanced VHS narrowing.", "Heavier band-limit/companding.", "Changes tone and dynamic range.", "Baked into tape audio."),
    "rt_record": _help("Paces recording in real time.", "Off records as fast as possible.", "On matches real-time recording.", "Use on for monitoring feel.", "Off improves throughput; on improves live monitoring.", "Recording workflow setting."),
    "audio_extract": _help("Extracts source audio into the tape.", "Off records silent/blank tape audio.", "On stores source audio.", "Use on when final export needs audio.", "Adds ffmpeg work during recording.", "Baked into tape audio."),
    "autosave": _help("Automatically saves the active bundle after recording.", "Off keeps changes in memory.", "On writes bundle after record.", "Use on for safer long sessions.", "Adds disk write time after recording.", "Workflow setting."),
    "edit_live_preview": _help("Runs the editor preview continuously.", "Off saves CPU.", "On shows moving preview.", "Use on while tuning dub settings.", "Consumes CPU but does not alter tape.", "Preview-only setting."),
    "dub_realtime": _help("Paces save/dub in real time.", "Off processes faster.", "On mimics real-time dub.", "Use on for monitoring behavior.", "Affects save/dub duration only.", "Workflow setting."),
    "export_video": _help("Exports output.mp4 after saving/dubbing.", "Off only saves bundle.", "On also renders video.", "Use on for shareable output.", "Adds export time and disk use.", "Export workflow setting."),
    "aspect_display": _help("Display aspect for playback preview/export.", "4:3 classic VHS frame.", "Use source-appropriate choice.", "16:9 widescreen presentation.", "Adds letter/pillarboxing as needed.", "Playback/export setting."),
    "tracking_knob": _help("Manual tracking control.", "Biases one side of tracking.", "Centered/neutral tracking.", "Biases the other side.", "Poor tracking increases tearing/snow.", "Playback-only setting."),
    "tracking_sensitivity": _help("How strongly tracking mismatch affects playback.", "Forgiving tape/player.", "Normal sensitivity.", "Touchy tracking.", "Higher values reveal tracking errors faster.", "Playback-only setting."),
    "tracking_artifacts": _help("Visibility of tracking-related artifacts.", "Subtle/hidden artifacts.", "Natural VHS instability.", "Strong tearing and crosstalk.", "Scales visual tracking damage.", "Playback-only setting."),
    "auto_tracking": _help("Consumer-style automatic tracking.", "Off uses manual knob only.", "Partial auto correction.", "On hunts for best lock.", "Can improve stability but may visibly hunt.", "Playback-only setting."),
    "auto_tracking_strength": _help("Aggressiveness of auto tracking.", "Slow gentle correction.", "Balanced correction.", "Fast, possibly jumpy correction.", "Changes how quickly tracking recenters.", "Playback-only setting."),
    "servo_recovery": _help("How fast playback lock recovers.", "Slow relock after edits/noise.", "Normal recovery.", "Fast relock.", "Higher values stabilize sooner but can hunt.", "Playback-only setting."),
    "sync_bias": _help("Bias applied to sync pulse strength.", "Harder to lock.", "Neutral sync behavior.", "Easier to lock.", "Changes vertical/sync stability.", "Playback-only setting."),
    "servo_hunt": _help("Low-frequency servo hunting amount.", "Stable transport.", "Subtle hunting.", "Visible rolling/wobble.", "Adds mechanical instability.", "Playback-only setting."),
    "servo_hunt_freq": _help("How often servo hunting appears.", "Rare hunting.", "Intermittent hunting.", "Frequent hunting.", "Controls variation frequency, not strength.", "Playback-only setting."),
    "head_switch_strength": _help("Bottom-of-frame head switching band strength.", "Minimal band.", "Classic VHS bottom noise.", "Heavy noisy band.", "Adds head-switch disturbance near frame bottom.", "Playback-only setting."),
    "head_switch_freq": _help("How often head switching noise appears.", "Rare.", "Occasional.", "Frequent.", "Controls event frequency, not band size.", "Playback-only setting."),
    "playback_timebase": _help("Playback timebase wobble amount.", "Stable geometry.", "Mild horizontal waving.", "Strong warping/flagging.", "Distorts frame geometry during playback.", "Playback-only setting."),
    "timebase_freq": _help("How often timebase wobble is active.", "Rare wobble.", "Intermittent wobble.", "Frequent wobble.", "Controls variation frequency.", "Playback-only setting."),
    "playback_rf_noise": _help("Noise introduced while reading tape.", "Cleaner playback.", "Light RF noise.", "Heavy noisy playback.", "Increases snow and signal stress.", "Playback-only setting."),
    "playback_dropouts": _help("Playback-side signal loss events.", "Few events.", "Occasional dropouts.", "Frequent dropouts.", "Adds read errors without rewriting tape.", "Playback-only setting."),
    "dropout_freq": _help("How often playback dropouts occur.", "Rare.", "Intermittent.", "Frequent.", "Controls dropout event cadence.", "Playback-only setting."),
    "interference": _help("External interference amount.", "Clean picture.", "Mild bars/buzz.", "Strong interference.", "Adds moving brightness/noise bands.", "Playback-only setting."),
    "interference_freq": _help("How often interference is active.", "Rare.", "Intermittent.", "Frequent.", "Controls interference cadence.", "Playback-only setting."),
    "snow": _help("Visible RF snow amount.", "Clean image.", "Light sparkle.", "Heavy snow.", "Adds crisp analog speckle/noise.", "Playback-only setting."),
    "snow_freq": _help("How often snow appears.", "Rare bursts.", "Intermittent snow.", "Near-constant snow.", "Controls snow cadence.", "Playback-only setting."),
    "variance": _help("Randomness of playback instability.", "Predictable effects.", "Natural variation.", "Chaotic variation.", "Changes how much effects fluctuate over time.", "Playback-only setting."),
    "chroma_shift_x": _help("Horizontal color delay.", "Aligned color.", "Mild color offset.", "Strong color smear/offset.", "Moves chroma relative to luma.", "Playback-only setting."),
    "chroma_shift_y": _help("Vertical color delay.", "Aligned color.", "Mild vertical offset.", "Strong vertical color misregistration.", "Moves chroma up/down.", "Playback-only setting."),
    "chroma_phase": _help("Color phase error.", "Accurate hue.", "Mild hue instability.", "Strong hue rotation.", "Shifts VHS color decoding.", "Playback-only setting."),
    "chroma_noise": _help("Color-channel noise.", "Clean color.", "Mild speckles.", "Heavy color noise.", "Adds noise mainly to chroma.", "Playback-only setting."),
    "chroma_noise_freq": _help("How often chroma noise appears.", "Rare.", "Intermittent.", "Frequent.", "Controls color-noise cadence.", "Playback-only setting."),
    "chroma_wobble": _help("Slow color phase/position wobble.", "Stable color.", "Subtle drift.", "Obvious color breathing.", "Animates color instability.", "Playback-only setting."),
    "chroma_wobble_freq": _help("How often chroma wobble is active.", "Rare.", "Intermittent.", "Frequent.", "Controls wobble cadence.", "Playback-only setting."),
    "scanline_strength": _help("Dark scanline overlay amount.", "No added scanlines.", "Subtle line texture.", "Strong line pattern.", "Changes preview/export texture.", "Playback-only setting."),
    "scanline_soften": _help("Vertical blending to soften scanlines.", "Sharper hard lines.", "Balanced blend.", "Smoothest scanlines.", "Reduces harsh interlace appearance.", "Playback-only setting."),
    "brightness": _help("Playback brightness adjustment.", "Darker image.", "Neutral brightness.", "Brighter image.", "Affects displayed/exported playback image.", "Playback-only setting."),
    "contrast": _help("Playback contrast adjustment.", "Flatter image.", "Moderate contrast.", "Punchier contrast.", "Changes tonal separation.", "Playback-only setting."),
    "saturation": _help("Playback color saturation.", "Muted/washed color.", "Normal VHS color.", "Strong saturated color.", "Changes color intensity.", "Playback-only setting."),
    "bloom": _help("Bright-area glow amount.", "No glow.", "Mild analog bloom.", "Heavy glow/halation.", "Softens bright highlights.", "Playback-only setting."),
    "sharpen": _help("Playback sharpening amount.", "Soft image.", "Mild edge lift.", "Strong sharpening artifacts.", "Improves apparent detail at risk of ringing.", "Playback-only setting."),
    "playback_blur": _help("Playback-side blur amount.", "Sharper playback.", "Mild softness.", "Heavy soft playback.", "Softens image without rewriting tape.", "Playback-only setting."),
    "playback_blur_freq": _help("How often playback blur varies.", "Rare.", "Intermittent.", "Frequent.", "Controls blur cadence.", "Playback-only setting."),
    "frame_jitter": _help("Whole-frame transport jitter.", "Stable frame.", "Subtle shake.", "Strong frame shake.", "Moves the whole frame slightly.", "Playback-only setting."),
    "frame_jitter_freq": _help("How often frame jitter appears.", "Rare.", "Intermittent.", "Frequent.", "Controls jitter cadence.", "Playback-only setting."),
    "composite_view": _help("Adds composite-video style color artifacts.", "Off keeps cleaner separation.", "On adds composite-like crawl/shift.", "Use on for dirtier analog output.", "Changes playback preview/export only.", "Playback-only setting."),
    "ap_hiss": _help("Playback audio hiss.", "Cleaner playback audio.", "Light hiss.", "Heavy hiss.", "Adds noise while playing/exporting audio.", "Playback-only setting."),
    "ap_pops": _help("Playback audio pops/clicks.", "Few pops.", "Occasional clicks.", "Frequent pops.", "Adds transient audio defects.", "Playback-only setting."),
    "play_audio": _help("Enables audio while using the Player.", "Off keeps UI silent.", "On follows tape audio.", "Use on for sync checks.", "Uses Windows audio backend when available.", "Playback-only setting."),
    "proxy_use": _help("Uses RAM proxy frames for smoother playback.", "Off renders live.", "On uses built proxy when available.", "Use on for heavy effects/tapes.", "Improves smoothness but may be lower fidelity.", "Playback performance setting."),
    "proxy_seconds": _help("Length of RAM proxy to build.", "Short proxy, low memory.", "Normal preview range.", "Long proxy, high memory/time.", "Affects proxy build time and RAM use.", "Playback performance setting."),
    "rf_fm_depth": _help("Luma FM deviation depth for RF model.", "Softer/less robust carrier.", "Neutral carrier depth.", "Hotter/more robust carrier.", "Affects RF stability and detail.", "Baked into recording when RF is on."),
    "rf_am_depth": _help("RF amplitude modulation depth.", "Little envelope ripple.", "Natural RF amplitude movement.", "Strong amplitude instability.", "Adds analog carrier-level variation.", "Baked into recording when RF is on."),
    "rf_phase_noise": _help("RF phase jitter.", "Stable carrier phase.", "Mild phase noise.", "Strong phase instability.", "Can soften and destabilize detail/color.", "Baked into recording when RF is on."),
    "rf_carrier_noise": _help("Additional RF carrier noise.", "Clean carrier.", "Moderate noise.", "Noisy carrier.", "Adds carrier-level noise before decode.", "Baked into recording when RF is on."),
    "rf_nonlinearity": _help("RF saturation/nonlinearity.", "Linear carrier.", "Mild saturation.", "Heavy distortion.", "Adds analog compression/clipping character.", "Baked into recording when RF is on."),
    "rf_chroma_fc_frac": _help("Chroma subcarrier position as a sample-rate fraction.", "Lower carrier position.", "Default color-under behavior.", "Higher carrier position.", "Changes chroma RF behavior; extremes may look odd.", "Baked into recording when RF is on."),
    "rf_chroma_lpf": _help("Chroma demodulation low-pass strength.", "More chroma detail/noise.", "Balanced filtering.", "Softer cleaner chroma.", "Trades color detail for stability.", "Baked into recording when RF is on."),
    "luma_chroma_bleed": _help("Cross-talk between brightness and color.", "Clean separation.", "Mild cross-talk.", "Heavy bleed/dot-crawl feel.", "Adds analog recombination artifacts.", "Playback-only setting."),
    "rf_playback_model": _help("Applies RF-like channel effects during playback.", "Off uses faster playback path.", "On adds carrier-level read degradation.", "Use on for deeper analog instability.", "Costs CPU and changes playback only.", "Playback-only setting."),
    "rf_playback_fm_depth": _help("Playback RF luma FM depth.", "Less robust playback carrier.", "Neutral.", "More robust/hotter carrier.", "Changes RF playback degradation.", "Playback-only setting."),
    "rf_playback_am_depth": _help("Playback RF AM depth.", "Stable amplitude.", "Moderate ripple.", "Strong amplitude instability.", "Changes carrier-level brightness stability.", "Playback-only setting."),
    "rf_playback_phase_noise": _help("Playback RF phase noise.", "Stable carrier.", "Mild phase jitter.", "Strong phase jitter.", "Destabilizes detail/color during playback.", "Playback-only setting."),
    "rf_playback_carrier_noise": _help("Playback RF carrier noise.", "Clean read.", "Moderate carrier noise.", "Noisy read.", "Adds RF noise before decode.", "Playback-only setting."),
    "rf_playback_nonlinearity": _help("Playback RF nonlinearity.", "Linear read path.", "Mild saturation.", "Heavy distortion.", "Adds analog read distortion.", "Playback-only setting."),
    "live_cam": _help("Camera index used by Live mode.", "First detected camera.", "Choose another capture device.", "Higher indexes are additional devices.", "Affects live input source only.", "Live workflow setting."),
    "live_bufsec": _help("Length of live ring-buffer tape.", "Lower latency/less buffer.", "Balanced buffer.", "Longer buffer/more memory.", "Controls live tape capacity and memory.", "Live-only setting."),
    "live_downscale_width": _help("Live input recording width.", "Faster, softer live output.", "Balanced live quality.", "Sharper, higher CPU live output.", "Strongly affects live performance.", "Live-only recording setting."),
    "live_mode": _help("Turns live VHS processing on.", "Off stops camera processing.", "On starts live pipeline.", "Use with overlay for output.", "Consumes camera and CPU while on.", "Live workflow setting."),
    "live_overlay": _help("Fullscreen live output window.", "Off keeps output inside app.", "On mirrors live output fullscreen.", "Use for display/capture workflows.", "Does not change recorded signal.", "Live display setting."),
}

class GradientCanvas(tk.Canvas):
    def __init__(self, parent, color_a="#071019", color_b="#162536", color_c="#0b0f14", **kwargs):
        super().__init__(parent, highlightthickness=0, bd=0, **kwargs)
        self.colors = (color_a, color_b, color_c)
        self.bind("<Configure>", self._draw)

    @staticmethod
    def _hex_to_rgb(value: str) -> tuple[int, int, int]:
        value = value.lstrip("#")
        return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
        return "#%02x%02x%02x" % rgb

    def _draw(self, _evt=None):
        self.delete("gradient")
        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())
        a = self._hex_to_rgb(self.colors[0])
        b = self._hex_to_rgb(self.colors[1])
        c = self._hex_to_rgb(self.colors[2])
        for y in range(h):
            t = y / max(1, h - 1)
            if t < 0.55:
                k = t / 0.55
                rgb = tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))
            else:
                k = (t - 0.55) / 0.45
                rgb = tuple(int(b[i] + (c[i] - b[i]) * k) for i in range(3))
            self.create_line(0, y, w, y, fill=self._rgb_to_hex(rgb), tags=("gradient",))
        self.lower("gradient")


class HelpTooltip:
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<FocusIn>", self.show, add="+")
        widget.bind("<FocusOut>", self.hide, add="+")
        widget.bind("<Escape>", self.hide, add="+")

    def show(self, _evt=None):
        if self.tip is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 22
        y = self.widget.winfo_rooty() + 24
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        self.tip.configure(bg=ACCENT)
        body = tk.Label(
            self.tip,
            text=self.text,
            justify="left",
            wraplength=360,
            bg=TOOLTIP_BG,
            fg=TEXT,
            padx=12,
            pady=10,
            font=("Segoe UI", 9),
        )
        body.pack(padx=1, pady=1)

    def hide(self, _evt=None):
        if self.tip is not None:
            try:
                self.tip.destroy()
            except Exception:
                pass
            self.tip = None


class PageNavigator:
    def __init__(self, app):
        self.app = app

    def select(self, page):
        self.app._show_page(page)


class VScrollFrame(ctk.CTkFrame):
    def __init__(self, parent, width=340, **kwargs):
        super().__init__(parent, fg_color="transparent", corner_radius=0, **kwargs)
        self.inner = ctk.CTkScrollableFrame(
            self,
            width=width,
            fg_color=CARD_BG,
            scrollbar_button_color="#26384c",
            scrollbar_button_hover_color=ACCENT,
            corner_radius=14,
        )
        self.inner.pack(fill="both", expand=True)

class DigitalVCRApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title(f"Digital VCR {APP_VERSION}")
        self.root.geometry("1240x760")
        self.root.minsize(1080, 640)
        self.root.configure(bg=DARK_BG)

        self.uiq = queue.Queue()
        self.lock = threading.Lock()

        self.recorder = Recorder()
        self.editor = Editor()
        self.player = VCRPlayer()
        self.audio_player = AudioPlayer()

        # Active tape in memory
        self.tape_live = TapeImage(cart=TapeCartridge(length_tracks=18000))
        self.tape_edited = None
        self.tape_loaded = None

        # If a bundle folder is active, record/save will update it
        self.bundle_path: str | None = None

        self.loaded_settings = None

        self.rec_def = RecordDefects()
        self.pb_def = PlaybackDefects()  # defaults include softer scanlines
        self.ar_def = AudioRecordDefects()
        self.ap_def = AudioPlaybackDefects()
        self.rec_opts = RecordOptions()
        self.crt_settings = consumer_tv_preset()
        self.crt_renderer = CRTFrameRenderer()
        self._crt_available = None
        self._crt_last_error = ""
        self._last_crt_direct_player = False
        self._last_crt_direct_live = False

        self._live_preview = False
        self._preview_pos = 0
        self._record_pos = 0

        # Player worker (already threaded)
        self._play_worker_stop = threading.Event()
        self._latest_play_frame = None
        self._play_worker_thread = threading.Thread(target=self._play_worker_loop, daemon=True)
        self._play_worker_thread.start()

        # Scrub worker (prevents freezing while dragging slider)
        self._scrub_req = None
        self._scrub_stop = threading.Event()
        self._scrub_thread = threading.Thread(target=self._scrub_worker_loop, daemon=True)
        self._scrub_thread.start()

        # Editor preview stop flag (was missing in some builds)
        self._edit_stop = threading.Event()

        # Live-mode cached UI values (worker-safe snapshots)
        self._cached_live_downscale_width = 640
        self._cached_live_tape_mode = "SP"
        self._live_last_good = None
        # Live mode worker (camera input -> VHS pipeline)
        self._live_worker_stop = threading.Event()
        self._latest_live_frame = None
        self._live_on = False
        self._live_cap = None
        self._live_tape = None
        self._live_cam_index = 0
        self._live_seg_id = int(time.time()*1000) & 0x7fffffff
        self.live_player = VCRPlayer()
        self._live_worker_thread = threading.Thread(target=self._live_worker_loop, daemon=True)
        self._live_worker_thread.start()

        # Cached defect objects (updated on main thread; worker threads never touch Tk vars)
        self._cached_rec_def = self.rec_def
        self._cached_pb_def = self.pb_def
        self._cached_ap_def = self.ap_def
        self._cached_crt_settings = self.crt_settings
        self._cached_proxy_use = False
        self._cached_live_bufsec = 6.0
        # Optional RAM proxy (JPEG frames) for smooth playback
        self._proxy = None

        self._setup_theme()
        self._build_ui()
        self._start_cache_loop()
        self._poll_uiq()
        self._player_ui_loop()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self._play_worker_stop.set()
        try:
            self._live_worker_stop.set()
        except Exception:
            pass
        try:
            if self._live_cap is not None:
                self._live_cap.release()
        except Exception:
            pass
        try:
            self.audio_player.stop()
        except Exception:
            pass
        try:
            self.crt_renderer.close()
        except Exception:
            pass

        self._scrub_stop.set()
        self._edit_stop.set()
        try:
            self.root.destroy()
        except Exception:
            pass

    def _q(self, kind: str, payload):
        self.uiq.put((kind, payload))

    
    def _poll_uiq(self):
        # IMPORTANT: Don't drain the entire queue in one go (it can starve Tk and look like a freeze).
        # Also coalesce preview frames: only display the latest one per tick.
        max_items = 40
        last_rec = None
        last_edit = None
        try:
            for _ in range(max_items):
                kind, payload = self.uiq.get_nowait()
                if kind == "status_rec":
                    self.rec_status.set(payload)
                elif kind == "status_edit":
                    if hasattr(self, "edit_status"):
                        self.edit_status.set(payload)
                elif kind == "status_play":
                    self.play_status.set(payload)
                elif kind == "status_live":
                    if hasattr(self, "live_status"):
                        self.live_status.set(payload)
                elif kind == "status_crt":
                    if hasattr(self, "crt_status"):
                        self.crt_status.set(payload)
                elif kind == "preview_rec":
                    last_rec = payload
                elif kind == "preview_edit":
                    last_edit = payload
                elif kind == "call":
                    try:
                        payload()
                    except Exception:
                        traceback.print_exc()
        except queue.Empty:
            pass

        if last_rec is not None:
            self._show_image(self.rec_canvas, last_rec)
        if last_edit is not None and hasattr(self, "edit_canvas"):
            self._show_image(self.edit_canvas, last_edit)

        # If queue is still large, poll faster to catch up, otherwise normal pace.
        delay = 15 if self.uiq.qsize() > 80 else 50
        self.root.after(delay, self._poll_uiq)


    def _setup_theme(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background=PANEL_BG, foreground=TEXT, fieldbackground=SURFACE_BG)
        style.configure("TFrame", background=PANEL_BG)
        style.configure("Root.TFrame", background=DARK_BG)
        style.configure("Panel.TFrame", background=PANEL_BG)
        style.configure("Header.TFrame", background=DARK_BG)
        style.configure("TLabel", background=PANEL_BG, foreground=TEXT)
        style.configure("Muted.TLabel", background=PANEL_BG, foreground=MUTED)
        style.configure("HeaderTitle.TLabel", background=DARK_BG, foreground=TEXT, font=("Segoe UI", 15, "bold"))
        style.configure("HeaderMeta.TLabel", background=DARK_BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Section.TLabel", background=PANEL_BG, foreground=ACCENT_ACTIVE, font=("Segoe UI", 10, "bold"))
        style.configure("Status.TLabel", background=PANEL_BG, foreground=MUTED)
        style.configure("TButton", background=PANEL_BG_2, foreground=TEXT, bordercolor=BORDER, focusthickness=0, padding=(10, 6))
        style.map("TButton", background=[("active", "#1d2a38"), ("pressed", "#203246")], foreground=[("disabled", "#66717e")])
        style.configure("Accent.TButton", background=ACCENT, foreground="#07111d", bordercolor=ACCENT, padding=(12, 7))
        style.map("Accent.TButton", background=[("active", ACCENT_ACTIVE), ("pressed", "#318ce7")])
        style.configure("TCheckbutton", background=PANEL_BG, foreground=TEXT)
        style.map("TCheckbutton", background=[("active", PANEL_BG)], foreground=[("disabled", "#66717e")])
        style.configure("TRadiobutton", background=PANEL_BG, foreground=TEXT)
        style.configure("TSeparator", background=BORDER)
        style.configure("TNotebook", background=DARK_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background="#151d27", foreground=MUTED, padding=(14, 8), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", PANEL_BG), ("active", PANEL_BG_2)], foreground=[("selected", TEXT), ("active", TEXT)])
        style.configure("TEntry", fieldbackground=SURFACE_BG, foreground=TEXT, insertcolor=TEXT, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
        style.configure("TCombobox", fieldbackground=SURFACE_BG, background=PANEL_BG_2, foreground=TEXT, arrowcolor=TEXT, bordercolor=BORDER)
        style.map("TCombobox", fieldbackground=[("readonly", SURFACE_BG)], foreground=[("readonly", TEXT)])
        style.configure("TSpinbox", fieldbackground=SURFACE_BG, foreground=TEXT, bordercolor=BORDER, arrowsize=12)
        style.configure("Horizontal.TScale", background=PANEL_BG, troughcolor="#263343", bordercolor=PANEL_BG, lightcolor=PANEL_BG, darkcolor=PANEL_BG)
        try:
            self.root.option_add("*Font", "{Segoe UI} 9")
            self.root.option_add("*TCombobox*Listbox.background", SURFACE_BG)
            self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
            self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
            self.root.option_add("*TCombobox*Listbox.selectForeground", "#07111d")
        except Exception:
            pass


    def _start_cache_loop(self):
        # periodically snapshot Tk variables into plain dataclasses (main thread only)
        self._cache_defects_mainthread()
        self.root.after(120, self._start_cache_loop)

    def _cache_defects_mainthread(self):
        try:
            r, p, ap = self._sync_edit_defects()
            self._cached_rec_def = r
            self._cached_pb_def = p
            self._cached_ap_def = ap

            try:
                self._cached_live_downscale_width = int(getattr(self, "var_live_downscale_width").get())
            except Exception:
                pass
            try:
                self._cached_live_tape_mode = str(getattr(self, "live_tape_mode_var").get())
            except Exception:
                pass
            try:
                self._cached_live_bufsec = float(getattr(self, "live_bufsec_var").get())
            except Exception:
                pass
            try:
                self._cached_proxy_use = bool(getattr(self, "proxy_use_var").get())
            except Exception:
                pass
            try:
                crt = self._sync_crt_settings()
                self._cached_crt_settings = crt
                if self._last_crt_direct_player and not crt.direct_player:
                    self.crt_renderer.close_direct("player")
                if self._last_crt_direct_live and not crt.direct_live:
                    self.crt_renderer.close_direct("live")
                self._last_crt_direct_player = bool(crt.direct_player)
                self._last_crt_direct_live = bool(crt.direct_live)
            except Exception:
                pass
        except Exception:
            # If UI vars not ready yet, keep last cached
            pass

    def _show_image(self, label: tk.Label, bgr: np.ndarray):
        if bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        lw = max(1, label.winfo_width())
        lh = max(1, label.winfo_height())
        pil = pil.resize(self._fit(pil.size, (lw, lh)), Image.BILINEAR)
        imgtk = ImageTk.PhotoImage(pil)
        label.configure(image=imgtk, text="")
        label.image = imgtk

    def _fit(self, size, box):
        w, h = size
        bw, bh = box
        s = min(bw/max(1,w), bh/max(1,h))
        return (max(1, int(w*s)), max(1, int(h*s)))

    def _crt_float_var(self, name: str, default: float):
        existing = getattr(self, f"var_crt_{name}", None)
        if existing is not None:
            return existing
        var = tk.DoubleVar(value=float(default))
        setattr(self, f"var_crt_{name}", var)
        return var

    def _crt_slider(self, parent, label: str, name: str, frm: float, to: float, help_key: str | None = None):
        default = float(getattr(self.crt_settings, name, 0.0))
        var = self._crt_float_var(name, default)
        self._setting_header(parent, label, help_key or f"crt_{name}")
        ctk.CTkSlider(
            parent,
            from_=frm,
            to=to,
            variable=var,
            progress_color=ACCENT,
            button_color=ACCENT_ACTIVE,
        ).pack(anchor="w", fill="x", pady=(0, 7))
        return var

    def _sync_crt_settings(self) -> CRTSettings:
        c0 = getattr(self, "crt_settings", consumer_tv_preset())

        def _b(name: str, default: bool) -> bool:
            try:
                return bool(getattr(self, name).get())
            except Exception:
                return bool(default)

        def _f(name: str, default: float) -> float:
            try:
                return float(getattr(self, name).get())
            except Exception:
                return float(default)

        def _s(name: str, default: str) -> str:
            try:
                return str(getattr(self, name).get())
            except Exception:
                return str(default)

        def _i(name: str, default: int) -> int:
            try:
                return int(float(getattr(self, name).get()))
            except Exception:
                return int(default)

        settings = CRTSettings(
            enabled=_b("crt_enabled_var", c0.enabled),
            preview_enabled=_b("crt_preview_var", c0.preview_enabled),
            live_enabled=_b("crt_live_var", c0.live_enabled),
            export_enabled=_b("crt_export_var", c0.export_enabled),
            direct_player=_b("crt_direct_player_var", c0.direct_player),
            direct_live=_b("crt_direct_live_var", c0.direct_live),
            preset=_s("crt_preset_var", c0.preset),
            quality=_s("crt_quality_var", c0.quality),
            render_width=_i("crt_render_width_var", c0.render_width),
            mask_type=_s("crt_mask_type_var", c0.mask_type),
            mask_strength=_f("var_crt_mask_strength", c0.mask_strength),
            scanline_strength=_f("var_crt_scanline_strength", c0.scanline_strength),
            beam_sharpness=_f("var_crt_beam_sharpness", c0.beam_sharpness),
            bloom=_f("var_crt_bloom", c0.bloom),
            halation=_f("var_crt_halation", c0.halation),
            glass_diffusion=_f("var_crt_glass_diffusion", c0.glass_diffusion),
            curvature=_f("var_crt_curvature", c0.curvature),
            overscan=_f("var_crt_overscan", c0.overscan),
            vignette=_f("var_crt_vignette", c0.vignette),
            edge_focus=_f("var_crt_edge_focus", c0.edge_focus),
            phosphor_decay=_f("var_crt_phosphor_decay", c0.phosphor_decay),
            convergence_x=_f("var_crt_convergence_x", c0.convergence_x),
            convergence_y=_f("var_crt_convergence_y", c0.convergence_y),
            brightness=_f("var_crt_brightness", c0.brightness),
            contrast=_f("var_crt_contrast", c0.contrast),
            saturation=_f("var_crt_saturation", c0.saturation),
        ).validated()
        self.crt_settings = settings
        return settings

    def _set_crt_vars_from_settings(self, settings: CRTSettings):
        s = settings.validated()

        def _set(name: str, value):
            try:
                getattr(self, name).set(value)
            except Exception:
                pass

        _set("crt_enabled_var", bool(s.enabled))
        _set("crt_preview_var", bool(s.preview_enabled))
        _set("crt_live_var", bool(s.live_enabled))
        _set("crt_export_var", bool(s.export_enabled))
        _set("crt_direct_player_var", bool(s.direct_player))
        _set("crt_direct_live_var", bool(s.direct_live))
        _set("crt_preset_var", s.preset)
        _set("crt_quality_var", s.quality)
        _set("crt_render_width_var", int(s.render_width))
        _set("crt_mask_type_var", s.mask_type)
        for key, value in crt_settings_to_dict(s).items():
            if isinstance(value, (int, float)) and hasattr(self, f"var_crt_{key}"):
                _set(f"var_crt_{key}", float(value))
        self.crt_settings = s
        self._cached_crt_settings = s

    def _apply_crt_preset(self, name: str):
        current = self._sync_crt_settings()
        preset = preset_by_name(name)
        preset.enabled = current.enabled
        preset.preview_enabled = current.preview_enabled
        preset.live_enabled = current.live_enabled
        preset.export_enabled = current.export_enabled
        preset.direct_player = current.direct_player
        preset.direct_live = current.direct_live
        self._set_crt_vars_from_settings(preset)
        self._q("status_crt", f"Loaded CRT preset: {name}")

    def _test_crt_gpu(self):
        frame = np.zeros((72, 96, 3), dtype=np.uint8)
        frame[:, :32] = (32, 32, 220)
        frame[:, 32:64] = (32, 220, 32)
        frame[:, 64:] = (220, 32, 32)
        try:
            settings = self._sync_crt_settings()
            settings.enabled = True
            self.crt_renderer.render_frame(frame, settings, output_size=(320, 240), timeout=8.0)
            self._crt_available = True
            self._crt_last_error = ""
            self._q("status_crt", "GPU CRT renderer ready (ModernGL + GLFW OpenGL 3.3).")
        except CRTGPUUnavailable as exc:
            self._crt_available = False
            self._crt_last_error = str(exc)
            self._q("status_crt", f"GPU CRT unavailable: {exc}")
        except Exception as exc:
            self._crt_available = False
            self._crt_last_error = str(exc)
            self._q("status_crt", f"GPU CRT test failed: {exc}")

    def _report_crt_error_once(self, message: str):
        if message == getattr(self, "_crt_last_error", ""):
            return
        self._crt_available = False
        self._crt_last_error = message
        self._q("status_crt", f"GPU CRT unavailable: {message}")

    def _apply_crt_to_frame(self, frame: np.ndarray | None, mode: str) -> np.ndarray | None:
        if frame is None:
            return None
        settings = getattr(self, "_cached_crt_settings", getattr(self, "crt_settings", consumer_tv_preset())).validated()
        if not settings.enabled:
            return frame

        try:
            if mode == "player" and settings.direct_player:
                self.crt_renderer.submit_direct("player", frame, settings, "Digital VCR - CRT Player")
            elif mode == "live" and settings.direct_live:
                self.crt_renderer.submit_direct("live", frame, settings, "Digital VCR - CRT Live")
        except CRTGPUUnavailable as exc:
            self._report_crt_error_once(str(exc))
        except Exception as exc:
            self._report_crt_error_once(str(exc))

        integrated = (mode == "player" and settings.preview_enabled) or (mode == "live" and settings.live_enabled)
        if not integrated:
            return frame
        try:
            out = self.crt_renderer.render_frame(frame, settings, timeout=1.2)
            if self._crt_available is not True:
                self._q("status_crt", "GPU CRT renderer active.")
            self._crt_available = True
            self._crt_last_error = ""
            return out
        except CRTGPUUnavailable as exc:
            self._report_crt_error_once(str(exc))
            return frame
        except Exception as exc:
            self._report_crt_error_once(str(exc))
            return frame

    def _active_tape(self):
        if self.tape_loaded is not None:
            return self.tape_loaded
        if self.tape_edited is not None:
            return self.tape_edited
        return self.tape_live

    # ---------- UI ----------
    def _build_ui(self):
        self.gradient = GradientCanvas(self.root, bg=DARK_BG)
        self.gradient.place(relx=0, rely=0, relwidth=1, relheight=1)

        shell = ctk.CTkFrame(self.root, fg_color="transparent", corner_radius=0)
        shell.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.sidebar = ctk.CTkFrame(shell, width=210, fg_color=SIDEBAR_BG, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.content = ctk.CTkFrame(shell, fg_color="transparent", corner_radius=0)
        self.content.pack(side="left", fill="both", expand=True, padx=16, pady=16)

        ctk.CTkLabel(self.sidebar, text="Digital VCR", font=("Segoe UI", 20, "bold"), text_color=TEXT).pack(anchor="w", padx=18, pady=(20, 2))
        ctk.CTkLabel(self.sidebar, text=f"{APP_VERSION}", font=("Segoe UI", 11), text_color=MUTED).pack(anchor="w", padx=18, pady=(0, 22))

        self.tab_rec = ctk.CTkFrame(self.content, fg_color="transparent", corner_radius=0)
        self.tab_play = ctk.CTkFrame(self.content, fg_color="transparent", corner_radius=0)
        self.tab_vhs = ctk.CTkFrame(self.content, fg_color="transparent", corner_radius=0)
        self.tab_crt = ctk.CTkFrame(self.content, fg_color="transparent", corner_radius=0)
        self.tab_live = ctk.CTkFrame(self.content, fg_color="transparent", corner_radius=0)
        self._pages = {
            "Recorder": self.tab_rec,
            "Player": self.tab_play,
            "VHS Tape": self.tab_vhs,
            "CRT TV": self.tab_crt,
            "Live": self.tab_live,
        }
        self._nav_buttons = {}
        self.nb = PageNavigator(self)
        for name, page in self._pages.items():
            btn = ctk.CTkButton(
                self.sidebar,
                text=name,
                anchor="w",
                height=40,
                corner_radius=10,
                fg_color="transparent",
                hover_color="#172436",
                text_color=TEXT,
                command=lambda p=page: self._show_page(p),
            )
            btn.pack(fill="x", padx=12, pady=4)
            self._nav_buttons[page] = btn

        ctk.CTkLabel(
            self.sidebar,
            text="Studio Console\nDark gradient interface\nHover ? for setting help",
            justify="left",
            font=("Segoe UI", 10),
            text_color=MUTED,
        ).pack(side="bottom", anchor="w", padx=18, pady=18)

        self._build_recorder_tab()
        self._build_player_tab()
        self._build_vhs_tab()
        self._build_crt_tab()
        self._build_live_tab()
        self._show_page(self.tab_rec)

    def _show_page(self, page):
        for pg in getattr(self, "_pages", {}).values():
            pg.pack_forget()
        page.pack(fill="both", expand=True)
        for pg, btn in getattr(self, "_nav_buttons", {}).items():
            if pg is page:
                btn.configure(fg_color="#1f6aa5", text_color="#ffffff")
            else:
                btn.configure(fg_color="transparent", text_color=TEXT)

    def _help_text(self, name: str, label: str, key: str | None = None) -> str:
        candidates = []
        if key:
            candidates.append(f"{key}_{name}")
        candidates.append(name)
        for cand in candidates:
            if cand in SETTING_HELP:
                return SETTING_HELP[cand]
        return _help(
            f"Controls {label}.",
            "Lower values reduce the effect.",
            "Middle values are balanced.",
            "Higher values increase the effect.",
            "Changes the current tape workflow or output as labeled.",
            "UI setting.",
        )

    def _help_button(self, parent, name: str, label: str, key: str | None = None):
        btn = ctk.CTkButton(
            parent,
            text="?",
            width=24,
            height=24,
            corner_radius=12,
            fg_color="#233448",
            hover_color=ACCENT,
            text_color=TEXT,
            font=("Segoe UI", 11, "bold"),
        )
        HelpTooltip(btn, self._help_text(name, label, key))
        return btn

    def _section_title(self, parent, text: str):
        ctk.CTkLabel(parent, text=text, text_color=ACCENT_ACTIVE, font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(14, 6))

    def _setting_header(self, parent, label: str, help_key: str, key: str | None = None, value_text: str | None = None):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(anchor="w", fill="x", pady=(7, 1))
        ctk.CTkLabel(row, text=label, text_color=TEXT, font=("Segoe UI", 10, "bold")).pack(side="left")
        self._help_button(row, help_key, label, key).pack(side="left", padx=(7, 0))
        val_lbl = ctk.CTkLabel(row, text=value_text or "", text_color=MUTED, font=("Segoe UI", 10))
        val_lbl.pack(side="right")
        return val_lbl

    def _button(self, parent, text: str, command, *, accent: bool = False, width: int | None = None):
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            width=width or 120,
            height=34,
            corner_radius=10,
            fg_color=ACCENT if accent else CARD_BG_2,
            hover_color=ACCENT_ACTIVE if accent else "#223247",
            text_color="#07111d" if accent else TEXT,
        )

    def _setting_switch(self, parent, label: str, variable, help_key: str):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(anchor="w", fill="x", pady=5)
        sw = ctk.CTkSwitch(
            row,
            text=label,
            variable=variable,
            progress_color=ACCENT,
            button_color="#d8e8ff",
            fg_color="#2b3848",
            text_color=TEXT,
        )
        sw.pack(side="left")
        self._help_button(row, help_key, label).pack(side="left", padx=(8, 0))
        return sw

    def _setting_combo(self, parent, label: str, variable, values, help_key: str, width: int = 130):
        self._setting_header(parent, label, help_key)
        combo = ctk.CTkComboBox(
            parent,
            values=list(values),
            variable=variable,
            width=width,
            fg_color=SURFACE_BG,
            border_color=BORDER,
            button_color=CARD_BG_2,
            button_hover_color=ACCENT,
            dropdown_fg_color=SURFACE_BG,
            dropdown_hover_color="#223247",
            text_color=TEXT,
            dropdown_text_color=TEXT,
        )
        combo.pack(anchor="w", pady=(0, 5))
        return combo

    def _setting_entry(self, parent, label: str, variable, help_key: str, width: int = 120):
        self._setting_header(parent, label, help_key)
        entry = ctk.CTkEntry(parent, textvariable=variable, width=width, fg_color=SURFACE_BG, border_color=BORDER, text_color=TEXT)
        entry.pack(anchor="w", pady=(0, 5))
        return entry

    def _slider(self, parent, label, name, frm, to, key: str):
        return self._setting_slider(parent, label, name, frm, to, key)

    def _setting_slider(self, parent, label, name, frm, to, key: str, help_key: str | None = None):
        if key == "rec":
            init = float(getattr(self.rec_def, name))
            existing = getattr(self, f"var_rec_{name}", None)
            var = existing if existing is not None else tk.DoubleVar(value=init)
            setattr(self, f"var_rec_{name}", var)
        elif key == "rec_edit":
            init = float(getattr(self.rec_def, name))
            existing = getattr(self, f"var_rec_edit_{name}", None)
            var = existing if existing is not None else tk.DoubleVar(value=init)
            setattr(self, f"var_rec_edit_{name}", var)
        elif key == "pb":
            init = float(getattr(self.pb_def, name, 0.0))
            existing = getattr(self, f"var_pb_{name}", None)
            var = existing if existing is not None else tk.DoubleVar(value=init)
            setattr(self, f"var_pb_{name}", var)
        elif key == "ar":
            init = float(getattr(self.ar_def, name))
            existing = getattr(self, f"var_ar_{name}", None)
            var = existing if existing is not None else tk.DoubleVar(value=init)
            setattr(self, f"var_ar_{name}", var)
        else:
            init = float(getattr(self.ap_def, name))
            existing = getattr(self, f"var_ap_{name}", None)
            var = existing if existing is not None else tk.DoubleVar(value=init)
            setattr(self, f"var_ap_{name}", var)
        val_lbl = self._setting_header(parent, label, help_key or name, key)
        scale = ctk.CTkSlider(parent, from_=frm, to=to, variable=var, progress_color=ACCENT, button_color=ACCENT_ACTIVE, button_hover_color="#9ccdff")
        scale.pack(anchor="w", fill="x", pady=(0, 6))

        def _fmt_value():
            v = float(var.get())
            rng = float(to - frm) if float(to - frm) != 0.0 else 1.0
            pct = (v - float(frm)) / rng
            pct = max(0.0, min(1.0, pct))
            val_lbl.configure(text=f"{v:.3f}  ({pct*100:.0f}%)")

        try:
            var.trace_add("write", lambda *args: _fmt_value())
        except Exception:
            pass
        _fmt_value()


    # ---------- bundle loading/creation ----------
    def _load_bundle_async(self):
        folder = filedialog.askdirectory(title="Select tape bundle folder")
        if not folder:
            return
        self._q("status_rec", "Loading bundle…")
        self._q("status_edit", "Loading bundle…")
        self._q("status_play", "Loading bundle…")

        def worker():
            try:
                tape, settings = load_bundle(folder)
                rec_def, pb_def, ar_def, ap_def = settings_from_dict(settings)
                with self.lock:
                    self.tape_loaded = tape
                    self.tape_edited = None
                    self.bundle_path = folder
                    self.loaded_settings = settings
                    self.rec_def = rec_def
                    self.pb_def = pb_def
                    self.ar_def = ar_def
                    self.ap_def = ap_def
                    self._record_pos = 0
                    # jump editor preview to first recorded track
                    self._preview_pos = int(sorted(tape.cart.tracks.keys())[0]) if tape.cart.tracks else 0
                self._q("call", lambda settings=settings: (self._apply_settings_dict_to_ui(settings), self._update_scrub_range(), self.recpos_var.set("0"), self.scrub_var.set(0)))
                self._q("status_rec", f"Loaded bundle. Tracks={tape.cart.recorded_count()} len={tape.cart.length_tracks}")
                self._q("status_edit", "Loaded bundle into Editor.")
                self._q("status_play", "Loaded bundle. Insert → Play.")
            except Exception as e:
                self._q("status_rec", f"Load failed: {e}")
                self._q("status_edit", f"Load failed: {e}")
                self._q("status_play", f"Load failed: {e}")

        threading.Thread(target=worker, daemon=True).start()


    def _new_bundle_blank_tape(self):
        # Create a new tape bundle in the configured base folder (NO folder dialog by default).
        # This avoids Windows folder-dialog stalls and keeps the UI responsive.
        base_dir = self.base_dir_var.get().strip() if hasattr(self, "base_dir_var") else ""
        if not base_dir:
            base_dir = str(Path.cwd() / "tapes")

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = str(Path(base_dir) / f"tape_{stamp}")

        mins = max(1, int(self.tape_minutes.get()))
        length_tracks = int(30*2*60*mins)

        with self.lock:
            self.tape_live = TapeImage(cart=TapeCartridge(length_tracks=length_tracks))
            self.tape_edited = None
            self.tape_loaded = self.tape_live
            self.bundle_path = folder
            self.loaded_settings = None
            self._record_pos = 0

        self.recpos_var.set("0")
        self.scrub_var.set(0)
        self._update_scrub_range()
        self._q("status_rec", f"Creating new tape bundle… {folder}")

        def worker():
            try:
                settings = settings_to_dict(self.rec_def, self.pb_def, self.ar_def, self.ap_def)
                # Minimal blank bundle write (very fast)
                create_blank_bundle(folder, length_tracks=length_tracks, settings=settings)
                self._q("status_rec", f"Created new tape bundle on disk: {folder}")
                self._q("status_edit", f"Active bundle: {folder}")
                self._q("status_play", f"Active bundle: {folder}")
            except Exception as e:
                self._q("status_rec", f"Failed to create bundle: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def _use_memory(self):
        with self.lock:
            self.tape_loaded = None
            self.bundle_path = None
            self.loaded_settings = None
        self._update_scrub_range()
        self._q("status_play", "Using live/edited tape in memory (no bundle folder).")

    # -------- Recorder tab --------
    def _build_recorder_tab(self):
        left_sf = VScrollFrame(self.tab_rec, width=410)
        right = ctk.CTkFrame(self.tab_rec, fg_color=CARD_BG, corner_radius=16)
        left_sf.pack(side="left", fill="y", padx=(0, 14), pady=0)
        right.pack(side="right", fill="both", expand=True, padx=0, pady=0)
        left = left_sf.inner

        self._section_title(left, "Recorder")
        self._button(left, "Load tape bundle folder...", self._load_bundle_async).pack(anchor="w", fill="x", pady=(0,6))
        self._button(left, "Go to Player tab", lambda: self.nb.select(self.tab_play)).pack(anchor="w", fill="x", pady=(0,6))
        self._button(left, "Use memory (no disk)", self._use_memory).pack(anchor="w", fill="x", pady=(0,6))

        self._section_title(left, "Tape")

        self.base_dir_var = tk.StringVar(value=str(Path.cwd() / "tapes"))
        self._setting_header(left, "Tape base folder", "base_dir")
        rowb = ctk.CTkFrame(left, fg_color="transparent"); rowb.pack(fill="x", pady=4)
        ctk.CTkEntry(rowb, textvariable=self.base_dir_var, width=250, fg_color=SURFACE_BG, border_color=BORDER, text_color=TEXT).pack(side="left", padx=(0,6), fill="x", expand=True)
        self._button(rowb, "Browse...", self._browse_base_dir, width=86).pack(side="left")

        self.tape_minutes = tk.IntVar(value=5)
        self._setting_entry(left, "Blank tape length (min)", self.tape_minutes, "tape_minutes", width=90)
        self._button(left, "New tape (create in base folder)", self._new_bundle_blank_tape, accent=True).pack(anchor="w", fill="x", pady=(0, 6))

        self.recpos_var = tk.StringVar(value="0")
        self._setting_header(left, "Record position", "recpos")
        row2 = ctk.CTkFrame(left, fg_color="transparent"); row2.pack(fill="x", pady=4)
        ctk.CTkEntry(row2, textvariable=self.recpos_var, width=100, fg_color=SURFACE_BG, border_color=BORDER, text_color=TEXT).pack(side="left", padx=(0,6))
        self._button(row2, "Set", self._set_record_pos, width=62).pack(side="left")
        self._button(row2, "Rewind", lambda: self._set_record_pos_value(0), width=82).pack(side="left", padx=4)

        self.scrub_var = tk.IntVar(value=0)
        self._setting_header(left, "Scrub tape location", "scrub")
        self.scrub_scale = ctk.CTkSlider(left, from_=0, to=1000, variable=self.scrub_var, command=self._on_scrub, progress_color=ACCENT, button_color=ACCENT_ACTIVE)
        self.scrub_scale.pack(anchor="w", fill="x", pady=(0,4))
        self.scrub_lbl = tk.StringVar(value="0 / 0 tracks")
        ctk.CTkLabel(left, textvariable=self.scrub_lbl, text_color=MUTED).pack(anchor="w")

        self._section_title(left, "Source")

        self.src_var = tk.StringVar(value="")
        self._setting_header(left, "Source file", "source_file")
        ctk.CTkEntry(left, textvariable=self.src_var, width=340, fg_color=SURFACE_BG, border_color=BORDER, text_color=TEXT).pack(anchor="w", fill="x", pady=4)
        self._button(left, "Browse...", self._browse_source).pack(anchor="w", fill="x")

        self.monitor_var = tk.StringVar(value="tape")
        self._setting_combo(left, "Record monitor", self.monitor_var, ["tape","input"], "monitor_mode", width=130)

        self.down_w = tk.IntVar(value=self.rec_opts.downscale_width)
        self._setting_header(left, "Downscale width", "downscale_width")
        ctk.CTkSlider(left, from_=200, to=720, variable=self.down_w, progress_color=ACCENT, button_color=ACCENT_ACTIVE).pack(anchor="w", fill="x")

        self.sampling_var = tk.StringVar(value=getattr(self.rec_opts, "field_sampling", "progressive"))
        self._setting_combo(left, "Field sampling", self.sampling_var, ["progressive","interlaced"], "field_sampling", width=160)

        self.encode_threads_var = tk.IntVar(value=int(getattr(self.rec_opts, 'encode_threads', 0)))
        self._setting_entry(left, "Encode threads (0=auto)", self.encode_threads_var, "encode_threads", width=90)

        # Real RF modulation (FM+AM carrier round-trip)
        self.real_rf_var = tk.BooleanVar(value=bool(getattr(self.rec_def, 'real_rf_modulation', False)))
        self._setting_switch(left, "Real RF modulation (FM+AM)", self.real_rf_var, "real_rf_modulation")

        self.rec_mode = tk.StringVar(value=self.rec_def.tape_mode)
        self._setting_combo(left, "Tape mode (baked quality)", self.rec_mode, ["SP","LP","EP"], "tape_mode", width=110)

        self._section_title(left, "Record Defects")
        self._slider(left, "Luma bandwidth", "luma_bw", 0.35, 1.0, key="rec")
        self._slider(left, "Chroma bandwidth", "chroma_bw", 0.20, 1.0, key="rec")
        self._slider(left, "Record blur", "record_blur", 0.0, 1.0, key="rec")
        self._slider(left, "Record jitter", "record_jitter", 0.0, 1.0, key="rec")
        self._slider(left, "Record RF noise", "record_rf_noise", 0.0, 0.15, key="rec")
        self._slider(left, "Record dropouts", "record_dropouts", 0.0, 0.10, key="rec")

        self._section_title(left, "Audio (baked if ffmpeg available)")
        self._slider(left, "Audio wow/flutter", "wow", 0.0, 1.0, key="ar")
        self._slider(left, "Audio hiss", "hiss", 0.0, 1.0, key="ar")
        self._slider(left, "Audio dropouts", "dropouts", 0.0, 0.25, key="ar")
        self._slider(left, "Audio compression", "compression", 0.0, 1.0, key="ar")

        self.rt_var = tk.BooleanVar(value=True)
        self._setting_switch(left, "Enforce real-time record", self.rt_var, "rt_record")

        self.audio_extract_var = tk.BooleanVar(value=True)
        self._setting_switch(left, "Extract audio (ffmpeg)", self.audio_extract_var, "audio_extract")

        self.autosave_var = tk.BooleanVar(value=True)
        self._setting_switch(left, "Auto-save bundle after record", self.autosave_var, "autosave")

        self._button(left, "Record (overwrite)", self._record, accent=True).pack(anchor="w", fill="x", pady=(14,4))

        self.rec_status = tk.StringVar(value="Idle.")
        ctk.CTkLabel(left, textvariable=self.rec_status, wraplength=390, text_color=MUTED, justify="left").pack(anchor="w", pady=(10,0))

        self.rec_canvas = tk.Label(right, text="Live record monitor / scrub preview", background="#05070a", foreground=MUTED, bd=1, relief="solid", highlightbackground=BORDER)
        self.rec_canvas.pack(fill="both", expand=True)

        self._update_scrub_range()

    def _update_scrub_range(self):
        with self.lock:
            tape = self._active_tape()
            mx = max(2, int(tape.cart.length_tracks-2))
        try:
            self._scrub_max = mx
            self.scrub_scale.configure(to=mx)
            self.scrub_lbl.set(f"{int(self.scrub_var.get())} / {mx} tracks")
        except Exception:
            pass

    def _on_scrub(self, _val=None):
        v = int(self.scrub_var.get())
        self.scrub_lbl.set(f"{v} / {int(getattr(self, '_scrub_max', 1000))} tracks")
        self._scrub_req = v  # worker picks latest; no UI-thread decode

    def _scrub_worker_loop(self):
        last_done = -999999
        last_time = 0.0
        while not self._scrub_stop.is_set():
            req = self._scrub_req
            if req is None or req == last_done:
                time.sleep(0.02)
                continue
            # throttle while user is dragging
            if time.time() - last_time < 0.08:
                time.sleep(0.02)
                continue
            last_time = time.time()
            last_done = req
            with self.lock:
                tape = self._active_tape()
                pos = int(max(0, min(tape.cart.length_tracks-2, req)))
                a = tape.cart.get(pos)
                b = tape.cart.get(pos+1)
            if a is None or b is None:
                continue
            try:
                f0 = decode_field_bgr(a.y_dphi8, a.c_u8, a.meta)
                f1 = decode_field_bgr(b.y_dphi8, b.c_u8, b.meta)
                hh = min(f0.shape[0], f1.shape[0])
                ww = min(f0.shape[1], f1.shape[1])
                f0 = f0[:hh,:ww]; f1 = f1[:hh,:ww]
                out = np.zeros((hh*2, ww, 3), dtype=np.uint8)
                out[0::2] = f0
                out[1::2] = f1
                # downscale for preview to keep UI smooth
                if out.shape[1] > 900:
                    new_w = 900
                    new_h = int(out.shape[0] * (new_w / out.shape[1]))
                    out = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_AREA)
                self._q("preview_rec", out)
            except Exception:
                continue

    def _set_record_pos_value(self, v: int):
        with self.lock:
            mx = self._active_tape().cart.length_tracks-2
            self._record_pos = int(max(0, min(mx, v)))
        self.recpos_var.set(str(self._record_pos))
        self.scrub_var.set(self._record_pos)
        self._q("status_rec", f"Record position set to track {self._record_pos}.")

    def _set_record_pos(self):
        try:
            v = int(self.recpos_var.get())
        except Exception:
            v = 0
        self._set_record_pos_value(v)

    def _browse_base_dir(self):
        # Optional: choose base directory. If your system freezes on folder dialogs,
        # you can just type a path into the Tape base folder box instead.
        p = filedialog.askdirectory(title="Select base folder for new tapes")
        if p:
            self.base_dir_var.set(p)


    def _browse_source(self):
        p = filedialog.askopenfilename(title="Select video",
                                       filetypes=[("Video", "*.mp4 *.mov *.mkv *.avi *.webm"), ("All", "*.*")])
        if p:
            self.src_var.set(p)

    def _sync_rec_def(self):
        self.rec_def.tape_mode = self.rec_mode.get()
        for k in ["luma_bw","chroma_bw","record_blur","record_jitter","record_rf_noise","record_dropouts"]:
            setattr(self.rec_def, k, float(getattr(self, f"var_rec_{k}").get()))
        try:
            self.rec_def.real_rf_modulation = bool(self.real_rf_var.get())
        except Exception:
            pass

        # RF parameters (from VHS tab sliders, if available)
        for k, default in [
            ("rf_fm_depth", 1.0),
            ("rf_am_depth", 0.25),
            ("rf_phase_noise", 0.10),
            ("rf_carrier_noise", 0.20),
            ("rf_nonlinearity", 0.25),
            ("rf_chroma_fc_frac", 0.12),
            ("rf_chroma_lpf", 0.35),
        ]:
            try:
                var = getattr(self, f"var_rec_{k}")
                setattr(self.rec_def, k, float(var.get()))
            except Exception:
                # keep prior value/default
                if not hasattr(self.rec_def, k):
                    setattr(self.rec_def, k, float(default))
        self.rec_opts.downscale_width = int(self.down_w.get())
        self.rec_opts.enforce_real_time = bool(self.rt_var.get())
        self.rec_opts.extract_audio = bool(self.audio_extract_var.get())
        try:
            self.rec_opts.encode_threads = int(getattr(self, "encode_threads_var").get())
        except Exception:
            pass
        try:
            self.rec_opts.field_sampling = str(self.sampling_var.get())
        except Exception:
            pass
        self.ar_def.wow = float(getattr(self, "var_ar_wow").get())
        self.ar_def.hiss = float(getattr(self, "var_ar_hiss").get())
        self.ar_def.dropouts = float(getattr(self, "var_ar_dropouts").get())
        self.ar_def.compression = float(getattr(self, "var_ar_compression").get())

    def _record(self):
        path = self.src_var.get().strip()
        if not path:
            messagebox.showerror("Missing", "Choose a source file.")
            return
        self._sync_rec_def()
        monitor = self.monitor_var.get()
        autosave_on = bool(self.autosave_var.get())
        rec_def_for_save, pb_def_for_save, ap_def_for_save = self._sync_edit_defects()
        settings_for_save = settings_to_dict(rec_def_for_save, pb_def_for_save, self.ar_def, ap_def_for_save)

        def worker():
            self._q("status_rec", "Recording…")
            with self.lock:
                tape = self._active_tape()
                start = int(self._record_pos)
                bundle = self.bundle_path
            ok, endpos = self.recorder.record_from_file(
                path, tape, start,
                self.rec_opts, self.rec_def, self.ar_def,
                progress_cb=lambda f,t: self._q("status_rec", f"Recording… frames={f} pos_track={t}"),
                preview_cb=lambda img: self._q("preview_rec", img),
                monitor_mode=monitor
            )
            if not ok:
                self._q("status_rec", f"Error: {self.recorder.last_error}")
                return
            with self.lock:
                self._record_pos = endpos
            self._q("call", lambda: (self.recpos_var.set(str(endpos)), self.scrub_var.set(endpos), self._update_scrub_range()))

            msg = f"Done. Recorded up to track={endpos} | Stored tracks={tape.cart.recorded_count()}"
            if self.recorder.last_audio_error:
                msg += f" | Audio: {self.recorder.last_audio_error}"
            elif tape.audio.pcm16 is not None:
                msg += " | Audio: extracted"

            # Auto-save if we have a bundle folder
            if bundle and autosave_on:
                try:
                    save_bundle(bundle, tape, settings_for_save)
                    msg += " | Auto-saved bundle"
                except Exception as e:
                    msg += f" | Auto-save failed: {e}"

            self._q("status_rec", msg)

        threading.Thread(target=worker, daemon=True).start()

    # -------- Editor tab --------
    
    def _build_editor_tab(self):
        left_sf = VScrollFrame(self.tab_edit, width=460)
        right = ttk.Frame(self.tab_edit)
        left_sf.pack(side="left", fill="y", padx=10, pady=10)
        right.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        left = left_sf.inner

        ttk.Button(left, text="Load tape bundle folder…", command=self._load_bundle_async).pack(anchor="w", pady=(0,6))
        ttk.Button(left, text="Go to Player tab (final playback + export)", command=lambda: self.nb.select(self.tab_play)).pack(anchor="w", pady=(0,6))
        ttk.Button(left, text="Use memory (no disk)", command=self._use_memory).pack(anchor="w", pady=(0,6))

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(left, text="Editor = (re-)recording / baked defects").pack(anchor="w")
        ttk.Label(left, text="Playback/VCR effects are controlled in the Player tab (final preview).").pack(anchor="w", pady=(2,8))

        ttk.Label(left, text="Baked / recording defects (require Save/Dub)").pack(anchor="w")
        self.edit_mode = tk.StringVar(value=self.rec_def.tape_mode)
        ttk.Label(left, text="Tape mode").pack(anchor="w")
        ttk.Combobox(left, values=["SP","LP","EP"], textvariable=self.edit_mode, width=8, state="readonly").pack(anchor="w")

        self._slider(left, "Luma bandwidth", "luma_bw", 0.35, 1.0, key="rec_edit")
        self._slider(left, "Record blur", "record_blur", 0.0, 1.0, key="rec_edit")
        self._slider(left, "Record jitter", "record_jitter", 0.0, 1.0, key="rec_edit")
        self._slider(left, "Record RF noise", "record_rf_noise", 0.0, 0.15, key="rec_edit")
        self._slider(left, "Record dropouts", "record_dropouts", 0.0, 0.10, key="rec_edit")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(left, text="Audio (baked on Record, playback in Player)").pack(anchor="w")
        self._slider(left, "Audio wow/flutter", "wow", 0.0, 1.0, key="ar")
        self._slider(left, "Audio hiss (record)", "hiss", 0.0, 1.0, key="ar")
        self._slider(left, "Audio dropouts (record)", "dropouts", 0.0, 0.25, key="ar")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        self.live_prev_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Live Preview (threaded)", variable=self.live_prev_var, command=self._toggle_live_preview).pack(anchor="w")

        self.dub_rt = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Save/Dub in real time", variable=self.dub_rt).pack(anchor="w", pady=(6,0))

        self.export_video_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Export output.mp4 (toggle)", variable=self.export_video_var).pack(anchor="w", pady=(2,0))

        ttk.Button(left, text="Save bundle (and maybe export)", command=self._save_bundle).pack(anchor="w", pady=(10,4))

        self.edit_status = tk.StringVar(value="Enable Live Preview for real-time updates.")
        ttk.Label(left, textvariable=self.edit_status, wraplength=430).pack(anchor="w", pady=(10,0))

        self.edit_canvas = tk.Label(right, text="Editor preview", background="#05070a", foreground=MUTED, bd=1, relief="solid", highlightbackground=BORDER)
        self.edit_canvas.pack(fill="both", expand=True)

    def _toggle_live_preview(self):
        self._live_preview = bool(self.live_prev_var.get())
        if self._live_preview:
            # start preview at the first recorded area so user sees something immediately
            with self.lock:
                tape = self._active_tape()
                if tape.cart.tracks:
                    self._preview_pos = int(sorted(tape.cart.tracks.keys())[0])
                else:
                    self._preview_pos = 0
            self._q("status_edit", "Live preview ON")
        else:
            self._q("status_edit", "Live preview OFF")

    def _editor_worker_loop(self):
        # Threaded editor preview loop (prevents UI freezes).
        # DO NOT touch Tk variables here. Use cached defects updated on the main thread.
        while not self._edit_stop.is_set():
            if not getattr(self, '_live_preview', False):
                time.sleep(0.05)
                continue
            try:
                with self.lock:
                    tape = self._active_tape()
                if not tape.cart.tracks:
                    time.sleep(0.08)
                    continue

                rec_def = getattr(self, "_cached_rec_def", self.rec_def)
                pb_def = getattr(self, "_cached_pb_def", self.pb_def)

                frame, nxt = self.editor.preview_step(tape, int(self._preview_pos), rec_def, pb_def, lock=0.95)
                self._preview_pos = int(nxt)

                # Avoid building a backlog of preview frames
                if frame is not None and self.uiq.qsize() < 30:
                    if frame.shape[1] > 900:
                        new_w = 900
                        new_h = int(frame.shape[0] * (new_w / frame.shape[1]))
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    self._q('preview_edit', frame)

                time.sleep(1/15)
            except Exception:
                time.sleep(0.08)


    def _sync_edit_defects(self):
        """Build defect objects from Player tab vars (no separate editor)."""
        def _v(attr_name: str, default: float) -> float:
            try:
                return float(getattr(self, attr_name).get())
            except Exception:
                return float(default)
    
        pb0 = self.pb_def
        # Record-side defects (baked into tape) come from Recorder/Live sliders
        r0 = self.rec_def
        try:
            mode = self.rec_mode.get()
        except Exception:
            mode = r0.tape_mode
        rec = RecordDefects(
            tape_mode=mode,
            luma_bw=_v("var_rec_luma_bw", r0.luma_bw),
            chroma_bw=_v("var_rec_chroma_bw", getattr(r0, "chroma_bw", 1.0)),
            record_blur=_v("var_rec_record_blur", r0.record_blur),
            record_jitter=_v("var_rec_record_jitter", r0.record_jitter),
            record_rf_noise=_v("var_rec_record_rf_noise", r0.record_rf_noise),
            record_dropouts=_v("var_rec_record_dropouts", r0.record_dropouts),
            real_rf_modulation=bool(getattr(self, 'real_rf_var', tk.BooleanVar(value=bool(getattr(r0,'real_rf_modulation',False)))).get()),
            rf_fm_depth=_v("var_rec_rf_fm_depth", getattr(r0, "rf_fm_depth", 1.0)),
            rf_am_depth=_v("var_rec_rf_am_depth", getattr(r0, "rf_am_depth", 0.25)),
            rf_phase_noise=_v("var_rec_rf_phase_noise", getattr(r0, "rf_phase_noise", 0.10)),
            rf_carrier_noise=_v("var_rec_rf_carrier_noise", getattr(r0, "rf_carrier_noise", 0.20)),
            rf_nonlinearity=_v("var_rec_rf_nonlinearity", getattr(r0, "rf_nonlinearity", 0.25)),
            rf_chroma_fc_frac=_v("var_rec_rf_chroma_fc_frac", getattr(r0, "rf_chroma_fc_frac", 0.12)),
            rf_chroma_lpf=_v("var_rec_rf_chroma_lpf", getattr(r0, "rf_chroma_lpf", 0.35)),
        )

        try:
            aspect = self.pb_aspect_play.get()
        except Exception:
            aspect = pb0.aspect_display
        try:
            comp = bool(self.comp_play_var.get())
        except Exception:
            comp = pb0.composite_view
    
        pb = PlaybackDefects(
            aspect_display=aspect,
            tracking_knob=_v("var_pb_tracking_knob", pb0.tracking_knob),
            tracking_sensitivity=_v("var_pb_tracking_sensitivity", pb0.tracking_sensitivity),
            tracking_artifacts=_v("var_pb_tracking_artifacts", pb0.tracking_artifacts),
            auto_tracking=_v("var_pb_auto_tracking", getattr(pb0, "auto_tracking", 0.0)),
            auto_tracking_strength=_v("var_pb_auto_tracking_strength", getattr(pb0, "auto_tracking_strength", 0.70)),
            servo_recovery=_v("var_pb_servo_recovery", pb0.servo_recovery),
            sync_bias=_v("var_pb_sync_bias", pb0.sync_bias),
            servo_hunt=_v("var_pb_servo_hunt", getattr(pb0, "servo_hunt", 0.22)),
            servo_hunt_freq=_v("var_pb_servo_hunt_freq", getattr(pb0, "servo_hunt_freq", 0.55)),
            head_switch_strength=_v("var_pb_head_switch_strength", getattr(pb0, "head_switch_strength", 0.22)),
            head_switch_freq=_v("var_pb_head_switch_freq", getattr(pb0, "head_switch_freq", 0.70)),
            playback_timebase=_v("var_pb_playback_timebase", pb0.playback_timebase),
            timebase_freq=_v("var_pb_timebase_freq", getattr(pb0, "timebase_freq", 0.70)),
            playback_rf_noise=_v("var_pb_playback_rf_noise", pb0.playback_rf_noise),
            playback_dropouts=_v("var_pb_playback_dropouts", pb0.playback_dropouts),
            dropout_freq=_v("var_pb_dropout_freq", getattr(pb0, "dropout_freq", 0.65)),
            interference=_v("var_pb_interference", pb0.interference),
            interference_freq=_v("var_pb_interference_freq", getattr(pb0, "interference_freq", 0.75)),
            snow=_v("var_pb_snow", pb0.snow),
            snow_freq=_v("var_pb_snow_freq", getattr(pb0, "snow_freq", 0.85)),
            variance=_v("var_pb_variance", pb0.variance),
            composite_view=comp,
            chroma_shift_x=_v("var_pb_chroma_shift_x", pb0.chroma_shift_x),
            chroma_shift_y=_v("var_pb_chroma_shift_y", pb0.chroma_shift_y),
            chroma_phase=_v("var_pb_chroma_phase", pb0.chroma_phase),
            chroma_noise=_v("var_pb_chroma_noise", pb0.chroma_noise),
            chroma_noise_freq=_v("var_pb_chroma_noise_freq", getattr(pb0, "chroma_noise_freq", 0.55)),
            chroma_wobble=_v("var_pb_chroma_wobble", getattr(pb0, "chroma_wobble", 0.10)),
            chroma_wobble_freq=_v("var_pb_chroma_wobble_freq", getattr(pb0, "chroma_wobble_freq", 0.55)),
            brightness=_v("var_pb_brightness", pb0.brightness),
            contrast=_v("var_pb_contrast", pb0.contrast),
            saturation=_v("var_pb_saturation", pb0.saturation),
            bloom=_v("var_pb_bloom", pb0.bloom),
            sharpen=_v("var_pb_sharpen", pb0.sharpen),
            playback_blur=_v("var_pb_playback_blur", getattr(pb0, "playback_blur", 0.10)),
            playback_blur_freq=_v("var_pb_playback_blur_freq", getattr(pb0, "playback_blur_freq", 0.65)),
            frame_jitter=_v("var_pb_frame_jitter", getattr(pb0, "frame_jitter", 0.12)),
            frame_jitter_freq=_v("var_pb_frame_jitter_freq", getattr(pb0, "frame_jitter_freq", 0.65)),
            scanline_strength=_v("var_pb_scanline_strength", getattr(pb0, "scanline_strength", 0.0)),
            scanline_soften=_v("var_pb_scanline_soften", pb0.scanline_soften),
            luma_chroma_bleed=_v("var_pb_luma_chroma_bleed", getattr(pb0, "luma_chroma_bleed", 0.0)),
            rf_playback_model=bool(getattr(self, 'rf_play_var', tk.BooleanVar(value=bool(getattr(pb0,'rf_playback_model',False)))).get()),
            rf_playback_fm_depth=_v("var_pb_rf_playback_fm_depth", getattr(pb0, "rf_playback_fm_depth", 1.0)),
            rf_playback_am_depth=_v("var_pb_rf_playback_am_depth", getattr(pb0, "rf_playback_am_depth", 0.18)),
            rf_playback_phase_noise=_v("var_pb_rf_playback_phase_noise", getattr(pb0, "rf_playback_phase_noise", 0.12)),
            rf_playback_carrier_noise=_v("var_pb_rf_playback_carrier_noise", getattr(pb0, "rf_playback_carrier_noise", 0.20)),
            rf_playback_nonlinearity=_v("var_pb_rf_playback_nonlinearity", getattr(pb0, "rf_playback_nonlinearity", 0.20)),
        )
    
        ap0 = self.ap_def
        ap = AudioPlaybackDefects(
            hiss=_v("var_ap_hiss", ap0.hiss),
            pops=_v("var_ap_pops", ap0.pops),
        )
        return rec, pb, ap
    
    def _save_bundle(self):
        with self.lock:
            tape = self._active_tape()
            bundle = self.bundle_path
        if tape.cart.length_tracks < 2:
            messagebox.showerror("No tape", "Record or load a tape first.")
            return

        rec_def, pb_def, ap_def = self._sync_edit_defects()
        crt_def = self._sync_crt_settings()
        dub_realtime = bool(self.dub_rt.get()) if hasattr(self, "dub_rt") else False
        export_video = bool(self.export_video_var.get()) if hasattr(self, "export_video_var") else True

        # Prefer current bundle folder if present; otherwise ask
        folder = bundle
        if folder is None:
            folder = filedialog.askdirectory(title="Choose / create bundle folder")
            if not folder:
                return
        folder = str(Path(folder))

        def worker():
            self._q("status_edit", "Dubbing / re-recording…")
            out_tape = self.editor.dub_rerecord(
                tape, rec_def,
                DubOptions(enforce_real_time=dub_realtime),
                progress_cb=lambda a,b: self._q("status_edit", f"Dubbing… {a}/{b}"),
                preview_cb=lambda fr: self._q("preview_edit", fr)
            )
            with self.lock:
                self.tape_edited = out_tape
                self.tape_loaded = out_tape
                self.bundle_path = folder

            settings = settings_to_dict(rec_def, pb_def, self.ar_def, ap_def)
            settings["crt"] = crt_settings_to_dict(crt_def)
            save_bundle(folder, out_tape, settings)

            if not export_video:
                self._q("status_edit", f"Saved bundle (no video export): {folder}")
                return

            self._q("status_edit", "Exporting output.mp4…")
            out_mp4 = str(Path(folder) / "output.mp4")
            try:
                ok = export_playback_video_mp4(
                    out_tape, pb_def,
                    ExportOptions(out_mp4=out_mp4, fps=30.0, upscale_width=960),
                    progress_cb=lambda a,b: self._q("status_edit", f"Exporting video… {a}/{b}"),
                    crt=crt_def,
                    crt_renderer=self.crt_renderer,
                )
            except CRTGPUUnavailable as exc:
                self._q("status_edit", f"CRT export failed: {exc}")
                self._q("status_crt", f"GPU CRT unavailable: {exc}")
                return
            if not ok:
                self._q("status_edit", "Export failed (VideoWriter).")
                return

            if out_tape.audio.pcm16 is not None and out_tape.audio.pcm16.size > 0:
                wav = str(Path(folder) / "audio_playback.wav")
                out_mux = str(Path(folder) / "output_with_audio.mp4")
                ok2, err = export_audio_and_optional_mux(out_tape, wav, out_mp4, out_mux, ap=ap_def)
                if ok2:
                    self._q("status_edit", f"Done. output.mp4 + output_with_audio.mp4 in: {folder}")
                else:
                    self._q("status_edit", f"Done. output.mp4 saved. Audio mux failed: {err}")
            else:
                self._q("status_edit", f"Done. output.mp4 saved in: {folder} (no audio)")

        threading.Thread(target=worker, daemon=True).start()

    # -------- Player tab --------
    
    def _build_live_tab(self):
        left_sf = VScrollFrame(self.tab_live, width=410)
        right = ctk.CTkFrame(self.tab_live, fg_color=CARD_BG, corner_radius=16)
        left_sf.pack(side="left", fill="y", padx=(0, 14), pady=0)
        right.pack(side="right", fill="both", expand=True, padx=0, pady=0)
        left = left_sf.inner

        self._section_title(left, "Live Mode")
        ctk.CTkLabel(left, text="Camera input -> VHS pipeline -> preview", text_color=MUTED).pack(anchor="w", pady=(0,8))

        self.live_cam_var = tk.StringVar(value=str(int(getattr(self, "_live_cam_index", 0))))
        self._setting_header(left, "Camera index", "live_cam")
        cam_row = ctk.CTkFrame(left, fg_color="transparent"); cam_row.pack(anchor="w", fill="x", pady=4)
        self.live_cam_combo = ctk.CTkComboBox(cam_row, width=110, values=["0"], variable=self.live_cam_var, fg_color=SURFACE_BG, border_color=BORDER, button_color=CARD_BG_2, text_color=TEXT)
        self.live_cam_combo.pack(side="left", padx=(0,6))
        self._button(cam_row, "Refresh", self._refresh_cameras, width=92).pack(side="left")
        self._refresh_cameras()

        self.live_bufsec_var = tk.DoubleVar(value=6.0)
        self._setting_header(left, "Live tape buffer (seconds)", "live_bufsec")
        ctk.CTkSlider(left, from_=2.0, to=20.0, variable=self.live_bufsec_var, progress_color=ACCENT, button_color=ACCENT_ACTIVE).pack(anchor="w", fill="x", pady=(0,8))
        self.live_tape_mode_var = tk.StringVar(value="SP")
        self._setting_combo(left, "Live tape speed / mode", self.live_tape_mode_var, ["SP","LP","EP"], "tape_mode", width=110)


        self.live_toggle_var = tk.BooleanVar(value=False)
        live_sw = self._setting_switch(left, "Live mode ON", self.live_toggle_var, "live_mode")
        live_sw.configure(command=self._toggle_live)

        self.live_overlay_var = tk.BooleanVar(value=False)
        overlay_sw = self._setting_switch(left, "Overlay fullscreen output", self.live_overlay_var, "live_overlay")
        overlay_sw.configure(command=self._toggle_live_overlay)
        ctk.CTkLabel(left, text="Tip: Press ESC to close overlay.", text_color=MUTED).pack(anchor="w", pady=(0,8))

        self._section_title(left, "Live Controls (Recording)")
        # Live preview downscale (controls the record-side horizontal resolution)
        self.var_live_downscale_width = tk.IntVar(value=640)
        row = ctk.CTkFrame(left, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text="Downscale width (px)", text_color=TEXT, font=("Segoe UI", 10, "bold")).pack(side="left")
        self._help_button(row, "live_downscale_width", "Downscale width (px)").pack(side="left", padx=(7, 0))
        ctk.CTkLabel(row, textvariable=self.var_live_downscale_width, text_color=MUTED).pack(side="right")
        ctk.CTkSlider(left, from_=240, to=960, variable=self.var_live_downscale_width, progress_color=ACCENT, button_color=ACCENT_ACTIVE).pack(fill="x")
        ctk.CTkLabel(left, text="Higher = sharper, more CPU.", text_color=MUTED).pack(anchor="w", pady=(0,4))

        # Video record-side sliders
        self._slider(left, "Luma bandwidth", "luma_bw", 0.35, 1.0, key="rec")
        self._slider(left, "Record blur", "record_blur", 0.0, 1.0, key="rec")
        self._slider(left, "Record jitter", "record_jitter", 0.0, 1.0, key="rec")
        self._slider(left, "Record RF noise", "record_rf_noise", 0.0, 0.15, key="rec")
        self._slider(left, "Record dropouts", "record_dropouts", 0.0, 0.10, key="rec")

        self._section_title(left, "Live Controls (Audio Record)")

        # Audio record-side sliders
        self._slider(left, "Audio wow/flutter", "wow", 0.0, 1.0, key="ar")
        self._slider(left, "Audio hiss", "hiss", 0.0, 1.0, key="ar")
        self._slider(left, "Audio dropouts", "dropouts", 0.0, 0.25, key="ar")
        self._slider(left, "Audio compression", "compression", 0.0, 1.0, key="ar")

        self._section_title(left, "Live Controls (Playback / Tracking)")

        # Playback-side sliders (same as Player tab)
        self._slider(left, "Tracking knob", "tracking_knob", 0.0, 1.0, key="pb")
        self._slider(left, "Tracking sensitivity", "tracking_sensitivity", 0.0, 1.0, key="pb")
        self._slider(left, "Tracking artifacts", "tracking_artifacts", 0.0, 2.0, key="pb")
        self._slider(left, "Auto tracking (0=off, 1=on)", "auto_tracking", 0.0, 1.0, key="pb")
        self._slider(left, "Auto tracking strength", "auto_tracking_strength", 0.0, 1.0, key="pb")
        self._slider(left, "Servo recovery", "servo_recovery", 0.0, 1.0, key="pb")
        self._slider(left, "Sync bias", "sync_bias", 0.0, 1.0, key="pb")
        self._slider(left, "Servo hunt (amount)", "servo_hunt", 0.0, 1.0, key="pb")
        self._slider(left, "Servo hunt frequency", "servo_hunt_freq", 0.0, 1.0, key="pb")
        self._slider(left, "Head switch (amount)", "head_switch_strength", 0.0, 1.0, key="pb")
        self._slider(left, "Head switch frequency", "head_switch_freq", 0.0, 1.0, key="pb")
        self._slider(left, "Timebase (amount)", "playback_timebase", 0.0, 1.0, key="pb")
        self._slider(left, "Timebase frequency", "timebase_freq", 0.0, 1.0, key="pb")
        self._slider(left, "Playback RF noise", "playback_rf_noise", 0.0, 0.25, key="pb")
        self._slider(left, "Dropouts (amount)", "playback_dropouts", 0.0, 0.12, key="pb")
        self._slider(left, "Dropouts frequency", "dropout_freq", 0.0, 1.0, key="pb")
        self._slider(left, "Interference (amount)", "interference", 0.0, 1.0, key="pb")
        self._slider(left, "Interference frequency", "interference_freq", 0.0, 1.0, key="pb")
        self._slider(left, "Variance / instability", "variance", 0.0, 1.0, key="pb")
        self._slider(left, "Snow (amount)", "snow", 0.0, 1.0, key="pb")
        self._slider(left, "Snow frequency", "snow_freq", 0.0, 1.0, key="pb")

        self._slider(left, "Chroma delay (horizontal)", "chroma_shift_x", 0.0, 1.0, key="pb")
        self._slider(left, "Chroma delay (vertical)", "chroma_shift_y", 0.0, 1.0, key="pb")
        self._slider(left, "Chroma phase error", "chroma_phase", 0.0, 1.0, key="pb")
        self._slider(left, "Chroma phase noise", "chroma_noise", 0.0, 1.0, key="pb")
        self._slider(left, "Chroma noise frequency", "chroma_noise_freq", 0.0, 1.0, key="pb")
        self._slider(left, "Chroma wobble", "chroma_wobble", 0.0, 1.0, key="pb")
        self._slider(left, "Chroma wobble frequency", "chroma_wobble_freq", 0.0, 1.0, key="pb")

        self._slider(left, "Scanlines", "scanline_strength", 0.0, 1.0, key="pb")
        self._slider(left, "Scanline soften", "scanline_soften", 0.0, 1.0, key="pb")
        self._slider(left, "Brightness", "brightness", -1.0, 1.0, key="pb")
        self._slider(left, "Contrast", "contrast", 0.0, 1.0, key="pb")
        self._slider(left, "Saturation", "saturation", 0.0, 1.0, key="pb")
        self._slider(left, "Bloom", "bloom", 0.0, 1.0, key="pb")
        self._slider(left, "Sharpen", "sharpen", 0.0, 1.0, key="pb")
        self._slider(left, "Playback blur", "playback_blur", 0.0, 1.0, key="pb")
        self._slider(left, "Blur frequency", "playback_blur_freq", 0.0, 1.0, key="pb")
        self._slider(left, "Frame jitter", "frame_jitter", 0.0, 1.0, key="pb")
        self._slider(left, "Jitter frequency", "frame_jitter_freq", 0.0, 1.0, key="pb")

        self._section_title(left, "Live Controls (Audio Playback)")
        self._slider(left, "Audio hiss", "hiss", 0.0, 1.0, key="ap")
        self._slider(left, "Audio pops", "pops", 0.0, 1.0, key="ap")

        self._section_title(left, "Quick Access")
        self._button(left, "Go to Recorder", lambda: self.nb.select(self.tab_rec)).pack(anchor="w", fill="x", pady=2)
        self._button(left, "Go to Player", lambda: self.nb.select(self.tab_play)).pack(anchor="w", fill="x", pady=2)

        self.live_status = tk.StringVar(value="Select a camera, then enable Live mode.")
        ctk.CTkLabel(left, textvariable=self.live_status, wraplength=360, text_color=MUTED, justify="left").pack(anchor="w", pady=(12,0))

        self.live_canvas = tk.Label(right, text="Live output", background="#05070a", foreground=MUTED, bd=1, relief="solid", highlightbackground=BORDER)
        self.live_canvas.pack(fill="both", expand=True)

        # Hotkey: F11 toggles overlay
        try:
            self.root.bind('<F11>', lambda e: (self.live_overlay_var.set(not bool(self.live_overlay_var.get())), self._toggle_live_overlay()))
        except Exception:
            pass

        self._live_ui_loop()

    def _refresh_cameras(self):
        avail = []
        old_log_level = None
        try:
            if hasattr(cv2, "getLogLevel") and hasattr(cv2, "setLogLevel"):
                old_log_level = cv2.getLogLevel()
                cv2.setLogLevel(0)
        except Exception:
            old_log_level = None
        try:
            for i in range(0, 8):
                cap = None
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap is not None and cap.isOpened():
                        avail.append(i)
                except Exception:
                    pass
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
        finally:
            try:
                if old_log_level is not None:
                    cv2.setLogLevel(old_log_level)
            except Exception:
                pass
        if not avail:
            avail = [0]
        values = [str(i) for i in avail]
        self.live_cam_combo.configure(values=values)
        cur = str(self.live_cam_var.get()) if hasattr(self, "live_cam_var") else str(avail[0])
        if cur not in values:
            cur = str(avail[0])
        self.live_cam_combo.set(cur)
        try:
            self.live_cam_var.set(str(cur))
        except Exception:
            pass
        self.live_cam_combo.configure(command=lambda value: self.live_cam_var.set(str(value)))

    def _toggle_live(self):
        try:
            on = bool(self.live_toggle_var.get())
        except Exception:
            on = False
        try:
            self._live_cam_index = int(self.live_cam_var.get())
        except Exception:
            pass
        try:
            self._cached_live_bufsec = float(self.live_bufsec_var.get())
        except Exception:
            pass
        self._live_on = on
        old_cap = self._live_cap
        self._live_cap = None
        if not on and old_cap is not None:
            try:
                old_cap.release()
            except Exception:
                pass
        if on:
            try:
                self.live_status.set("Live mode: starting…")
            except Exception:
                pass
        else:
            try:
                self.live_status.set("Live mode: off")
            except Exception:
                pass

    def _toggle_live_overlay(self):
        try:
            on = bool(self.live_overlay_var.get())
        except Exception:
            on = False
        if on:
            self._open_live_overlay()
        else:
            self._close_live_overlay()

    def _open_live_overlay(self):
        if getattr(self, "_live_overlay_win", None) is not None:
            try:
                self._live_overlay_win.lift()
                self._live_overlay_win.focus_force()
            except Exception:
                pass
            return
        win = tk.Toplevel(self.root)
        win.title("VCR Live Overlay")
        win.configure(bg="#000")
        try:
            win.attributes("-fullscreen", True)
        except Exception:
            pass
        try:
            win.attributes("-topmost", True)
        except Exception:
            pass
        lbl = tk.Label(win, text="", bg="#000")
        lbl.pack(fill="both", expand=True)
        self._live_overlay_win = win
        self._live_overlay_label = lbl

        def _close_evt(event=None):
            self._close_live_overlay()
        win.bind("<Escape>", _close_evt)
        win.bind("<Button-1>", lambda e: win.focus_force())
        try:
            win.protocol("WM_DELETE_WINDOW", self._close_live_overlay)
        except Exception:
            pass
        try:
            win.lift()
            win.focus_force()
        except Exception:
            pass

    def _close_live_overlay(self):
        try:
            if hasattr(self, "live_overlay_var"):
                self.live_overlay_var.set(False)
        except Exception:
            pass
        win = getattr(self, "_live_overlay_win", None)
        self._live_overlay_win = None
        self._live_overlay_label = None
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass

    def _live_worker_loop(self):
        # Worker thread: capture camera -> encode to tape tracks -> decode via player -> store latest frame
        t_last = time.perf_counter()
        drop_n = 0
        while not self._live_worker_stop.is_set():
            try:
                if not getattr(self, "_live_on", False):
                    time.sleep(0.05)
                    continue

                # Open/reopen capture if needed
                if self._live_cap is None:
                    cam_idx = int(getattr(self, "_live_cam_index", 0))
                    self._live_cam_index = cam_idx

                    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        self._q("status_live", f"Live: could not open camera {cam_idx}")
                        self._live_on = False
                        time.sleep(0.2)
                        continue
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    self._live_cap = cap

                    # Create ring tape buffer
                    try:
                        bufsec = float(getattr(self, "_cached_live_bufsec", 6.0))
                    except Exception:
                        bufsec = 6.0
                    length_tracks = int(max(180, min(2400, bufsec*60.0))) + 8
                    self._live_tape = TapeImage(TapeCartridge(length_tracks=length_tracks))

                    self.live_player.insert()
                    self.live_player.play()
                    self._live_seg_id = int(time.time()*1000) & 0x7fffffff

                    self._q("status_live", f"Live: camera {cam_idx} opened")
                    t_last = time.perf_counter()

                # snapshot defs/options (worker never touches Tk vars)
                with self.lock:
                    rec_def = getattr(self, "_cached_rec_def", self.rec_def)
                    pb_def = getattr(self, "_cached_pb_def", self.pb_def)
                    opts = self.rec_opts
                    tape = self._live_tape

                # Live-only overrides (so Live can differ from Recorder tab)
                try:
                    live_mode = str(getattr(self, "_cached_live_tape_mode", getattr(rec_def, "tape_mode", "SP")))
                except Exception:
                    live_mode = getattr(rec_def, "tape_mode", "SP")
                try:
                    if live_mode:
                        rec_def = RecordDefects(**{**rec_def.__dict__, "tape_mode": live_mode})
                except Exception:
                    pass
                try:
                    live_w = int(getattr(self, "_cached_live_downscale_width", 0))
                except Exception:
                    live_w = 0


                if tape is None:
                    time.sleep(0.05)
                    continue

                # Update servo (advances tape position)
                self.live_player.update(tape, pb_def)
                base = int(self.live_player.state.pos_tracks)

                # Wrap ring buffer cleanly
                if base >= tape.cart.length_tracks - 4:
                    self.live_player.state.pos_tracks = 0.0
                    self.live_player.state._last_pos_tracks = 0.0
                    self.live_player._cache.clear()
                    base = 0

                # Capture frame (drop buffered frames if we fall behind to avoid 'fast old frames')
                try:
                    for _ in range(int(drop_n)):
                        self._live_cap.grab()
                except Exception:
                    pass
                ok, frame = self._live_cap.read()
                if not ok or frame is None:
                    # Camera frame-drop -> write a weak/garbled control track for this moment.
                    # This makes Live behave like a real VCR losing RF/sync briefly instead of just "skipping".
                    try:
                        last = getattr(self, "_live_last_good", None)
                        if isinstance(last, dict) and ("y0" in last) and ("y1" in last):
                            # Synthesize noisy/unreadable RF for this field pair.
                            y0b = (128 + np.random.randn(*last["y0"].shape).astype(np.float32) * 42.0).clip(0, 255).astype(np.uint8)
                            c0b = (128 + np.random.randn(*last["c0"].shape).astype(np.float32) * 28.0).clip(0, 255).astype(np.uint8)
                            y1b = (128 + np.random.randn(*last["y1"].shape).astype(np.float32) * 42.0).clip(0, 255).astype(np.uint8)
                            c1b = (128 + np.random.randn(*last["c1"].shape).astype(np.float32) * 28.0).clip(0, 255).astype(np.uint8)

                            # Control track becomes weak; vertical jitter spikes.
                            sync_u8 = int(np.clip(np.random.randint(8, 42), 0, 255))
                            vjit_u8 = int(np.clip(np.random.randint(180, 255), 0, 255))
                            dt_field = 1.0 / 60.0

                            for field_i, (yy, cc, meta_src) in enumerate([
                                (y0b, c0b, last.get("meta0", {})),
                                (y1b, c1b, last.get("meta1", {})),
                            ]):
                                idx = base + field_i
                                meta = dict(meta_src) if isinstance(meta_src, dict) else {}
                                meta.update({
                                    "dt": dt_field,
                                    "seg_id": int(self._live_seg_id),
                                    "ctl_sync_u8": int(sync_u8),
                                    "ctl_vjit_u8": int(vjit_u8),
                                    "tape_mode": str(rec_def.tape_mode),
                                    "field": int(field_i),
                                    "capture_drop": True,
                                })
                                tape.cart.set(idx, TapeTrack(y_dphi8=yy, c_u8=cc, meta=meta))
                        else:
                            # No last frame yet -> leave track empty (brief black / unlock).
                            try:
                                tape.cart.tracks.pop(base, None)
                                tape.cart.tracks.pop(base + 1, None)
                            except Exception:
                                pass

                        out = self.live_player.get_frame(tape, pb_def)
                        self._latest_live_frame = self._apply_crt_to_frame(out, "live")
                    except Exception:
                        pass
                    time.sleep(0.005)
                    continue

                # Downscale to recorder width (preserve aspect)
                try:
                    target_w = int(max(64, (live_w if int(live_w)>0 else getattr(opts, "downscale_width", 360))))
                except Exception:
                    target_w = 360
                if frame.shape[1] != target_w and frame.shape[1] > 0:
                    scale = float(target_w) / float(frame.shape[1])
                    nh = max(2, int(frame.shape[0] * scale))
                    frame = cv2.resize(frame, (target_w, nh), interpolation=cv2.INTER_AREA)

                # Even height for field split
                if frame.shape[0] % 2 == 1:
                    frame = frame[:-1, :, :]

                f0, f1 = sample_fields_from_frame(frame, getattr(opts, 'field_sampling', 'interlaced'))

                f0b = apply_record_defects_to_field(f0, rec_def)
                f1b = apply_record_defects_to_field(f1, rec_def)

                y0, c0, meta0 = encode_field_bgr(
                    f0b,
                    sample_rate=opts.sample_rate,
                    chroma_subsample=opts.chroma_subsample,
                    luma_bw=rec_def.luma_bw,
                    chroma_bw=float(getattr(rec_def, 'chroma_bw', 1.0)),
                )
                y1, c1, meta1 = encode_field_bgr(
                    f1b,
                    sample_rate=opts.sample_rate,
                    chroma_subsample=opts.chroma_subsample,
                    luma_bw=rec_def.luma_bw,
                    chroma_bw=float(getattr(rec_def, 'chroma_bw', 1.0)),
                )

                y0 = apply_rf_defects_y_dphi_u8(y0, rec_def.record_rf_noise, rec_def.record_dropouts, rec_def.tape_mode)
                c0 = apply_rf_defects_chroma_u8(c0, rec_def.record_rf_noise, rec_def.record_dropouts, rec_def.tape_mode)
                y1 = apply_rf_defects_y_dphi_u8(y1, rec_def.record_rf_noise, rec_def.record_dropouts, rec_def.tape_mode)
                c1 = apply_rf_defects_chroma_u8(c1, rec_def.record_rf_noise, rec_def.record_dropouts, rec_def.tape_mode)

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

                sync_u8, vjit_u8 = self.recorder._control_track_values(rec_def)
                dt_field = 1.0/60.0

                for field_i, (yy, cc, meta) in enumerate([(y0, c0, meta0), (y1, c1, meta1)]):
                    idx = base + field_i
                    meta.update({
                        "dt": dt_field,
                        "seg_id": int(self._live_seg_id),
                        "ctl_sync_u8": int(sync_u8),
                        "ctl_vjit_u8": int(vjit_u8),
                        "tape_mode": str(rec_def.tape_mode),
                        "field": int(field_i),
                        "real_rf_modulation": bool(getattr(rec_def, 'real_rf_modulation', False)),
                        "rf_fm_depth": float(getattr(rec_def, 'rf_fm_depth', 1.0)),
                        "rf_chroma_fc_frac": float(getattr(rec_def, 'rf_chroma_fc_frac', 0.12)),
                        "rf_chroma_lpf": float(getattr(rec_def, 'rf_chroma_lpf', 0.35)),
                    })
                    tape.cart.set(idx, TapeTrack(y_dphi8=yy, c_u8=cc, meta=meta))


                try:
                    self._live_last_good = {"y0": y0, "c0": c0, "meta0": meta0, "y1": y1, "c1": c1, "meta1": meta1}
                except Exception:
                    pass

                out = self.live_player.get_frame(tape, pb_def)
                self._latest_live_frame = self._apply_crt_to_frame(out, "live")

                # Pace slightly (avoid hogging CPU)
                now = time.perf_counter()
                _ = now - t_last
                t_last = now
                time.sleep(0.001)

            except Exception:
                try:
                    self._live_worker_error = traceback.format_exc()
                    print(self._live_worker_error)
                except Exception:
                    pass
                time.sleep(0.02)

        try:
            if self._live_cap is not None:
                self._live_cap.release()
        except Exception:
            pass

    def _live_ui_loop(self):
        try:
            fr = getattr(self, "_latest_live_frame", None)
            if fr is not None and hasattr(self, "live_canvas"):
                self._show_image(self.live_canvas, fr)
                # Mirror to overlay window if open
                ol = getattr(self, '_live_overlay_label', None)
                if ol is not None:
                    try:
                        self._show_image(ol, fr)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            self.root.after(33, self._live_ui_loop)
        except Exception:
            pass

    def _build_player_tab(self):
        left_sf = VScrollFrame(self.tab_play, width=390)
        right = ctk.CTkFrame(self.tab_play, fg_color=CARD_BG, corner_radius=16)
        left_sf.pack(side="left", fill="y", padx=(0, 14), pady=0)
        right.pack(side="right", fill="both", expand=True, padx=0, pady=0)
        left = left_sf.inner

        self._section_title(left, "Player")
        self._button(left, "Load tape bundle folder...", self._load_bundle_async).pack(anchor="w", fill="x", pady=4)
        self._button(left, "Use memory (no disk)", self._use_memory).pack(anchor="w", fill="x", pady=4)

        self._section_title(left, "Transport")

        row_transport_1 = ctk.CTkFrame(left, fg_color="transparent"); row_transport_1.pack(fill="x", pady=2)
        self._button(row_transport_1, "Insert", self._player_insert, width=82).pack(side="left", padx=(0, 6))
        self._button(row_transport_1, "Eject", self._player_eject, width=82).pack(side="left")

        row_transport_2 = ctk.CTkFrame(left, fg_color="transparent"); row_transport_2.pack(fill="x", pady=2)
        self._button(row_transport_2, "Play", self._player_play, accent=True, width=78).pack(side="left", padx=(0, 6))
        self._button(row_transport_2, "Stop", self._player_stop, width=78).pack(side="left", padx=(0, 6))
        self._button(row_transport_2, "FF", self._player_ff, width=58).pack(side="left", padx=(0, 6))
        self._button(row_transport_2, "REW", self._player_rew, width=62).pack(side="left")

        self._section_title(left, "Playback / VCR Effects")

        self.pb_aspect_play = tk.StringVar(value=self.pb_def.aspect_display)
        self._setting_combo(left, "Display aspect", self.pb_aspect_play, ["4:3","16:9"], "aspect_display", width=110)

        # Tracking controls moved here
        # Playback signal / VCR effects
        self._slider(left, "Tracking knob", "tracking_knob", 0.0, 1.0, key="pb")
        self._slider(left, "Tracking sensitivity", "tracking_sensitivity", 0.0, 1.0, key="pb")
        self._slider(left, "Tracking artifacts", "tracking_artifacts", 0.0, 2.0, key="pb")
        self._slider(left, "Auto tracking (0=off, 1=on)", "auto_tracking", 0.0, 1.0, key="pb")
        self._slider(left, "Auto tracking strength", "auto_tracking_strength", 0.0, 1.0, key="pb")
        self._slider(left, "Servo recovery", "servo_recovery", 0.0, 1.0, key="pb")
        self._slider(left, "Sync bias", "sync_bias", 0.0, 1.0, key="pb")

        self._section_title(left, "Servo / Head Switching")
        self._slider(left, "Servo hunt (amount)", "servo_hunt", 0.0, 1.0, key="pb")
        self._slider(left, "Servo hunt frequency", "servo_hunt_freq", 0.0, 1.0, key="pb")
        self._slider(left, "Head switch (amount)", "head_switch_strength", 0.0, 1.0, key="pb")
        self._slider(left, "Head switch frequency", "head_switch_freq", 0.0, 1.0, key="pb")

        self._slider(left, "Timebase (amount)", "playback_timebase", 0.0, 1.0, key="pb")
        self._slider(left, "Timebase frequency", "timebase_freq", 0.0, 1.0, key="pb")
        self._slider(left, "Playback RF noise", "playback_rf_noise", 0.0, 0.25, key="pb")
        self._slider(left, "Dropouts (amount)", "playback_dropouts", 0.0, 0.12, key="pb")
        self._slider(left, "Dropouts frequency", "dropout_freq", 0.0, 1.0, key="pb")
        self._slider(left, "Interference (amount)", "interference", 0.0, 1.0, key="pb")
        self._slider(left, "Interference frequency", "interference_freq", 0.0, 1.0, key="pb")
        self._slider(left, "Snow (amount)", "snow", 0.0, 1.0, key="pb")
        self._slider(left, "Snow frequency", "snow_freq", 0.0, 1.0, key="pb")
        self._slider(left, "Variance / instability", "variance", 0.0, 1.0, key="pb")

        self._section_title(left, "Chroma")
        self._slider(left, "Chroma delay (horizontal)", "chroma_shift_x", 0.0, 1.0, key="pb")
        self._slider(left, "Chroma delay (vertical)", "chroma_shift_y", 0.0, 1.0, key="pb")
        self._slider(left, "Chroma phase error", "chroma_phase", 0.0, 1.0, key="pb")
        self._slider(left, "Chroma phase noise", "chroma_noise", 0.0, 1.0, key="pb")
        self._slider(left, "Chroma noise frequency", "chroma_noise_freq", 0.0, 1.0, key="pb")
        self._slider(left, "Chroma wobble", "chroma_wobble", 0.0, 1.0, key="pb")
        self._slider(left, "Chroma wobble frequency", "chroma_wobble_freq", 0.0, 1.0, key="pb")

        self._section_title(left, "Scanlines")
        self._slider(left, "Scanlines", "scanline_strength", 0.0, 1.0, key="pb")
        self._slider(left, "Scanline soften", "scanline_soften", 0.0, 1.0, key="pb")

        self._section_title(left, "Image Controls")
        self._slider(left, "Brightness", "brightness", -1.0, 1.0, key="pb")
        self._slider(left, "Contrast", "contrast", 0.0, 1.0, key="pb")
        self._slider(left, "Saturation", "saturation", 0.0, 1.0, key="pb")
        self._slider(left, "Bloom", "bloom", 0.0, 1.0, key="pb")
        self._slider(left, "Sharpen", "sharpen", 0.0, 1.0, key="pb")
        self._slider(left, "Playback blur", "playback_blur", 0.0, 1.0, key="pb")
        self._slider(left, "Blur frequency", "playback_blur_freq", 0.0, 1.0, key="pb")
        self._slider(left, "Frame jitter", "frame_jitter", 0.0, 1.0, key="pb")
        self._slider(left, "Jitter frequency", "frame_jitter_freq", 0.0, 1.0, key="pb")

        self.comp_play_var = tk.BooleanVar(value=self.pb_def.composite_view)
        self._setting_switch(left, "Composite-ish view", self.comp_play_var, "composite_view")

        self._section_title(left, "Audio (Player)")
        # Create audio playback vars here too (in case user never opens Editor)
        self._slider(left, "Audio hiss", "hiss", 0.0, 1.0, key="ap")
        self._slider(left, "Audio pops", "pops", 0.0, 1.0, key="ap")

        self.play_audio_var = tk.BooleanVar(value=True)
        self._setting_switch(left, "Play audio (Windows built-in)", self.play_audio_var, "play_audio")
        self._button(left, "Preview tape audio", self._play_audio_preview).pack(anchor="w", fill="x", pady=(4,2))

        self._section_title(left, "Prefetch / Proxy")
        self.proxy_use_var = tk.BooleanVar(value=False)
        self._setting_switch(left, "Use proxy for playback (if built)", self.proxy_use_var, "proxy_use")
        self.proxy_seconds_var = tk.DoubleVar(value=30.0)
        self._setting_header(left, "Proxy seconds", "proxy_seconds")
        ctk.CTkSlider(left, from_=5.0, to=600.0, variable=self.proxy_seconds_var, progress_color=ACCENT, button_color=ACCENT_ACTIVE).pack(anchor="w", fill="x")
        self._button(left, "Build proxy in RAM...", self._build_proxy).pack(anchor="w", fill="x", pady=4)
        self._button(left, "Export final MP4...", self._export_from_player, accent=True).pack(anchor="w", fill="x", pady=4)

        self.play_status = tk.StringVar(value="Load tape, then Insert → Play.")
        ctk.CTkLabel(left, textvariable=self.play_status, wraplength=360, text_color=MUTED, justify="left").pack(anchor="w", pady=(10,0))

        self.play_canvas = tk.Label(right, text="Player output (video + audio)", background="#05070a", foreground=MUTED, bd=1, relief="solid", highlightbackground=BORDER)
        self.play_canvas.pack(fill="both", expand=True)



    # -------- Presets (settings save/load) --------
    def _preset_default_dir(self) -> Path:
        d = Path.cwd() / "presets"
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return d

    def _collect_current_settings_dict(self) -> dict:
        # Sync record-side options (downscale, audio, rf toggle, etc.)
        try:
            self._sync_rec_def()
        except Exception:
            pass
        # Build fresh defect objects from UI (safe snapshot)
        rec_def, pb_def, ap_def = self._sync_edit_defects()
        settings = settings_to_dict(rec_def, pb_def, self.ar_def, ap_def)
        rec_opts = {
            "downscale_width": int(getattr(self, 'down_w').get()) if hasattr(self, 'down_w') else int(getattr(self.rec_opts, 'downscale_width', 360)),
            "enforce_real_time": bool(getattr(self, 'rt_var').get()) if hasattr(self, 'rt_var') else bool(getattr(self.rec_opts, 'enforce_real_time', True)),
            "extract_audio": bool(getattr(self, 'audio_extract_var').get()) if hasattr(self, 'audio_extract_var') else bool(getattr(self.rec_opts, 'extract_audio', True)),
            "field_sampling": str(getattr(self, 'sampling_var').get()) if hasattr(self, 'sampling_var') else str(getattr(self.rec_opts, 'field_sampling', 'progressive')),
            "encode_threads": int(getattr(self, 'encode_threads_var').get()) if hasattr(self, 'encode_threads_var') else int(getattr(self.rec_opts, 'encode_threads', 0)),
            "use_src_timestamps": bool(getattr(self.rec_opts, 'use_src_timestamps', True)),
        }
        try:
            settings["crt"] = crt_settings_to_dict(self._sync_crt_settings())
        except Exception:
            settings["crt"] = crt_settings_to_dict(getattr(self, "crt_settings", consumer_tv_preset()))
        return {
            "version": "digital_vcr_preset_v1",
            "created": datetime.datetime.now().isoformat(timespec='seconds'),
            "settings": settings,
            "record_options": rec_opts,
        }

    def _apply_settings_dict_to_ui(self, data: dict) -> None:
        # Accept either full preset JSON or a raw settings dict.
        settings = data.get('settings', data)
        rec_def, pb_def, ar_def, ap_def = settings_from_dict(settings)

        def _set(name: str, value):
            try:
                var = getattr(self, name)
                var.set(value)
            except Exception:
                pass

        # Recorder tab
        _set('rec_mode', getattr(rec_def, 'tape_mode', 'SP'))
        for k in ['luma_bw','chroma_bw','record_blur','record_jitter','record_rf_noise','record_dropouts',
                  'rf_fm_depth','rf_am_depth','rf_phase_noise','rf_carrier_noise','rf_nonlinearity','rf_chroma_fc_frac','rf_chroma_lpf']:
            try:
                v = float(getattr(rec_def, k))
                _set(f'var_rec_{k}', v)
            except Exception:
                pass
        _set('real_rf_var', bool(getattr(rec_def, 'real_rf_modulation', False)))

        # Audio record
        for k in ['wow','hiss','dropouts','compression']:
            try:
                _set(f'var_ar_{k}', float(getattr(ar_def, k)))
            except Exception:
                pass

        # Player tab
        _set('pb_aspect_play', getattr(pb_def, 'aspect_display', '4:3'))
        _set('comp_play_var', bool(getattr(pb_def, 'composite_view', False)))
        for k, v in pb_def.__dict__.items():
            if k in ('aspect_display','composite_view'):
                continue
            # booleans
            if k == 'rf_playback_model':
                _set('rf_play_var', bool(v))
                continue
            # floats
            if isinstance(v, (int, float)):
                _set(f'var_pb_{k}', float(v))

        # Audio playback
        for k in ['hiss','pops']:
            try:
                _set(f'var_ap_{k}', float(getattr(ap_def, k)))
            except Exception:
                pass

        # Record options (if present)
        ro = data.get('record_options', {}) if isinstance(data, dict) else {}
        if isinstance(ro, dict):
            if 'downscale_width' in ro:
                _set('down_w', int(ro['downscale_width']))
            if 'enforce_real_time' in ro:
                _set('rt_var', bool(ro['enforce_real_time']))
            if 'extract_audio' in ro:
                _set('audio_extract_var', bool(ro['extract_audio']))
            if 'field_sampling' in ro:
                _set('sampling_var', str(ro['field_sampling']))
            if 'encode_threads' in ro:
                _set('encode_threads_var', int(ro['encode_threads']))

        # CRT settings are optional for older presets/bundles.
        try:
            crt_def = crt_settings_from_dict(settings)
            self._set_crt_vars_from_settings(crt_def)
        except Exception:
            pass

        # Refresh cached dataclasses used by worker threads
        try:
            self._cache_defects_mainthread()
        except Exception:
            pass

    def _save_preset(self):
        d = self._preset_default_dir()
        pth = filedialog.asksaveasfilename(
            title='Save preset',
            initialdir=str(d),
            defaultextension='.json',
            filetypes=[('Preset JSON','*.json')]
        )
        if not pth:
            return
        data = self._collect_current_settings_dict()
        try:
            Path(pth).write_text(json.dumps(data, indent=2), encoding='utf-8')
            messagebox.showinfo('Preset saved', f'Saved preset\n{pth}')

        
        except Exception as e:
            messagebox.showerror('Save failed', str(e))

    def _load_preset(self):
        d = self._preset_default_dir()
        pth = filedialog.askopenfilename(
            title='Load preset',
            initialdir=str(d),
            filetypes=[('Preset JSON','*.json'), ('All','*.*')]
        )
        if not pth:
            return
        try:
            raw = Path(pth).read_text(encoding='utf-8')
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError('Preset file is not a JSON object')
            self._apply_settings_dict_to_ui(data)
            messagebox.showinfo('Preset loaded', f'Loaded preset\n{pth}')
        except Exception as e:
            messagebox.showerror('Load failed', str(e))

    def _build_crt_tab(self):
        left_sf = VScrollFrame(self.tab_crt, width=410)
        right = ctk.CTkFrame(self.tab_crt, fg_color=CARD_BG, corner_radius=16)
        left_sf.pack(side="left", fill="y", padx=(0, 14), pady=0)
        right.pack(side="right", fill="both", expand=True, padx=0, pady=0)
        left = left_sf.inner

        s = self.crt_settings.validated()
        self._section_title(left, "CRT TV Simulator")
        self.crt_enabled_var = tk.BooleanVar(value=s.enabled)
        self.crt_preview_var = tk.BooleanVar(value=s.preview_enabled)
        self.crt_live_var = tk.BooleanVar(value=s.live_enabled)
        self.crt_export_var = tk.BooleanVar(value=s.export_enabled)
        self.crt_direct_player_var = tk.BooleanVar(value=s.direct_player)
        self.crt_direct_live_var = tk.BooleanVar(value=s.direct_live)

        self._setting_switch(left, "Enable CRT simulation", self.crt_enabled_var, "crt_enabled")
        self._setting_switch(left, "Apply inside Player preview", self.crt_preview_var, "crt_preview")
        self._setting_switch(left, "Apply inside Live preview / overlay", self.crt_live_var, "crt_live")
        self._setting_switch(left, "Bake CRT into MP4 exports", self.crt_export_var, "crt_export")
        self._setting_switch(left, "Direct OpenGL Player window", self.crt_direct_player_var, "crt_direct")
        self._setting_switch(left, "Direct OpenGL Live window", self.crt_direct_live_var, "crt_direct")

        self._section_title(left, "Presets / Resolution")
        rowp = ctk.CTkFrame(left, fg_color="transparent"); rowp.pack(anchor="w", fill="x", pady=(0, 6))
        self._button(rowp, "Consumer TV", lambda: self._apply_crt_preset("Consumer TV"), width=112).pack(side="left")
        self._button(rowp, "Pro Monitor", lambda: self._apply_crt_preset("Pro Monitor"), width=112).pack(side="left", padx=6)
        self._button(rowp, "Test GPU", self._test_crt_gpu, width=92).pack(side="left")

        self.crt_preset_var = tk.StringVar(value=s.preset)
        self._setting_combo(left, "CRT preset", self.crt_preset_var, ["Consumer TV", "Pro Monitor"], "crt_preset", width=160)
        self.crt_quality_var = tk.StringVar(value=s.quality)
        self._setting_combo(left, "Quality preset", self.crt_quality_var, list(CRT_QUALITIES), "crt_quality", width=140)
        self.crt_mask_type_var = tk.StringVar(value=s.mask_type)
        self._setting_combo(left, "Phosphor mask", self.crt_mask_type_var, list(CRT_MASK_TYPES), "crt_mask", width=140)

        self.crt_render_width_var = tk.IntVar(value=int(s.render_width))
        row = ctk.CTkFrame(left, fg_color="transparent")
        row.pack(anchor="w", fill="x", pady=(7, 1))
        ctk.CTkLabel(row, text="Simulated phosphor width", text_color=TEXT, font=("Segoe UI", 10, "bold")).pack(side="left")
        self._help_button(row, "crt_render_width", "Simulated phosphor width").pack(side="left", padx=(7, 0))
        ctk.CTkLabel(row, textvariable=self.crt_render_width_var, text_color=MUTED).pack(side="right")
        ctk.CTkSlider(left, from_=320, to=4096, variable=self.crt_render_width_var, progress_color=ACCENT, button_color=ACCENT_ACTIVE).pack(anchor="w", fill="x", pady=(0, 8))

        self._section_title(left, "Phosphors / Beam")
        self._crt_slider(left, "Mask strength", "mask_strength", 0.0, 1.0)
        self._crt_slider(left, "Scanline strength", "scanline_strength", 0.0, 1.0)
        self._crt_slider(left, "Beam sharpness", "beam_sharpness", 0.0, 1.0)
        self._crt_slider(left, "Phosphor decay", "phosphor_decay", 0.0, 1.0)
        self._crt_slider(left, "Convergence X", "convergence_x", -4.0, 4.0)
        self._crt_slider(left, "Convergence Y", "convergence_y", -4.0, 4.0)

        self._section_title(left, "Glass / Tube")
        self._crt_slider(left, "Bloom", "bloom", 0.0, 1.0)
        self._crt_slider(left, "Halation", "halation", 0.0, 1.0)
        self._crt_slider(left, "Glass diffusion", "glass_diffusion", 0.0, 1.0)
        self._crt_slider(left, "Curvature", "curvature", 0.0, 1.0)
        self._crt_slider(left, "Overscan", "overscan", 0.0, 0.18)
        self._crt_slider(left, "Vignette", "vignette", 0.0, 1.0)
        self._crt_slider(left, "Edge focus loss", "edge_focus", 0.0, 1.0)

        self._section_title(left, "CRT Image Trim")
        self._crt_slider(left, "Brightness", "brightness", -1.0, 1.0)
        self._crt_slider(left, "Contrast", "contrast", -0.5, 1.0)
        self._crt_slider(left, "Saturation", "saturation", -1.0, 1.0)

        self.crt_status = tk.StringVar(value="GPU CRT idle. Requires ModernGL + GLFW with OpenGL 3.3+.")
        ctk.CTkLabel(left, textvariable=self.crt_status, wraplength=360, text_color=MUTED, justify="left").pack(anchor="w", pady=(12, 0))

        msg = (
            "The CRT pass runs after the VCR playback image. It can be displayed in the app, mirrored to a "
            "direct OpenGL window, and baked into MP4 exports without rewriting the tape bundle."
        )
        ctk.CTkLabel(right, text="CRT TV", font=("Segoe UI", 18, "bold"), text_color=TEXT).pack(anchor="nw", padx=22, pady=(22, 8))
        ctk.CTkLabel(right, text=msg, justify="left", wraplength=520, text_color=MUTED).pack(anchor="nw", padx=22)

    def _build_vhs_tab(self):
        """Advanced RF / tape modelling controls.

        This tab is intentionally separated so Recorder/Player tabs stay usable.
        """
        left_sf = VScrollFrame(self.tab_vhs, width=390)
        right = ctk.CTkFrame(self.tab_vhs, fg_color=CARD_BG, corner_radius=16)
        left_sf.pack(side="left", fill="y", padx=(0, 14), pady=0)
        right.pack(side="right", fill="both", expand=True, padx=0, pady=0)
        left = left_sf.inner

        self._section_title(left, "RF / Tape Model")
        ctk.CTkLabel(left, text="Advanced FM/AM carrier controls before decode.", text_color=MUTED, wraplength=340, justify="left").pack(anchor="w", pady=(0, 8))

        rowp = ctk.CTkFrame(left, fg_color="transparent"); rowp.pack(anchor="w", fill="x", pady=(0,6))
        self._button(rowp, "Save preset...", self._save_preset, width=112).pack(side="left")
        self._button(rowp, "Load preset...", self._load_preset, width=112).pack(side="left", padx=6)

        # Recorder-side toggle is also on the Recorder tab; reuse the same tk var.
        if not hasattr(self, 'real_rf_var'):
            self.real_rf_var = tk.BooleanVar(value=bool(getattr(self.rec_def, 'real_rf_modulation', False)))
        self._setting_switch(left, "Enable Real RF modulation", self.real_rf_var, "real_rf_modulation")

        self._section_title(left, "Record-side RF Parameters")
        self._slider(left, "RF FM depth (luma deviation)", "rf_fm_depth", 0.50, 1.50, key="rec")
        self._slider(left, "RF AM depth", "rf_am_depth", 0.0, 1.0, key="rec")
        self._slider(left, "RF phase noise", "rf_phase_noise", 0.0, 1.0, key="rec")
        self._slider(left, "RF carrier noise", "rf_carrier_noise", 0.0, 1.0, key="rec")
        self._slider(left, "RF nonlinearity", "rf_nonlinearity", 0.0, 1.0, key="rec")
        self._slider(left, "Chroma carrier (fc fraction)", "rf_chroma_fc_frac", 0.02, 0.49, key="rec")
        self._slider(left, "Chroma demod low-pass", "rf_chroma_lpf", 0.0, 1.0, key="rec")

        self._section_title(left, "Playback-side RF / Recombination")

        if not hasattr(self, 'rf_play_var'):
            self.rf_play_var = tk.BooleanVar(value=bool(getattr(self.pb_def, 'rf_playback_model', False)))
        self._setting_switch(left, "Enable RF playback model", self.rf_play_var, "rf_playback_model")

        self._slider(left, "Luma/Chroma bleed", "luma_chroma_bleed", 0.0, 1.0, key="pb")
        self._slider(left, "RF FM depth (playback)", "rf_playback_fm_depth", 0.50, 1.50, key="pb")
        self._slider(left, "RF AM depth (playback)", "rf_playback_am_depth", 0.0, 1.0, key="pb")
        self._slider(left, "RF phase noise (playback)", "rf_playback_phase_noise", 0.0, 1.0, key="pb")
        self._slider(left, "RF carrier noise (playback)", "rf_playback_carrier_noise", 0.0, 1.0, key="pb")
        self._slider(left, "RF nonlinearity (playback)", "rf_playback_nonlinearity", 0.0, 1.0, key="pb")

        msg = (
            "Tip: If you want 'clean' separation, keep Luma/Chroma bleed at 0.\n"
            "Then raise it slowly to introduce controlled cross-talk."
        )
        ctk.CTkLabel(right, text="VHS Tape", font=("Segoe UI", 18, "bold"), text_color=TEXT).pack(anchor="nw", padx=22, pady=(22, 8))
        ctk.CTkLabel(right, text=msg, justify="left", wraplength=420, text_color=MUTED).pack(anchor="nw", padx=22)

    def _player_insert(self):
        with self.lock:
            tape = self._active_tape()
        if tape.cart.length_tracks < 2:
            self._q("status_play", "No tape available.")
            return
        with self.lock:
            self.player.insert()
        self._q("status_play", "Inserted. Press Play.")

    def _player_eject(self):
        with self.lock:
            self.player.eject()
        self._q("status_play", "Ejected.")
        try:
            self.audio_player.stop()
        except Exception:
            pass

    def _player_play(self):
        with self.lock:
            if not self.player.state.inserted:
                self.player.insert()
            self.player.play()
            tape = self._active_tape()
        status = "Play"

        # Audio playback (optional)
        try:
            if bool(self.play_audio_var.get()):
                audio_msg = self._start_player_audio_stream(tape)
                if audio_msg:
                    status = f"{status} ({audio_msg})"
        except Exception as e:
            status = f"{status} (audio failed: {e})"
        self._q("status_play", status)

    def _start_player_audio_stream(self, tape: TapeImage) -> str | None:
        if not self.audio_player.available:
            return "audio backend unavailable"
        if tape.audio.pcm16 is None or tape.audio.pcm16.size == 0:
            return "no audio on tape"
        _pb, ap_def = self._current_playback_defects()
        self.audio_player.start_stream(
            tape,
            get_pos_sec=lambda: float(self.player.state.pos_tracks) / 60.0,
            get_lock=lambda: (float(getattr(self.player.state,'lock',0.0)) * (1.0 - 0.85*min(1.0, float(getattr(self.player.state,'switch_confuse_timer',0.0))/1.35))),
            ap_def=ap_def,
            chunk_sec=0.18,
        )
        if self.audio_player.last_error:
            return self.audio_player.last_error
        return "audio on"

    def _play_audio_preview(self):
        with self.lock:
            tape = self._active_tape()
            pos_tracks = float(self.player.state.pos_tracks)
        if tape.audio.pcm16 is None or tape.audio.pcm16.size == 0:
            self._q("status_play", "No audio is stored on this tape.")
            return
        start_sec = max(0.0, pos_tracks / 60.0)
        self.audio_player.play_from_seconds(tape.audio.pcm16, int(tape.audio.sample_rate or 44100), start_sec)
        if self.audio_player.last_error:
            self._q("status_play", f"Audio preview failed: {self.audio_player.last_error}")
        else:
            self._q("status_play", f"Playing tape audio from {start_sec:.2f}s inside the UI.")

    def _player_stop(self):
        with self.lock:
            self.player.stop()
        self._q("status_play", "Stop")
        try:
            self.audio_player.stop()
        except Exception:
            pass

    def _player_ff(self):
        with self.lock:
            self.player.ff(10.0)
        self._q("status_play", "Fast forward")
        try:
            self.audio_player.stop()
        except Exception:
            pass

    def _player_rew(self):
        with self.lock:
            self.player.rew(10.0)
        self._q("status_play", "Rewind")
        try:
            self.audio_player.stop()
        except Exception:
            pass




    def _current_playback_defects(self):
        """Return playback (pb) + audio playback (ap) defects based on Player tab controls."""
        try:
            _r, pb_def, ap_def = self._sync_edit_defects()
        except Exception:
            pb_def = self.pb_def
            ap_def = self.ap_def

        try:
            pb_def.aspect_display = self.pb_aspect_play.get()
        except Exception:
            pass
        try:
            pb_def.composite_view = bool(self.comp_play_var.get())
        except Exception:
            pass
        return pb_def, ap_def
    def _build_proxy(self):
        with self.lock:
            tape = self._active_tape()
            pos_tracks = float(self.player.state.pos_tracks)
        if tape.cart.length_tracks < 2:
            messagebox.showerror("No tape", "Load or record a tape first.")
            return

        secs = float(self.proxy_seconds_var.get()) if hasattr(self, "proxy_seconds_var") else 30.0
        secs = max(5.0, min(3600.0, secs))
        fps = 30.0
        frames_target = int(secs * fps)

        pb_def, _ap = self._current_playback_defects()
        start_tracks = int(max(0, (pos_tracks // 2) * 2))
        step_tracks = 2  # 2 fields per frame

        def worker():
            self._q("status_play", f"Building proxy… 0/{frames_target}")
            frames = []
            proxy_player = VCRPlayer()
            proxy_player.insert()
            proxy_player.play()
            proxy_player.state.inserting_timer = 1.2
            proxy_player.state.lock = 1.0
            for i in range(frames_target):
                base = start_tracks + i * step_tracks
                if base >= tape.cart.length_tracks - 2:
                    break
                proxy_player.state.pos_tracks = float(base)
                proxy_player.state._last_pos_tracks = float(base)
                proxy_player.update(tape, pb_def)
                fr = proxy_player.get_frame(tape, pb_def)
                if fr is None:
                    continue
                # downscale proxy
                if fr.shape[1] > 640:
                    nw = 640
                    nh = int(fr.shape[0] * (nw / fr.shape[1]))
                    fr = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_AREA)
                # encode JPEG to save memory
                ok, buf = cv2.imencode(".jpg", fr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    frames.append(buf.tobytes())
                if i % 10 == 0:
                    self._q("status_play", f"Building proxy… {i}/{frames_target}")
            with self.lock:
                self._proxy = {"start_tracks": start_tracks, "fps": fps, "frames": frames}
            self._q("status_play", f"Proxy ready ({len(frames)} frames). Enable 'Use proxy for playback'.")

        threading.Thread(target=worker, daemon=True).start()


    def _export_from_player(self):
        with self.lock:
            tape = self._active_tape()
        if tape.cart.length_tracks < 2:
            messagebox.showerror("No tape", "Load or record a tape first.")
            return

        out = filedialog.asksaveasfilename(
            title="Export final MP4",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")]
        )
        if not out:
            return

        pb_def, ap_def = self._current_playback_defects()
        crt_def = self._sync_crt_settings()

        def worker():
            self._q("status_play", "Exporting MP4…")
            try:
                ok = export_playback_video_mp4(
                    tape, pb_def,
                    ExportOptions(out_mp4=out, fps=30.0, upscale_width=960),
                    progress_cb=lambda a,b: self._q("status_play", f"Exporting… {a}/{b}"),
                    crt=crt_def,
                    crt_renderer=self.crt_renderer,
                )
            except CRTGPUUnavailable as exc:
                self._q("status_play", f"CRT export failed: {exc}")
                self._q("status_crt", f"GPU CRT unavailable: {exc}")
                return
            if not ok:
                self._q("status_play", "Export failed (VideoWriter).")
                return

            if tape.audio.pcm16 is not None and tape.audio.pcm16.size > 0:
                wav = str(Path(out).with_suffix(".audio_playback.wav"))
                out_mux = str(Path(out).with_name(Path(out).stem + "_with_audio.mp4"))
                ok2, err = export_audio_and_optional_mux(tape, wav, out, out_mux, ap=ap_def)
                if ok2:
                    self._q("status_play", f"Done. Exported: {out_mux}")
                else:
                    self._q("status_play", f"Done. Video exported. Audio mux failed: {err}")
            else:
                self._q("status_play", f"Done. Exported: {out} (no audio in tape)")

        threading.Thread(target=worker, daemon=True).start()

    def _play_worker_loop(self):
        while not self._play_worker_stop.is_set():
            try:
                with self.lock:
                    tape = self._active_tape()
                    pb_def = getattr(self, '_cached_pb_def', self.pb_def)
                    self.player.update(tape, pb_def)
                    if self.player.state.inserted:
                        frame = None
                        # Use proxy if enabled and available
                        try:
                            use_proxy = bool(getattr(self, '_cached_proxy_use', False))
                        except Exception:
                            use_proxy = False
                        if use_proxy and self._proxy is not None:
                            pr = self._proxy
                            rel = (float(self.player.state.pos_tracks) - float(pr['start_tracks'])) / 2.0
                            fi = int(rel)
                            if 0 <= fi < len(pr['frames']):
                                b = np.frombuffer(pr['frames'][fi], dtype=np.uint8)
                                frame = cv2.imdecode(b, cv2.IMREAD_COLOR)
                        if frame is None:
                            frame = self.player.get_frame(tape, pb_def)
                    else:
                        frame = None
                frame = self._apply_crt_to_frame(frame, "player")
                self._latest_play_frame = frame
            except Exception:
                self._latest_play_frame = None
                try:
                    self._play_worker_error = traceback.format_exc()
                    print(self._play_worker_error)
                except Exception:
                    pass
            time.sleep(1/30)

    def _player_ui_loop(self):
        try:
            fr = self._latest_play_frame
            if fr is not None:
                self._show_image(self.play_canvas, fr)
            else:
                # If inserted but frame decode failed, at least show black (prevents "blank canvas" confusion)
                try:
                    inserted = bool(self.player.state.inserted)
                except Exception:
                    inserted = False
                if inserted:
                    if not hasattr(self, "_black_frame"):
                        self._black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    self._show_image(self.play_canvas, self._black_frame)
        except Exception:
            pass
        self.root.after(33, self._player_ui_loop)
    def _current_audio_record_defects(self):
        def _g(name, default):
            try:
                return float(getattr(self, name).get())
            except Exception:
                return float(default)
        return AudioRecordDefects(
            wow=_g("var_ar_wow", getattr(self.ar_def, "wow", 0.0)),
            hiss=_g("var_ar_hiss", getattr(self.ar_def, "hiss", 0.0)),
            dropouts=_g("var_ar_dropouts", getattr(self.ar_def, "dropouts", 0.0)),
            compression=_g("var_ar_compression", getattr(self.ar_def, "compression", 0.55)),

        )
