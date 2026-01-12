from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import cv2
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
import time
import traceback
import datetime

from ..tape import TapeImage, TapeCartridge, TapeTrack
from ..bundle import save_bundle, load_bundle, create_blank_bundle
from ..recorder import Recorder, RecordOptions
from ..modulation import encode_field_bgr
from ..defects import apply_record_defects_to_field, apply_rf_defects_y_dphi_u8, apply_rf_defects_chroma_u8
from ..editor import Editor, DubOptions
from ..player import VCRPlayer
from ..audio_player import AudioPlayer
from ..exporter import export_playback_video_mp4, ExportOptions, export_audio_and_optional_mux
from ..defects import (
    RecordDefects, PlaybackDefects, AudioRecordDefects, AudioPlaybackDefects,
    settings_to_dict, settings_from_dict
)
from ..modulation import decode_field_bgr

class VScrollFrame(ttk.Frame):
    def __init__(self, parent, width=340, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, width=width)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.inner = ttk.Frame(self.canvas)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Scroll wheel support
        self.canvas.bind("<Enter>", lambda _e: self.canvas.focus_set())
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind("<Button-5>", self._on_mousewheel_linux)

    def _on_inner_configure(self, _evt=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, _evt=None):
        self.canvas.itemconfigure(self.inner_id, width=self.canvas.winfo_width())

    def _on_mousewheel(self, evt):
        delta = -1 * int(evt.delta / 120) if evt.delta else 0
        if delta:
            self.canvas.yview_scroll(delta, "units")

    def _on_mousewheel_linux(self, evt):
        if evt.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif evt.num == 5:
            self.canvas.yview_scroll(1, "units")

class DigitalVCRApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Digital VCR (v6) — tape bundles + nonfreezing UI")
        self.root.geometry("1120x690")
        self.root.minsize(980, 560)

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
        # Optional RAM proxy (JPEG frames) for smooth playback
        self._proxy = None

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
                elif kind == "preview_rec":
                    last_rec = payload
                elif kind == "preview_edit":
                    last_edit = payload
        except queue.Empty:
            pass

        if last_rec is not None:
            self._show_image(self.rec_canvas, last_rec)
        if last_edit is not None and hasattr(self, "edit_canvas"):
            self._show_image(self.edit_canvas, last_edit)

        # If queue is still large, poll faster to catch up, otherwise normal pace.
        delay = 15 if self.uiq.qsize() > 80 else 50
        self.root.after(delay, self._poll_uiq)



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

    def _active_tape(self):
        if self.tape_loaded is not None:
            return self.tape_loaded
        if self.tape_edited is not None:
            return self.tape_edited
        return self.tape_live

    # ---------- UI ----------
    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        self.nb = nb
        nb.pack(fill="both", expand=True)
        self.tab_rec = ttk.Frame(nb)
        self.tab_play = ttk.Frame(nb)
        self.tab_live = ttk.Frame(nb)
        nb.add(self.tab_rec, text="Recorder")
        nb.add(self.tab_play, text="Player")
        nb.add(self.tab_live, text="Live")
        self._build_recorder_tab()
        self._build_player_tab()
        self._build_live_tab()

    def _slider(self, parent, label, name, frm, to, key: str):
        ttk.Label(parent, text=label).pack(anchor="w")
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
        row = ttk.Frame(parent)
        row.pack(anchor="w", fill="x")
        # Move the label into the row (left) and add a live value readout (right)
        # Note: we already created a label above; keep it for layout stability on older saved UIs.
        val_lbl = ttk.Label(row, text="")
        val_lbl.pack(side="right")
        scale = ttk.Scale(parent, from_=frm, to=to, variable=var, orient="horizontal")
        scale.pack(anchor="w", fill="x", pady=(0,6))

        def _fmt_value():
            v = float(var.get())
            rng = float(to - frm) if float(to - frm) != 0.0 else 1.0
            pct = (v - float(frm)) / rng
            pct = max(0.0, min(1.0, pct))
            val_lbl.config(text=f"{v:.3f}  ({pct*100:.0f}%)")

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
                self._update_scrub_range()
                self.recpos_var.set("0")
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
        right = ttk.Frame(self.tab_rec)
        left_sf.pack(side="left", fill="y", padx=10, pady=10)
        right.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        left = left_sf.inner

        ttk.Button(left, text="Load tape bundle folder…", command=self._load_bundle_async).pack(anchor="w", pady=(0,6))
        ttk.Button(left, text="Go to Player tab (final playback + export)", command=lambda: self.nb.select(self.tab_play)).pack(anchor="w", pady=(0,6))
        ttk.Button(left, text="Use memory (no disk)", command=self._use_memory).pack(anchor="w", pady=(0,6))

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(left, text="Tape base folder").pack(anchor="w")
        self.base_dir_var = tk.StringVar(value=str(Path.cwd() / "tapes"))
        rowb = ttk.Frame(left); rowb.pack(fill="x", pady=4)
        ttk.Entry(rowb, textvariable=self.base_dir_var, width=36).pack(side="left", padx=(0,6))
        ttk.Button(rowb, text="Browse…", command=self._browse_base_dir).pack(side="left")

        ttk.Label(left, text="Blank tape").pack(anchor="w")
        self.tape_minutes = tk.IntVar(value=5)
        row = ttk.Frame(left); row.pack(fill="x", pady=4)
        ttk.Label(row, text="Length (min)").pack(side="left")
        ttk.Entry(row, textvariable=self.tape_minutes, width=6).pack(side="left", padx=6)
        ttk.Button(row, text="New tape (create in base folder)", command=self._new_bundle_blank_tape).pack(side="left")

        ttk.Label(left, text="Record position").pack(anchor="w", pady=(8,0))
        self.recpos_var = tk.StringVar(value="0")
        row2 = ttk.Frame(left); row2.pack(fill="x", pady=4)
        ttk.Label(row2, text="Track idx").pack(side="left")
        ttk.Entry(row2, textvariable=self.recpos_var, width=10).pack(side="left", padx=6)
        ttk.Button(row2, text="Set", command=self._set_record_pos).pack(side="left")
        ttk.Button(row2, text="Rewind", command=lambda: self._set_record_pos_value(0)).pack(side="left", padx=4)

        ttk.Label(left, text="Scrub tape to find location (nonfreezing)").pack(anchor="w", pady=(10,0))
        self.scrub_var = tk.IntVar(value=0)
        self.scrub_scale = ttk.Scale(left, from_=0, to=1000, variable=self.scrub_var, orient="horizontal", command=self._on_scrub)
        self.scrub_scale.pack(anchor="w", fill="x", pady=(0,4))
        self.scrub_lbl = tk.StringVar(value="0 / 0 tracks")
        ttk.Label(left, textvariable=self.scrub_lbl).pack(anchor="w")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        self.src_var = tk.StringVar(value="")
        ttk.Label(left, text="Source file (record/overwrite from record position)").pack(anchor="w")
        ttk.Entry(left, textvariable=self.src_var, width=52).pack(anchor="w", pady=4)
        ttk.Button(left, text="Browse…", command=self._browse_source).pack(anchor="w")

        ttk.Label(left, text="Record monitor").pack(anchor="w", pady=(8,0))
        self.monitor_var = tk.StringVar(value="tape")
        ttk.Combobox(left, values=["tape","input"], textvariable=self.monitor_var, width=10, state="readonly").pack(anchor="w")

        ttk.Label(left, text="Downscale width").pack(anchor="w", pady=(8,0))
        self.down_w = tk.IntVar(value=self.rec_opts.downscale_width)
        ttk.Scale(left, from_=200, to=720, variable=self.down_w, orient="horizontal").pack(anchor="w", fill="x")

        ttk.Label(left, text="Tape mode (baked quality)").pack(anchor="w", pady=(8,0))
        self.rec_mode = tk.StringVar(value=self.rec_def.tape_mode)
        ttk.Combobox(left, values=["SP","LP","EP"], textvariable=self.rec_mode, width=8, state="readonly").pack(anchor="w")

        self._slider(left, "Luma bandwidth", "luma_bw", 0.35, 1.0, key="rec")
        self._slider(left, "Record blur", "record_blur", 0.0, 1.0, key="rec")
        self._slider(left, "Record jitter", "record_jitter", 0.0, 1.0, key="rec")
        self._slider(left, "Record RF noise", "record_rf_noise", 0.0, 0.15, key="rec")
        self._slider(left, "Record dropouts", "record_dropouts", 0.0, 0.10, key="rec")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(left, text="Audio (baked if ffmpeg available)").pack(anchor="w")
        self._slider(left, "Audio wow/flutter", "wow", 0.0, 1.0, key="ar")
        self._slider(left, "Audio hiss", "hiss", 0.0, 1.0, key="ar")
        self._slider(left, "Audio dropouts", "dropouts", 0.0, 0.25, key="ar")
        self._slider(left, "Audio compression", "compression", 0.0, 1.0, key="ar")

        self.rt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Enforce real-time record", variable=self.rt_var).pack(anchor="w", pady=6)

        self.audio_extract_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Extract audio (ffmpeg)", variable=self.audio_extract_var).pack(anchor="w", pady=2)

        self.autosave_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Auto-save bundle after record (if using bundle)", variable=self.autosave_var).pack(anchor="w", pady=2)

        ttk.Button(left, text="Record (overwrite)", command=self._record).pack(anchor="w", pady=(10,4))

        self.rec_status = tk.StringVar(value="Idle.")
        ttk.Label(left, textvariable=self.rec_status, wraplength=390).pack(anchor="w", pady=(10,0))

        self.rec_canvas = tk.Label(right, text="Live record monitor / scrub preview", background="#111", foreground="#ddd")
        self.rec_canvas.pack(fill="both", expand=True)

        self._update_scrub_range()

    def _update_scrub_range(self):
        with self.lock:
            tape = self._active_tape()
            mx = max(2, int(tape.cart.length_tracks-2))
        try:
            self.scrub_scale.configure(to=mx)
            self.scrub_lbl.set(f"{int(self.scrub_var.get())} / {mx} tracks")
        except Exception:
            pass

    def _on_scrub(self, _val=None):
        v = int(self.scrub_var.get())
        self.scrub_lbl.set(f"{v} / {int(float(self.scrub_scale.cget('to')))} tracks")
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
        for k in ["luma_bw","record_blur","record_jitter","record_rf_noise","record_dropouts"]:
            setattr(self.rec_def, k, float(getattr(self, f"var_rec_{k}").get()))
        self.rec_opts.downscale_width = int(self.down_w.get())
        self.rec_opts.enforce_real_time = bool(self.rt_var.get())
        self.rec_opts.extract_audio = bool(self.audio_extract_var.get())
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
            self.recpos_var.set(str(endpos))
            self.scrub_var.set(endpos)

            msg = f"Done. Recorded up to track={endpos} | Stored tracks={tape.cart.recorded_count()}"
            if self.recorder.last_audio_error:
                msg += f" | Audio: {self.recorder.last_audio_error}"
            elif tape.audio.pcm16 is not None:
                msg += " | Audio: extracted"

            # Auto-save if we have a bundle folder
            if bundle and bool(self.autosave_var.get()):
                try:
                    settings = settings_to_dict(self.rec_def, self.pb_def, self.ar_def, self.ap_def)
                    save_bundle(bundle, tape, settings)
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

        self.edit_canvas = tk.Label(right, text="Editor preview", background="#111", foreground="#ddd")
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
            record_blur=_v("var_rec_record_blur", r0.record_blur),
            record_jitter=_v("var_rec_record_jitter", r0.record_jitter),
            record_rf_noise=_v("var_rec_record_rf_noise", r0.record_rf_noise),
            record_dropouts=_v("var_rec_record_dropouts", r0.record_dropouts),
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
                DubOptions(enforce_real_time=bool(self.dub_rt.get())),
                progress_cb=lambda a,b: self._q("status_edit", f"Dubbing… {a}/{b}"),
                preview_cb=lambda fr: self._q("preview_edit", fr)
            )
            with self.lock:
                self.tape_edited = out_tape
                self.tape_loaded = out_tape
                self.bundle_path = folder

            settings = settings_to_dict(rec_def, pb_def, self.ar_def, ap_def)
            save_bundle(folder, out_tape, settings)

            if not bool(self.export_video_var.get()):
                self._q("status_edit", f"Saved bundle (no video export): {folder}")
                return

            self._q("status_edit", "Exporting output.mp4…")
            out_mp4 = str(Path(folder) / "output.mp4")
            ok = export_playback_video_mp4(
                out_tape, pb_def,
                ExportOptions(out_mp4=out_mp4, fps=30.0, upscale_width=960),
                progress_cb=lambda a,b: self._q("status_edit", f"Exporting video… {a}/{b}")
            )
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
        right = ttk.Frame(self.tab_live)
        left_sf.pack(side="left", fill="y", padx=10, pady=10)
        right.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        left = left_sf.inner

        ttk.Label(left, text="Live mode (camera input → VHS pipeline → preview)").pack(anchor="w")
        ttk.Label(left, text="Live uses the same Recorder/Player sliders. You can tweak them while live is running.").pack(anchor="w", pady=(0,8))

        ttk.Label(left, text="Camera index").pack(anchor="w")
        self.live_cam_var = tk.IntVar(value=int(getattr(self, "_live_cam_index", 0)))
        cam_row = ttk.Frame(left); cam_row.pack(anchor="w", fill="x", pady=4)
        self.live_cam_combo = ttk.Combobox(cam_row, width=10, state="readonly")
        self.live_cam_combo.pack(side="left", padx=(0,6))
        ttk.Button(cam_row, text="Refresh", command=self._refresh_cameras).pack(side="left")
        self._refresh_cameras()

        self.live_bufsec_var = tk.DoubleVar(value=6.0)
        ttk.Label(left, text="Live tape buffer (seconds)").pack(anchor="w")
        ttk.Scale(left, from_=2.0, to=20.0, variable=self.live_bufsec_var, orient="horizontal").pack(anchor="w", fill="x", pady=(0,8))
        ttk.Label(left, text="Live tape speed / mode (quality)").pack(anchor="w")
        self.live_tape_mode_var = tk.StringVar(value="SP")
        ttk.Combobox(left, values=["SP","LP","EP"], textvariable=self.live_tape_mode_var, width=8, state="readonly").pack(anchor="w", pady=(0,8))


        self.live_toggle_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Live mode ON", variable=self.live_toggle_var, command=self._toggle_live).pack(anchor="w", pady=4)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        # Overlay output window (fullscreen)
        self.live_overlay_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Overlay fullscreen output", variable=self.live_overlay_var,
                        command=self._toggle_live_overlay).pack(anchor="w", pady=2)
        ttk.Label(left, text="Tip: Press ESC to close overlay.").pack(anchor="w", pady=(0,8))

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(left, text="Live controls (recording)").pack(anchor="w")
        # Live preview downscale (controls the record-side horizontal resolution)
        self.var_live_downscale_width = tk.IntVar(value=640)
        row = ttk.Frame(left)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Downscale width (px)").pack(side="left")
        ttk.Label(row, textvariable=self.var_live_downscale_width).pack(side="right")
        tk.Scale(left, from_=240, to=960, resolution=10, orient="horizontal", showvalue=False,
                 variable=self.var_live_downscale_width, length=320).pack(fill="x")
        ttk.Label(left, text="(Higher = sharper/less compressed live preview)").pack(anchor="w", pady=(0,4))

        # Video record-side sliders
        self._slider(left, "Luma bandwidth", "luma_bw", 0.35, 1.0, key="rec")
        self._slider(left, "Record blur", "record_blur", 0.0, 1.0, key="rec")
        self._slider(left, "Record jitter", "record_jitter", 0.0, 1.0, key="rec")
        self._slider(left, "Record RF noise", "record_rf_noise", 0.0, 0.15, key="rec")
        self._slider(left, "Record dropouts", "record_dropouts", 0.0, 0.10, key="rec")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(left, text="Live controls (audio record)").pack(anchor="w")

        # Audio record-side sliders
        self._slider(left, "Audio wow/flutter", "wow", 0.0, 1.0, key="ar")
        self._slider(left, "Audio hiss", "hiss", 0.0, 1.0, key="ar")
        self._slider(left, "Audio dropouts", "dropouts", 0.0, 0.25, key="ar")
        self._slider(left, "Audio compression", "compression", 0.0, 1.0, key="ar")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(left, text="Live controls (playback / tracking)").pack(anchor="w")

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

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(left, text="Live controls (audio playback)").pack(anchor="w")
        self._slider(left, "Audio hiss", "hiss", 0.0, 1.0, key="ap")
        self._slider(left, "Audio pops", "pops", 0.0, 1.0, key="ap")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(left, text="Quick access").pack(anchor="w")
        ttk.Button(left, text="Go to Recorder tab", command=lambda: self.nb.select(self.tab_rec)).pack(anchor="w", pady=2)
        ttk.Button(left, text="Go to Player tab", command=lambda: self.nb.select(self.tab_play)).pack(anchor="w", pady=2)

        self.live_status = tk.StringVar(value="Select a camera, then enable Live mode.")
        ttk.Label(left, textvariable=self.live_status, wraplength=360).pack(anchor="w", pady=(12,0))

        self.live_canvas = tk.Label(right, text="Live output", background="#111", foreground="#ddd")
        self.live_canvas.pack(fill="both", expand=True)

        # Hotkey: F11 toggles overlay
        try:
            self.root.bind('<F11>', lambda e: (self.live_overlay_var.set(not bool(self.live_overlay_var.get())), self._toggle_live_overlay()))
        except Exception:
            pass

        self._live_ui_loop()

    def _refresh_cameras(self):
        avail = []
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
        if not avail:
            avail = [0]
        self.live_cam_combo["values"] = [str(i) for i in avail]
        cur = str(self.live_cam_var.get()) if hasattr(self, "live_cam_var") else str(avail[0])
        if cur not in self.live_cam_combo["values"]:
            cur = str(avail[0])
        self.live_cam_combo.set(cur)
        try:
            self.live_cam_var.set(int(cur))
        except Exception:
            pass
        self.live_cam_combo.bind("<<ComboboxSelected>>", lambda _e: self.live_cam_var.set(int(self.live_cam_combo.get())))

    def _toggle_live(self):
        try:
            on = bool(self.live_toggle_var.get())
        except Exception:
            on = False
        self._live_on = on
        self._live_cap = None
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
                    try:
                        cam_idx = int(self.live_cam_var.get())
                    except Exception:
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
                        bufsec = float(self.live_bufsec_var.get())
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
                        self._latest_live_frame = out
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

                f0 = frame[0::2].copy()
                f1 = frame[1::2].copy()

                f0b = apply_record_defects_to_field(f0, rec_def)
                f1b = apply_record_defects_to_field(f1, rec_def)

                y0, c0, meta0 = encode_field_bgr(f0b, sample_rate=opts.sample_rate,
                                                 chroma_subsample=opts.chroma_subsample,
                                                 luma_bw=rec_def.luma_bw)
                y1, c1, meta1 = encode_field_bgr(f1b, sample_rate=opts.sample_rate,
                                                 chroma_subsample=opts.chroma_subsample,
                                                 luma_bw=rec_def.luma_bw)

                y0 = apply_rf_defects_y_dphi_u8(y0, rec_def.record_rf_noise, rec_def.record_dropouts, rec_def.tape_mode)
                c0 = apply_rf_defects_chroma_u8(c0, rec_def.record_rf_noise, rec_def.record_dropouts, rec_def.tape_mode)
                y1 = apply_rf_defects_y_dphi_u8(y1, rec_def.record_rf_noise, rec_def.record_dropouts, rec_def.tape_mode)
                c1 = apply_rf_defects_chroma_u8(c1, rec_def.record_rf_noise, rec_def.record_dropouts, rec_def.tape_mode)

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
                    })
                    tape.cart.set(idx, TapeTrack(y_dphi8=yy, c_u8=cc, meta=meta))


                try:
                    self._live_last_good = {"y0": y0, "c0": c0, "meta0": meta0, "y1": y1, "c1": c1, "meta1": meta1}
                except Exception:
                    pass

                out = self.live_player.get_frame(tape, pb_def)
                self._latest_live_frame = out

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
        right = ttk.Frame(self.tab_play)
        left_sf.pack(side="left", fill="y", padx=10, pady=10)
        right.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        left = left_sf.inner

        ttk.Button(left, text="Load tape bundle folder…", command=self._load_bundle_async).pack(anchor="w", pady=4)
        ttk.Button(left, text="Use memory (no disk)", command=self._use_memory).pack(anchor="w", pady=4)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        ttk.Button(left, text="Insert", command=self._player_insert).pack(anchor="w", pady=2)
        ttk.Button(left, text="Eject", command=self._player_eject).pack(anchor="w", pady=2)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        ttk.Button(left, text="Play", command=self._player_play).pack(anchor="w", pady=2)
        ttk.Button(left, text="Stop", command=self._player_stop).pack(anchor="w", pady=2)
        ttk.Button(left, text="FF", command=self._player_ff).pack(anchor="w", pady=2)
        ttk.Button(left, text="REW", command=self._player_rew).pack(anchor="w", pady=2)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(left, text="Playback / VCR effects (final preview)").pack(anchor="w")

        ttk.Label(left, text="Display aspect").pack(anchor="w")
        self.pb_aspect_play = tk.StringVar(value=self.pb_def.aspect_display)
        ttk.Combobox(left, values=["4:3","16:9"], textvariable=self.pb_aspect_play, width=8, state="readonly").pack(anchor="w")

        # Tracking controls moved here
        # Playback signal / VCR effects
        self._slider(left, "Tracking knob", "tracking_knob", 0.0, 1.0, key="pb")
        self._slider(left, "Tracking sensitivity", "tracking_sensitivity", 0.0, 1.0, key="pb")
        self._slider(left, "Tracking artifacts", "tracking_artifacts", 0.0, 2.0, key="pb")
        self._slider(left, "Auto tracking (0=off, 1=on)", "auto_tracking", 0.0, 1.0, key="pb")
        self._slider(left, "Auto tracking strength", "auto_tracking_strength", 0.0, 1.0, key="pb")
        self._slider(left, "Servo recovery", "servo_recovery", 0.0, 1.0, key="pb")
        self._slider(left, "Sync bias", "sync_bias", 0.0, 1.0, key="pb")

        ttk.Label(left, text="Servo / head switching").pack(anchor="w", pady=(8,0))
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

        ttk.Label(left, text="Chroma").pack(anchor="w", pady=(8,0))
        self._slider(left, "Chroma delay (horizontal)", "chroma_shift_x", 0.0, 1.0, key="pb")
        self._slider(left, "Chroma delay (vertical)", "chroma_shift_y", 0.0, 1.0, key="pb")
        self._slider(left, "Chroma phase error", "chroma_phase", 0.0, 1.0, key="pb")
        self._slider(left, "Chroma phase noise", "chroma_noise", 0.0, 1.0, key="pb")
        self._slider(left, "Chroma noise frequency", "chroma_noise_freq", 0.0, 1.0, key="pb")
        self._slider(left, "Chroma wobble", "chroma_wobble", 0.0, 1.0, key="pb")
        self._slider(left, "Chroma wobble frequency", "chroma_wobble_freq", 0.0, 1.0, key="pb")

        ttk.Label(left, text="Scanline soften (turn up to hide scanlines)").pack(anchor="w", pady=(8,0))
        self._slider(left, "Scanlines", "scanline_strength", 0.0, 1.0, key="pb")
        self._slider(left, "Scanline soften", "scanline_soften", 0.0, 1.0, key="pb")

        ttk.Label(left, text="Image controls").pack(anchor="w", pady=(8,0))
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
        ttk.Checkbutton(left, text="Composite-ish view", variable=self.comp_play_var).pack(anchor="w", pady=4)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(left, text="Audio (Player)").pack(anchor="w")
        # Create audio playback vars here too (in case user never opens Editor)
        self._slider(left, "Audio hiss", "hiss", 0.0, 1.0, key="ap")
        self._slider(left, "Audio pops", "pops", 0.0, 1.0, key="ap")

        self.play_audio_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Play audio (Windows built-in)", variable=self.play_audio_var).pack(anchor="w", pady=2)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(left, text="Prefetch / Proxy (smooth playback)").pack(anchor="w")
        self.proxy_use_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Use proxy for playback (if built)", variable=self.proxy_use_var).pack(anchor="w", pady=2)
        self.proxy_seconds_var = tk.DoubleVar(value=30.0)
        ttk.Label(left, text="Proxy seconds").pack(anchor="w")
        ttk.Scale(left, from_=5.0, to=600.0, orient="horizontal", variable=self.proxy_seconds_var).pack(anchor="w", fill="x")
        ttk.Button(left, text="Build proxy in RAM…", command=self._build_proxy).pack(anchor="w", pady=4)
        ttk.Button(left, text="Export final MP4…", command=self._export_from_player).pack(anchor="w", pady=4)

        self.play_status = tk.StringVar(value="Load tape, then Insert → Play.")
        ttk.Label(left, textvariable=self.play_status, wraplength=360).pack(anchor="w", pady=(10,0))

        self.play_canvas = tk.Label(right, text="Player output (video + audio)", background="#111", foreground="#ddd")
        self.play_canvas.pack(fill="both", expand=True)

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
            self.player.play()
            tape = self._active_tape()
            pos_tracks = float(self.player.state.pos_tracks)
        self._q("status_play", "Play")

        # Audio playback (optional)
        try:
            if bool(self.play_audio_var.get()):
                if not self.audio_player.available:
                    self._q("status_play", "Play (audio disabled — Windows audio backend unavailable)")
                elif tape.audio.pcm16 is not None and tape.audio.pcm16.size > 0:
                    seconds = max(0.0, pos_tracks / 60.0)
                    _pb, ap_def = self._current_playback_defects()
                    self.audio_player.start_stream(
                        tape,
                        get_pos_sec=lambda: float(self.player.state.pos_tracks) / 60.0,
                        get_lock=lambda: (float(getattr(self.player.state,'lock',0.0)) * (1.0 - 0.85*min(1.0, float(getattr(self.player.state,'switch_confuse_timer',0.0))/1.35))),
                        ap_def=ap_def,
                        chunk_sec=0.25,
                    )
        except Exception:
            pass

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
            for i in range(frames_target):
                base = start_tracks + i * step_tracks
                if base >= tape.cart.length_tracks - 2:
                    break
                fr = self.player.get_frame(tape, pb_def)
                # We want deterministic: set position then get_frame
                with self.lock:
                    self.player.state.pos_tracks = float(base)
                fr = self.player.get_frame(tape, pb_def)
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

        def worker():
            self._q("status_play", "Exporting MP4…")
            ok = export_playback_video_mp4(
                tape, pb_def,
                ExportOptions(out_mp4=out, fps=30.0, upscale_width=960),
                progress_cb=lambda a,b: self._q("status_play", f"Exporting… {a}/{b}")
            )
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
                            use_proxy = bool(getattr(self, 'proxy_use_var').get()) if hasattr(self, 'proxy_use_var') else False
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

