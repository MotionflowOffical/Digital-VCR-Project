# Digital VCR (V6_13_8)

A desktop VHS-style video recorder and playback simulator built with **CustomTkinter + OpenCV + NumPy**.
It records source video into a custom tape format, plays it back with tracking/RF defects,
and exports the result as MP4.

This build extends the earlier v6 line with a more advanced **RF / tape model**, a **split bundle format**,
better **field pairing**, and improved handling for **progressive / variable-frame-rate sources**.

## Highlights

- **Recorder, Player, VHS Tape, and Live pages** in one desktop app.
- **Modern Studio Console UI** with left-sidebar navigation, dark gradients, and per-setting `?` hover help.
- **Tape bundle workflow**: create, load, save, and keep working directly from a bundle folder.
- **Threaded UI paths** for playback, scrubbing, preview, and bundle loading.
- **Optional RF carrier round-trip model** for more analog-like luma/chroma degradation.
- **Required-GPU CRT simulator** using ModernGL + GLFW for phosphor masks, beam shape, bloom, curvature, persistence, and export baking.
- **Built-in live camera path** with fullscreen overlay output.
- **Windows built-in audio playback** plus MP4 export with optional audio mux.
- **Backward-compatible bundle loading** for older `tape.npz` tapes.

## What changed vs v6.13.1

### V6_13_8
- Added a dedicated **CRT TV** tab with Consumer TV and Pro Monitor presets.
- Added a GPU CRT renderer built on **ModernGL + GLFW / OpenGL 3.3**.
- CRT simulation can be enabled independently for Player preview, Live preview/overlay, direct OpenGL Player/Live windows, and MP4 exports.
- CRT export bakes the display simulation into the rendered video only; tape tracks and bundle media remain unchanged.
- Added simulated phosphor resolution controls so masks are rendered at a higher internal resolution before display/downsample.
- Simulated phosphor masks, scanline beam profile, convergence, curvature, overscan, edge focus, bloom, halation, vignette, and phosphor decay.
- Added CRT settings persistence in presets/bundles while keeping older settings files compatible.
- Added GPU smoke/export tests for the CRT path.

### V6_13_7
- Rebuilt the desktop UI with **CustomTkinter**, left-sidebar navigation, a dark Studio Console palette, and gradient-backed app shell.
- Added `?` hover help beside user-adjustable settings so ranges explain low/mid/high behavior, visual/audio/performance impact, and whether changes are baked or playback-only.
- Refreshed the desktop UI with a professional dark theme and clearer playback controls.
- Fixed unsafe worker-thread access to Tk variables during load, record, live capture, and proxy playback.
- Added in-app tape audio preview and clearer playback audio status, so audio does not need to be exported just to check it.
- Removed duplicated ffmpeg audio extraction during recording for faster record startup and lower CPU/disk load.
- Fixed live-mode status updates and camera release behavior.

### v6.13.6
- Added a dedicated **VHS Tape** tab for advanced RF / tape modelling controls.
- Added **real RF modulation** on record as an optional luma/chroma carrier round-trip:
  - luma: FM-style carrier simulation
  - chroma: color-under style subcarrier simulation
  - channel effects include AM ripple, phase noise, carrier noise, nonlinearity, and dropouts
- Added **RF playback model** controls so playback can also apply carrier-level degradation before decode.
- Added **Luma/Chroma bleed** control for controlled cross-talk at recombination.
- Added **chroma bandwidth** control during encoding so chroma softness can be tuned independently from luma.
- Recorder now stores **frame pairing metadata** (`frame_base_track`, `field_in_frame`) so playback pairs fields correctly even if recording starts on an odd track index.
- Player now uses that pairing metadata to avoid mismatched field weaving, reducing false combing/tearing and improving fine-detail readability.
- Recording now defaults to **progressive field sampling** instead of simple even/odd weaving for progressive sources.
- Recording can use **source timestamps** for variable-frame-rate inputs instead of relying only on a fixed FPS assumption.
- Bundle format now prefers **split tape containers**:
  - `tape_luma.npz`
  - `tape_chroma.npz`
- Bundle save/load now persists RF-related per-track metadata so RF settings survive round-trips through disk.
- Embedded audio save/load now prefers compact **`audio_tape.npz`** storage, with fallback support retained.
- Legacy bundles are still supported:
  - older single-file `tape.npz` loads normally
  - `audio.wav` remains a fallback when compact embedded audio is unavailable
- RF control wiring and persistence were cleaned up so the new advanced controls are exposed and usable from the UI.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

CRT implementation files live in `vcr/crt.py` and `vcr/crt_renderer.py`.

## Requirements

- Python 3.10+
- NumPy
- OpenCV
- Pillow
- imageio-ffmpeg
- CustomTkinter
- ModernGL
- GLFW

Installed from `requirements.txt`:

```txt
numpy>=1.24
opencv-python>=4.7
Pillow>=10.0
imageio-ffmpeg>=0.4.9
customtkinter>=5.2.2
moderngl>=5.12.0
glfw>=2.10.0
```

## Main sidebar pages

### Recorder
Use this to:
- create a new tape in memory or as a real bundle folder
- record a source video into tape tracks
- tune record-side defects such as blur, jitter, RF noise, and dropouts
- auto-save back into the active bundle

### Player
Use this to:
- insert/eject/play/stop/FF/REW
- preview final VHS-style playback
- tune tracking, sync, snow, interference, chroma instability, scanlines, blur, jitter, and image controls
- build a RAM playback proxy for smoother real-time playback
- export the final MP4

### VHS Tape
Advanced modelling tab for:
- enabling **Real RF modulation (FM+AM round-trip)** on record
- adjusting record-side RF parameters:
  - FM depth
  - AM depth
  - phase noise
  - carrier noise
  - nonlinearity
  - chroma carrier fraction
  - chroma demod low-pass
- enabling the **RF playback model**
- adjusting playback-side RF behaviour and luma/chroma bleed

### CRT TV
GPU display simulation tab for:
- enabling CRT output for Player, Live, direct OpenGL windows, and MP4 export
- choosing Consumer TV or Pro Monitor presets
- selecting aperture grille, slot mask, or shadow mask phosphors
- tuning simulated phosphor render width, scanlines, beam sharpness, bloom, halation, curvature, overscan, convergence, vignette, and phosphor decay

### Live
Use a live camera as the source path with:
- live VHS-style preview
- fullscreen overlay output
- quick access to record-side, playback-side, and audio controls

## Tape bundle format

### Current preferred bundle layout
- `tape_info.json` — global tape info and decode parameters
- `tape_luma.npz` — luma tracks + per-track metadata arrays
- `tape_chroma.npz` — chroma tracks
- `settings.json` — saved UI settings
- `audio_tape.npz` — compact embedded tape audio when available
- `audio.wav` — fallback / export-friendly audio file
- `output.mp4` — exported video
- `output_with_audio.mp4` — muxed export when audio is available

### Compatibility
The loader still supports older bundle layouts, including:
- legacy single-file `tape.npz`
- fallback `audio.wav`

## Where is my tape saved?

- **New tape bundle** creates a real folder immediately.
- Recording updates the in-memory tape and can **auto-save back into that same bundle**.
- Saving writes the active tape state to the current bundle folder.
- Newer saves prefer the split luma/chroma bundle format, but older bundles can still be loaded.

## Playback / modelling notes

- **Real RF modulation** is optional. If disabled, the app uses the faster byte-domain defect path.
- If a track was recorded with RF round-trip metadata, playback can automatically use the RF-aware path.
- **Progressive field sampling** is intended to reduce combing artifacts that come from weaving progressive frames into interlaced-style fields.
- **Field-pair alignment** is now track-aware, which is important for insert/edit-style cases and for tapes that do not begin exactly on an even track boundary.

## Audio notes

- Player audio uses **Windows built-in audio playback** when available.
- Tape audio is stored compactly when possible and decoded back in memory on load.
- Export can generate video-only or muxed video+audio outputs.

## Changelog from earlier v6 builds

### v6.4 fix
- Editor Live Preview now works after load (fixed missing Audio Playback slider vars).
- Live Preview jumps to first recorded track automatically.

### v6.5
- Tracking controls moved to the Player tab (final preview).
- Player can export final MP4 (with optional audio mux).
- Player can play tape audio.
- Tape bundles embed audio on tape and keep `audio.wav` as a fallback.

### v6.6.2
- Fixed crash when exporting/saving when playback vars were not built yet.
- Live Preview no longer silently stops due to missing tracking vars.

### v6.6.4
- Fixed Editor Live Preview freezing: worker threads no longer call Tk variable `.get()`.
- Coalesced preview frames + bounded queue drain to prevent UI starvation.

### v6.7
- Reduced full-frame white flicker.
- Snow model became sparser and less "fireworks"-like.
- Added Sync bias + Servo recovery controls.
- Audio storage became smaller / more encoded.
- Audio playback improved with a Windows fallback.
- Added optional RAM proxy builder for smoother playback.

### v6.8
- Audio playback switched to Windows built-in support.
- UI simplified around Recorder + Player flow.
- Added Tracking artifacts + Variance controls.
- Snow + interference became more visible and smoother to tune.

## Project structure

```text
main.py
requirements.txt
README.md

vcr/
├── audio.py
├── audio_player.py
├── bundle.py
├── defects.py
├── editor.py
├── exporter.py
├── modulation.py
├── player.py
├── recorder.py
├── rf_model.py
├── tape.py
└── gui/
    └── app.py

tools/
└── capture_screen.py
```
