
# Digital VCR (V7_1)
=======
# Digital VCR (V7)


A desktop VHS-style video recorder, live camera processor, CRT display simulator, and MP4 exporter built with CustomTkinter, OpenCV, NumPy, ModernGL, and GLFW.

V7_1 focuses on live-mode stability, simpler camera input selection, CRT thread safety, and a smoother desktop UI.

## Highlights

- Recorder, Player, VHS Tape, CRT TV, and Live pages in one desktop app.
- Tape bundle workflow for creating, loading, saving, and exporting virtual tapes.
- Threaded playback, scrubbing, editing preview, camera capture, live processing, and CRT rendering paths.
- Optional RF carrier round-trip model for analog-like luma/chroma degradation.
- GPU CRT simulator using ModernGL and GLFW for phosphor masks, beam shape, bloom, curvature, persistence, and export baking.
- Live camera path with a simple camera-index selector, fullscreen overlay, and direct OpenGL CRT output.
- Windows built-in audio playback plus MP4 export with optional audio mux.
- Backward-compatible bundle loading for older tape bundle layouts.


## V7_1 Updates

- Simplified Live camera selection back to plain camera indexes such as `0`, `1`, and `2`.
- Camera refresh discovers connected inputs in a background worker so the UI does not freeze.
- Camera backend fallback now happens internally instead of cluttering the dropdown with backend names.
- Live mode no longer turns brief camera read hiccups into periodic static bursts; short misses are dropped, sustained loss still triggers signal-loss behavior.
- Live on/off no longer blocks the UI while waiting for camera release.
- Live processing uses a bounded latest-frame queue so stale camera frames are dropped instead of piling up.
- Safe OpenCL/UMat preprocessing is used for live resize when available, with CPU fallback.
- CRT output remains isolated through the CRT renderer thread; no Live worker touches ModernGL or GLFW directly.
- Removed the expensive per-pixel gradient redraw from the app shell to reduce resize, drag, and scroll lag.
- Updated CRT and Live setting help text.
=======
## What changed

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

## Main Pages

### Recorder

Use this to create tapes, record source video into tape tracks, tune record-side defects, and save bundle folders.

### Player

Use this to play, scrub, tune playback defects, build a RAM proxy, preview audio, and export final MP4 files.

### VHS Tape

Advanced modelling tab for RF record/playback controls, luma/chroma carrier behavior, crosstalk, and tape-style degradation.

### CRT TV

GPU display simulation tab for Player, Live, direct OpenGL windows, and MP4 export. It includes Consumer TV and Pro Monitor presets plus phosphor masks, scanlines, beam sharpness, bloom, halation, curvature, overscan, convergence, vignette, and phosphor decay controls.

### Live

Use a live camera as the source path with:

- simple camera-index selection
- background camera discovery
- live VHS-style preview
- fullscreen overlay output
- direct CRT output through the CRT renderer thread
- quick access to record-side, playback-side, and audio controls

## Tape Bundle Format

Current preferred bundle layout:

- `tape_info.json` - global tape info and decode parameters
- `tape_luma.npz` - luma tracks plus per-track metadata arrays
- `tape_chroma.npz` - chroma tracks
- `settings.json` - saved UI settings
- `audio_tape.npz` - compact embedded tape audio when available
- `audio.wav` - fallback or export-friendly audio file
- `output.mp4` - exported video
- `output_with_audio.mp4` - muxed export when audio is available

The loader still supports older bundle layouts, including legacy single-file `tape.npz` and fallback `audio.wav`.

## Notes

- Real RF modulation is optional. If disabled, the app uses the faster byte-domain defect path.
- If a track was recorded with RF round-trip metadata, playback can automatically use the RF-aware path.
- Progressive field sampling reduces combing artifacts from progressive sources.
- Field-pair alignment is track-aware, which is important for insert/edit-style cases and tapes that do not begin exactly on an even track boundary.
- Tape audio is stored compactly when possible and decoded back in memory on load.
- Export can generate video-only or muxed video+audio outputs.


## Project Structure
=======
## Project structure


```text
main.py
requirements.txt
README.md

vcr/
  audio.py
  audio_player.py
  bundle.py
  crt.py
  crt_renderer.py
  defects.py
  editor.py
  exporter.py
  modulation.py
  player.py
  recorder.py
  rf_model.py
  tape.py
  gui/
    app.py

tools/
  capture_screen.py
```
