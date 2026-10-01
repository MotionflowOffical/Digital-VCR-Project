# Digital VCR (V8.1)


A desktop VHS-style video recorder, live camera processor, CRT display simulator, and MP4 exporter built with CustomTkinter, OpenCV, NumPy, ModernGL, and GLFW.

V8.1 adds a Windows installer build workflow, application/installer branding, and consistent Windows version metadata while retaining the V8.0 playback, CRT, RF, and stability work.

## Highlights

- Recorder, Player, VHS Tape, CRT TV, and Live pages in one desktop app.
- Tape bundle workflow for creating, loading, saving, and exporting virtual tapes.
- Threaded playback, scrubbing, editing preview, camera capture, live processing, and CRT rendering paths.
- Optional RF carrier round-trip model for analog-like luma/chroma degradation.
- GPU CRT simulator using ModernGL and GLFW for phosphor masks, beam shape, bloom, curvature, persistence, and export baking.
- Live camera path with a simple camera-index selector, fullscreen overlay, and direct OpenGL CRT output.
- Windows built-in audio playback plus MP4 export with optional audio mux.
- Backward-compatible bundle loading for older tape bundle layouts.

## V8.0 Updates

- Restored the missing CRT TV application wiring on top of the optimized playback/RF branch: Player preview, Live preview, direct OpenGL windows, settings persistence, and CRT-baked exports are connected again.
- Fixed the CRT phosphor-history path so previous-frame history is copied framebuffer-to-framebuffer entirely on the GPU instead of falling back to a GPU→CPU→GPU round-trip every frame.
- CRT source upload can now use texture channel swizzling to consume OpenCV BGR frames directly, avoiding a full-frame BGR→RGB allocation on supported OpenGL drivers.
- Static CRT shader uniforms are cached and resent only when settings/resolution change; the shader equations and visual model are unchanged.
- CRT remains GPU-native; the optional Rust core is used only for CPU-side VHS/RF hot loops where it provides a measurable benefit.
- Simplified Live camera selection back to plain camera indexes such as `0`, `1`, and `2`.
- Camera refresh discovers connected inputs in a background worker so the UI does not freeze.
- Camera backend fallback now happens internally instead of cluttering the dropdown with backend names.
- Live mode no longer turns brief camera read hiccups into periodic static bursts; short misses are dropped, sustained loss still triggers signal-loss behavior.
- Live on/off no longer blocks the UI while waiting for camera release.
- Live CRT processing uses a bounded latest-frame queue so stale GPU jobs are dropped instead of piling up.
- Live preprocessing uses one shared resize/even-field path; OpenCV can be extended to OpenCL/UMat without changing the tape equations.
- CRT output remains isolated through the CRT renderer thread; no Live worker touches ModernGL or GLFW directly.
- Updated CRT and Live setting help text.

## V8.1 Updates

- Added a Windows installer builder (`build_installer.bat`) using Inno Setup 6.
- Embedded V8.1 into the Windows executable file-version/product-version metadata.
- Updated package/UI version identifiers to V8.1.
- The installer includes the complete PyInstaller application folder and creates Start Menu and optional Desktop shortcuts.
- 
## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

CRT implementation files live in `vcr/crt.py` and `vcr/crt_renderer.py`.

## Build the Windows installer

Run from the project root:

```bat
build_installer.bat
```

The script builds the PyInstaller application first, locates Inno Setup 6, and creates:

```text
installer\output\Digital-VCR-V8.1-Setup.exe
```

If the EXE was already built and you only want to rebuild the installer:

```bat
build_installer.bat --skip-exe
```

The installer places Digital VCR in Program Files, creates a Start Menu shortcut, offers an optional Desktop shortcut, and includes the V8.1 uninstaller entry.

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
  native_core.py
  native/
    digital_vcr_core.dll   # generated on Windows after build_native.bat
  player.py
  recorder.py
  rf_model.py
  tape.py
  gui/
    app.py

native/
  digital_vcr_core/
    Cargo.toml
    src/lib.rs

tools/
  capture_screen.py
```
## Optional Rust Native Core

Digital VCR can use a small Rust native library for hot loops that were previously
executed in Python. The signal model itself remains in the existing Python/NumPy/
OpenCV pipeline; the native core currently accelerates exact row-shift copies and
RF smooth-noise interpolation.

On Windows, install the Rust toolchain, then run:

```bat
build_native.bat
```

This places `digital_vcr_core.dll` in `vcr/native/`. `build_exe.bat` and
`build_exe.ps1` automatically build and package the native core when `cargo` is
available. If the DLL is missing or fails its runtime numerical self-check, the
application automatically uses the Python reference implementation instead.

The performance changes are designed not to reduce simulation quality: the RF
noise batching consumes the same NumPy random sequence as the former per-scanline
implementation, and the native path is guarded against numerical divergence.
Playback also uses deadline-based 30 fps pacing so render time is counted inside
the frame budget rather than added on top of a fixed 33 ms sleep.
