# Digital VCR (v6)

### What changed vs v5
- **New tape creates a real bundle folder immediately** (you choose a parent folder).
- **Recorder auto-saves the bundle after recording** (toggle).
- **Loading bundles is threaded** (no UI freeze).
- **Scrub preview + Editor live preview are threaded** (dragging sliders shouldn't lock up).
- **Bundle format is compact** (no giant meta.json). Faster load, smaller files.
- **Export output.mp4 can be toggled off** in Editor.
- **Scanlines are much less visible by default** (and can be hidden by increasing *Scanline soften*).

### Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Where is my tape saved?
- If you click **New tape bundle (creates folder now)**, the tape is created on disk immediately.
- Recording updates the in-memory tape and then **auto-saves back into that same bundle** (if enabled).
- Editor **Save bundle** writes to the active bundle folder (or asks for one if you're in memory mode).

### Files inside a bundle folder
- `tape.npz` — stored tape tracks + compact per-track metadata
- `tape_info.json` — global decode params + tape length
- `settings.json` — UI settings
- `audio.wav` — if extracted
- `output.mp4`, `output_with_audio.mp4` — if export enabled


### v6.4 fix
- Editor Live Preview now works after load (fixed missing Audio Playback slider vars).
- Live Preview jumps to first recorded track automatically.


### v6.5
- Tracking controls moved to the Player tab (final preview).
- Player can export final MP4 (with optional audio mux).
- Player can play tape audio (requires `simpleaudio`).
- Tape bundles embed audio inside `tape.npz` (and still keep audio.wav as a fallback).


### v6.6.2
- Fixed crash when exporting/saving: playback vars may not exist until Player tab builds; now uses safe fallbacks.
- Live Preview no longer silently stops due to missing tracking vars.


### v6.6.4
- Fixed Editor Live Preview freezing: worker threads no longer call Tk variable `.get()`.
- Coalesced preview frames + bounded queue drain to prevent UI starvation.


### v6.7
- Reduced full-frame white flicker: RF drift is now low-frequency and noise-dependent (no unconditional random-walk).
- Snow model no longer 'fireworks': speckle is sparse/clustered with subtle streaks.
- Added Sync bias + Servo recovery sliders to Player for easier lock/sync tuning.
- Audio is stored on tape as 8-bit μ-law (smaller, more 'encoded'); decoded to PCM in memory.
- Player audio playback improved with Windows winsound fallback if simpleaudio isn't available.
- Added optional RAM proxy builder for smooth real-time playback without exporting.


### v6.8
- Audio playback now uses Windows built-in winsound (no simpleaudio / no MSVC build tools required).
- Removed Editor tab (Recorder + Player only).
- Added Tracking artifacts + Variance sliders, and wired them into playback.
- Snow + interference are more visible and respond smoothly to slider values.
