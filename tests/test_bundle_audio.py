import numpy as np

from vcr.bundle import save_bundle, load_bundle
from vcr.tape import TapeAudio, TapeCartridge, TapeImage


def test_bundle_embedded_audio_roundtrip_ulaw(tmp_path):
    pcm16 = np.linspace(-30000, 30000, num=256, dtype=np.int16)
    sample_rate = 8000
    tape = TapeImage(
        cart=TapeCartridge(length_tracks=1),
        audio=TapeAudio(sample_rate=sample_rate, pcm16=pcm16),
    )

    save_bundle(str(tmp_path), tape, settings={}, embed_audio=True)
    loaded_tape, _settings = load_bundle(str(tmp_path))

    assert loaded_tape.audio.pcm16 is not None
    assert loaded_tape.audio.pcm16.size == pcm16.size
    assert loaded_tape.audio.sample_rate == sample_rate
    assert np.any(loaded_tape.audio.pcm16 != pcm16)
