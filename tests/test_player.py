from vcr.defects import PlaybackDefects
from vcr.player import _playback_interference_strength


def test_interference_strength_is_zero_when_amount_slider_is_zero():
    pb = PlaybackDefects(interference=0.0)

    strength = _playback_interference_strength(pb, sync=0.0, stress=1.0, gate=1.0)

    assert strength == 0.0

