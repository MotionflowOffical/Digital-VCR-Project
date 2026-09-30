import numpy as np

from vcr.crt import CRTSettings
from vcr.crt_renderer import _ModernGLCRTBackend


class _Use:
    def use(self, *args, **kwargs):
        pass


class _VAO:
    def render(self, *args, **kwargs):
        pass


class _Ctx:
    TRIANGLE_STRIP = 5

    def __init__(self):
        self.viewport = None
        self.copies = []

    def clear(self, *args, **kwargs):
        pass

    def copy_framebuffer(self, dst, src):
        self.copies.append((dst, src))


def test_phosphor_history_uses_gpu_framebuffer_copy():
    b = _ModernGLCRTBackend.__new__(_ModernGLCRTBackend)
    b.ctx = _Ctx()
    b.source_tex = _Use()
    b.prev_tex = _Use()
    b.prev_fbo = object()
    b.fbo = _Use()
    b.vao = _VAO()
    b._ensure_targets = lambda w, h: None
    b._upload_source = lambda frame: None
    b._set_uniforms = lambda frame, settings, w, h: None

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    b._render_to_fbo(frame, CRTSettings().validated(), 8, 8)

    assert b.ctx.copies == [(b.prev_fbo, b.fbo)]
