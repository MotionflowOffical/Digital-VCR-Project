from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .crt import CRTSettings


class CRTGPUUnavailable(RuntimeError):
    pass


@dataclass
class _RenderJob:
    frame: np.ndarray
    settings: CRTSettings
    output_size: tuple[int, int] | None
    result: queue.Queue
    cancelled: threading.Event


@dataclass
class _DirectJob:
    key: str
    frame: np.ndarray
    settings: CRTSettings
    title: str
    size: tuple[int, int] | None


@dataclass
class _CloseDirectJob:
    key: str


class _ModernGLCRTBackend:
    VERTEX_SHADER = """
        #version 330
        in vec2 in_pos;
        in vec2 in_uv;
        out vec2 v_uv;
        void main() {
            gl_Position = vec4(in_pos, 0.0, 1.0);
            v_uv = in_uv;
        }
    """

    FRAGMENT_SHADER = """
        #version 330
        uniform sampler2D u_source;
        uniform sampler2D u_prev;
        uniform vec2 u_input_size;
        uniform vec2 u_output_size;
        uniform float u_time;
        uniform int u_mask_type;
        uniform float u_mask_strength;
        uniform float u_scanline_strength;
        uniform float u_beam_sharpness;
        uniform float u_bloom;
        uniform float u_halation;
        uniform float u_glass_diffusion;
        uniform float u_curvature;
        uniform float u_overscan;
        uniform float u_vignette;
        uniform float u_edge_focus;
        uniform float u_decay;
        uniform vec2 u_convergence;
        uniform float u_brightness;
        uniform float u_contrast;
        uniform float u_saturation;
        in vec2 v_uv;
        out vec4 fragColor;

        vec3 sampleConv(vec2 uv) {
            vec2 px = 1.0 / max(u_output_size, vec2(1.0));
            vec2 c = u_convergence * px;
            float r = texture(u_source, uv + c).r;
            float g = texture(u_source, uv).g;
            float b = texture(u_source, uv - c).b;
            return vec3(r, g, b);
        }

        vec3 satAdjust(vec3 c, float sat) {
            float y = dot(c, vec3(0.299, 0.587, 0.114));
            return mix(vec3(y), c, 1.0 + sat);
        }

        vec3 maskPattern(vec2 frag) {
            float pitch = max(1.0, floor(u_output_size.x / 720.0));
            float x = floor(frag.x / pitch);
            float y = floor(frag.y / max(1.0, pitch));
            float tri = mod(x, 3.0);
            vec3 m = vec3(0.70);

            if (u_mask_type == 0) {
                if (tri < 1.0) m = vec3(1.18, 0.58, 0.58);
                else if (tri < 2.0) m = vec3(0.58, 1.18, 0.58);
                else m = vec3(0.58, 0.58, 1.18);
            } else if (u_mask_type == 1) {
                float row = mod(y, 4.0);
                if (row > 2.0) {
                    m = vec3(0.50);
                } else if (tri < 1.0) {
                    m = vec3(1.16, 0.55, 0.55);
                } else if (tri < 2.0) {
                    m = vec3(0.55, 1.16, 0.55);
                } else {
                    m = vec3(0.55, 0.55, 1.16);
                }
            } else {
                vec2 cell = vec2(mod(x, 6.0), mod(y, 4.0));
                float dotGate = smoothstep(2.2, 0.4, length(cell - vec2(1.5, 1.3)));
                float dotGate2 = smoothstep(2.2, 0.4, length(cell - vec2(4.5, 2.7)));
                float d = max(dotGate, dotGate2);
                if (tri < 1.0) m = mix(vec3(0.46), vec3(1.20, 0.54, 0.54), d);
                else if (tri < 2.0) m = mix(vec3(0.46), vec3(0.54, 1.20, 0.54), d);
                else m = mix(vec3(0.46), vec3(0.54, 0.54, 1.20), d);
            }
            return mix(vec3(1.0), m, u_mask_strength);
        }

        void main() {
            vec2 p = v_uv * 2.0 - 1.0;
            p *= 1.0 + u_overscan;
            float r2 = dot(p, p);
            vec2 warped = p * (1.0 + 0.22 * u_curvature * r2);
            vec2 uv = warped * 0.5 + 0.5;
            if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
                return;
            }

            vec2 srcPx = 1.0 / max(u_input_size, vec2(1.0));
            float edge = smoothstep(0.55, 1.25, length(warped));
            float focus = u_edge_focus * edge;

            vec3 c = sampleConv(uv);
            vec3 blur = vec3(0.0);
            blur += sampleConv(uv + vec2( srcPx.x * 1.8, 0.0));
            blur += sampleConv(uv + vec2(-srcPx.x * 1.8, 0.0));
            blur += sampleConv(uv + vec2(0.0,  srcPx.y * 1.6));
            blur += sampleConv(uv + vec2(0.0, -srcPx.y * 1.6));
            blur *= 0.25;
            c = mix(c, blur, clamp(u_glass_diffusion + focus, 0.0, 1.0));

            float lum = dot(c, vec3(0.299, 0.587, 0.114));
            vec3 wide = (
                sampleConv(uv + vec2( srcPx.x * 5.0,  srcPx.y * 2.0)) +
                sampleConv(uv + vec2(-srcPx.x * 5.0,  srcPx.y * 2.0)) +
                sampleConv(uv + vec2( srcPx.x * 5.0, -srcPx.y * 2.0)) +
                sampleConv(uv + vec2(-srcPx.x * 5.0, -srcPx.y * 2.0))
            ) * 0.25;
            vec3 bright = max(wide - vec3(0.52), vec3(0.0));
            c += bright * (0.55 + 1.6 * lum) * u_bloom;
            c += bright.rgg * vec3(1.35, 0.55, 0.35) * u_halation;

            float line = uv.y * u_input_size.y;
            float dist = abs(fract(line) - 0.5);
            float beamWidth = mix(0.09, 0.40, lum) + (1.0 - u_beam_sharpness) * 0.20;
            float beam = exp(-(dist * dist) / max(0.0008, beamWidth * beamWidth));
            c *= mix(1.0, 0.26 + 0.92 * beam, u_scanline_strength);

            c *= maskPattern(gl_FragCoord.xy);

            float vig = smoothstep(1.25, 0.20, length(p));
            c *= mix(1.0 - 0.65 * u_vignette, 1.0, vig);

            c = (c - 0.5) * (1.0 + u_contrast) + 0.5 + u_brightness;
            c = satAdjust(c, u_saturation);
            c *= 1.0 + 0.0015 * sin(u_time * 59.94);

            vec3 prev = texture(u_prev, v_uv).rgb;
            c = max(c, prev * u_decay);
            c = pow(clamp(c, 0.0, 1.0), vec3(0.92));
            fragColor = vec4(c, 1.0);
        }
    """

    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=self.VERTEX_SHADER, fragment_shader=self.FRAGMENT_SHADER)
        quad = np.array(
            [
                -1.0, -1.0, 0.0, 1.0,
                 1.0, -1.0, 1.0, 1.0,
                -1.0,  1.0, 0.0, 0.0,
                 1.0,  1.0, 1.0, 0.0,
            ],
            dtype="f4",
        )
        self.vbo = ctx.buffer(quad.tobytes())
        self.vao = ctx.vertex_array(self.program, [(self.vbo, "2f 2f", "in_pos", "in_uv")])
        self.source_tex = None
        self.color_tex = None
        self.prev_tex = None
        self.prev_fbo = None
        self.fbo = None
        self.size = (0, 0)
        self._source_bgr_swizzle = False
        self._uniform_key = None

    def release(self):
        for obj in (self.fbo, self.prev_fbo, self.color_tex, self.prev_tex, self.source_tex, self.vao, self.vbo, self.program):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass

    def _ensure_targets(self, width: int, height: int):
        width = int(max(1, width))
        height = int(max(1, height))
        if self.size == (width, height) and self.fbo is not None:
            return
        for obj in (self.fbo, self.prev_fbo, self.color_tex, self.prev_tex):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass
        self.color_tex = self.ctx.texture((width, height), 4)
        self.color_tex.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self.prev_tex = self.ctx.texture((width, height), 4)
        self.prev_tex.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self.prev_tex.write(bytes(width * height * 4))
        self.fbo = self.ctx.framebuffer(color_attachments=[self.color_tex])
        # A texture is not a valid destination for Context.copy_framebuffer().
        # The previous implementation therefore fell through to a full GPU->CPU
        # readback and CPU->GPU upload on every frame.  Attach the history
        # texture to its own FBO so persistence stays entirely on the GPU.
        self.prev_fbo = self.ctx.framebuffer(color_attachments=[self.prev_tex])
        self.size = (width, height)

    def _upload_source(self, frame_bgr: np.ndarray):
        # OpenCV frames are BGR.  ModernGL textures can swizzle channels while
        # sampling, so upload the contiguous BGR bytes directly and expose them
        # to the shader as RGB.  This removes one full-frame cvtColor allocation
        # without changing sampled values.  Fall back to the old conversion on
        # drivers/backends that do not expose texture swizzles.
        if not frame_bgr.flags.c_contiguous:
            frame_bgr = np.ascontiguousarray(frame_bgr)
        h, w = frame_bgr.shape[:2]
        if self.source_tex is not None:
            try:
                if self.source_tex.size != (w, h):
                    self.source_tex.release()
                    self.source_tex = None
            except Exception:
                self.source_tex = None
        if self.source_tex is None:
            self.source_tex = self.ctx.texture((w, h), 3)
            self.source_tex.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
            self._source_bgr_swizzle = False
            try:
                self.source_tex.swizzle = "BGR1"
                self._source_bgr_swizzle = True
            except Exception:
                self._source_bgr_swizzle = False
        if self._source_bgr_swizzle:
            self.source_tex.write(frame_bgr.tobytes(), alignment=1)
        else:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.source_tex.write(rgb.tobytes(), alignment=1)

    def _set_uniforms(self, frame_bgr: np.ndarray, s: CRTSettings, width: int, height: int):
        # Everything except time is static until the CRT settings or frame/target
        # dimensions change.  Avoid ~20 Python->OpenGL uniform calls per frame.
        mask_id = {"aperture": 0, "slot": 1, "shadow": 2}.get(s.mask_type, 1)
        key = (
            int(frame_bgr.shape[1]), int(frame_bgr.shape[0]), int(width), int(height), int(mask_id),
            float(s.mask_strength), float(s.scanline_strength), float(s.beam_sharpness),
            float(s.bloom), float(s.halation), float(s.glass_diffusion), float(s.curvature),
            float(s.overscan), float(s.vignette), float(s.edge_focus), float(s.phosphor_decay),
            float(s.convergence_x), float(s.convergence_y), float(s.brightness),
            float(s.contrast), float(s.saturation),
        )
        if key != self._uniform_key:
            self.program["u_source"].value = 0
            self.program["u_prev"].value = 1
            self.program["u_input_size"].value = (float(frame_bgr.shape[1]), float(frame_bgr.shape[0]))
            self.program["u_output_size"].value = (float(width), float(height))
            self.program["u_mask_type"].value = int(mask_id)
            self.program["u_mask_strength"].value = float(s.mask_strength)
            self.program["u_scanline_strength"].value = float(s.scanline_strength)
            self.program["u_beam_sharpness"].value = float(s.beam_sharpness)
            self.program["u_bloom"].value = float(s.bloom)
            self.program["u_halation"].value = float(s.halation)
            self.program["u_glass_diffusion"].value = float(s.glass_diffusion)
            self.program["u_curvature"].value = float(s.curvature)
            self.program["u_overscan"].value = float(s.overscan)
            self.program["u_vignette"].value = float(s.vignette)
            self.program["u_edge_focus"].value = float(s.edge_focus)
            self.program["u_decay"].value = float(s.phosphor_decay)
            self.program["u_convergence"].value = (float(s.convergence_x), float(s.convergence_y))
            self.program["u_brightness"].value = float(s.brightness)
            self.program["u_contrast"].value = float(s.contrast)
            self.program["u_saturation"].value = float(s.saturation)
            self._uniform_key = key
        self.program["u_time"].value = float(time.perf_counter())

    def _render_to_fbo(self, frame_bgr: np.ndarray, settings: CRTSettings, width: int, height: int):
        self._ensure_targets(width, height)
        self._upload_source(frame_bgr)
        self.source_tex.use(0)
        self.prev_tex.use(1)
        self._set_uniforms(frame_bgr, settings, width, height)
        self.fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render(mode=self.ctx.TRIANGLE_STRIP)
        try:
            self.ctx.copy_framebuffer(self.prev_fbo, self.fbo)
        except Exception:
            # Compatibility fallback only.  Normal ModernGL backends use the
            # GPU-side framebuffer copy above and never cross the PCIe bus here.
            data = self.fbo.read(components=4, alignment=1)
            self.prev_tex.write(data, alignment=1)

    def render_to_array(
        self,
        frame_bgr: np.ndarray,
        settings: CRTSettings,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        s = settings
        fh, fw = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        width = int(s.render_width)
        height = int(max(1, round(fh * (width / float(max(1, fw))))))
        self._render_to_fbo(frame_bgr, s, width, height)
        data = self.fbo.read(components=3, alignment=1)
        rgb = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        rgb = np.flipud(rgb)
        if output_size is not None:
            ow, oh = int(output_size[0]), int(output_size[1])
            if ow > 0 and oh > 0 and (ow != width or oh != height):
                rgb = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def render_to_screen(self, frame_bgr: np.ndarray, settings: CRTSettings, width: int, height: int):
        self._render_to_fbo(frame_bgr, settings, width, height)
        try:
            self.ctx.screen.use()
            self.ctx.copy_framebuffer(self.ctx.screen, self.fbo)
        except Exception:
            self.ctx.screen.use()
            self.color_tex.use(0)
            self.program["u_source"].value = 0
            self.vao.render(mode=self.ctx.TRIANGLE_STRIP)


class CRTFrameRenderer:
    def __init__(self):
        # Off-screen render/control jobs are ordered and must not be starved.
        # Direct presentation is intentionally *not* queued frame-by-frame: a
        # window only needs the newest frame. Keeping old direct frames caused
        # queue growth, preview timeouts and playback-clock jumps when a GLFW
        # window was resized/minimized or VSync blocked.
        self._jobs: queue.Queue = queue.Queue()
        self._direct_pending: dict[str, _DirectJob] = {}
        self._direct_lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._lock = threading.Lock()

    def _start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive() and self._ready.is_set() and not self._error:
                return
            # A failed GLFW/GL context used to leave a dead thread object behind.
            # Later renders would enqueue work to that dead thread and wait for
            # timeouts.  Reset the worker state so a recoverable driver/context
            # failure can be retried cleanly.
            if self._thread is not None and not self._thread.is_alive():
                self._thread = None
            if self._thread is None:
                self._ready.clear()
                self._error = None
                self._stop.clear()
                self._thread = threading.Thread(target=self._thread_main, name="CRTFrameRenderer", daemon=True)
                self._thread.start()
        self._ready.wait(timeout=8.0)
        if self._error:
            raise CRTGPUUnavailable(self._error)
        if not self._ready.is_set():
            raise CRTGPUUnavailable("Timed out while creating the OpenGL CRT renderer.")

    def _thread_main(self):
        hidden_window = None
        hidden_backend = None
        direct: dict[str, tuple[Any, _ModernGLCRTBackend]] = {}
        glfw = None
        try:
            import glfw as _glfw
            import moderngl

            glfw = _glfw
            if not glfw.init():
                raise CRTGPUUnavailable("GLFW initialization failed.")
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            hidden_window = glfw.create_window(64, 64, "Digital VCR CRT Renderer", None, None)
            if not hidden_window:
                raise CRTGPUUnavailable("Could not create hidden OpenGL 3.3 window.")
            glfw.make_context_current(hidden_window)
            ctx = moderngl.create_context(require=330)
            hidden_backend = _ModernGLCRTBackend(ctx)
            self._ready.set()

            while not self._stop.is_set():
                job = None
                try:
                    # Render/control work has priority.  Direct windows consume
                    # only their latest pending frame below, so they can never
                    # build a backlog in front of integrated preview renders.
                    job = self._jobs.get(timeout=0.004)
                except queue.Empty:
                    pass

                if isinstance(job, _RenderJob):
                    if not job.cancelled.is_set():
                        try:
                            glfw.make_context_current(hidden_window)
                            out = hidden_backend.render_to_array(job.frame, job.settings, job.output_size)
                            if not job.cancelled.is_set():
                                job.result.put_nowait((True, out))
                        except Exception as exc:
                            if not job.cancelled.is_set():
                                try:
                                    job.result.put_nowait((False, exc))
                                except queue.Full:
                                    pass
                elif isinstance(job, _CloseDirectJob):
                    self._close_direct_key(glfw, job.key, direct)
                    try:
                        glfw.make_context_current(hidden_window)
                    except Exception:
                        pass

                # Present at most the newest frame for each direct window.
                # Pulling from a small dict rather than a FIFO is the key to
                # keeping window presentation from starving preview rendering.
                with self._direct_lock:
                    direct_jobs = list(self._direct_pending.values())
                    self._direct_pending.clear()
                for djob in direct_jobs:
                    try:
                        self._handle_direct_job(glfw, djob, direct)
                    except Exception:
                        try:
                            self._close_direct_key(glfw, djob.key, direct)
                        except Exception:
                            pass
                    finally:
                        try:
                            glfw.make_context_current(hidden_window)
                        except Exception:
                            pass

                if glfw is not None:
                    glfw.poll_events()
                    for key in list(direct.keys()):
                        win, _backend = direct[key]
                        if glfw.window_should_close(win):
                            self._close_direct_key(glfw, key, direct)
                            try:
                                glfw.make_context_current(hidden_window)
                            except Exception:
                                pass

        except Exception as exc:
            self._error = str(exc)
            self._ready.set()
        finally:
            for key in list(direct.keys()):
                try:
                    self._close_direct_key(glfw, key, direct)
                except Exception:
                    pass
            if hidden_backend is not None:
                hidden_backend.release()
            if glfw is not None and hidden_window is not None:
                try:
                    glfw.destroy_window(hidden_window)
                except Exception:
                    pass
            if glfw is not None:
                try:
                    glfw.terminate()
                except Exception:
                    pass

    def _handle_direct_job(self, glfw: Any, job: _DirectJob, direct: dict[str, tuple[Any, _ModernGLCRTBackend]]):
        import moderngl

        s = job.settings
        if job.key not in direct:
            if job.size is not None:
                width, height = job.size
            else:
                fh, fw = int(job.frame.shape[0]), int(job.frame.shape[1])
                width = int(s.render_width)
                height = int(max(1, round(fh * (width / float(max(1, fw))))))
            glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
            window = glfw.create_window(int(width), int(height), job.title, None, None)
            if not window:
                raise CRTGPUUnavailable(f"Could not create CRT window '{job.title}'.")
            glfw.make_context_current(window)
            glfw.swap_interval(1)
            ctx = moderngl.create_context(require=330)
            direct[job.key] = (window, _ModernGLCRTBackend(ctx))

        window, backend = direct[job.key]
        glfw.make_context_current(window)
        width, height = glfw.get_framebuffer_size(window)
        if width <= 0 or height <= 0:
            return
        backend.render_to_screen(job.frame, s, int(width), int(height))
        glfw.swap_buffers(window)

    def _close_direct_key(self, glfw: Any, key: str, direct: dict[str, tuple[Any, _ModernGLCRTBackend]]):
        item = direct.pop(key, None)
        if item is None:
            return
        window, backend = item
        # ModernGL objects must be released while the context that owns them is
        # current. Releasing after another context became current could leave a
        # driver in a bad state and make the hidden preview context render black.
        if glfw is not None:
            try:
                glfw.make_context_current(window)
            except Exception:
                pass
        backend.release()
        if glfw is not None:
            try:
                glfw.destroy_window(window)
            except Exception:
                pass

    def render_frame(
        self,
        frame_bgr: np.ndarray,
        settings: CRTSettings,
        output_size: tuple[int, int] | None = None,
        timeout: float = 2.0,
    ) -> np.ndarray:
        s = settings.validated()
        if not s.enabled:
            if output_size is not None:
                return cv2.resize(frame_bgr, output_size, interpolation=cv2.INTER_AREA)
            return frame_bgr.copy()

        self._start()
        result: queue.Queue = queue.Queue(maxsize=1)
        cancelled = threading.Event()
        job = _RenderJob(frame_bgr.copy(), s, output_size, result, cancelled)
        self._jobs.put(job)
        try:
            ok, payload = result.get(timeout=max(0.1, float(timeout)))
        except queue.Empty as exc:
            # Do not let timed-out frames become expensive stale work later.
            # The worker will cheaply skip this job if it has not started yet.
            cancelled.set()
            raise CRTGPUUnavailable("Timed out while rendering CRT frame.") from exc
        if not ok:
            raise CRTGPUUnavailable(str(payload))
        return payload

    def submit_direct(
        self,
        key: str,
        frame_bgr: np.ndarray,
        settings: CRTSettings,
        title: str,
        size: tuple[int, int] | None = None,
    ) -> None:
        s = settings.validated()
        if not s.enabled:
            return
        self._start()
        # Latest-frame mailbox: overwrite an older unsent direct frame instead
        # of queueing every playback frame.  This keeps latency bounded.
        djob = _DirectJob(str(key), frame_bgr.copy(), s, str(title), size)
        with self._direct_lock:
            self._direct_pending[str(key)] = djob

    def close_direct(self, key: str) -> None:
        key = str(key)
        with self._direct_lock:
            self._direct_pending.pop(key, None)
        if self._thread is not None:
            self._jobs.put(_CloseDirectJob(key))

    def close(self):
        self._stop.set()
        if self._thread is not None:
            try:
                self._jobs.put(_CloseDirectJob("__none__"))
            except Exception:
                pass
            self._thread.join(timeout=2.0)
            if not self._thread.is_alive():
                self._thread = None


def render_crt_frame_sync(
    frame_bgr: np.ndarray,
    settings: CRTSettings,
    output_size: tuple[int, int] | None = None,
    timeout: float = 2.0,
) -> np.ndarray:
    renderer = CRTFrameRenderer()
    try:
        return renderer.render_frame(frame_bgr, settings, output_size=output_size, timeout=timeout)
    finally:
        renderer.close()
