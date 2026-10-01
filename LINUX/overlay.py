"""Small click-through X Shape overlay that never paints the meter crop."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sharkman.config import CalibrationProfile
    from sharkman.models import EngineSnapshot, MeterObservation, WindowInfo


Rect = tuple[int, int, int, int]
Primitive = tuple[int, int, int, int, str]


def intersects(first: Rect, second: Rect) -> bool:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def overlay_primitives(
    window: WindowInfo,
    profile: CalibrationProfile,
    observation: MeterObservation,
    screen_width: int,
    screen_height: int,
) -> list[Primitive]:
    """Draw around, never inside, the detector's calibrated rectangle."""
    left, top, right, bottom = profile.meter_roi.pixels(window.width, window.height)
    x0, y0 = window.left + left, window.top + top
    x1, y1 = window.left + right, window.top + bottom
    protected = (x0, y0, x1 - x0, y1 - y0)
    color = "#34d399" if observation.present else "#fb7185"
    proposed: list[Primitive] = [
        (x0 - 4, y0 - 4, x1 - x0 + 8, 2, color),
        (x0 - 4, y1 + 2, x1 - x0 + 8, 2, color),
        (x0 - 4, y0, 2, y1 - y0, color),
        (x1 + 2, y0, 2, y1 - y0, color),
    ]
    if observation.zone_bounds is not None:
        for zone_y in observation.zone_bounds:
            proposed.append((x1 + 6, y0 + round(zone_y) - 1, 12, 2, "#34d399"))
    if observation.marker_center is not None:
        proposed.append((x0 - 19, y0 + round(observation.marker_center) - 1, 13, 3, "#ffffff"))
    shapes: list[Primitive] = []
    for x, y, width, height, shade in proposed:
        # Fail closed at the screen edge instead of asking X Shape to clip a
        # rectangle into the protected meter region.
        if (width > 0 and height > 0 and x >= 0 and y >= 0
                and x + width <= screen_width and y + height <= screen_height
                and not intersects((x, y, width, height), protected)):
            shapes.append((x, y, width, height, shade))
    return shapes


def status_pill(
    window: WindowInfo, profile: CalibrationProfile,
    screen_width: int, screen_height: int,
) -> Rect | None:
    left, top, right, bottom = profile.meter_roi.pixels(window.width, window.height)
    protected = (window.left + left, window.top + top, right - left, bottom - top)
    x = max(0, protected[0] - 140)
    y = protected[1] - 28
    candidate = (x, y, 136, 22)
    if (y < 0 or x + candidate[2] > screen_width or y + candidate[3] > screen_height
            or intersects(candidate, protected)):
        return None
    return candidate


class NullOverlay:
    visible = False
    error = "X Shape overlay is unavailable; F8 still records diagnostics."

    def show(self, _window: WindowInfo) -> None:
        pass

    def update(
        self, _window: WindowInfo, _profile: CalibrationProfile,
        _observation: MeterObservation, _snapshot: EngineSnapshot,
    ) -> None:
        pass

    def hide(self) -> None:
        pass

    def close(self) -> None:
        pass


class X11Overlay:
    """An empty input shape guarantees that the overlay cannot take a click."""

    def __init__(self, _parent: object = None) -> None:
        from Xlib import X
        from Xlib import display as xdisplay
        from Xlib.ext import shape

        self._X = X
        self._shape = shape
        self._display = xdisplay.Display()
        if not self._display.has_extension("SHAPE"):
            self._display.close()
            raise RuntimeError("X11 Shape extension is unavailable")
        self._root = self._display.screen().root
        geometry = self._root.get_geometry()
        self._width, self._height = int(geometry.width), int(geometry.height)
        self._window = None
        self.visible = False
        self.error = ""
        self._colors: dict[str, int] = {}

    def _shape_rectangles(self, kind: int, rectangles: list[Rect]) -> None:
        assert self._window is not None
        method = getattr(self._window, "shape_rectangles", None)
        if method is None:
            raise RuntimeError("X11 Shape window methods are unavailable")
        method(self._shape.SO.Set, kind, 0, 0, 0, rectangles)

    def show(self, _window: WindowInfo) -> None:
        if self.visible or self.error:
            return
        try:
            screen = self._display.screen()
            self._window = self._root.create_window(
                0, 0, self._width, self._height, 0,
                self._X.CopyFromParent, self._X.InputOutput, self._X.CopyFromParent,
                override_redirect=1, background_pixel=screen.black_pixel,
                event_mask=0,
            )
            self._shape_rectangles(self._shape.SK.Input, [])
            self._shape_rectangles(self._shape.SK.Bounding, [])
            self._window.map()
            self._window.configure(stack_mode=self._X.Above)
            self._display.sync()
            self.visible = True
        except Exception as exc:
            self.error = f"X11 overlay could not start: {exc}"
            self.hide()

    def update(
        self, window: WindowInfo, profile: CalibrationProfile,
        observation: MeterObservation, snapshot: EngineSnapshot,
    ) -> None:
        if not self.visible or self._window is None:
            return
        try:
            shapes = overlay_primitives(window, profile, observation, self._width, self._height)
            pill = status_pill(window, profile, self._width, self._height)
            rectangles = [item[:4] for item in shapes]
            if pill is not None:
                rectangles.append(pill)
            self._shape_rectangles(self._shape.SK.Bounding, rectangles)
            self._window.clear_area(0, 0, 0, 0)
            colormap = self._display.screen().default_colormap
            for x, y, width, height, shade in shapes:
                if shade not in self._colors:
                    self._colors[shade] = colormap.alloc_named_color(shade).pixel
                graphics = self._window.create_gc(foreground=self._colors[shade])
                self._window.fill_rectangle(graphics, x, y, width, height)
                graphics.free()
            if pill is not None:
                x, y, width, height = pill
                foreground_shade = "#34d399" if observation.present else "#fb7185"
                for shade in ("#07111f", foreground_shade):
                    if shade not in self._colors:
                        self._colors[shade] = colormap.alloc_named_color(shade).pixel
                graphics = self._window.create_gc(foreground=self._colors["#07111f"])
                self._window.fill_rectangle(graphics, x, y, width, height)
                graphics.free()
                state = getattr(getattr(snapshot, "state", None), "value", "TRACKING")
                label = f"{state} {observation.confidence:.0%}"[:21]
                graphics = self._window.create_gc(foreground=self._colors[foreground_shade])
                self._window.draw_text(graphics, x + 5, y + 15, label)
                graphics.free()
            self._display.flush()
        except Exception as exc:
            self.error = f"X11 overlay failed: {exc}"
            self.hide()

    def hide(self) -> None:
        self.visible = False
        if self._window is not None:
            try:
                self._window.destroy()
                self._display.flush()
            except Exception:
                pass
            self._window = None

    def close(self) -> None:
        self.hide()
        self._display.close()
