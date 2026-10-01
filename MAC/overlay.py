"""Small native nonactivating panels, strictly outside the calibrated crop."""
from __future__ import annotations


def intersects(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def overlay_rects(binding, profile, observation, screen):
    left, top, right, bottom = profile.meter_roi.pixels(binding.info.width, binding.info.height)
    scale = binding.scale
    x, y = binding.x + left / scale, binding.y + top / scale
    width, height = (right - left) / scale, (bottom - top) / scale
    protected = (x, y, width, height)
    color = "green" if observation.present else "red"
    shapes = [(x - 4, y - 4, width + 8, 2, color),
              (x - 4, y + height + 2, width + 8, 2, color),
              (x - 4, y, 2, height, color), (x + width + 2, y, 2, height, color)]
    if observation.marker_center is not None:
        shapes.append((x - 17, y + observation.marker_center / scale - 1, 11, 2, "white"))
    if observation.zone_bounds:
        for boundary in observation.zone_bounds:
            shapes.append((x + width + 6, y + boundary / scale - 1, 11, 2, "green"))
    shapes.append((x, y - 30, 180, 22, "status"))
    sx, sy, sw, sh = screen
    return [shape for shape in shapes
            if not intersects(shape[:4], protected)
            and shape[0] >= sx and shape[1] >= sy
            and shape[0] + shape[2] <= sx + sw and shape[1] + shape[3] <= sy + sh]


def cocoa_rect(rect, main_height):
    x, y, width, height = rect
    return (x, main_height - y - height), (width, height)


class NullOverlay:
    visible = False

    def __init__(self, error="Mac overlay unavailable; diagnostic logging remains available."):
        self.error = error

    def show(self, _window):
        pass

    def update(self, *_args):
        pass

    def hide(self):
        pass

    def close(self):
        pass


_PANEL_CLASS = None


def panel_class():
    global _PANEL_CLASS
    if _PANEL_CLASS is None:
        import AppKit as A

        class SharkmanDiagnosticPanel(A.NSPanel):
            def canBecomeKeyWindow(self):
                return False

            def canBecomeMainWindow(self):
                return False

        _PANEL_CLASS = SharkmanDiagnosticPanel
    return _PANEL_CLASS


class MacOverlay:
    def __init__(self, backend):
        import AppKit as A
        import Quartz as Q

        self.A, self.Q = A, Q
        self.panel_type = panel_class()
        self.backend = backend
        self.panels = []
        self.visible = False
        self.error = ""

    def show(self, _window):
        if not self.error:
            self.visible = True

    def update(self, window, profile, observation, snapshot):
        if not self.visible:
            return
        try:
            binding = self.backend.bound
            if binding is None or not self.backend.is_foreground(window):
                for panel, _label in self.panels:
                    panel.orderOut_(None)
                return
            from .native import _rect

            A, Q = self.A, self.Q
            screen = _rect(Q.CGDisplayBounds(binding.display_id))
            main_height = Q.CGDisplayBounds(Q.CGMainDisplayID()).size.height
            shapes = overlay_rects(binding, profile, observation, screen)
            while len(self.panels) < len(shapes):
                panel = self.panel_type.alloc().initWithContentRect_styleMask_backing_defer_(
                    ((0, 0), (1, 1)), A.NSWindowStyleMaskBorderless | A.NSWindowStyleMaskNonactivatingPanel,
                    A.NSBackingStoreBuffered, False)
                panel.setOpaque_(False)
                panel.setHasShadow_(False)
                panel.setIgnoresMouseEvents_(True)
                panel.setHidesOnDeactivate_(False)
                panel.setLevel_(A.NSFloatingWindowLevel)
                panel.setCollectionBehavior_(A.NSWindowCollectionBehaviorCanJoinAllSpaces |
                                             A.NSWindowCollectionBehaviorFullScreenAuxiliary)
                panel.setReleasedWhenClosed_(False)
                label = A.NSTextField.labelWithString_("")
                label.setFont_(A.NSFont.systemFontOfSize_(11))
                label.setTextColor_(A.NSColor.whiteColor())
                panel.contentView().addSubview_(label)
                self.panels.append((panel, label))
            colors = {"green": (0.2, 0.83, 0.6), "red": (0.98, 0.44, 0.52),
                      "white": (1, 1, 1), "status": (0.03, 0.07, 0.12)}
            for index, (panel, label) in enumerate(self.panels):
                if index >= len(shapes):
                    panel.orderOut_(None)
                    continue
                *rect, color = shapes[index]
                panel.setFrame_display_(cocoa_rect(rect, main_height), True)
                panel.setBackgroundColor_(A.NSColor.colorWithSRGBRed_green_blue_alpha_(*colors[color], 1))
                label.setHidden_(color != "status")
                if color == "status":
                    label.setFrame_(((4, 2), (172, 18)))
                    label.setStringValue_(f"{snapshot.state.value}  {observation.confidence:.0%}")
                panel.orderFrontRegardless()
        except Exception as exc:
            self.error = f"Mac overlay disabled: {exc}; diagnostic logging still works."
            self.hide()

    def hide(self):
        self.visible = False
        for panel, _label in self.panels:
            panel.orderOut_(None)

    def close(self):
        self.hide()
        for panel, _label in self.panels:
            panel.close()
        self.panels.clear()
