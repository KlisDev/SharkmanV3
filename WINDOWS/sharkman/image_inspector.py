"""Read-only original-resolution image inspection for calibration."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass

import customtkinter as ctk
from PIL import Image, ImageTk

from .ui_platform import wheel_steps


@dataclass
class ImageViewport:
    image_width: int
    image_height: int
    width: int = 1
    height: int = 1
    scale: float = 1.0
    x: float = 0.0
    y: float = 0.0

    @property
    def fit_scale(self) -> float:
        return min(self.width / self.image_width, self.height / self.image_height, 1.0)

    def fit(self) -> None:
        self.scale = self.fit_scale
        self.x = (self.width - self.image_width * self.scale) / 2
        self.y = (self.height - self.image_height * self.scale) / 2

    def source_at(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.x) / self.scale, (y - self.y) / self.scale

    def clamp(self) -> None:
        for origin, extent, pixels in (
            ("x", self.width, self.image_width),
            ("y", self.height, self.image_height),
        ):
            size = pixels * self.scale
            setattr(
                self,
                origin,
                (extent - size) / 2 if size <= extent
                else min(0.0, max(extent - size, getattr(self, origin))),
            )

    def zoom(self, factor: float, x: float, y: float) -> None:
        sx, sy = self.source_at(x, y)
        self.scale = min(8.0, max(self.fit_scale, self.scale * factor))
        self.x, self.y = x - sx * self.scale, y - sy * self.scale
        self.clamp()

    def resize(self, width: int, height: int) -> None:
        fitted = abs(self.scale - self.fit_scale) < 1e-8
        sx, sy = self.source_at(self.width / 2, self.height / 2)
        self.width, self.height = max(1, width), max(1, height)
        if fitted:
            self.fit()
        else:
            self.scale = max(self.fit_scale, self.scale)
            self.x = self.width / 2 - sx * self.scale
            self.y = self.height / 2 - sy * self.scale
            self.clamp()


def viewport_image(source: Image.Image, view: ImageViewport) -> Image.Image:
    """Render only the visible viewport, not a huge zoomed source bitmap."""
    left, top = view.source_at(0, 0)
    right, bottom = view.source_at(view.width, view.height)
    return source.transform(
        (view.width, view.height),
        Image.Transform.EXTENT,
        (left, top, right, bottom),
        resample=Image.Resampling.BILINEAR,
        fillcolor="#08111f",
    )


class ImageInspector(ctk.CTkToplevel):
    """Detached snapshot viewer with no calibration editing callbacks.

    overlay is optionally ("box" or "dot", source-pixel coordinates, color, label).
    """

    def __init__(
        self,
        master: tk.Misc,
        image: Image.Image,
        title: str,
        overlay: tuple[str, tuple[int, ...], str, str] | None = None,
        *,
        redraw_delay_ms: int,
    ) -> None:
        super().__init__(master)
        self.title(title)
        self.geometry("1000x720")
        self.minsize(600, 400)
        self.configure(fg_color="#08111f")
        self.transient(master)
        self.source = image.convert("RGB")
        self.view = ImageViewport(*self.source.size)
        self.view.fit()
        self.overlay = overlay
        self.redraw_delay_ms = redraw_delay_ms
        self.show_overlay = True
        self._pending: str | None = None
        self._drag: tuple[int, int] | None = None
        self._closed = False
        self._photo: ImageTk.PhotoImage | None = None

        header = ctk.CTkFrame(
            self, fg_color="#101d31", corner_radius=16,
            border_width=1, border_color="#27415e",
        )
        header.pack(fill="x", padx=16, pady=(16, 8))
        ctk.CTkLabel(
            header, text="IMAGE INSPECTOR  /  VIEW ONLY",
            text_color="#38bdf8", font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(10, 2))
        ctk.CTkLabel(
            header,
            text="Scroll to zoom · Drag to pan · Esc to close\n"
                 "Inspect here; move the meter region and sample colors in the calibration workspace.",
            text_color="#c9dbea", justify="left", wraplength=550,
        ).pack(anchor="w", padx=16, pady=(0, 10))

        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.pack(fill="x", padx=16, pady=(0, 8))
        for label, command in (
            ("−", lambda: self._zoom(0.8)),
            ("+", lambda: self._zoom(1.25)),
            ("Fit", self._fit_view),
            ("1:1", self._actual_size),
        ):
            ctk.CTkButton(
                controls, text=label, width=52, height=30,
                fg_color="#192e49", hover_color="#244765", command=command,
            ).pack(side="left", padx=(0, 6))
        if overlay is not None:
            self.outline_button = ctk.CTkButton(
                controls, text="Outline: on", width=100, height=30,
                fg_color="#192e49", command=self._toggle_outline,
            )
            self.outline_button.pack(side="left", padx=(0, 6))
        self.percent = ctk.CTkLabel(controls, text="100%", text_color="#7dd3fc")
        self.percent.pack(side="right")

        self.canvas = tk.Canvas(
            self, bg="#08111f", bd=0, highlightthickness=1,
            highlightbackground="#27415e", cursor="fleur",
        )
        self.canvas.pack(fill="both", expand=True, padx=16)
        self._image_item = self.canvas.create_image(0, 0, anchor="nw")
        caption = f"{title} · {self.source.width} × {self.source.height} source pixels"
        if overlay is not None:
            caption += f" · {overlay[3]}"
        ctk.CTkLabel(
            self, text=caption, text_color="#9eb5ce",
            wraplength=560, justify="left",
        ).pack(fill="x", padx=16, pady=(8, 12))

        self.canvas.bind("<Configure>", self._resize)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", self._wheel)
        self.canvas.bind("<Button-5>", self._wheel)
        self.canvas.bind("<Button-1>", self._pan_start)
        self.canvas.bind("<B1-Motion>", self._pan)
        self.canvas.bind("<ButtonRelease-1>", self._pan_end)
        self.bind("<Escape>", lambda _event: self.destroy())
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        # Tk needs to finish mapping the child before an immediate parent close.
        self.update_idletasks()

    def _resize(self, event: tk.Event) -> None:
        self.view.resize(event.width, event.height)
        self._schedule_render()

    def _schedule_render(self) -> None:
        if not self._closed and self._pending is None:
            self._pending = self.after(self.redraw_delay_ms, self._render_view)

    def _render_view(self) -> None:
        self._pending = None
        if self._closed:
            return
        rendered = viewport_image(self.source, self.view)
        self._photo = ImageTk.PhotoImage(rendered, master=self.canvas)
        self.canvas.itemconfigure(self._image_item, image=self._photo)
        self.canvas.delete("outline")
        if self.overlay is not None and self.show_overlay:
            kind, points, color, _label = self.overlay
            values = [
                (self.view.x if index % 2 == 0 else self.view.y) + value * self.view.scale
                for index, value in enumerate(points)
            ]
            if kind == "box":
                self.canvas.create_rectangle(
                    *values, outline=color, width=2, tags="outline",
                )
            elif kind == "dot":
                x, y = values
                self.canvas.create_oval(
                    x - 7, y - 7, x + 7, y + 7,
                    outline=color, width=2, tags="outline",
                )
                self.canvas.create_line(x - 11, y, x + 11, y, fill=color, tags="outline")
                self.canvas.create_line(x, y - 11, x, y + 11, fill=color, tags="outline")
        self.percent.configure(text=f"{self.view.scale * 100:.0f}%")

    def _zoom(self, factor: float, x: float | None = None, y: float | None = None) -> None:
        self.view.zoom(
            factor,
            self.view.width / 2 if x is None else x,
            self.view.height / 2 if y is None else y,
        )
        self._schedule_render()

    def _wheel(self, event: tk.Event) -> str:
        steps = wheel_steps(event)
        if not steps:
            return "break"
        self._zoom(1.2 ** steps, event.x, event.y)
        return "break"

    def _fit_view(self) -> None:
        self.view.fit()
        self._schedule_render()

    def _actual_size(self) -> None:
        self._zoom(1 / self.view.scale)

    def _pan_start(self, event: tk.Event) -> str:
        self._drag = event.x, event.y
        return "break"

    def _pan(self, event: tk.Event) -> str:
        if self._drag is not None:
            self.view.x += event.x - self._drag[0]
            self.view.y += event.y - self._drag[1]
            self.view.clamp()
            self._drag = event.x, event.y
            self._schedule_render()
        return "break"

    def _pan_end(self, _event: tk.Event) -> str:
        self._drag = None
        return "break"

    def _toggle_outline(self) -> None:
        self.show_overlay = not self.show_overlay
        self.outline_button.configure(text="Outline: on" if self.show_overlay else "Outline: off")
        self._schedule_render()

    def destroy(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pending is not None:
            try:
                self.after_cancel(self._pending)
            except tk.TclError:
                pass
            self._pending = None
        self.withdraw()
        self.transient("")
        super().destroy()
        self._photo = None
        self.source.close()
