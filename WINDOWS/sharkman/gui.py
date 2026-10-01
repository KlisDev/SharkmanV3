from __future__ import annotations

import ctypes
import sys
import tkinter as tk
from datetime import UTC, datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog
from typing import Any

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from . import __version__
from .adaptive import LEARNING_BATCH, LIVE_ADAPTIVE_FEEDBACK_SUPPORTED
from .backends import BackendError, PlatformBackend, create_backend
from .backends.base import WindowSelectionRequired
from .config import (
    SAMPLE_GROUPS,
    TIMING_BY_NAME,
    CalibrationProfile,
    ColorSample,
    ProfileStore,
    TimingConfig,
    timing_groups,
)
from .image_inspector import ImageInspector
from .models import EngineSnapshot, FireAction, MeterObservation, NormalizedRect, WindowInfo
from .runner import LiveRunner
from .ui_platform import CANVAS_FONT, wheel_steps
from .vision import (
    MeterDetector,
    color_mask_bgr,
    color_sample_error,
    sample_median_rgb,
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")
DEFAULT_SESSION_MODE = "Live input"

APP_BG = "#08111f"
BG_CARD = "#111d31"
BG_CARD_RAISED = "#162642"
BG_SOFT = "#0c1729"
BORDER = "#253956"
TEXT = "#eff6ff"
MUTED = "#91a4c3"
ACCENT = "#38bdf8"
ACCENT_HOVER = "#149bd7"
SUCCESS = "#34d399"
WARNING = "#fbbf24"
DANGER = "#fb7185"
COLOR_ACCENT = "#e879f9"

CALIBRATION_ASSETS = Path(__file__).resolve().parent / "assets" / "calibration"
CALIBRATION_REFERENCE_FILES = {
    "region": "complete_meter.png",
    "green": "green_target.png",
    "marker": "black_marker.png",
    "track": "track_gradient.png",
}
_REFERENCE_CACHE: dict[tuple[str, int, int, int], ctk.CTkImage] = {}
_BLANK_REFERENCE: ctk.CTkImage | None = None


def _blank_reference() -> ctk.CTkImage:
    global _BLANK_REFERENCE
    if _BLANK_REFERENCE is None:
        blank = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        _BLANK_REFERENCE = ctk.CTkImage(
            light_image=blank, dark_image=blank, size=(1, 1))
    return _BLANK_REFERENCE


def _calibration_reference(task: str, width: int = 250, height: int = 138) -> ctk.CTkImage | None:
    """Load a replaceable calibration guide without using it for detection."""
    path = _calibration_reference_path(task)
    if path is None:
        return None
    try:
        modified = path.stat().st_mtime_ns
    except OSError:
        return None
    cache_key = (str(path), width, height, modified)
    if cache_key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[cache_key]
    for stale in tuple(_REFERENCE_CACHE):
        if stale[:3] == cache_key[:3]:
            _REFERENCE_CACHE.pop(stale, None)
    try:
        with Image.open(path) as source:
            image = source.convert("RGBA")
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image.thumbnail((width, height), resampling)
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, image.width - 1, image.height - 1),
            radius=min(14, max(2, image.height // 6)),
            fill=255,
        )
        image.putalpha(mask)
        rendered = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=(image.width, image.height),
        )
        _REFERENCE_CACHE[cache_key] = rendered
        return rendered
    except (OSError, ValueError):
        return None


def _calibration_reference_path(task: str) -> Path | None:
    """Resolve the same original asset used by the reference thumbnail."""
    filename = CALIBRATION_REFERENCE_FILES.get(task)
    return CALIBRATION_ASSETS / filename if filename else None


def _sample_text_color(rgb: tuple[int, int, int]) -> str:
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "#07111f" if luminance >= 145 else "#ffffff"


def _sample_preview_rgb(
    image_bgr: np.ndarray, profile: CalibrationProfile, selected: str,
) -> np.ndarray:
    """Render selected sample matches without modifying the captured screenshot."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if selected not in SAMPLE_GROUPS or not profile.meter_roi.valid:
        return rgb
    height, width = image_bgr.shape[:2]
    left, top, right, bottom = profile.meter_roi.pixels(width, height)
    roi = image_bgr[top:bottom, left:right]
    mask = color_mask_bgr(roi, profile.samples[selected])
    if mask.any():
        view = rgb[top:bottom, left:right].copy()
        view[mask] = (232, 80, 222)
        rgb[top:bottom, left:right] = (
            rgb[top:bottom, left:right].astype(np.float32) * 0.42
            + view.astype(np.float32) * 0.58
        ).astype(np.uint8)
    return rgb


def speed_scroll(frame: ctk.CTkScrollableFrame, factor: int = 2) -> None:
    if sys.platform == "darwin":
        # CTk already scales native trackpad deltas. A second global handler
        # would double-scroll every Mac gesture; zoom uses wheel_steps below.
        return
    canvas = getattr(frame, "_parent_canvas", None)
    if canvas is None:
        return

    remainder = 0.0

    def on_wheel(event: tk.Event) -> None:
        nonlocal remainder
        try:
            widget = frame.winfo_containing(event.x_root, event.y_root)
        except Exception:
            return
        while widget is not None:
            if widget is frame:
                remainder += factor * wheel_steps(event)
                units = int(remainder)
                remainder -= units
                if units:
                    canvas.yview_scroll(-units, "units")
                return
            widget = getattr(widget, "master", None)

    frame.bind_all("<MouseWheel>", on_wheel, add=True)
    frame.bind_all("<Button-4>", on_wheel, add=True)
    frame.bind_all("<Button-5>", on_wheel, add=True)


class Card(ctk.CTkFrame):
    def __init__(self, master: Any, title: str, hint: str, number: int | None = None) -> None:
        super().__init__(master, fg_color=BG_CARD, corner_radius=18,
                         border_width=1, border_color=BORDER)
        self.grid_columnconfigure(0, weight=1)
        title_row = ctk.CTkFrame(self, fg_color="transparent")
        title_row.grid(row=0, column=0, sticky="ew", padx=18, pady=(15, 1))
        title_row.grid_columnconfigure(1, weight=1)
        if number is not None:
            ctk.CTkLabel(
                title_row, text=f"{number:02d}", width=28, height=24,
                corner_radius=8, fg_color="#16314c", text_color=ACCENT,
                font=ctk.CTkFont(size=11, weight="bold"),
            ).grid(row=0, column=0, padx=(0, 10))
        ctk.CTkLabel(
            title_row, text=title, text_color=TEXT, anchor="w", justify="left",
            wraplength=520, font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=1, sticky="ew")
        if hint:
            ctk.CTkLabel(
                self, text=hint, text_color=MUTED, anchor="w", justify="left",
                wraplength=560, font=ctk.CTkFont(size=12),
            ).grid(row=1, column=0, sticky="ew", padx=18, pady=(1, 10))
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 16))


class PreparationItem(ctk.CTkFrame):
    PRIORITY_COLORS = {"required": DANGER, "important": WARNING, "helpful": ACCENT}

    def __init__(
        self, master: Any, index: int, title: str, priority: str,
        description: str, steps: tuple[str, ...], check: str,
        *, expanded: bool = False,
    ) -> None:
        super().__init__(master, fg_color=BG_CARD, corner_radius=16,
                         border_width=1, border_color=BORDER)
        self._expanded = False
        summary = ctk.CTkFrame(self, fg_color="transparent")
        summary.pack(fill="x", padx=14, pady=11)
        ctk.CTkLabel(
            summary, text=f"{index:02d}", width=31, height=27,
            corner_radius=8, fg_color="#16314c", text_color=ACCENT,
            font=ctk.CTkFont(size=11, weight="bold"),
        ).pack(side="left", padx=(0, 10))
        words = ctk.CTkFrame(summary, fg_color="transparent")
        words.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(
            words, text=priority.upper(), text_color=self.PRIORITY_COLORS[priority],
            anchor="w", font=ctk.CTkFont(size=9, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            words, text=title, text_color=TEXT, anchor="w", justify="left",
            wraplength=760, font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w")
        self.arrow = ctk.CTkButton(
            summary, text="⌄", width=34, height=30, fg_color=BG_CARD_RAISED,
            hover_color="#203858", command=self.toggle,
        )
        self.arrow.pack(side="right")
        self.body = ctk.CTkFrame(self, fg_color=BG_SOFT, corner_radius=12)
        ctk.CTkLabel(
            self.body, text=description, text_color=MUTED, anchor="w",
            justify="left", wraplength=900,
        ).pack(fill="x", padx=14, pady=(12, 8))
        for step in steps:
            ctk.CTkLabel(
                self.body, text=f"•  {step}", text_color=TEXT, anchor="w",
                justify="left", wraplength=890,
            ).pack(fill="x", padx=14, pady=2)
        ctk.CTkLabel(
            self.body, text=f"CHECK  ·  {check}", text_color=SUCCESS,
            anchor="w", justify="left", wraplength=890,
            font=ctk.CTkFont(size=11, weight="bold"),
        ).pack(fill="x", padx=14, pady=(9, 12))
        if expanded:
            self.toggle()

    def toggle(self) -> None:
        self._expanded = not self._expanded
        if self._expanded:
            self.body.pack(fill="x", padx=10, pady=(0, 10))
            self.arrow.configure(text="⌃")
            self.configure(border_color="#365f88")
        else:
            self.body.pack_forget()
            self.arrow.configure(text="⌄")
            self.configure(border_color=BORDER)


class DebugOverlay:
    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        self.parts: dict[str, tk.Toplevel] = {}
        self.labels: dict[str, tk.Label] = {}
        self._active = False

    @property
    def visible(self) -> bool:
        return self._active

    def hide(self) -> None:
        for part in self.parts.values():
            try:
                part.destroy()
            except tk.TclError:
                pass
        self.parts.clear()
        self.labels.clear()
        self._active = False

    def show(self, _window: WindowInfo) -> None:
        self.hide()
        self._active = True

    @staticmethod
    def _geometry(
        window: WindowInfo, profile: CalibrationProfile,
        observation: MeterObservation,
    ) -> dict[str, tuple[int, int, int, int]]:
        left, top, right, bottom = profile.meter_roi.pixels(window.width, window.height)
        x0 = window.left + left - 3
        y0 = window.top + top - 3
        x1 = window.left + right + 3
        y1 = window.top + bottom + 3
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        layout = {
            "outline_top": (x0, y0, width, 2),
            "outline_bottom": (x0, y1 - 2, width, 2),
            "outline_left": (x0, y0, 2, height),
            "outline_right": (x1 - 2, y0, 2, height),
            "status": (max(window.left, x0 - 140), max(window.top, y0 - 26), 136, 22),
        }
        if observation.zone_bounds:
            zone_top, zone_bottom = observation.zone_bounds
            layout["zone_top"] = (
                window.left + right + 6, window.top + top + zone_top - 1, 12, 2,
            )
            layout["zone_bottom"] = (
                window.left + right + 6, window.top + top + zone_bottom - 1, 12, 2,
            )
        if observation.marker_center is not None:
            layout["marker"] = (
                window.left + left - 18,
                int(round(window.top + top + observation.marker_center)) - 1,
                13,
                3,
            )
        return layout

    def _part(self, name: str, *, with_label: bool = False) -> tk.Toplevel:
        existing = self.parts.get(name)
        if existing is not None and existing.winfo_exists():
            return existing
        top = tk.Toplevel(self.root)
        top.overrideredirect(True)
        top.attributes("-topmost", True)
        top.withdraw()
        if with_label:
            label = tk.Label(top, borderwidth=0, padx=4, pady=0,
                             font=(CANVAS_FONT, 9, "bold"))
            label.pack(fill="both", expand=True)
            self.labels[name] = label
        self.parts[name] = top
        top.update_idletasks()
        if hasattr(ctypes, "windll"):
            hwnd = top.winfo_id()
            get_long = ctypes.windll.user32.GetWindowLongW
            set_long = ctypes.windll.user32.SetWindowLongW
            exstyle = get_long(hwnd, -20)
            set_long(hwnd, -20, exstyle | 0x20 | 0x80)
        return top

    def _place(
        self, name: str, geometry: tuple[int, int, int, int], color: str,
        *, text: str | None = None, text_color: str = "#ffffff",
    ) -> None:
        top = self._part(name, with_label=text is not None)
        x, y, width, height = geometry
        top.configure(bg=color)
        if text is not None:
            self.labels[name].configure(text=text, bg=color, fg=text_color)
        top.geometry(f"{max(1, width)}x{max(1, height)}{x:+d}{y:+d}")
        top.deiconify()
        top.lift()

    def update(
        self, window: WindowInfo, profile: CalibrationProfile,
        observation: MeterObservation, snapshot: EngineSnapshot,
    ) -> None:
        if not self.visible:
            return
        layout = self._geometry(window, profile, observation)
        color = SUCCESS if observation.present else DANGER
        used = set(layout)
        for name in ("outline_top", "outline_bottom", "outline_left", "outline_right"):
            self._place(name, layout[name], color)
        for name in ("zone_top", "zone_bottom"):
            if name in layout:
                self._place(name, layout[name], SUCCESS)
        if "marker" in layout:
            self._place("marker", layout["marker"], "#ffffff")
        self._place(
            "status", layout["status"], "#07111f",
            text=f"{snapshot.state.value}  {observation.confidence:.0%}", text_color=color,
        )
        for name, part in self.parts.items():
            if name not in used:
                part.withdraw()


class CooldownEditor(ctk.CTkToplevel):
    def __init__(self, master: SharkmanApp) -> None:
        super().__init__(master)
        self.master_app = master
        self.title(f"SharkmanV3 v{__version__} · Advanced cooldowns")
        self.geometry("820x760")
        self.minsize(680, 520)
        self.configure(fg_color=APP_BG)
        self.transient(master)
        self.entries: dict[str, ctk.CTkEntry] = {}
        shell = ctk.CTkFrame(self, fg_color="transparent")
        shell.pack(fill="both", expand=True, padx=20, pady=18)
        head = ctk.CTkFrame(shell, fg_color=BG_CARD, corner_radius=18,
                            border_width=1, border_color=BORDER)
        head.pack(fill="x", pady=(0, 12))
        words = ctk.CTkFrame(head, fg_color="transparent")
        words.pack(side="left", fill="x", expand=True, padx=17, pady=12)
        ctk.CTkLabel(words, text="ADVANCED COOLDOWNS", text_color=ACCENT,
                     anchor="w", font=ctk.CTkFont(size=10, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(words, text="Every wait and reaction setting", text_color=TEXT,
                     anchor="w", font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w")
        ctk.CTkButton(head, text="Reset all", width=100, command=self.reset_all,
                      fg_color=BG_CARD_RAISED, hover_color="#203858").pack(side="right", padx=14)
        body = ctk.CTkScrollableFrame(
            shell, fg_color=BG_SOFT, corner_radius=18, border_width=1,
            border_color=BORDER,
        )
        body.pack(fill="both", expand=True)
        speed_scroll(body)
        adaptive = ctk.CTkFrame(body, fg_color=BG_CARD, corner_radius=12)
        adaptive.pack(fill="x", padx=10, pady=(12, 5))
        available = LIVE_ADAPTIVE_FEEDBACK_SUPPORTED and sys.platform == "win32"
        self.adaptive_var = ctk.BooleanVar(value=master.profile.adaptive.enabled and available)
        self.adaptive_switch = ctk.CTkSwitch(
            adaptive, text="Adaptive timing — disabled",
            variable=self.adaptive_var, command=self._toggle_adaptive,
            state="normal" if available else "disabled",
            text_color=TEXT, progress_color=SUCCESS,
        )
        self.adaptive_switch.pack(anchor="w", padx=14, pady=(12, 4))
        self.adaptive_status = ctk.CTkLabel(
            adaptive, text="", text_color=MUTED, anchor="w", justify="left",
            wraplength=690,
        )
        self.adaptive_status.pack(fill="x", padx=14, pady=(0, 5))
        adaptive_actions = ctk.CTkFrame(adaptive, fg_color="transparent")
        adaptive_actions.pack(fill="x", padx=14, pady=(0, 12))
        ctk.CTkButton(
            adaptive_actions, text="Undo last adjustment", width=160,
            command=self._undo_adaptive, fg_color=BG_CARD_RAISED,
            hover_color="#203858", text_color=TEXT,
            state="normal" if available else "disabled",
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            adaptive_actions, text="Reset learning", width=130,
            command=self._reset_adaptive, fg_color=BG_CARD_RAISED,
            hover_color="#4c2a38", text_color=TEXT,
            state="normal" if available else "disabled",
        ).pack(side="left")
        self._refresh_adaptive_status()
        timing = master.profile.timing
        for group, specs in timing_groups():
            ctk.CTkLabel(body, text=group.upper(), text_color=ACCENT,
                         anchor="w", font=ctk.CTkFont(size=10, weight="bold")).pack(
                             fill="x", padx=16, pady=(16, 5))
            for spec in specs:
                is_input_latency = spec.name == "input_latency_ms"
                row = ctk.CTkFrame(
                    body, fg_color="#292315" if is_input_latency else BG_CARD,
                    corner_radius=12, border_width=2 if is_input_latency else 0,
                    border_color=WARNING if is_input_latency else BORDER,
                )
                row.pack(fill="x", padx=10, pady=4)
                row_words = ctk.CTkFrame(row, fg_color="transparent")
                row_words.pack(side="left", fill="x", expand=True, padx=13, pady=9)
                ctk.CTkLabel(row_words, text=spec.label,
                             text_color=WARNING if is_input_latency else TEXT, anchor="w",
                             font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
                ctk.CTkLabel(row_words, text=spec.help, text_color=MUTED, anchor="w",
                             justify="left", wraplength=500,
                             font=ctk.CTkFont(size=10)).pack(anchor="w")
                entry = ctk.CTkEntry(row, width=92)
                if is_input_latency:
                    entry.configure(border_color=WARNING, text_color=TEXT)
                entry.insert(0, f"{getattr(timing, spec.name):g}")
                entry.pack(side="left", padx=5)
                self.entries[spec.name] = entry
                ctk.CTkLabel(row, text=spec.unit, width=32,
                             text_color=MUTED).pack(side="left")
                ctk.CTkButton(
                    row, text="Reset", width=64,
                    command=lambda name=spec.name: self.reset_one(name),
                    fg_color=BG_CARD_RAISED, hover_color="#203858",
                ).pack(side="left", padx=(3, 12))
        foot = ctk.CTkFrame(shell, fg_color="transparent")
        foot.pack(fill="x", pady=(12, 0))
        self.message = ctk.CTkLabel(
            foot, text="Only values changed from defaults are saved.", text_color=MUTED)
        self.message.pack(side="left")
        ctk.CTkButton(
            foot, text="Save", width=120, command=self.save, fg_color=ACCENT,
            hover_color=ACCENT_HOVER, text_color="#07111f",
            font=ctk.CTkFont(weight="bold"),
        ).pack(side="right")

    def _refresh_adaptive_status(self) -> None:
        state = self.master_app.profile.adaptive
        if not LIVE_ADAPTIVE_FEEDBACK_SUPPORTED:
            message = (
                "Adaptive timing is disabled: live runs did not provide a reliable frozen meter position. "
                "Saved 'enabled' settings are ignored; no automatic cooldown changes occur. "
                "Manual cooldowns still work."
            )
        elif sys.platform != "win32":
            message = "Linux/Sober adaptation remains disabled until its live acceptance test."
        else:
            progress = state.sample_count % LEARNING_BATCH
            if state.sample_count == 0:
                last = state.last_reason
                advice = (
                    " More rounds alone will not help while the target changes after Space."
                    if "Target changed" in last else ""
                )
                message = (
                    f"Progress 0/{LEARNING_BATCH}: no eligible frozen positions were saved, "
                    f"so latency remains {self.master_app.profile.timing.input_latency_ms:g} ms. "
                    f"Last status: {last}{advice}"
                )
            else:
                message = (
                    f"Progress {progress}/{LEARNING_BATCH} this batch · {state.sample_count} eligible total · "
                    f"saved latency {self.master_app.profile.timing.input_latency_ms:g} ms. "
                    f"Last: {state.last_reason} Frozen position is not a game hit grade."
                )
        self.adaptive_status.configure(
            text=message,
            text_color=(WARNING if LIVE_ADAPTIVE_FEEDBACK_SUPPORTED
                        and state.enabled and state.sample_count == 0 else MUTED),
        )

    def _toggle_adaptive(self) -> None:
        if not LIVE_ADAPTIVE_FEEDBACK_SUPPORTED or sys.platform != "win32":
            self.adaptive_var.set(False)
            return
        profile = self.master_app.profile
        profile.adaptive.enabled = bool(self.adaptive_var.get())
        if profile.adaptive.baseline_latency_ms is None:
            profile.adaptive.baseline_latency_ms = profile.timing.input_latency_ms
        self.master_app.store.save(profile)
        self._refresh_adaptive_status()
        self.message.configure(text="Experimental visual centering saved for this profile.", text_color=SUCCESS)

    def _undo_adaptive(self) -> None:
        if not LIVE_ADAPTIVE_FEEDBACK_SUPPORTED:
            return
        profile = self.master_app.profile
        previous = profile.adaptive.previous_latency_ms
        if previous is None:
            self.message.configure(text="No saved adaptive adjustment to undo.", text_color=WARNING)
            return
        timing = profile.timing
        timing.set("input_latency_ms", previous)
        profile.timing_overrides = timing.overrides()
        profile.adaptive.clear(previous)
        self.master_app.store.save(profile)
        self.reset_one("input_latency_ms")
        self.entries["input_latency_ms"].delete(0, "end")
        self.entries["input_latency_ms"].insert(0, f"{previous:g}")
        self._refresh_adaptive_status()
        self.message.configure(text="Previous latency restored; learning evidence cleared.", text_color=SUCCESS)

    def _reset_adaptive(self) -> None:
        if not LIVE_ADAPTIVE_FEEDBACK_SUPPORTED:
            return
        profile = self.master_app.profile
        baseline = profile.adaptive.baseline_latency_ms
        if baseline is None:
            baseline = profile.timing.input_latency_ms
        timing = profile.timing
        timing.set("input_latency_ms", baseline)
        profile.timing_overrides = timing.overrides()
        profile.adaptive.clear(baseline)
        self.master_app.store.save(profile)
        self.entries["input_latency_ms"].delete(0, "end")
        self.entries["input_latency_ms"].insert(0, f"{baseline:g}")
        self._refresh_adaptive_status()
        self.message.configure(text="Manual baseline restored; learning reset.", text_color=SUCCESS)

    def reset_one(self, name: str) -> None:
        entry = self.entries[name]
        entry.delete(0, "end")
        entry.insert(0, f"{TIMING_BY_NAME[name].default:g}")

    def reset_all(self) -> None:
        self._reset_all_requested = True
        for name in self.entries:
            self.reset_one(name)
        self.message.configure(
            text="Defaults restored in the editor. Press Save to apply.",
            text_color=WARNING,
        )

    def save(self) -> None:
        timing = TimingConfig()
        try:
            for name, entry in self.entries.items():
                timing.set(name, float(entry.get().strip()))
        except (ValueError, KeyError) as exc:
            self.message.configure(text=str(exc), text_color=DANGER)
            return
        profile = self.master_app.profile
        changed_latency = timing.input_latency_ms != profile.timing.input_latency_ms
        profile.timing_overrides = timing.overrides()
        if changed_latency or getattr(self, "_reset_all_requested", False):
            profile.adaptive.clear(timing.input_latency_ms, disable=getattr(self, "_reset_all_requested", False))
            self.adaptive_var.set(profile.adaptive.enabled and LIVE_ADAPTIVE_FEEDBACK_SUPPORTED)
        self._reset_all_requested = False
        self.master_app.store.save(profile)
        self._refresh_adaptive_status()
        self.message.configure(text="Cooldowns saved.", text_color=SUCCESS)


class Calibrator(ctk.CTkToplevel):
    TASKS = {
        "region": (
            "REGION", "Complete timing meter",
            "Drag around the full vertical red/yellow/green gauge.",
            "Leave a small margin around the gauge and include its full height.",
            "Do not include the character, HUD, or nearby scenery.",
            "The outline contains only the complete timing meter.",
        ),
        "green": (
            "SAMPLE", "Green target",
            "Click a plain green pixel inside the target band.",
            "Click away from outlines; extra clicks add lighting variants.",
            "Do not sample the black marker or yellow transition.",
            "The magenta preview covers the green target and little else.",
        ),
        "marker": (
            "SAMPLE", "Black marker",
            "Click the solid black moving horizontal marker.",
            "Capture it while it is clearly visible across the gauge.",
            "Do not click a shadow, border, or dark scenery pixel.",
            "The magenta preview follows the black bar across the meter.",
        ),
        "track": (
            "SAMPLE", "Track / gradient",
            "Click a saturated red or yellow part of the vertical track.",
            "Add another sample when lighting changes the track color.",
            "Do not sample the background outside the meter region.",
            "The magenta preview stays on the colored gauge track.",
        ),
    }
    HANDLE = 14

    @property
    def game_name(self) -> str:
        return getattr(getattr(self, "master_app", None), "game_name", "Roblox")

    def __init__(self, master: SharkmanApp) -> None:
        super().__init__(master)
        self.master_app = master
        self.profile = CalibrationProfile.from_dict(master.profile.to_dict())
        self.saved_signature = (
            self.profile.validation_signature if self.profile.validation_current else ""
        )
        self.title(f"SharkmanV3 v{__version__} · Calibrate")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{min(1440, max(900, screen_width - 80))}x"
                      f"{min(900, max(520, screen_height - 100))}")
        self.minsize(900, 520)
        self.configure(fg_color=APP_BG)
        self.selected = "region"
        self.selected_sample = 0
        self.window: WindowInfo | None = None
        self.image_bgr: np.ndarray | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.scale = 1.0
        self.offset = (0, 0)
        self.drag: dict[str, Any] | None = None
        self.observation: MeterObservation | None = None
        self._inspector: ImageInspector | None = None
        self._reference_key = self.selected
        self._compact_reference: bool | None = None
        self._setting_tolerance = False
        self._setting_inset = False
        self._build()
        self.bind("<Configure>", self._layout_for_size, add="+")
        self.after(0, self.shoot)

    def _build(self) -> None:
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=20, pady=18)
        outer.grid_columnconfigure(0, minsize=300)
        outer.grid_columnconfigure(1, weight=1)
        outer.grid_rowconfigure(1, weight=1)
        head = ctk.CTkFrame(outer, fg_color=BG_CARD, corner_radius=18,
                            border_width=1, border_color=BORDER)
        head.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        brand = ctk.CTkFrame(head, fg_color="transparent")
        brand.pack(side="left", padx=18, pady=12)
        ctk.CTkLabel(
            brand, text="✓", width=36, height=36, corner_radius=12,
            fg_color=ACCENT, text_color="#07111f",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(side="left")
        words = ctk.CTkFrame(brand, fg_color="transparent")
        words.pack(side="left", padx=11)
        ctk.CTkLabel(words, text="Calibration workspace", text_color=TEXT,
                     anchor="w", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(
            words,
            text="Capture the meter, place its region, then click all three required colors.",
            text_color=MUTED, anchor="w", font=ctk.CTkFont(size=11),
        ).pack(anchor="w")
        progress_area = ctk.CTkFrame(head, fg_color="transparent")
        progress_area.pack(side="right", padx=18, pady=13)
        self.progress_text = ctk.CTkLabel(
            progress_area, text="0 / 4 required steps complete",
            text_color="#b9d8eb", anchor="e",
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        self.progress_text.pack(anchor="e")
        self.progress = ctk.CTkProgressBar(
            progress_area, width=230, height=7, fg_color="#17243a",
            progress_color=SUCCESS,
        )
        self.progress.pack(pady=(5, 0))
        left_shell = ctk.CTkFrame(
            outer, fg_color=BG_CARD, corner_radius=20, border_width=1,
            border_color=BORDER,
        )
        left_shell.grid(row=1, column=0, sticky="nsew", padx=(0, 14))
        ctk.CTkLabel(left_shell, text="CALIBRATION MAP", text_color=ACCENT,
                     anchor="w", font=ctk.CTkFont(size=11, weight="bold")).pack(
                         fill="x", padx=16, pady=(16, 2))
        ctk.CTkLabel(left_shell, text="Choose one task at a time", text_color=TEXT,
                     anchor="w", font=ctk.CTkFont(size=15, weight="bold")).pack(
                         fill="x", padx=16)
        ctk.CTkLabel(
            left_shell,
            text="REGION = drag or resize · SAMPLE = click the screenshot",
            text_color=MUTED, anchor="w", justify="left", wraplength=260,
            font=ctk.CTkFont(size=10),
        ).pack(fill="x", padx=16, pady=(2, 10))
        nav = ctk.CTkScrollableFrame(
            left_shell, fg_color=BG_SOFT, corner_radius=14,
            scrollbar_button_color="#345074", scrollbar_button_hover_color=ACCENT_HOVER,
        )
        nav.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        speed_scroll(nav)
        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        for key, (kind, label, *_rest) in self.TASKS.items():
            button = ctk.CTkButton(
                nav, text=f"{kind}   {label}", anchor="w", height=42,
                corner_radius=10, fg_color="transparent", hover_color="#183353",
                text_color="#d7e8f7", font=ctk.CTkFont(size=11),
                command=lambda task=key: self.select(task),
            )
            button.pack(fill="x", padx=4, pady=3)
            self.nav_buttons[key] = button
        right = ctk.CTkScrollableFrame(
            outer, fg_color="transparent",
            scrollbar_button_color="#345074",
            scrollbar_button_hover_color=ACCENT_HOVER,
        )
        right.grid(row=1, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        speed_scroll(right)
        bar = ctk.CTkFrame(right, fg_color=BG_CARD, corner_radius=16,
                           border_width=1, border_color=BORDER)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 9))
        bar.grid_columnconfigure(0, weight=1)
        context = ctk.CTkFrame(bar, fg_color="transparent")
        context.grid(row=0, column=0, sticky="ew", padx=15, pady=(9, 0))
        ctk.CTkLabel(context, text="LIVE WORKSPACE", text_color=ACCENT,
                     anchor="w", font=ctk.CTkFont(size=10, weight="bold")).pack(anchor="w")
        self.interaction = ctk.CTkLabel(
            context, text=f"Capturing {self.game_name}…", text_color=MUTED, anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"), justify="left",
            wraplength=760,
        )
        self.interaction.pack(anchor="w", pady=(1, 0))
        bar_actions = ctk.CTkFrame(bar, fg_color="transparent")
        bar_actions.grid(row=1, column=0, sticky="ew", padx=12, pady=(6, 9))
        self.save_button = ctk.CTkButton(
            bar_actions, text="Save calibration", width=142, height=34, state="disabled",
            fg_color="#1b3652", hover_color=ACCENT_HOVER, text_color=TEXT,
            text_color_disabled="#d5e6f4",
            font=ctk.CTkFont(size=12, weight="bold"), command=self.save,
        )
        self.save_button.pack(side="right", padx=(7, 0))
        ctk.CTkButton(
            bar_actions, text="Re-shoot", width=90, height=34, fg_color=BG_CARD_RAISED,
            hover_color="#203858", border_width=1, border_color=BORDER,
            text_color=TEXT, command=self.shoot,
        ).pack(side="right", padx=(6, 0))
        ctk.CTkButton(
            bar_actions, text="Reset selected", width=112, height=34,
            fg_color=BG_CARD_RAISED, hover_color="#4c2a38", border_width=1,
            border_color=BORDER, text_color=TEXT, command=self.reset_selected,
        ).pack(side="right")
        canvas_shell = ctk.CTkFrame(
            right, height=380, fg_color="#050d1a", corner_radius=18, border_width=1,
            border_color=BORDER,
        )
        self.canvas_shell = canvas_shell
        canvas_shell.grid(row=1, column=0, sticky="ew")
        canvas_shell.grid_propagate(False)
        canvas_shell.grid_rowconfigure(1, weight=1)
        canvas_shell.grid_columnconfigure(0, weight=1)
        zoom_toolbar = ctk.CTkFrame(canvas_shell, fg_color="transparent")
        zoom_toolbar.grid(row=0, column=0, sticky="ew", padx=10, pady=(7, 0))
        self.zoom_shot_btn = ctk.CTkButton(
            zoom_toolbar, text="Zoom screenshot", width=140, height=29,
            state="disabled", fg_color=BG_CARD_RAISED, hover_color="#244765",
            text_color=TEXT, text_color_disabled=MUTED,
            command=self._inspect_screenshot,
        )
        self.zoom_shot_btn.pack(side="left", padx=(0, 7))
        self.zoom_ref_btn = ctk.CTkButton(
            zoom_toolbar, text="Zoom reference", width=140, height=29,
            state="disabled", fg_color=BG_CARD_RAISED, hover_color="#244765",
            text_color=TEXT, text_color_disabled=MUTED,
            command=self._inspect_reference,
        )
        self.zoom_ref_btn.pack(side="left")
        self.canvas = tk.Canvas(canvas_shell, bg="#08111f", highlightthickness=0,
                                bd=0, cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())
        self.canvas.bind("<Button-1>", self._down)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", self._up)
        detail = ctk.CTkFrame(right, fg_color=BG_CARD, corner_radius=18,
                              border_width=1, border_color=BORDER)
        self.detail_panel = detail
        detail.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        detail.grid_columnconfigure(0, weight=1)
        detail.grid_columnconfigure(1, minsize=276)
        self.eyebrow = ctk.CTkLabel(
            detail, text="GUIDED CALIBRATION", text_color=ACCENT, anchor="w",
            font=ctk.CTkFont(size=10, weight="bold"),
        )
        self.eyebrow.grid(row=0, column=0, sticky="ew", padx=16, pady=(13, 0))
        self.detail_title = ctk.CTkLabel(
            detail, text="Complete timing meter", text_color=TEXT, anchor="w",
            font=ctk.CTkFont(size=17, weight="bold"),
        )
        self.detail_title.grid(row=1, column=0, sticky="ew", padx=16, pady=(2, 0))
        self.detail_text = ctk.CTkLabel(
            detail, text="", text_color=MUTED, anchor="w", justify="left",
            wraplength=720,
        )
        self.detail_text.grid(row=2, column=0, sticky="ew", padx=16, pady=(3, 6))
        self.do_label = ctk.CTkLabel(
            detail, text="", text_color=SUCCESS, anchor="w", justify="left",
            wraplength=720, font=ctk.CTkFont(size=11, weight="bold"),
        )
        self.do_label.grid(row=3, column=0, sticky="ew", padx=16, pady=2)
        self.avoid_label = ctk.CTkLabel(
            detail, text="", text_color=WARNING, anchor="w", justify="left",
            wraplength=720, font=ctk.CTkFont(size=11, weight="bold"),
        )
        self.avoid_label.grid(row=4, column=0, sticky="ew", padx=16, pady=2)
        self.check_label = ctk.CTkLabel(
            detail, text="", text_color=ACCENT, anchor="w", justify="left",
            wraplength=720, font=ctk.CTkFont(size=11, weight="bold"),
        )
        self.check_label.grid(row=5, column=0, sticky="ew", padx=16, pady=2)
        self.controls = ctk.CTkFrame(detail, fg_color="transparent")
        self.controls.grid(row=6, column=0, sticky="ew", padx=16, pady=(6, 2))
        self.reference_shell = ctk.CTkFrame(
            detail, fg_color="#0b1728", corner_radius=13,
            border_width=1, border_color="#285073",
        )
        self.reference_shell.grid(
            row=0, column=1, rowspan=8, sticky="nsew", padx=(8, 14), pady=12)
        ctk.CTkLabel(
            self.reference_shell, text="REFERENCE EXAMPLE", text_color="#9ee4ff",
            font=ctk.CTkFont(size=9, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(9, 3))
        self.reference_image = ctk.CTkLabel(
            self.reference_shell, text="", text_color="#d7e8f7",
            justify="center", font=ctk.CTkFont(size=11, weight="bold"),
        )
        self.reference_image.pack(fill="both", expand=True, padx=10, pady=(0, 4))
        self.reference_image.bind("<Button-1>", lambda _event: self._inspect_reference())
        self.reference_note = ctk.CTkLabel(
            self.reference_shell, text="", text_color="#abc2d8",
            justify="center", wraplength=244, font=ctk.CTkFont(size=10),
        )
        self.reference_note.pack(fill="x", padx=10, pady=(0, 9))
        action_row = ctk.CTkFrame(detail, fg_color="transparent")
        action_row.grid(row=7, column=0, sticky="ew", padx=16, pady=(8, 13))
        self.test_button = ctk.CTkButton(
            action_row, text="Test detection on screenshot", width=220,
            command=self.test_detection, fg_color="#17563a", hover_color="#20714a",
        )
        self.test_button.pack(side="left")
        self.result_label = ctk.CTkLabel(
            action_row, text="Not tested", text_color=MUTED, anchor="w")
        self.result_label.pack(side="left", padx=12)
        self.select("region")

    def select(self, task: str) -> None:
        self.selected = task
        self._reference_key = task
        self.selected_sample = 0
        kind, label, description, do, avoid, check = self.TASKS[task]
        self.eyebrow.configure(
            text=f"{kind} CALIBRATION",
            text_color=ACCENT if task == "region" else COLOR_ACCENT,
        )
        self.detail_title.configure(text=label)
        self.detail_text.configure(text=description)
        self.do_label.configure(text=f"DO THIS  ·  {do}")
        self.avoid_label.configure(text=f"AVOID  ·  {avoid}")
        self.check_label.configure(text=f"CHECK  ·  {check}")
        self._build_reference()
        self._build_controls()
        self._refresh_state()
        self.redraw()

    def _build_controls(self) -> None:
        for child in self.controls.winfo_children():
            child.destroy()
        if self.selected == "region":
            row = ctk.CTkFrame(self.controls, fg_color="transparent")
            row.pack(fill="x")
            ctk.CTkLabel(
                row, text="SAFE-ZONE INSET", text_color=ACCENT,
                font=ctk.CTkFont(size=9, weight="bold"),
            ).pack(side="left")
            self.inset_value = ctk.CTkLabel(
                row, text=f"{self.profile.detection.safe_zone_inset:.0%}",
                text_color=TEXT, font=ctk.CTkFont(size=15, weight="bold"),
            )
            self.inset_value.pack(side="left", padx=10)
            slider = ctk.CTkSlider(
                row, from_=0.0, to=0.45, number_of_steps=45,
                width=180, command=self._set_inset,
            )
            self._setting_inset = True
            slider.set(self.profile.detection.safe_zone_inset)
            self._setting_inset = False
            slider.pack(side="left", padx=(0, 12))
            ctk.CTkLabel(
                self.controls,
                text="Drag inside the rectangle to move it.\n"
                     "Drag the bright lower-right grip to resize.",
                text_color="#b9cde0", justify="left", wraplength=300,
            ).pack(anchor="w", pady=(3, 0))
            return
        samples = self.profile.samples[self.selected]
        sample_row = ctk.CTkFrame(self.controls, fg_color="transparent")
        sample_row.pack(fill="x")
        ctk.CTkLabel(
            sample_row, text=f"COLOR VARIANTS  ·  {len(samples)}", text_color=COLOR_ACCENT,
            font=ctk.CTkFont(size=9, weight="bold"),
        ).pack(side="left", padx=(0, 8))
        for index, sample in enumerate(samples):
            color = "#%02x%02x%02x" % sample.rgb
            invalid = color_sample_error(self.selected, sample.rgb) is not None
            swatch = ctk.CTkButton(
                sample_row, text=f"!{index + 1}" if invalid else str(index + 1),
                width=36 if invalid else 32, height=28,
                fg_color=color, hover_color=color,
                text_color=_sample_text_color(sample.rgb), border_width=2,
                border_color=DANGER if invalid else
                "#ffffff" if index == self.selected_sample else BORDER,
                command=lambda selected=index: self._select_sample(selected),
            )
            swatch.pack(side="left", padx=2)
        if not samples:
            ctk.CTkLabel(
                sample_row, text="Click inside the meter to add the required color.",
                text_color="#c5d7e9",
            ).pack(side="left")
        adjustment_row = ctk.CTkFrame(self.controls, fg_color="transparent")
        adjustment_row.pack(fill="x", pady=(4, 0))
        self._setting_tolerance = True
        current = samples[self.selected_sample] if samples else ColorSample((0, 0, 0), 45)
        self.tolerance_value = ctk.CTkLabel(
            adjustment_row,
            text=f"RGB {current.rgb}  ·  tolerance {current.tolerance:g}",
            text_color="#dceaf7",
        )
        self.tolerance_value.pack(side="left", padx=(0, 6))
        self.tolerance_slider = ctk.CTkSlider(
            adjustment_row, from_=1, to=120, number_of_steps=119,
            width=130, command=self._set_tolerance,
        )
        self.tolerance_slider.set(current.tolerance)
        self.tolerance_slider.pack(side="left", padx=(0, 7))
        self._setting_tolerance = False
        ctk.CTkButton(
            adjustment_row, text="Remove",
            state="normal" if samples else "disabled", width=78, height=28,
            fg_color=BG_CARD_RAISED, hover_color="#4c2a38",
            text_color=TEXT, text_color_disabled="#b6c8d9",
            command=self.remove_sample,
        ).pack(side="left")
        invalid_count = sum(
            color_sample_error(self.selected, sample.rgb) is not None
            for sample in samples
        )
        if invalid_count:
            ctk.CTkButton(
                self.controls, text=f"Remove invalid ({invalid_count})",
                width=126, height=28, fg_color="#4b2235", hover_color="#713047",
                text_color="#ffe4e6", command=self.remove_invalid_samples,
            ).pack(anchor="w", pady=(5, 0))

    def _build_reference(self) -> None:
        compact = bool(self._compact_reference)
        image = _calibration_reference(
            self._reference_key, 420 if compact else 250,
            220 if compact else 180,
        )
        filename = CALIBRATION_REFERENCE_FILES[self._reference_key]
        self.zoom_ref_btn.configure(state="normal" if image is not None else "disabled")
        if image is None:
            blank = _blank_reference()
            self.reference_image.configure(
                image=blank,
                text=f"Add your guide image here\n\n{filename}",
                fg_color="#101f34",
                corner_radius=10,
            )
            self.reference_image.image = blank
            self.reference_note.configure(
                text="Visual aid only. Calibration always uses the live screenshot.")
            return
        self.reference_image.configure(image=image, text="", fg_color="transparent")
        self.reference_image.image = image
        self.reference_note.configure(
            text="Match the named game element, not any cursor shown in the image.")

    def _layout_for_size(self, event: tk.Event) -> None:
        if event.widget is not self or not hasattr(self, "reference_shell"):
            return
        compact = event.width < 1200 or event.height < 740
        preview_height = max(380, min(560, event.height - 410))
        if self.canvas_shell.cget("height") != preview_height:
            self.canvas_shell.configure(height=preview_height)
        if compact == self._compact_reference:
            return
        self._compact_reference = compact
        self.detail_panel.grid_columnconfigure(1, minsize=0 if compact else 276)
        if compact:
            self.reference_shell.grid_configure(
                row=8, column=0, rowspan=1, sticky="ew", padx=16, pady=(0, 13),
            )
        else:
            self.reference_shell.grid_configure(
                row=0, column=1, rowspan=8, sticky="nsew",
                padx=(8, 14), pady=12,
            )
        self._build_reference()

    def _close_inspector(self) -> None:
        viewer = getattr(self, "_inspector", None)
        self._inspector = None
        if viewer is not None and viewer.winfo_exists():
            viewer.destroy()

    def _inspect_screenshot(self) -> None:
        if self.image_bgr is None:
            self._set_status(f"Capture {self.game_name} before zooming the screenshot.", WARNING)
            return
        self._close_inspector()
        height, width = self.image_bgr.shape[:2]
        overlay = None
        if self.profile.meter_roi.valid:
            box = self.profile.meter_roi.pixels(width, height)
            overlay = ("box", box, ACCENT, "Complete timing meter")
        source = Image.fromarray(_sample_preview_rgb(
            self.image_bgr, self.profile, self.selected,
        ))
        try:
            self._inspector = ImageInspector(
                self, source, f"{self.game_name} screenshot snapshot", overlay,
                redraw_delay_ms=int(self.profile.timing.inspector_redraw_ms),
            )
        finally:
            source.close()

    def _inspect_reference(self) -> None:
        path = _calibration_reference_path(self._reference_key)
        if path is None:
            self._set_status("No reference image is assigned to this task.", WARNING)
            return
        try:
            with Image.open(path) as source:
                self._close_inspector()
                self._inspector = ImageInspector(
                    self, source, f"Reference: {self.TASKS[self._reference_key][1]}",
                    redraw_delay_ms=int(self.profile.timing.inspector_redraw_ms),
                )
        except (OSError, ValueError) as exc:
            self._set_status(f"Reference image could not be opened: {exc}", WARNING)

    def destroy(self) -> None:
        self._close_inspector()
        super().destroy()

    def _select_sample(self, index: int) -> None:
        self.selected_sample = index
        self._build_controls()
        self.redraw()

    def _set_tolerance(self, value: float) -> None:
        if self._setting_tolerance or self.selected not in SAMPLE_GROUPS:
            return
        samples = self.profile.samples[self.selected]
        if not samples:
            return
        index = min(self.selected_sample, len(samples) - 1)
        samples[index] = ColorSample(samples[index].rgb, int(float(value)))
        self.tolerance_value.configure(
            text=f"RGB {samples[index].rgb}  ·  tolerance {int(float(value))}")
        self.profile.invalidate_validation()
        self.observation = None
        self._refresh_state()
        self.redraw()

    def _set_inset(self, value: float) -> None:
        if self._setting_inset:
            return
        self.profile.detection.safe_zone_inset = float(value)
        if hasattr(self, "inset_value"):
            self.inset_value.configure(text=f"{float(value):.0%}")
        self.profile.invalidate_validation()
        self._refresh_state()
        self.redraw()

    def shoot(self) -> None:
        self._close_inspector()
        if not self.master_app.backend:
            self._set_status(self.master_app.backend_error, DANGER)
            return
        self.withdraw()
        self.master_app.withdraw()
        delay = int(self.profile.timing.calibration_capture_delay_ms)

        def restore() -> None:
            self.master_app.deiconify()
            self.deiconify()
            self.lift()
            self._refresh_state()
            self.redraw()

        def failed(exc: Exception) -> None:
            self.window = None
            self.image_bgr = None
            self.zoom_shot_btn.configure(state="disabled")
            self._set_status(
                f"{self.game_name} capture failed: {exc} Keep the game visible and unobstructed, "
                "then press Re-shoot.", DANGER)
            restore()

        def take(window: WindowInfo) -> None:
            try:
                if not self.master_app.backend.is_foreground(window):
                    raise BackendError(
                        f"{self.game_name} did not remain in the foreground after activation.")
                self.master_app.backend.configure_timing(self.profile.timing)
                image = self.master_app.backend.capture(window)
                if image.size == 0:
                    raise BackendError(f"{self.game_name} returned an empty capture.")
                self.window = window
                self.image_bgr = image
                self.zoom_shot_btn.configure(state="normal")
                self.profile.platform = sys.platform
                self.profile.fingerprint = window.fingerprint
                self.profile.invalidate_validation()
                self.observation = None
                self._set_status(
                    f"Captured {window.title!r} at {window.width}×{window.height}. "
                    "Choose a task on the left.", SUCCESS)
            except Exception as exc:
                failed(exc)
                return
            finally:
                self.master_app.backend.stop_capture()
            restore()

        def activate() -> None:
            try:
                window = self.master_app.backend.find_window(self.profile.window_title)
                if not self.master_app.backend.activate_window(window):
                    raise BackendError(f"Could not bring {self.game_name} to the foreground.")
            except Exception as exc:
                failed(exc)
                return
            self.after(delay, lambda: take(window))

        self.after(0, activate)

    def _set_status(self, text: str, color: str = MUTED) -> None:
        self.interaction.configure(text=text, text_color=color)

    def _image_point(self, event: tk.Event) -> tuple[int, int] | None:
        if self.image_bgr is None or self.scale <= 0:
            return None
        ox, oy = self.offset
        x = int((event.x - ox) / self.scale)
        y = int((event.y - oy) / self.scale)
        height, width = self.image_bgr.shape[:2]
        return (x, y) if 0 <= x < width and 0 <= y < height else None

    def _down(self, event: tk.Event) -> None:
        point = self._image_point(event)
        if point is None or self.image_bgr is None:
            return
        if self.selected in SAMPLE_GROUPS:
            if not self.profile.meter_roi.valid:
                self._set_status("Draw the complete meter region before sampling colors.", WARNING)
                return
            height, width = self.image_bgr.shape[:2]
            left, top, right, bottom = self.profile.meter_roi.pixels(width, height)
            if not (left <= point[0] < right and top <= point[1] < bottom):
                self._set_status("Color samples must be clicked inside the meter region.", WARNING)
                return
            rgb = sample_median_rgb(self.image_bgr, *point)
            error = color_sample_error(self.selected, rgb)
            if error:
                self._set_status(f"Sample rejected: {error}", DANGER)
                return
            if any(sum(abs(a - b) for a, b in zip(sample.rgb, rgb, strict=True)) <= 9
                   for sample in self.profile.samples[self.selected]):
                self._set_status(
                    f"RGB {rgb} is already represented in this group.", WARNING)
                return
            self.profile.samples[self.selected].append(ColorSample(rgb, 45))
            self.selected_sample = len(self.profile.samples[self.selected]) - 1
            self.profile.invalidate_validation()
            self.observation = None
            self._set_status(
                f"Added {self.selected} sample RGB {rgb}. Magenta shows its matches.",
                SUCCESS,
            )
            self._build_controls()
            self._refresh_state()
            self.redraw()
            return
        height, width = self.image_bgr.shape[:2]
        px = self.profile.meter_roi.pixels(width, height)
        left, top, right, bottom = px
        if (self.profile.meter_roi.valid and abs(point[0] - right) <= 18
                and abs(point[1] - bottom) <= 18):
            mode = "resize"
        elif (self.profile.meter_roi.valid and left <= point[0] <= right
              and top <= point[1] <= bottom):
            mode = "move"
        else:
            mode = "new"
        self.drag = {"mode": mode, "start": point, "rect": px}

    def _move(self, event: tk.Event) -> None:
        if not self.drag or self.image_bgr is None:
            return
        point = self._image_point(event)
        if point is None:
            return
        height, width = self.image_bgr.shape[:2]
        sx, sy = self.drag["start"]
        left, top, right, bottom = self.drag["rect"]
        if self.drag["mode"] == "new":
            left, right = sorted((sx, point[0]))
            top, bottom = sorted((sy, point[1]))
        elif self.drag["mode"] == "move":
            dx, dy = point[0] - sx, point[1] - sy
            box_w, box_h = right - left, bottom - top
            left = max(0, min(width - box_w, left + dx))
            top = max(0, min(height - box_h, top + dy))
            right, bottom = left + box_w, top + box_h
        else:
            right = max(left + 8, min(width, point[0]))
            bottom = max(top + 20, min(height, point[1]))
        if right - left >= 8 and bottom - top >= 20:
            self.profile.meter_roi = NormalizedRect(
                left / width, top / height, right / width, bottom / height)
            self.profile.invalidate_validation()
            self.observation = None
            self._refresh_state()
            self.redraw()

    def _up(self, _event: tk.Event) -> None:
        if self.drag:
            self.drag = None
            self._set_status(
                "Meter region updated. Capture all three color samples, then test detection.",
                SUCCESS,
            )

    def remove_sample(self) -> None:
        if self.selected not in SAMPLE_GROUPS:
            return
        samples = self.profile.samples[self.selected]
        if not samples:
            return
        removed = samples.pop(min(self.selected_sample, len(samples) - 1))
        self.selected_sample = max(0, min(self.selected_sample, len(samples) - 1))
        self.profile.invalidate_validation()
        self.observation = None
        self._set_status(
            f"Removed RGB {removed.rgb}. Unsaved change — test detection again, "
            "then save calibration.", WARNING)
        self._build_controls()
        self._refresh_state()
        self.redraw()

    def remove_invalid_samples(self) -> None:
        if self.selected not in SAMPLE_GROUPS:
            return
        samples = self.profile.samples[self.selected]
        kept = [
            sample for sample in samples
            if color_sample_error(self.selected, sample.rgb) is None
        ]
        removed = len(samples) - len(kept)
        if not removed:
            return
        self.profile.samples[self.selected] = kept
        self.selected_sample = max(0, min(self.selected_sample, len(kept) - 1))
        self.profile.invalidate_validation()
        self.observation = None
        self._set_status(
            f"Removed {removed} invalid sample{'s' if removed != 1 else ''}. "
            "Unsaved change — test detection again, then save calibration.", WARNING)
        self._build_controls()
        self._refresh_state()
        self.redraw()

    def reset_selected(self) -> None:
        if self.selected == "region":
            self.profile.meter_roi = NormalizedRect(0.0, 0.0, 0.0, 0.0)
            self.profile.detection.safe_zone_inset = 0.15
        else:
            self.profile.samples[self.selected] = []
            self.selected_sample = 0
        self.profile.invalidate_validation()
        self.observation = None
        self._set_status(
            "Selected calibration item reset. Changes are not saved yet.", WARNING)
        self._build_controls()
        self._refresh_state()
        self.redraw()

    def _completed(self) -> set[str]:
        completed = set()
        if self.profile.meter_roi.valid:
            completed.add("region")
        completed.update(key for key in SAMPLE_GROUPS if self.profile.samples.get(key))
        return completed

    def _refresh_state(self) -> None:
        completed = self._completed()
        self.progress_text.configure(text=f"{len(completed)} / 4 required steps complete")
        self.progress.set(len(completed) / 4)
        for key, button in self.nav_buttons.items():
            kind, label, *_ = self.TASKS[key]
            prefix = "✓" if key in completed else kind
            active = key == self.selected
            button.configure(
                text=f"{prefix}   {label}",
                fg_color="#183353" if active else "transparent",
                text_color=SUCCESS if key in completed and not active else TEXT,
            )
        validated = self.profile.validation_current
        saved = validated and self.saved_signature == self.profile.validation_signature
        self.save_button.configure(
            text="✓ Saved" if saved else
            "Save calibration" if validated else "Test detection to save",
            state="normal" if validated and not saved else "disabled",
            fg_color="#17563a" if saved else ACCENT if validated else "#1b3652",
            text_color="#ffffff" if saved else "#07111f" if validated else TEXT,
            text_color_disabled="#ffffff" if saved else "#d5e6f4",
        )
        if saved:
            self.result_label.configure(
                text=f"Saved to {self.profile.name}", text_color=SUCCESS)
        elif validated:
            self.result_label.configure(text="Validated — ready to save", text_color=SUCCESS)
        elif self.profile.calibration_input_errors():
            self.result_label.configure(text="Complete all four tasks", text_color=MUTED)
        else:
            self.result_label.configure(
                text="Unsaved changes — test detection again", text_color=WARNING)

    def test_detection(self) -> None:
        if self.image_bgr is None:
            self._set_status(f"Re-shoot {self.game_name} while the timing meter is visible.", DANGER)
            return
        errors = self.profile.calibration_input_errors()
        if errors:
            self._set_status(" ".join(errors), WARNING)
            return
        crop = self.profile.meter_roi.crop(self.image_bgr)
        observation = MeterDetector(self.profile).detect(crop, 0.0)
        self.observation = observation
        passed = (
            observation.complete
            and observation.track_bounds is not None
            and observation.confidence >= self.profile.detection.required_confidence
        )
        if passed:
            self.profile.mark_validated()
            self._set_status(
                "Detection passed. Marker, green band, track, and confidence are confirmed.",
                SUCCESS,
            )
        else:
            self.profile.invalidate_validation()
            reasons = "; ".join(observation.reasons) or "required confidence was not reached"
            self._set_status(f"Detection failed: {reasons}", DANGER)
        self._refresh_state()
        self.redraw()

    def save(self) -> None:
        if not self.profile.validation_current:
            self._set_status("Run Test detection successfully before saving.", DANGER)
            return
        prior_profile = self.master_app.profile
        if prior_profile is not None and self.profile.calibration_signature() != prior_profile.calibration_signature():
            self.profile.adaptive.clear(self.profile.timing.input_latency_ms, disable=True)
        try:
            self.master_app.store.save(self.profile)
        except (OSError, ValueError) as exc:
            self._set_status(f"Calibration could not be saved: {exc}", DANGER)
            return
        self.saved_signature = self.profile.validation_signature
        self.master_app.profile = self.profile
        self.master_app._build_form()
        self._refresh_state()
        self._set_status(f"✓ Saved to profile {self.profile.name!r}.", SUCCESS)

    def redraw(self) -> None:
        if not hasattr(self, "canvas"):
            return
        self.canvas.delete("all")
        if self.image_bgr is None:
            self.canvas.create_text(
                22, 22, text=f"Start {self.game_name} and press Re-shoot", fill=MUTED,
                anchor="nw", font=(CANVAS_FONT, 16, "bold"),
            )
            return
        rgb = _sample_preview_rgb(self.image_bgr, self.profile, self.selected)
        height, width = self.image_bgr.shape[:2]
        left, top, right, bottom = self.profile.meter_roi.pixels(width, height)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        observation = self.observation
        if observation and self.profile.meter_roi.valid:
            if observation.track_bounds:
                y0, y1 = observation.track_bounds
                draw.rectangle((left, top + y0, right, top + y1),
                               outline=(56, 189, 248), width=2)
            if observation.zone_bounds:
                y0, y1 = observation.zone_bounds
                draw.rectangle((left, top + y0, right, top + y1),
                               outline=(52, 211, 153), width=3)
                inset = (y1 - y0) * self.profile.detection.safe_zone_inset
                safe0, safe1 = int(top + y0 + inset), int(top + y1 - inset)
                draw.line((left, safe0, right, safe0), fill=(255, 255, 255), width=2)
                draw.line((left, safe1, right, safe1), fill=(255, 255, 255), width=2)
            if observation.marker_span:
                y0, y1 = observation.marker_span
                draw.rectangle((left, top + y0, right, top + y1),
                               outline=(251, 113, 133), width=3)
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        scale = min(cw / width, ch / height)
        dw, dh = max(1, int(width * scale)), max(1, int(height * scale))
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        self.scale, self.offset = scale, (ox, oy)
        resized = pil.resize((dw, dh), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(ox, oy, image=self.photo, anchor="nw")
        if self.profile.meter_roi.valid:
            x0, y0 = ox + left * scale, oy + top * scale
            x1, y1 = ox + right * scale, oy + bottom * scale
            active = self.selected == "region"
            self.canvas.create_rectangle(
                x0, y0, x1, y1, outline=ACCENT if active else "#64748b",
                width=3 if active else 1, dash=() if active else (3, 4),
            )
            if active:
                label_left = x0 + 5
                label_top = y0 + 3
                self.canvas.create_rectangle(
                    label_left, label_top, label_left + 205, label_top + 25,
                    fill="#07111f", outline=ACCENT, width=1,
                )
                self.canvas.create_rectangle(
                    x1 - self.HANDLE, y1 - self.HANDLE, x1, y1,
                    outline="#ffffff", fill=ACCENT, width=2,
                )
                self.canvas.create_text(
                    label_left + 8, label_top + 13, text="TIMING METER · drag to move",
                    fill="#ffffff", anchor="w", font=(CANVAS_FONT, 10, "bold"),
                )


class SharkmanApp(ctk.CTk):
    def __init__(self, store: ProfileStore | None = None,
                 backend: PlatformBackend | None = None) -> None:
        super().__init__()
        # PhotoImages belong to one Tcl interpreter; do not reuse them after
        # an application instance has closed (including native CI smoke tests).
        global _BLANK_REFERENCE
        _REFERENCE_CACHE.clear()
        _BLANK_REFERENCE = None
        self.title(f"SharkmanV3 v{__version__}")
        self.geometry("1080x740")
        self.minsize(780, 560)
        self.configure(fg_color=APP_BG)
        self.store = store or ProfileStore()
        self.profile = self.store.active()
        self.backend_error = ""
        try:
            self.backend = backend or create_backend()
        except Exception as exc:
            self.backend = None
            self.backend_error = str(exc)
        self.game_name = self.backend.game_name if self.backend else "Roblox"
        self.default_window_title = self.backend.default_window_title if self.backend else "Roblox"
        if (self.backend and self.profile.platform == sys.platform
                and self.profile.fingerprint.width <= 0
                and self.profile.window_title == "Roblox"):
            self.profile.window_title = self.default_window_title
        self.runner: LiveRunner | None = None
        platform_overlay = self.backend.create_overlay(self) if self.backend else None
        self.overlay = platform_overlay if platform_overlay is not None else DebugOverlay(self)
        self.session_mode = ctk.StringVar(value=DEFAULT_SESSION_MODE)
        self.diagnostics_enabled = ctk.BooleanVar(value=False)
        self.record_crops = ctk.BooleanVar(value=False)
        self._page = "setup"
        self._unbind_hotkeys = None
        self.hotkey_error = ""
        self._build_form()
        self._bind_hotkeys()
        self._refresh_platform_readiness()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _clear(self) -> None:
        for child in self.winfo_children():
            if isinstance(child, tk.Toplevel):
                continue
            child.destroy()

    def _header(self, master: Any, eyebrow: str, title: str,
                subtitle: str) -> ctk.CTkFrame:
        head = ctk.CTkFrame(master, fg_color=BG_CARD, corner_radius=18,
                            border_width=1, border_color=BORDER)
        head.pack(fill="x", pady=(0, 16))
        brand = ctk.CTkFrame(head, fg_color="transparent")
        brand.pack(side="left", padx=18, pady=13)
        ctk.CTkLabel(
            brand, text="S", width=36, height=36, corner_radius=12,
            fg_color=ACCENT, text_color="#07111f",
            font=ctk.CTkFont(size=19, weight="bold"),
        ).pack(side="left")
        words = ctk.CTkFrame(brand, fg_color="transparent")
        words.pack(side="left", padx=11)
        title_row = ctk.CTkFrame(words, fg_color="transparent")
        title_row.pack(anchor="w")
        ctk.CTkLabel(title_row, text=title, text_color=TEXT, anchor="w",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")
        self.version_label = ctk.CTkLabel(
            title_row, text=f"v{__version__}", text_color=ACCENT,
            fg_color=BG_CARD_RAISED, corner_radius=7, width=48, height=24,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.version_label.pack(side="left", padx=(10, 0))
        ctk.CTkLabel(words, text=f"{eyebrow}  ·  {subtitle}", text_color=MUTED,
                     anchor="w", font=ctk.CTkFont(size=10, weight="bold")).pack(anchor="w")
        return head

    def _build_form(self) -> None:
        self._page = "setup"
        self._clear()
        shell = ctk.CTkFrame(self, fg_color="transparent")
        shell.pack(fill="both", expand=True, padx=22, pady=20)
        head = self._header(
            shell, "SETUP WORKSPACE", "SharkmanV3",
            "configure one safe timing session",
        )
        self.setup_state = ctk.CTkLabel(
            head, text="●  Ready to configure", corner_radius=10,
            fg_color="#12372f", text_color="#87efc3", height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.setup_state.pack(side="right", padx=16, pady=12)
        content = ctk.CTkFrame(shell, fg_color="transparent")
        content.pack(fill="both", expand=True)
        content.grid_columnconfigure(0, weight=5, minsize=430)
        content.grid_columnconfigure(1, weight=2, minsize=260)
        content.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(
            content, fg_color=BG_SOFT, corner_radius=20, border_width=1,
            border_color=BORDER, scrollbar_button_color="#345074",
            scrollbar_button_hover_color=ACCENT_HOVER,
        )
        body.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        speed_scroll(body)
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="SESSION SETUP", text_color=ACCENT, anchor="w",
                     font=ctk.CTkFont(size=11, weight="bold")).grid(
                         row=0, column=0, sticky="ew", padx=18, pady=(18, 2))
        ctk.CTkLabel(body, text="Build a safe timing run.", text_color=TEXT,
                     anchor="w", font=ctk.CTkFont(size=25, weight="bold")).grid(
                         row=1, column=0, sticky="ew", padx=18)
        ctk.CTkLabel(
            body,
            text=f"Choose a profile, bind {self.game_name}, calibrate once, then review the pre-flight checks.",
            text_color=MUTED, anchor="w", justify="left", wraplength=590,
        ).grid(row=2, column=0, sticky="ew", padx=18, pady=(2, 12))
        card = Card(
            body, "Which calibration profile?",
            "Use a separate profile for every resolution, DPI, or display mode.", 1)
        card.grid(row=3, column=0, sticky="ew", padx=18, pady=6)
        profiles = self.store.list()
        self.profile_labels = {f"{p.name} · {p.profile_id[:4]}": p for p in profiles}
        active_label = next(
            (label for label, profile in self.profile_labels.items()
             if profile.profile_id == self.profile.profile_id),
            next(iter(self.profile_labels), "Default"),
        )
        self.profile_menu = ctk.CTkOptionMenu(
            card.body, values=list(self.profile_labels) or ["Default"],
            command=self._select_profile, fg_color=BG_CARD_RAISED,
            button_color="#2b496c", button_hover_color=ACCENT_HOVER,
        )
        self.profile_menu.set(active_label)
        self.profile_menu.pack(fill="x", pady=(0, 8))
        actions = ctk.CTkFrame(card.body, fg_color="transparent")
        actions.pack(fill="x")
        for text, command in (
            ("New", self._new_profile), ("Rename", self._rename_profile),
            ("Delete", self._delete_profile), ("Import", self._import_profile),
            ("Export", self._export_profile),
        ):
            ctk.CTkButton(
                actions, text=text, width=74, height=30,
                fg_color=BG_CARD_RAISED, hover_color="#203858", command=command,
            ).pack(side="left", padx=(0, 5))
        card = Card(
            body, f"Which {self.game_name} window?",
            "Bind the exact client size, DPI, and fullscreen/windowed mode used for calibration.",
            2,
        )
        card.grid(row=4, column=0, sticky="ew", padx=18, pady=6)
        self.window_title = ctk.StringVar(value=self.profile.window_title)
        ctk.CTkEntry(
            card.body, textvariable=self.window_title,
            placeholder_text=f"Window title contains, usually {self.default_window_title}",
            fg_color="#091426", border_color=BORDER,
        ).pack(fill="x", pady=(0, 8))
        bind_row = ctk.CTkFrame(card.body, fg_color="transparent")
        bind_row.pack(fill="x")
        ctk.CTkButton(
            bind_row, text="Find and bind current window", width=190,
            command=self._bind_window, fg_color=ACCENT,
            hover_color=ACCENT_HOVER, text_color="#07111f",
        ).pack(side="left")
        self.window_status = ctk.CTkLabel(
            bind_row, text=self._fingerprint_text(), text_color=MUTED, anchor="w")
        self.window_status.pack(side="left", padx=10)
        card = Card(
            body, "How should this session run?",
            "Choose what happens when the marker reaches the green target.",
            3,
        )
        card.grid(row=5, column=0, sticky="ew", padx=18, pady=6)
        ctk.CTkSegmentedButton(
            card.body, values=["Dry run", "Live input"], variable=self.session_mode,
            selected_color=ACCENT, selected_hover_color=ACCENT_HOVER,
            unselected_color=BG_CARD_RAISED, unselected_hover_color="#203858",
            command=self._session_mode_changed,
        ).pack(fill="x")
        ctk.CTkLabel(
            card.body,
            text="DRY RUN · Watches the meter and logs ‘WOULD PRESS’. "
                 "It never presses Space; you control the game yourself.",
            text_color="#dceaf7", anchor="w", justify="left", wraplength=560,
            font=ctk.CTkFont(size=11),
        ).pack(fill="x", pady=(10, 2))
        ctk.CTkLabel(
            card.body,
            text="LIVE INPUT · Watches the meter and presses Space for you at "
                 f"the predicted time, only while {self.game_name} is focused.",
            text_color="#dceaf7", anchor="w", justify="left", wraplength=560,
            font=ctk.CTkFont(size=11),
        ).pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(
            card.body,
            text="F2 starts the selected mode. Every new app launch starts in Live input; it never starts automatically.",
            text_color=ACCENT, anchor="w", justify="left", wraplength=560,
            font=ctk.CTkFont(size=10, weight="bold"),
        ).pack(fill="x")
        card = Card(
            body, "Diagnostics and recordings",
            "Diagnostics are local. Meter-crop recording is opt-in and never captures the full game window.",
            4,
        )
        card.grid(row=6, column=0, sticky="ew", padx=18, pady=6)
        ctk.CTkSwitch(
            card.body, text="Enable F8 diagnostics log and overlay",
            variable=self.diagnostics_enabled, progress_color=SUCCESS,
        ).pack(anchor="w", pady=3)
        ctk.CTkSwitch(
            card.body, text="Record calibrated meter crops",
            variable=self.record_crops, progress_color=SUCCESS,
        ).pack(anchor="w", pady=3)
        card = Card(
            body, "Advanced cooldowns",
            "Fine-tune every delay, timeout, scan rate, latency value, and key duration.",
            5,
        )
        card.grid(row=7, column=0, sticky="ew", padx=18, pady=(6, 18))
        ctk.CTkButton(
            card.body, text="⚙  Open cooldown editor", width=190,
            command=self._open_cooldowns, fg_color=BG_CARD_RAISED,
            hover_color="#203858",
        ).pack(anchor="w")
        summary = ctk.CTkScrollableFrame(
            content, fg_color=BG_CARD, corner_radius=20, border_width=1,
            border_color=BORDER, scrollbar_button_color="#345074",
            scrollbar_button_hover_color=ACCENT_HOVER,
        )
        summary.grid(row=0, column=1, sticky="nsew")
        speed_scroll(summary)
        if self.backend and self.backend.readiness():
            permissions = Card(summary, "Mac readiness", "Experimental · gameplay-unverified")
            permissions.pack(fill="x", padx=12, pady=12)
            self.platform_status = ctk.CTkLabel(
                permissions.body, text="", justify="left", anchor="w",
                text_color=WARNING, wraplength=210,
            )
            self.platform_status.pack(fill="x")
            for label, callback in self.backend.permission_actions().items():
                ctk.CTkButton(
                    permissions.body, text=label, height=28,
                    command=lambda action=callback: self._permission_action(action),
                ).pack(fill="x", pady=3)
            ctk.CTkButton(
                permissions.body, text="Check again", command=self._recheck_platform,
            ).pack(fill="x", pady=3)
            self._refresh_platform_readiness()
        ctk.CTkLabel(summary, text="RUN PROFILE", text_color=ACCENT,
                     font=ctk.CTkFont(size=11, weight="bold")).pack(
                         anchor="w", padx=18, pady=(18, 3))
        ctk.CTkLabel(summary, text="At a glance", text_color=TEXT,
                     font=ctk.CTkFont(size=20, weight="bold")).pack(
                         anchor="w", padx=18)
        self.summary_profile = self._summary_row(summary, "Profile", self.profile.name)
        self.summary_geometry = self._summary_row(
            summary, "Geometry", self._fingerprint_text())
        self.summary_calibration = self._summary_row(
            summary, "Calibration",
            "Validated" if self.profile.validation_current else "Action required",
        )
        self.summary_mode = self._summary_row(
            summary, "Session mode", self.session_mode.get())
        calibration_card = ctk.CTkFrame(
            summary, fg_color="#0d2238", corner_radius=14,
            border_width=1, border_color="#1e4c6c",
        )
        calibration_card.pack(fill="x", padx=15, pady=(18, 9))
        ctk.CTkLabel(calibration_card, text="CALIBRATION", text_color="#83d8ff",
                     font=ctk.CTkFont(size=10, weight="bold")).pack(
                         anchor="w", padx=13, pady=(12, 2))
        ctk.CTkLabel(
            calibration_card,
            text="Draw the meter and click all required colors in a guided workspace.",
            text_color="#c5d7e9", justify="left", wraplength=220,
        ).pack(anchor="w", padx=13, pady=(0, 10))
        ctk.CTkButton(
            calibration_card, text="Open calibration  →", height=32,
            fg_color="#173a59", hover_color="#21557d",
            command=self._open_calibration,
        ).pack(fill="x", padx=11, pady=(0, 11))
        prep_card = ctk.CTkFrame(
            summary, fg_color="#102319", corner_radius=14,
            border_width=1, border_color="#236244",
        )
        prep_card.pack(fill="x", padx=15, pady=(0, 9))
        ctk.CTkLabel(prep_card, text="PREPARATION", text_color="#86efac",
                     font=ctk.CTkFont(size=10, weight="bold")).pack(
                         anchor="w", padx=13, pady=(11, 2))
        ctk.CTkLabel(
            prep_card,
            text="Review the required before-you-start checks and confirm the run mode.",
            text_color="#c5e7d3", justify="left", wraplength=220,
        ).pack(anchor="w", padx=13, pady=(0, 9))
        ctk.CTkButton(
            prep_card, text="Open preparation guide  →", height=32,
            fg_color="#17563a", hover_color="#20714a", command=self._apply_setup,
        ).pack(fill="x", padx=11, pady=(0, 11))
        foot = ctk.CTkFrame(shell, fg_color="transparent")
        foot.pack(fill="x", pady=(15, 0))
        self.setup_error = ctk.CTkLabel(
            foot, text="", text_color=DANGER,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.setup_error.pack(side="left")
        ctk.CTkButton(
            foot, text="Continue to pre-flight  →", width=202, height=42,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#07111f",
            font=ctk.CTkFont(size=13, weight="bold"), command=self._apply_setup,
        ).pack(side="right")
        self._refresh_setup_status()

    @staticmethod
    def _summary_row(master: Any, label: str, value: str) -> ctk.CTkLabel:
        row = ctk.CTkFrame(master, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=5)
        ctk.CTkLabel(row, text=label.upper(), text_color=MUTED, anchor="w",
                     font=ctk.CTkFont(size=10, weight="bold")).pack(anchor="w")
        widget = ctk.CTkLabel(
            row, text=value, text_color=TEXT, anchor="w", justify="left",
            wraplength=220, font=ctk.CTkFont(size=14, weight="bold"),
        )
        widget.pack(anchor="w", pady=(1, 1))
        ctk.CTkFrame(row, height=1, fg_color=BORDER).pack(fill="x", pady=(7, 0))
        return widget

    def _fingerprint_text(self) -> str:
        fp = self.profile.fingerprint
        return "Not bound" if fp.width <= 0 else (
            f"{fp.width}×{fp.height} @ {fp.dpi} DPI · {fp.display_mode}")

    def _session_mode_changed(self, value: str) -> None:
        if hasattr(self, "summary_mode") and self.summary_mode.winfo_exists():
            self.summary_mode.configure(text=value)

    def _refresh_setup_status(self) -> None:
        if self.profile.calibration_errors():
            self.setup_state.configure(
                text="●  Calibration required", fg_color="#462536",
                text_color="#fecdd3",
            )
        else:
            self.setup_state.configure(
                text="●  Ready for pre-flight", fg_color="#12372f",
                text_color="#87efc3",
            )

    def _select_profile(self, label: str) -> None:
        profile = self.profile_labels.get(label)
        if profile is not None:
            self.profile = profile
            self.store.set_active(profile.profile_id)
            self._build_form()

    def _new_profile(self) -> None:
        name = simpledialog.askstring("New profile", "Profile name:", parent=self)
        if not name:
            return
        existing = {profile.name.casefold() for profile in self.store.list()}
        clean = name.strip()
        if clean.casefold() in existing:
            messagebox.showerror(
                "Duplicate profile", "Choose a unique profile name.", parent=self)
            return
        profile = CalibrationProfile(name=clean)
        profile.window_title = self.default_window_title
        self.store.save(profile)
        self.profile = profile
        self._build_form()

    def _rename_profile(self) -> None:
        name = simpledialog.askstring(
            "Rename profile", "New profile name:",
            initialvalue=self.profile.name, parent=self,
        )
        if not name:
            return
        clean = name.strip()
        if any(
            profile.profile_id != self.profile.profile_id
            and profile.name.casefold() == clean.casefold()
            for profile in self.store.list()
        ):
            messagebox.showerror(
                "Duplicate profile", "Choose a unique profile name.", parent=self)
            return
        self.profile.name = clean
        self.store.save(self.profile)
        self._build_form()

    def _delete_profile(self) -> None:
        if not messagebox.askyesno(
            "Delete profile", f"Delete {self.profile.name!r}?", parent=self):
            return
        self.store.delete(self.profile.profile_id)
        self.profile = self.store.active()
        self._build_form()

    def _import_profile(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, filetypes=[("Sharkman profile", "*.json")])
        if path:
            try:
                self.profile = self.store.import_profile(Path(path))
                self._build_form()
            except Exception as exc:
                messagebox.showerror("Import failed", str(exc), parent=self)

    def _export_profile(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".json",
            initialfile=f"{self.profile.name}.json",
            filetypes=[("JSON", "*.json")],
        )
        if path:
            self.store.export_profile(self.profile, Path(path))

    def _bind_window(self) -> None:
        if not self.backend:
            self.setup_error.configure(text=self.backend_error)
            return
        self.profile.window_title = self.window_title.get().strip() or self.default_window_title
        try:
            self.backend.select_window(None)
            try:
                window = self.backend.find_window(self.profile.window_title)
            except WindowSelectionRequired as selection:
                choices = "\n".join(
                    f"{i}: {w.title} — {w.width}×{w.height} (window {w.handle})"
                    for i, w in enumerate(selection.windows, 1))
                choice = simpledialog.askinteger(
                    "Select Roblox client", choices, parent=self,
                    minvalue=1, maxvalue=len(selection.windows))
                if choice is None:
                    return
                self.backend.select_window(selection.windows[choice - 1].handle)
                window = self.backend.find_window(self.profile.window_title)
            self.profile.platform = sys.platform
            self.profile.fingerprint = window.fingerprint
            if not self.profile.validation_current:
                self.profile.invalidate_validation()
            self.store.save(self.profile)
            self.window_status.configure(text=self._fingerprint_text(), text_color=SUCCESS)
            self.summary_geometry.configure(text=self._fingerprint_text())
            privilege = self.backend.privilege_error(window)
            self.setup_error.configure(
                text=privilege or f"Bound to {window.title!r}.",
                text_color=DANGER if privilege else SUCCESS,
            )
            self._refresh_setup_status()
        except Exception as exc:
            self.setup_error.configure(text=str(exc), text_color=DANGER)

    def _open_calibration(self) -> None:
        if not self.backend:
            self.setup_error.configure(text=self.backend_error)
            return
        existing = getattr(self, "_calibrator", None)
        if existing and existing.winfo_exists():
            existing.lift()
            return
        self.profile.window_title = self.window_title.get().strip() or self.default_window_title
        self._calibrator = Calibrator(self)

    def _open_cooldowns(self) -> None:
        existing = getattr(self, "_cooldown_editor", None)
        if existing and existing.winfo_exists():
            existing.lift()
            return
        self._cooldown_editor = CooldownEditor(self)

    def _save_setup(self) -> None:
        self.profile.window_title = self.window_title.get().strip() or self.default_window_title
        self.store.save(self.profile)

    def _preflight_errors(self) -> list[str]:
        errors = list(self.profile.calibration_errors())
        if self.profile.platform != sys.platform:
            errors.append(
                f"Profile belongs to {self.profile.platform}; this host is {sys.platform}.")
        if not self.backend:
            errors.append(self.backend_error or "Platform backend is unavailable.")
            return errors
        try:
            window = self.backend.find_window(self.profile.window_title)
            errors.extend(self.profile.geometry_errors(window.fingerprint))
            privilege = self.backend.privilege_error(window)
            if privilege:
                errors.append(privilege)
            if self.session_mode.get() == "Live input":
                if self.__dict__.get("hotkey_error", ""):
                    errors.append(f"Global F4 emergency hotkey is unavailable: {self.hotkey_error}")
                input_error = self.backend.input_error(window)
                if input_error:
                    errors.append(input_error)
        except Exception as exc:
            errors.append(str(exc))
        return list(dict.fromkeys(errors))

    def _apply_setup(self) -> None:
        self._save_setup()
        self._build_preparation()

    def _build_preparation(self) -> None:
        self._page = "preparation"
        self._clear()
        shell = ctk.CTkFrame(self, fg_color="transparent")
        shell.pack(fill="both", expand=True, padx=22, pady=20)
        self._header(
            shell, "REQUIRED PRE-FLIGHT", "Preparation guide",
            "verify the game before enabling the timing loop",
        )
        body = ctk.CTkScrollableFrame(
            shell, fg_color=BG_SOFT, corner_radius=18, border_width=1,
            border_color=BORDER, scrollbar_button_color="#345074",
            scrollbar_button_hover_color=ACCENT_HOVER,
        )
        body.pack(fill="both", expand=True)
        speed_scroll(body)
        body.grid_columnconfigure(0, weight=1)
        mode = self.session_mode.get()
        items = (
            (f"{self.game_name} window and permissions", "required",
             f"The macro must find the exact {self.game_name} client and the system must allow live input.",
             (f"Keep {self.game_name} open and not minimized.",
              "Live input needs /dev/uinput permission on Linux." if sys.platform.startswith("linux")
              else "Grant Screen Recording, Accessibility, and Input Monitoring in Mac readiness."
              if sys.platform == "darwin" else "Use the same privilege level as Roblox."),
             "Window discovery and privilege checks below show no error."),
            ("Match the calibrated geometry", "required",
             "Resolution, DPI, and fullscreen/windowed mode are part of the profile fingerprint.",
             (f"Do not resize {self.game_name} after calibration.",
              "Create another named profile for a different display layout."),
             "The current fingerprint exactly matches the saved profile."),
            ("Use a tested meter calibration", "required",
             "The full meter region and three screenshot-click colors must pass detection together.",
             ("Re-shoot with the meter visible after any display change.",
              "Confirm the marker and green overlays before saving."),
             "Calibration status is Validated."),
            ("Keep the game visible and unobstructed", "important",
             "The assistant reads visible pixels only.",
             (f"Bring {self.game_name} to the foreground before F2.",
              "Do not cover the meter with another window or menu."),
             "The timing meter will be fully visible when the prompt appears."),
            ("Prepare the Sharkman challenge", "important",
             "SharkmanV3 handles only the vertical timing prompt and one Space tap.",
             ("Start the challenge yourself.",
              "Do not expect navigation, Continue, dodge, or farming behavior."),
             "You are ready to handle every non-meter action yourself."),
            ("Choose stable performance", "helpful",
             "High frame rate and low ping improve the usable reaction window.",
             ("Aim for roughly 120 FPS when stable.",
              "Prefer a low-ping server and reduce resolution if capture cannot keep up."),
             "The game is responsive and the connection is stable."),
            (f"Confirm session mode: {mode}", "required",
             "Dry run logs when it would press Space, but leaves the game untouched. "
             "Live input presses Space for you when the calibrated meter reaches its target.",
             ("Live input is selected at launch; choose Dry run if you only want WOULD PRESS messages.",
              "Check the Run Profile before pressing F2."),
             f"The Run Profile intentionally says {mode}."),
            ("Know the emergency controls", "required",
             "F4 is the immediate stop and release path.",
             ("F2 starts, pauses, or resumes.",
              "Mac: use Fn/Globe + F2/F4/F8 if your keyboard uses media keys."
              if sys.platform == "darwin" else "F4 stops and releases Space; F8 toggles diagnostics."),
             f"You can reach F4 without moving focus away from {self.game_name}."),
        )
        for index, item in enumerate(items, 1):
            PreparationItem(body, index, *item, expanded=index == 1).grid(
                row=index - 1, column=0, sticky="ew", padx=12, pady=5)
        foot = ctk.CTkFrame(
            shell, fg_color=BG_CARD, corner_radius=16, border_width=1,
            border_color=BORDER,
        )
        foot.pack(fill="x", pady=(12, 0))
        ctk.CTkButton(
            foot, text="←  Back to setup", width=138, height=36,
            fg_color=BG_CARD_RAISED, hover_color="#203858",
            command=self._build_form,
        ).pack(side="left", padx=12, pady=10)
        self.preflight_ready = ctk.BooleanVar(value=False)
        self.preflight_check = ctk.CTkCheckBox(
            foot, text="I completed the required preparation checks",
            variable=self.preflight_ready, command=self._refresh_preflight_gate,
            text_color="#dce8f5", font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.preflight_check.pack(side="left", padx=10)
        self.preflight_message = ctk.CTkLabel(
            foot, text="", text_color=DANGER, justify="left", wraplength=360)
        self.preflight_message.pack(side="left", padx=(4, 8))
        self.finish_button = ctk.CTkButton(
            foot, text="Finish  →", width=164, height=38, state="disabled",
            fg_color="#1b3652", hover_color=ACCENT_HOVER, text_color=TEXT,
            text_color_disabled="#d5e6f4", font=ctk.CTkFont(size=13, weight="bold"),
            command=self._finish_preparation,
        )
        self.finish_button.pack(side="right", padx=12, pady=10)
        self._refresh_preflight_gate()

    def _refresh_preflight_gate(self) -> None:
        errors = self._preflight_errors()
        if errors:
            self.preflight_message.configure(text=errors[0], text_color=DANGER)
        else:
            self.preflight_message.configure(
                text="Automatic checks passed.", text_color=SUCCESS)
        enabled = self.preflight_ready.get() and not errors
        self.finish_button.configure(
            state="normal" if enabled else "disabled",
            fg_color=ACCENT if enabled else "#1b3652",
            text_color="#07111f" if enabled else TEXT,
            text_color_disabled="#d5e6f4",
        )

    def _finish_preparation(self) -> None:
        if not self.preflight_ready.get():
            self.preflight_message.configure(
                text="Confirm that you completed the preparation checks.",
                text_color=DANGER,
            )
            return
        errors = self._preflight_errors()
        if errors:
            self.preflight_message.configure(text=errors[0], text_color=DANGER)
            return
        self._build_runner()

    def _build_runner(self) -> None:
        self._page = "runner"
        self._clear()
        shell = ctk.CTkFrame(self, fg_color="transparent")
        shell.pack(fill="both", expand=True, padx=22, pady=20)
        head = self._header(
            shell, "RUN CONSOLE", "Sharkman timing terminal",
            f"{self.profile.name} · {self.session_mode.get()}",
        )
        self.run_badge = ctk.CTkLabel(
            head, text="●  IDLE", text_color=MUTED,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.run_badge.pack(side="right", padx=18)
        dry_run = self.session_mode.get() == "Dry run"
        ctk.CTkLabel(
            shell,
            text=("DRY RUN · Space will not be pressed. Watch for WOULD PRESS in the log."
                  if dry_run else
                  "LIVE INPUT · Space will be pressed automatically when the meter is ready."),
            fg_color="#15364a" if dry_run else "#17563a",
            text_color="#effaff", corner_radius=12, height=40,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(fill="x", pady=(0, 10))
        hotkeys = ctk.CTkFrame(
            shell, fg_color=BG_SOFT, corner_radius=14, border_width=1,
            border_color=BORDER,
        )
        hotkeys.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(hotkeys, text="CONTROLS", text_color=ACCENT,
                     font=ctk.CTkFont(size=10, weight="bold")).pack(
                         side="left", padx=(15, 10), pady=11)
        ctk.CTkLabel(
            hotkeys,
            text="F2: Start / Pause / Resume     F4: Stop + release Space     F8: Diagnostics",
            text_color="#dce8f5", font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left", pady=11)
        self.run_metrics = ctk.CTkLabel(
            hotkeys, text="IDLE · confidence — · prompts 0 · actions 0",
            text_color=MUTED, font=ctk.CTkFont(size=11, weight="bold"),
        )
        self.run_metrics.pack(side="right", padx=15)
        self.adaptive_run_status = ctk.CTkLabel(
            shell, text=("Adaptive timing: disabled (saved settings ignored)"
                         if not LIVE_ADAPTIVE_FEEDBACK_SUPPORTED else
                         "Visual centering: idle in Dry run (no learning)"
                         if self.profile.adaptive.enabled and self.session_mode.get() == "Dry run" else
                         "Visual centering: on (experimental; hit grade unknown)"
                         if self.profile.adaptive.enabled and LIVE_ADAPTIVE_FEEDBACK_SUPPORTED
                         and sys.platform == "win32" else
                         "Visual centering: unavailable on this platform"
                         if self.profile.adaptive.enabled else
                         "Visual centering: off"),
            text_color=WARNING if self.profile.adaptive.enabled and LIVE_ADAPTIVE_FEEDBACK_SUPPORTED else MUTED,
            anchor="w", font=ctk.CTkFont(size=11),
        )
        self.adaptive_run_status.pack(fill="x", pady=(0, 6))
        log_card = ctk.CTkFrame(
            shell, fg_color=BG_CARD, corner_radius=16, border_width=1,
            border_color=BORDER,
        )
        log_card.pack(fill="both", expand=True)
        ctk.CTkLabel(log_card, text="LIVE LOG", text_color=MUTED,
                     font=ctk.CTkFont(size=10, weight="bold")).pack(
                         anchor="w", padx=14, pady=(11, 4))
        self.logbox = ctk.CTkTextbox(
            log_card, fg_color="#08111f", text_color="#c7d7ea",
            border_width=0, corner_radius=10,
            font=ctk.CTkFont(family="Menlo" if sys.platform == "darwin" else "Consolas", size=12),
            activate_scrollbars=True,
        )
        self.logbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._log_lines = 0
        self._run_failed = False
        foot = ctk.CTkFrame(shell, fg_color="transparent")
        foot.pack(fill="x", pady=(12, 0))
        ctk.CTkButton(
            foot, text="←  Setup", width=110, height=40,
            fg_color=BG_CARD_RAISED, hover_color="#203858",
            command=self._back_to_setup,
        ).pack(side="left")
        self.stop_button = ctk.CTkButton(
            foot, text="Stop  (F4)", width=132, height=40,
            fg_color="#4b2235", hover_color="#713047", text_color="#ffe4e6",
            state="disabled", font=ctk.CTkFont(size=13, weight="bold"),
            command=self.stop_run,
        )
        self.stop_button.pack(side="right")
        self.start_button = ctk.CTkButton(
            foot, text="Start  (F2)", width=160, height=40,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#07111f",
            font=ctk.CTkFont(size=13, weight="bold"), command=self.toggle_run,
        )
        self.start_button.pack(side="right", padx=(0, 9))
        mode_message = (
            "DRY RUN: Space will not be pressed."
            if dry_run else "LIVE INPUT: Space may be pressed automatically."
        )
        self._log(f"Ready in {mode_message} Focus {self.game_name}, then press Start or F2.")

    def _log(self, text: str) -> None:
        if self._page != "runner" or not hasattr(self, "logbox"):
            return

        def append() -> None:
            if not self.logbox.winfo_exists():
                return
            stamp = datetime.now(UTC).astimezone().strftime("%H:%M:%S")
            self.logbox.insert("end", f"[{stamp}] {text}\n")
            self._log_lines += 1
            if self._log_lines > 1200:
                self.logbox.delete("1.0", "201.0")
                self._log_lines -= 200
            self.logbox.see("end")
            if text.startswith("Stopped:"):
                self._run_failed = True
                self.run_badge.configure(text="●  ERROR — SEE LOG", text_color=DANGER)
                self.run_metrics.configure(text=text, text_color=DANGER)
            if text.startswith("Monitoring stopped"):
                self.overlay.hide()
                if self._run_failed:
                    self.run_badge.configure(
                        text="●  ERROR — SEE LOG", text_color=DANGER)
                else:
                    self.run_badge.configure(text="●  IDLE", text_color=MUTED)
                self.start_button.configure(text="Start  (F2)")
                self.stop_button.configure(state="disabled")

        self.after(0, append)

    def _runner_update(
        self, observation: MeterObservation, snapshot: EngineSnapshot,
        _action: FireAction | None, window: WindowInfo,
    ) -> None:
        def apply() -> None:
            if self._page != "runner" or not self.run_metrics.winfo_exists():
                return
            suffix = " · DRY RUN" if self.session_mode.get() == "Dry run" else ""
            self.run_badge.configure(
                text=f"●  {snapshot.state.value}{suffix}",
                text_color=SUCCESS if snapshot.focused else WARNING,
            )
            self.run_metrics.configure(
                text=f"{snapshot.state.value} · confidence {observation.confidence:.0%} "
                     f"· prompts {snapshot.prompt_count} · actions {snapshot.action_count}")
            if self.profile.adaptive.enabled and LIVE_ADAPTIVE_FEEDBACK_SUPPORTED:
                state = self.profile.adaptive
                self.adaptive_run_status.configure(
                    text=f"Visual centering: {state.sample_count} eligible · saved latency "
                         f"{self.profile.timing.input_latency_ms:g} ms · "
                         f"{state.last_outcome} · {state.last_reason}",
                )
            if self.overlay.visible:
                self.overlay.update(window, self.profile, observation, snapshot)
            elif (self.diagnostics_enabled.get() and self.backend
                  and self.backend.overlay_supported):
                self.overlay.show(window)
                self.overlay.update(window, self.profile, observation, snapshot)
            overlay_error = getattr(self.overlay, "error", "")
            if overlay_error and self.backend and self.backend.overlay_supported:
                self.backend.overlay_supported = False
                self._log(f"Visible overlay unavailable: {overlay_error} F8 logging remains active.")

        self.after(0, apply)

    def toggle_run(self) -> None:
        if self._page != "runner" or not self.backend:
            return
        if self.runner and self.runner.running:
            self.runner.toggle_pause()
            paused = self.runner.paused
            if paused:
                self.overlay.hide()
            self.start_button.configure(
                text="Resume  (F2)" if paused else "Pause  (F2)")
            self.run_badge.configure(
                text="●  PAUSED" if paused else "●  RUNNING",
                text_color=WARNING if paused else SUCCESS,
            )
            return
        errors = self._preflight_errors()
        if errors:
            self._log("Start blocked: " + errors[0])
            return
        self._run_failed = False
        self.runner = LiveRunner(
            self.backend, self.profile, self._runner_update, self._log,
            dry_run=self.session_mode.get() == "Dry run",
            diagnostics_enabled=self.diagnostics_enabled.get(),
            record_crops=self.record_crops.get(),
            profile_store=self.store,
        )
        self.runner.start()
        self.run_badge.configure(text="●  STARTING", text_color=WARNING)
        self.start_button.configure(text="Pause  (F2)")
        self.stop_button.configure(state="normal")

    def stop_run(self) -> None:
        if self.runner:
            self.runner.stop()
        elif self.backend:
            try:
                self.backend.release_all()
            except Exception as exc:
                self._log(f"Stopped: Space release failed: {exc}")
        if self._page == "runner" and hasattr(self, "run_badge"):
            self.run_badge.configure(text="●  STOPPING", text_color=WARNING)
            self.start_button.configure(text="Start  (F2)")
            self.stop_button.configure(state="disabled")
        self.overlay.hide()

    def toggle_debug(self) -> None:
        if self._page != "runner":
            return
        self.diagnostics_enabled.set(not self.diagnostics_enabled.get())
        enabled = self.diagnostics_enabled.get()
        if self.runner:
            self.runner.diagnostics.enabled = enabled
        if not enabled:
            self.overlay.hide()
        elif self.backend and self.backend.overlay_error:
            self._log(f"Visible overlay unavailable: {self.backend.overlay_error} F8 logging remains active.")
        self._log("Diagnostics enabled." if enabled else "Diagnostics disabled.")

    def _back_to_setup(self) -> None:
        if self.runner and self.runner.running:
            messagebox.showwarning(
                "Monitoring active", "Stop monitoring before returning to Setup.",
                parent=self,
            )
            return
        self.overlay.hide()
        self._build_form()

    def _refresh_platform_readiness(self) -> None:
        label = self.__dict__.get("platform_status")
        if label is not None and label.winfo_exists() and self.backend:
            report = self.backend.readiness()
            lines = [f"{group}: {item}" for group, items in report.items() for item in items]
            label.configure(text="\n\n".join(lines) or "Permissions ready.")

    def _permission_action(self, action: Any) -> None:
        try:
            action()
        except Exception as exc:
            messagebox.showerror("Mac permissions", str(exc), parent=self)
        self._refresh_platform_readiness()

    def _recheck_platform(self) -> None:
        if self.runner and self.runner.running:
            messagebox.showwarning("Monitoring active", "Stop before rechecking permissions.", parent=self)
            return
        if self._unbind_hotkeys:
            self._unbind_hotkeys()
            self._unbind_hotkeys = None
        self.hotkey_error = ""
        self._bind_hotkeys()
        self._refresh_platform_readiness()

    def _emergency_hotkey(self) -> None:
        # Stop input immediately on the listener thread, even if Tk is busy.
        try:
            runner = self.__dict__.get("runner")
            if runner:
                runner.stop()
            elif self.backend:
                self.backend.release_all()
        finally:
            self.after(0, self.stop_run)

    def _bind_hotkeys(self) -> None:
        if not self.backend:
            return
        try:
            self._unbind_hotkeys = self.backend.bind_hotkeys(
                lambda: self.after(0, self.toggle_run),
                self._emergency_hotkey,
                lambda: self.after(0, self.toggle_debug),
            )
        except BackendError as exc:
            self.hotkey_error = str(exc)
            self.bind_all("<F2>", lambda _event: self.toggle_run())
            self.bind_all("<F4>", lambda _event: self.stop_run())
            self.bind_all("<F8>", lambda _event: self.toggle_debug())

    def close(self) -> None:
        if self.runner:
            self.runner.stop()
            self.runner.join()
        self.overlay.hide()
        overlay_close = getattr(self.overlay, "close", None)
        if overlay_close:
            overlay_close()
        if self._unbind_hotkeys:
            self._unbind_hotkeys()
        try:
            if self.backend:
                self.backend.close()
        finally:
            self.destroy()


def launch(
    backend: PlatformBackend | None = None,
    store: ProfileStore | None = None,
) -> int:
    app = SharkmanApp(store=store, backend=backend)
    app.mainloop()
    return 0
