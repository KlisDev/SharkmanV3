"""Opt-in own-window smoke test. Never discovers or sends input to Roblox."""
from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import messagebox

from .backend import MacBackend


def main() -> int:
    root = tk.Tk()
    root.title("SharkmanV3 synthetic Mac self-test")
    root.geometry("720x500")
    backend = MacBackend(test_pid=os.getpid())
    results = queue.SimpleQueue()
    received = [0]
    root.bind("<KeyPress-space>", lambda _event: received.__setitem__(0, received[0] + 1))
    canvas = tk.Canvas(root, bg="#08111f", height=270)
    canvas.pack(fill="both", expand=True)
    canvas.create_rectangle(340, 20, 380, 250, fill="#ff6611")
    canvas.create_rectangle(340, 110, 380, 155, fill="#20ef20")
    marker = canvas.create_rectangle(334, 75, 386, 82, fill="black")
    phase = [0]
    status = tk.Label(root, text="No checks have run. Permission-dependent results are reported separately.", wraplength=680)
    status.pack(pady=8)
    stopped = threading.Event()

    def emergency():
        stopped.set()
        backend.release_all()
        results.put("F4: stopped; Space released.")

    try:
        unbind = backend.bind_hotkeys(
            lambda: results.put("F2 received (self-test only; no gameplay)."), emergency,
            lambda: results.put("F8 received (self-test only)."))
    except Exception as exc:
        def unbind():
            return

        results.put(f"HOTKEYS UNAVAILABLE: {exc}")

    def work():
        try:
            window = backend.find_window("SharkmanV3 synthetic Mac self-test")
            if not backend.is_foreground(window):
                raise RuntimeError("Keep the synthetic test window focused.")
            image = backend.capture(window)
            results.put(f"CAPTURE PASS: own window {image.shape[1]}×{image.shape[0]}; no image saved.")
            blocker = backend.input_error(window)
            if blocker:
                results.put(f"INPUT UNAVAILABLE: {blocker}")
                return
            if stopped.is_set():
                return
            backend.begin_input_session()
            if stopped.is_set():
                backend.release_all()
                return
            backend.press_space(backend.timing.key_hold_ms)
            results.put("CHECK_RECEIPT")
        except Exception as exc:
            results.put(f"CHECK UNAVAILABLE/FAILED: {exc}")
        finally:
            backend.release_all()
            backend.stop_capture()
            results.put("DONE")

    def start():
        if not messagebox.askyesno(
            "Run own-window test?",
            "This test captures only this synthetic window and may send one Space to it. "
            "It does not capture or control Roblox. Allow this test?", parent=root):
            return
        received[0] = 0
        stopped.clear()
        button.configure(state="disabled")
        root.focus_force()
        root.after(int(backend.timing.calibration_capture_delay_ms),
                   lambda: threading.Thread(target=work, daemon=True).start())

    button = tk.Button(root, text="Test own-window capture + one Space", command=start)
    button.pack(pady=8)
    history = []

    def receipt():
        results.put("INPUT PASS: Space received by the synthetic window." if received[0]
                    else "INPUT FAILED/UNVERIFIED: the synthetic window did not receive Space.")

    def poll():
        # Keep the synthetic frame changing so freshness tests are meaningful.
        phase[0] = (phase[0] + 1) % 180
        y = 30 + phase[0]
        canvas.coords(marker, 334, y, 386, y + 7)
        while not results.empty():
            text = results.get()
            if text == "CHECK_RECEIPT":
                root.after(int(backend.timing.mac_frame_wait_ms), receipt)
            elif text == "DONE":
                button.configure(state="normal")
            else:
                history.append(text)
                print(text)
                status.configure(text="\n".join(history[-4:]))
        root.after(int(backend.timing.diagnostic_refresh_ms), poll)

    def close():
        emergency()
        unbind()
        backend.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    poll()
    root.mainloop()
    return 0
