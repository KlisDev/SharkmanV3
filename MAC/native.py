"""Native Apple API bridge. Imported only on macOS, never by shared code.

Stream/event callbacks must not let Python exceptions cross the native boundary.
"""
from __future__ import annotations

import math
import threading
import time

import AppKit
import ApplicationServices as AX
import CoreFoundation as CF
import CoreMedia as CM
import dispatch
import objc
import Quartz as Q
import ScreenCaptureKit as SC

from sharkman.backends.base import BackendError

from .backend import MacWindow
from .capture import bgra_to_bgr


def await_completion(invoke, timeout_ms: float):
    done = threading.Event()
    result = []

    def completed(*values):
        # Called on a native dispatch queue. Keep this callback exception-free.
        result.extend(values)
        done.set()

    invoke(completed)
    if not done.wait(timeout_ms / 1000):
        raise BackendError("Apple API timed out. Recheck permissions and restart the app.")
    if result and result[-1] is not None:
        raise BackendError(str(result[-1]))
    return result[0] if len(result) > 1 else None


def _rect(rect) -> tuple[float, float, float, float]:
    return float(rect.origin.x), float(rect.origin.y), float(rect.size.width), float(rect.size.height)


def _bounds(info) -> tuple[float, float, float, float]:
    rect = info[Q.kCGWindowBounds]
    return tuple(float(rect[key]) for key in ("X", "Y", "Width", "Height"))


def displays() -> list[tuple[int, tuple, tuple]]:
    error, identities, _count = Q.CGGetActiveDisplayList(32, None, None)
    if error:
        raise BackendError(f"Display enumeration failed ({error})")
    result = []
    for identity in identities:
        bounds = _rect(Q.CGDisplayBounds(identity))
        mode = Q.CGDisplayCopyDisplayMode(identity)
        signature = (bounds, Q.CGDisplayModeGetWidth(mode), Q.CGDisplayModeGetHeight(mode),
                     Q.CGDisplayModeGetPixelWidth(mode), Q.CGDisplayModeGetPixelHeight(mode))
        result.append((int(identity), bounds, signature))
    return result


def display_for(rect: tuple, screens: list[tuple]) -> tuple:
    x, y, w, h = rect

    def overlap(screen):
        sx, sy, sw, sh = screen[1]
        return max(0, min(x + w, sx + sw) - max(x, sx)) * max(0, min(y + h, sy + sh) - max(y, sy))

    if not screens or max(map(overlap, screens)) <= 0:
        raise BackendError("Roblox is outside all active displays.")
    return max(screens, key=overlap)


class SharkmanStreamOutput(AppKit.NSObject,
                          protocols=[objc.protocolNamed("SCStreamOutput"),
                                     objc.protocolNamed("SCStreamDelegate")]):
    def stream_didOutputSampleBuffer_ofType_(self, _stream, sample, kind):
        try:
            if kind != SC.SCStreamOutputTypeScreen:
                return
            attachments = CM.CMSampleBufferGetSampleAttachmentsArray(sample, False)
            if not attachments or int(attachments[0].get(SC.SCStreamFrameInfoStatus, -1)) != SC.SCFrameStatusComplete:
                self.sink.invalidate()
                return
            pts = CM.CMTimeGetSeconds(CM.CMSampleBufferGetPresentationTimeStamp(sample))
            host_now = CM.CMTimeGetSeconds(CM.CMClockGetTime(CM.CMClockGetHostTimeClock()))
            age = host_now - pts
            if not math.isfinite(age) or age < 0 or age > self.max_age_ms / 1000:
                self.sink.invalidate()
                return
            captured_at = time.monotonic() - age
            pixel = CM.CMSampleBufferGetImageBuffer(sample)
            if pixel is None or Q.CVPixelBufferGetPixelFormatType(pixel) != Q.kCVPixelFormatType_32BGRA:
                raise BackendError("ScreenCaptureKit did not deliver a BGRA pixel buffer")
            status = Q.CVPixelBufferLockBaseAddress(pixel, Q.kCVPixelBufferLock_ReadOnly)
            if status:
                raise BackendError(f"Pixel buffer lock failed ({status})")
            try:
                width = Q.CVPixelBufferGetWidth(pixel)
                height = Q.CVPixelBufferGetHeight(pixel)
                stride = Q.CVPixelBufferGetBytesPerRow(pixel)
                base = Q.CVPixelBufferGetBaseAddress(pixel)
                image = bgra_to_bgr(base.as_buffer(height * stride), width, height, stride)
            finally:
                Q.CVPixelBufferUnlockBaseAddress(pixel, Q.kCVPixelBufferLock_ReadOnly)
            self.sink.publish(image, pts, captured_at)
        except Exception as exc:
            self.sink.fail(f"Mac capture failed: {exc}")

    def stream_didStopWithError_(self, _stream, error):
        try:
            self.sink.fail(f"ScreenCaptureKit stopped: {error}")
        except Exception:
            pass


class EventHotkeys:
    """Session-level listen-only tap on a dedicated CFRunLoop (no root)."""

    def __init__(self, toggle, stop, debug, timing):
        self.callbacks = {120: toggle, 118: stop, 100: debug}  # F2, F4, F8
        self.emergency = stop
        self.timing = timing
        self.healthy = False
        self.error = ""
        self.tap = self.loop = None
        self.ready = threading.Event()
        self.closed = threading.Event()
        self.down = set()
        self.thread = threading.Thread(target=self._run, name="mac-hotkeys", daemon=True)
        self.thread.start()
        if not self.ready.wait(timing.mac_capture_start_ms / 1000) or not self.healthy:
            self.close()
            raise BackendError(self.error or "Global Mac hotkeys could not start. Check Input Monitoring.")

    def _callback(self, _proxy, kind, event, _context):
        try:
            if kind in (Q.kCGEventTapDisabledByTimeout, Q.kCGEventTapDisabledByUserInput):
                self.healthy = False
                self.error = "macOS disabled the hotkey listener; use Check again before live input."
                # The stop callback must not wait on Tk or a capture completion.
                threading.Thread(target=self._safe_emergency, daemon=True).start()
                return event
            if self.closed.is_set() or not self.healthy:
                return event
            code = Q.CGEventGetIntegerValueField(event, Q.kCGKeyboardEventKeycode)
            if kind == Q.kCGEventKeyUp:
                self.down.discard(code)
            elif code in self.callbacks and code not in self.down:
                self.down.add(code)
                if not Q.CGEventGetIntegerValueField(event, Q.kCGKeyboardEventAutorepeat):
                    # Tk can block while closing dialogs; do not block the event tap.
                    threading.Thread(target=self._invoke, args=(self.callbacks[code],), daemon=True).start()
        except Exception:
            self.healthy = False
            threading.Thread(target=self._safe_emergency, daemon=True).start()
        return event

    def _invoke(self, callback):
        try:
            callback()
        except Exception:
            self.healthy = False
            self._safe_emergency()

    def _safe_emergency(self):
        try:
            self.emergency()
        except Exception:
            pass

    def _run(self):
        source = None
        try:
            if not Q.CGPreflightListenEventAccess():
                raise BackendError("Input Monitoring permission is required for global F2/F4/F8.")
            mask = (1 << Q.kCGEventKeyDown) | (1 << Q.kCGEventKeyUp)
            self.tap = Q.CGEventTapCreate(Q.kCGSessionEventTap, Q.kCGHeadInsertEventTap,
                                       Q.kCGEventTapOptionListenOnly, mask, self._callback, None)
            if self.tap is None:
                raise BackendError("macOS denied the global hotkey event tap.")
            self.loop = CF.CFRunLoopGetCurrent()
            source = CF.CFMachPortCreateRunLoopSource(None, self.tap, 0)
            CF.CFRunLoopAddSource(self.loop, source, CF.kCFRunLoopCommonModes)
            Q.CGEventTapEnable(self.tap, True)
            self.healthy = not self.closed.is_set()
            self.ready.set()
            if not self.closed.is_set():
                CF.CFRunLoopRun()
        except Exception as exc:
            self.error = str(exc)
        finally:
            unexpected = not self.closed.is_set() and self.healthy
            self.healthy = False
            self.ready.set()
            if self.tap is not None:
                Q.CGEventTapEnable(self.tap, False)
                CF.CFMachPortInvalidate(self.tap)
            if source is not None and self.loop is not None:
                CF.CFRunLoopRemoveSource(self.loop, source, CF.kCFRunLoopCommonModes)
            if unexpected:
                self._safe_emergency()

    def close(self):
        self.closed.set()
        self.healthy = False
        if self.loop is not None:
            CF.CFRunLoopStop(self.loop)
        if self.thread is not threading.current_thread():
            self.thread.join(self.timing.shutdown_join_ms / 1000)


class NativeMac:
    def __init__(self):
        self.filters = {}
        self.screen_signatures = {}

    def permissions(self):
        return {"screen": bool(Q.CGPreflightScreenCaptureAccess()),
                "post": bool(Q.CGPreflightPostEventAccess() and AX.AXIsProcessTrusted()),
                "listen": bool(Q.CGPreflightListenEventAccess())}

    def request_permission(self, kind):
        if kind == "screen":
            Q.CGRequestScreenCaptureAccess()
        elif kind == "post":
            AX.AXIsProcessTrustedWithOptions({AX.kAXTrustedCheckOptionPrompt: True})
            Q.CGRequestPostEventAccess()
        elif kind == "listen":
            Q.CGRequestListenEventAccess()

    def open_settings(self):
        url = AppKit.NSURL.URLWithString_("x-apple.systempreferences:com.apple.preference.security")
        if not AppKit.NSWorkspace.sharedWorkspace().openURL_(url):
            raise BackendError("Open System Settings → Privacy & Security manually.")

    def windows(self, timeout_ms):
        content = await_completion(
            lambda cb: SC.SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(True, True, cb),
            timeout_ms,
        )
        screens = displays()
        result = []
        self.filters = {}
        for window in content.windows():
            app = window.owningApplication()
            if app is None or int(window.windowLayer()) != 0 or not window.isOnScreen():
                continue
            filter_ = SC.SCContentFilter.alloc().initWithDesktopIndependentWindow_(window)
            x, y, width, height = _rect(window.frame())
            # contentRect describes the exact captured window, including its native chrome.
            _, _, cw, ch = _rect(filter_.contentRect())
            if abs(cw - width) > 1 or abs(ch - height) > 1:
                # Never silently guess a title-bar/capture offset.
                continue
            scale = float(filter_.pointPixelScale())
            screen = display_for((x, y, width, height), screens)
            fullscreen = all(abs(a - b) <= 1 for a, b in zip((x, y, width, height), screen[1], strict=True))
            candidate = MacWindow(
                int(window.windowID()), int(app.processID()), str(app.bundleIdentifier() or ""),
                str(window.title() or app.applicationName() or ""), x, y, width, height,
                scale, screen[0], "fullscreen" if fullscreen else "windowed",
            )
            result.append(candidate)
            self.filters[candidate.handle] = filter_
            self.screen_signatures[candidate.handle] = screen[2]
        return result

    def matches(self, window):
        try:
            infos = Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements, Q.kCGNullWindowID)
            info = next((v for v in infos if int(v[Q.kCGWindowNumber]) == window.handle), None)
            if info is None or int(info[Q.kCGWindowOwnerPID]) != window.pid:
                return False
            rect = _bounds(info)
            if rect != (window.x, window.y, window.point_width, window.point_height):
                return False
            screen = display_for(rect, displays())
            return screen[0] == window.display_id and screen[2] == self.screen_signatures.get(window.handle)
        except Exception:
            return False

    def foreground(self, window):
        app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None or int(app.processIdentifier()) != window.pid:
            return False
        infos = Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements, Q.kCGNullWindowID)
        # Respect front-to-back order, rejecting another normal window even from the same app.
        front = next((v for v in infos if int(v.get(Q.kCGWindowLayer, -1)) == 0), None)
        return bool(front and int(front[Q.kCGWindowNumber]) == window.handle)

    def activate(self, window):
        app = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(window.pid)
        return bool(app and app.activateWithOptions_(AppKit.NSApplicationActivateIgnoringOtherApps))

    def start_stream(self, window, frames, timing):
        filter_ = self.filters.get(window.handle)
        if filter_ is None:
            raise BackendError("Mac capture binding expired; bind the window again.")
        config = SC.SCStreamConfiguration.alloc().init()
        config.setWidth_(window.info.width)
        config.setHeight_(window.info.height)
        config.setPixelFormat_(Q.kCVPixelFormatType_32BGRA)
        config.setMinimumFrameInterval_(CM.CMTimeMake(1, int(timing.mac_capture_hz)))
        config.setQueueDepth_(3)
        config.setShowsCursor_(False)
        config.setCapturesAudio_(False)
        config.setIgnoreShadowsSingleWindow_(True)
        config.setScalesToFit_(False)
        config.setColorSpaceName_(Q.kCGColorSpaceSRGB)
        output = SharkmanStreamOutput.alloc().init()
        output.sink = frames
        output.max_age_ms = timing.mac_frame_age_ms
        stream = SC.SCStream.alloc().initWithFilter_configuration_delegate_(filter_, config, output)
        queue = dispatch.dispatch_queue_create(b"sharkman.frames", None)
        ok, error = stream.addStreamOutput_type_sampleHandlerQueue_error_(output, SC.SCStreamOutputTypeScreen, queue, None)
        if not ok:
            raise BackendError(str(error or "ScreenCaptureKit refused the stream output"))
        handle = (stream, output, queue)  # Strong references for all callback lifetimes.
        try:
            await_completion(stream.startCaptureWithCompletionHandler_, timing.mac_capture_start_ms)
        except Exception:
            frames.close()
            self.stop_stream(handle, timing.mac_capture_stop_ms)
            raise
        return handle

    def stop_stream(self, handle, timeout_ms):
        stream, output, _queue = handle
        output.sink.close()
        try:
            await_completion(stream.stopCaptureWithCompletionHandler_, timeout_ms)
        except BackendError:
            # Never reuse this handle, even if the OS stop callback times out.
            pass

    def space(self, down):
        event = Q.CGEventCreateKeyboardEvent(None, 49, bool(down))
        if event is None:
            raise BackendError("Could not construct the native Space event")
        Q.CGEventSetFlags(event, 0)
        Q.CGEventPost(Q.kCGSessionEventTap, event)

    def hotkeys(self, toggle, stop, debug, timing):
        return EventHotkeys(toggle, stop, debug, timing)
