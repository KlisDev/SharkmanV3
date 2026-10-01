# SharkmanV3 for Linux/Sober — experimental, live-unverified

This is the same timing-meter application as Windows: named profiles, guided
click-to-sample calibration, zoom and magenta preview, detection validation,
preparation, every cooldown, dry run/live input, local diagnostics, and optional
meter-only crop recording. The UI and timing engine are shared from
`WINDOWS/sharkman`; this folder contains only the Sober/X11 platform adapter.

Sober is an experimental Flatpak application. Run SharkmanV3 **on the Linux
host**, not inside Sober's Flatpak. Only an actual X11/Xorg desktop session is
supported. Wayland/XWayland is not an accepted substitute for this version.

## Install on an X11 desktop

Install Python 3.11 or newer (3.12 recommended), Tk, and Xvfb if you want to
run the smoke test. Package names for common distributions:

| Distribution | System packages |
| --- | --- |
| Ubuntu/Debian | `sudo apt install python3 python3-venv python3-tk` (`xvfb` for tests) |
| Fedora | `sudo dnf install python3 python3-pip python3-tkinter` (`xorg-x11-server-Xvfb` for tests) |
| Arch | `sudo pacman -S python python-pip tk` (`xorg-server-xvfb` for tests) |

Choose an X11/Xorg session at the login screen and confirm `echo
$XDG_SESSION_TYPE` prints `x11`. Install Sober from Flathub if needed
(`flatpak install flathub org.vinegarhq.Sober`), then open its Roblox game and
leave it visible.
From the repository root, run:

```bash
python3 LINUX/easy_run_linux.py
```

The launcher creates `~/.cache/sharkman-v3/venv-linux` (or uses
`$XDG_CACHE_HOME`) and installs pinned dependencies there.
It does not modify system Python packages. It opens the same Setup →
Calibration → Preparation → Run workflow as Windows. The default window title
is `Sober`, but discovery also recognizes Sober's X11 window class when its
title says `Roblox`.

Live input is selected on launch, but nothing starts until Preparation and F2.
Choose Dry run before starting if you want to log `WOULD PRESS` without sending
Space. Dry run needs no input permission.
Adaptive timing is disabled on all platforms: Windows gameplay showed that
the post-press frozen meter is not reliable feedback. Manual cooldowns remain.
For live input only, install the one-time `/dev/uinput` rule, then log out and
back in:

```bash
sudo bash LINUX/install-udev.sh
test -w /dev/uinput && echo "uinput ready"
```

Do **not** launch SharkmanV3 as root. If `/dev/uinput` is unavailable, the GUI
and dry run still work, but Preparation will block Live input with a specific
explanation. F2 starts/pauses, F4 stops and releases Space, and F8 toggles
diagnostic logging and a click-through X Shape overlay. If the X server lacks
Shape, F8 logs without a visible overlay.

## Calibration and profiles

Use a fresh Linux profile. Windows profiles cannot arm on Linux: the window
fingerprint and colors must be captured and validated on Sober. The screenshot
comes only from Sober's client rectangle; a missing or black capture is
rejected. Re-shoot after changing window size, X11 scaling, monitor, or display
mode. Calibration examples are visual aids; the live screenshot is what gets
sampled and tested.

Profiles and diagnostics live outside the repository under
`${XDG_CONFIG_HOME:-~/.config}/sharkman-v3`. Recording is opt-in and saves only
the calibrated meter crop. Do not include personal profiles or captures in a
public bug report unless you deliberately sanitize them.

## Troubleshooting and verification status

- **Wayland or missing DISPLAY:** select a real X11 session; do not merely set
  environment variables to pretend one exists.
- **Sober not found:** open Sober, keep it visible, and use Find and bind in
  Setup. Its title can be `Sober` or `Roblox`.
- **Black screenshot:** verify the game is actually drawn in the X11 session;
  calibration and running stop rather than using a desktop fallback.
- **No live Space:** check the udev rule and `/dev/uinput` access; try Dry run
  first. F4 must always release the key.
- **No visible F8 boxes:** the X Shape extension may be unavailable. Diagnostic
  logging remains usable and no opaque overlay is opened.

CI exercises the shared tests and an Xvfb window/capture/overlay smoke test on
Ubuntu, Debian, Fedora, and Arch. This is **not** a real Sober gameplay test.
A Linux tester must still complete calibration, dry run, live input, repeated
prompts, focus-loss, and F2/F4/F8 checks before Linux can be called
gameplay-verified. Until then this port remains experimental.
