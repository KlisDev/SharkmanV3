# SharkmanV3 for macOS

**Experimental and gameplay-unverified.** Targets macOS 14 or newer on Apple
Silicon and Intel, using the native Roblox client. There has been no live Mac
gameplay acceptance test. Automated tests are not evidence of hit accuracy.

This folder supplies the Mac adapter. The complete interface, calibration,
zoom/magenta previews, profiles, cooldowns, timing engine, and diagnostics are
shared from `WINDOWS/sharkman`. Keep both folders together. Adaptive timing
remains disabled. No navigation, dodge, farming, or Continue automation is added.

## Install and launch

1. Install **Python 3.12 with Tk** using the universal2 macOS installer from
   [python.org](https://www.python.org/downloads/macos/). Python 3.11+ is accepted.
   Use native arm64 Python on Apple Silicon when possible, rather than Rosetta.
2. Download/extract the whole project. Open Terminal in its `MAC` folder.
3. Run `python3.12 easy_run_mac.py` (or `python3 easy_run_mac.py` if that is
   your installed supported Python).
4. For double-click launches, run `chmod +x "Start SharkmanV3.command"` once,
   then open that file in Finder. If macOS blocks it, review the file and use
   Finder's Open / the system's explicit approval flow. Do not disable Gatekeeper.

The launcher installs pinned dependencies into
`~/Library/Caches/SharkmanV3/venv-mac-py<version>-<architecture>`. It does not
modify global Python packages or store the environment inside the project.
An internet connection is required for the first dependency installation.
No `.app`, `.dmg`, signing certificate, or Apple developer account is required.
**Do not use sudo.**

## Permissions and first calibration

The **Mac readiness** panel separates capture blockers from live-input blockers.
Profile editing remains available when permissions are missing.

1. Use **Request Screen Recording**. Grant the executing Python/Terminal
   application access in System Settings → Privacy & Security → Screen Recording
   (called Screen & System Audio Recording on some releases). No audio is captured.
2. For live input, grant **Accessibility** and **Input Monitoring** using the
   corresponding request buttons. These allow Space posting and global hotkeys.
3. Relaunch if macOS requests it, then use **Check again**. Permission identity
   may change after replacing Python or rebuilding its environment. A button press
   does not prove permission was granted; the readiness status must clear.
4. Open Roblox, choose the title filter, and click **Find and bind current window**.
   If several matching Roblox clients exist, select the numbered window.
   Browsers and Roblox Studio are not eligible.
5. Calibrate on this Mac: frame the meter, click its three colors, test detection,
   and save. Existing Windows/Linux calibrations cannot arm on Mac.
6. Complete Preparation and deliberately press Start/F2. **Live input is the
   launch selection; monitoring never starts automatically.** Choose Dry run
   to check predictions without sending Space.

Capture uses the exact Roblox window returned by ScreenCaptureKit without its
shadow or mouse cursor. Native window chrome may be included; do not subtract
a guessed title-bar height. Only the calibrated meter crop is analysed/optionally
recorded. There is no desktop screenshot fallback.

Retina pixels and macOS screen points are different. Window movement, resizing,
display/scaling changes, or fullscreen transitions invalidate the active binding.
Stop, bind again, and recalibrate if the fingerprint changed. Keep Roblox visible,
unobstructed, and focused. A focus loss releases input; explicitly restart or
pause/resume after returning to Roblox.

## Controls, diagnostics, and files

- **F2:** start / pause / resume. **F4:** stop and release Space. **F8:** diagnostics.
  You may need **Fn/Globe + F2/F4/F8**, or enable standard function keys in
  Keyboard settings. [Apple's instructions](https://support.apple.com/en-ie/102439).
- Live input requires a healthy global F4 listener. If macOS disables it,
  monitoring stops. Return to Setup and use Check again; never bypass this guard.
- The overlay uses small, click-through, nonactivating panels outside the crop.
  If unavailable, diagnostic logging remains usable. It is not a full-screen layer.
- Mac profiles and diagnostics live under
  `~/Library/Application Support/SharkmanV3`. `SHARKMAN_DATA_DIR` is an optional
  explicit override. No telemetry or full-window recording is added.
- All Mac capture rates/timeouts appear in Advanced Cooldowns. The default
  240 Hz request is a ceiling, not delivered FPS. Fresh-frame wait is 100 ms;
  maximum accepted frame age is 50 ms; startup/stop timeouts are 2,000/1,000 ms.
  A scan delay of zero is still fastest. Increasing scan delay reduces precision.

## Safe local self-test

Run `python3.12 easy_run_mac.py --self-test`. It opens an animated synthetic
gauge and asks for consent before capture and one Space input. Only its own
process/window is eligible; Roblox is never selected. Keep this test focused.
The console distinguishes PASS, FAILED, and UNAVAILABLE checks. It does not
save images or change profiles. F2/F4/F8 can be checked manually in this window.
Restart the test after granting missing permissions.

This is not a gameplay test. macOS CI checks both architectures, imports/selectors,
shared GUI behavior, simulated backend failures, and replay regressions. It does
not grant privacy permissions or post keyboard events to the CI desktop.

## Troubleshooting and acceptance

- **Missing Tk:** reinstall Python with Tk; pip cannot supply Apple's Tk framework.
- **Black/stale capture or timeout:** check Screen Recording permission and the
  exact window binding, leave fullscreen transitions, then relaunch/rebind. Blank,
  idle, duplicate, and old frames are rejected instead of used for input.
- **No Space:** check Live input mode, calibration validity, Roblox focus, all
  readiness checks, and the run log. macOS does not acknowledge whether Roblox
  accepted a posted event, so a posted event is not proof of a game hit.
- **No fresh frame while idle:** expected on an unchanged window. Input remains
  blocked until a fresh complete frame arrives. Long gaps clear prediction history,
  but never count as the meter disappearing or unlock an already-fired prompt.
- **Wrong resolution:** use the same display/scaling/fullscreen mode used for
  calibration. Shared cooldown defaults may need manual tuning on Mac hardware.

Before calling this gameplay-verified, a future Mac tester must complete saved
calibration, dry run, repeated live prompts, F2/F4/F8, focus-loss, permission-loss,
Space-release and shutdown checks, including hard stages and FINISH. Record the
OS, processor, display scale, actual capture rate, and results. Existing Windows
replay expectations are retained; this port does not claim to fix unrelated misses.

Implementation references: [Apple ScreenCaptureKit](https://developer.apple.com/documentation/screencapturekit/sccontentfilter),
[PyObjC](https://pyobjc.readthedocs.io/en/latest/). Licensed by the single root MIT license.
