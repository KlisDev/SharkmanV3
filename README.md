# SharkmanV3

Current release: **1.0** (shared by Windows, Linux, and macOS).

**New here?** Download and extract the project, then open [GUIDE.html](GUIDE.html)
in your browser for the complete installation, calibration, and first-run guide.

SharkmanV3 is a calibrated timing-meter assistant for the Blox Fruits Sharkman
Master challenge. It watches only the region you calibrate and can send one
Space tap when the moving marker reaches the safe part of the green target.

The project is deliberately limited to that meter. It does not navigate,
fight, dodge, farm, or click Continue.

## Choose your platform

- **Windows 10/11:** open [`WINDOWS`](WINDOWS/README.md). This is the primary
  implementation; the historical perfect-replay acceptance target still needs
  a gameplay-validated update.
- **Linux/Sober:** open [`LINUX`](LINUX/README.md). The shared GUI and timing
  features have an X11 adapter, but real Sober gameplay is still unverified;
  Wayland is not yet supported.
- **macOS 14+ (Apple Silicon and Intel):** open [`MAC`](MAC/README.md).
  Native Roblox support is **experimental and gameplay-unverified**; no Mac
  tester has completed live acceptance. The same UI and engine are shared.

All launchers create an isolated virtual environment in your user cache,
outside the project folder. Profiles, logs, diagnostic
crops, screenshots, and recordings are stored outside the repository or
ignored by Git. Live input is selected on every launch but never starts
automatically; choose Dry run before F2 to check calibration without input.

## Workflow

1. Create or select a profile and bind the Roblox window.
2. Open Calibration, draw the full meter, and click the screenshot for all
   three required color samples.
3. Pass **Test detection**, then save.
4. Complete the required preparation checklist.
5. You can switch to Dry run to check predictions first; Live input is the
   launch default and requires Start or F2.

Hotkeys: **F2** starts/pauses, **F4** stops and releases Space, and **F8**
toggles diagnostics.

## Privacy and publishing

The application has no telemetry or automatic uploads. Launchers use pip to
download dependencies during installation/updates. Calibration and runtime
capture can read the selected game window into memory; detection uses the
calibrated region. Windows/X11 capture visible screen coordinates, so keep
other windows and notifications away from the game. Focus loss inhibits
capture, and frames are discarded when a post-capture focus check fails.

Diagnostic logging is local and optional. Image recording is separately opt-in
and saves only the calibrated crop, which can still contain personal information
if the region includes it or another window covers it. Inspect logs, exported
profiles, and crops before sharing them. Default storage locations are:

- Windows: `%LOCALAPPDATA%/SharkmanV3`
- Linux: `$XDG_CONFIG_HOME/sharkman-v3` (normally `~/.config/sharkman-v3`)
- macOS: `~/Library/Application Support/SharkmanV3`

`SHARKMAN_DATA_DIR` can override these locations. Git excludes local profiles,
diagnostics, media, editing projects, and common secret files. These exclusions
do **not** protect a manually zipped folder or remove previously committed files
from Git history. Before publishing, review the staged files and commit history,
including author email addresses. Never upload the entire working folder with
its private `video-edit` directory or `.git` directory as a ZIP.

Licensed under the [MIT License](LICENSE).
