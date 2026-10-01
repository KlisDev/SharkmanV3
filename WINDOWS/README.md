# SharkmanV3 for Windows

Windows 10/11 is the primary supported platform. Install Python 3.12, then
double-click `easy_run.py`. The launcher creates an isolated environment in
`%LOCALAPPDATA%\SharkmanV3\venv-windows`, installs the pinned dependencies
there, and opens the app without changing global Python packages or filling
the project folder.

## First run

1. Open Roblox and enter the Sharkman challenge.
2. Create a named profile, enter the Roblox window title, and choose
   **Find / Bind Window**.
3. Open Calibration. The app hides itself and captures the bound Roblox client;
   it will not fall back to a desktop screenshot.
4. Draw the complete vertical meter region. Select each color task and click a
   representative pixel in the screenshot for green, black marker, and track.
   Sampling uses the median of a 5×5 patch. Extra clicks add lighting variants.
   **Zoom screenshot** and **Zoom reference** open read-only, original-resolution
   viewers for closer inspection. Scroll to zoom, drag to pan, and use Fit or
   1:1; close the viewer to move the region or sample a color. Re-shoot closes
   the viewer and captures a new snapshot.
5. Use **Test detection** on a frame where the meter is visible. Saving remains
   locked until the meter geometry, green target, marker, and confidence pass.
6. Complete Preparation. You can switch to **Dry run** first. It watches the meter and
   writes `WOULD PRESS` in the Run console, but never presses Space. You control
   the game yourself during this practice mode.
7. Leave **Live input** selected when you want SharkmanV3 to press Space for
   you at the predicted time. It only sends input while Roblox is focused and
   the calibrated meter is confirmed.

Profiles are stored under `%LOCALAPPDATA%\SharkmanV3`, not in this repository.
Live input is selected whenever the app starts, but monitoring does not start
until you complete Preparation and press Start or F2. Switch to Dry run before
starting if you want predictions without Space input. F4 stops either mode and
releases Space.

## Adaptive timing is disabled

The frozen-meter feedback prototype accepted **0 eligible positions** in the
September 26–27 live runs: the target changed or vanished before feedback
settled. Advanced Cooldowns therefore shows the adaptive switch as disabled.
Previously saved enabled settings are ignored, and the runtime makes **no
automatic cooldown changes or adaptive-only firing changes**. The ordinary
center predictor and manual **Capture + input latency** control still work.

Prototype code and the private gauge-crop audit tool remain for investigation,
but further rounds with the same feedback signal will not train the macro.
Live learning requires a different, validated signal before it can return.

## Controls

- **F2:** start, pause, or resume
- **F4:** emergency stop and release Space
- **F8:** toggle diagnostics and overlay

## Troubleshooting

- **Roblox is not found:** keep the game open, use its actual window title, and
  press Find / Bind Window again.
- **Privilege/focus refusal:** run Roblox and SharkmanV3 at the same privilege
  level. Input is intentionally blocked while Roblox is not foreground.
- **Geometry mismatch:** restore the calibrated monitor, size, DPI scaling, and
  window/fullscreen mode, or rebind and recalibrate.
- **Test detection fails:** capture while the meter is visible, keep the region
  narrow around the full meter, and click solid interior colors rather than
  antialiased edges.
- **Launcher cannot find Python:** reinstall Python 3.12 and enable the Python
  launcher (`py.exe`) during installation.

Developer checks from this folder:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
ruff check sharkman tests tools easy_run.py ..\LINUX
```

CI always replays compact numerical observations extracted from the historical
meter-only recordings; no video pixels are distributed. This currently locks
the measured 3-action short trace and 35-action/42-detected-prompt long trace
against regressions. The older human annotations target 3 and 37 events, so
the original perfect-replay acceptance gate is **not yet met**. Do not report
the numerical regression test as proof of 37/37 gameplay accuracy.
