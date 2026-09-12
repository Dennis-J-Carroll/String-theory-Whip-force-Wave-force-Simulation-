# Run doc — String Theory: Whip & Wave-Force Simulation

Pure-Python (3.10) project. **Default preview surface: the Dash dashboard on
port 8050.** The self-contained HTML report is the fallback (no server needed).

## Run / view

### 1. Default — the DJC Wave Lab dashboard (port 8050)

> **Launch recipe that survives (Sep 11, 2026):** use a **systemd user unit**.
> The command runner reaps plain `setsid nohup ... &` children of *python*
> processes within seconds (a bare `sleep` survives — the reaping is
> selector-based), but a user unit owns the process outright:
>
> ```bash
> systemd-run --user --unit=djc-wave-lab --working-directory="$PWD" python dashboard_app.py
> sleep 5
> systemctl --user is-active djc-wave-lab.service     # expect: active
> systemctl --user show djc-wave-lab.service -p MainPID --value   # the pid to register
> ```
>
> Restart after code changes with `systemctl --user restart djc-wave-lab.service`;
> stop with `systemctl --user stop djc-wave-lab.service`. Logs:
> `journalctl --user -u djc-wave-lab.service -f`.

```bash
pip install -r requirements.txt   # includes dash, dash-bootstrap-components, plotly
```

Then verify before registering the preview:

```bash
pgrep -f "^python dashboard_app.py"        # grab the pid
kill -0 <pid>                              # must survive the launching shell
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8050/   # expect 200
```

Register `register_preview(url="http://127.0.0.1:8050/", pid=<pid>)`. Use
`setsid` — a plain `nohup ... &` from a SYNC command gets its process group
reaped when the command runner exits. The log lives at
`.freebuff/djc_dashboard.log`.

### 2. Fallback — the self-contained HTML report

If the dashboard cannot start (missing deps, port conflict, headless
constraints), register the report by absolute path instead — no process:

```
register_preview(htmlPath="<workspace>/output/wave_report.html")
```

The report is a single self-contained HTML file (JSON + canvas renderer,
now carrying the same escape-analysis readout as the dashboard's PHYSICS
card — see README "Interactive HTML report").
works from `file://`), so it needs no server, port, or install.

## Reproduce the artifacts

From the repo root:

```bash
pip install -r requirements.txt   # numpy, matplotlib, pytest + dash/plotly extras
MPLBACKEND=Agg python main.py     # ~30 s: all plots, WAVs, and output/wave_report.html
```

`main.py` regenerates every artifact in `output/`, including `wave_report.html`
(the fallback report), the WAV exports (`whip_crack.wav`, `string_tone.wav`),
and the themed PNGs. Optional extras:

```bash
python accuracy_lab.py        # output/accuracy_lab.png + CLI convergence tables
python accuracy_lab.py --selftest   # headless smoke test of the lab
python playground.py --selftest     # headless render of the playground UI
python missions.py --demo           # scripted Break-It-On-Purpose arcs (CLI only)
python interactive_demo.py          # six themed Plotly HTML artifacts → output/
python dashboard_app.py             # the DJC Wave Lab on http://localhost:8050
python tests/e2e_pluck_landscapes.py   # Playwright E2E: pluck click + Landscapes tab (in-process server on 8099)
```

All commands are safe to run headless (`MPLBACKEND=Agg`) and need no `.env` or
secrets — nothing to copy from the main checkout (this workspace IS the main
checkout). If the dashboard is unreachable, check the log first, then re-run
the two commands in "Reproduce the artifacts".
