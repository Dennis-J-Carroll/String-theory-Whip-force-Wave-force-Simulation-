# Run doc — String Theory: Whip & Wave-Force Simulation

Pure-Python (3.10) matplotlib project. **No dev server needed**: the browser artifact is a
self-contained HTML file. Register `output/wave_report.html` directly by absolute path with
`register_preview` (no process, port, or install required).

## Reproduce the artifacts

From the repo root:

```bash
pip install -r requirements.txt   # numpy, matplotlib, pytest
MPLBACKEND=Agg python main.py     # ~30 s: all plots, WAVs, and output/wave_report.html
```

`main.py` regenerates every artifact in `output/`, including `wave_report.html`
(self-contained interactive report), the WAV exports (`whip_crack.wav`, `string_tone.wav`),
and the themed PNGs. Optional extras:

```bash
python accuracy_lab.py        # output/accuracy_lab.png + CLI convergence tables
python accuracy_lab.py --selftest   # headless smoke test of the lab
python playground.py --selftest     # headless render of the playground UI
python missions.py --demo           # scripted Break-It-On-Purpose arcs (CLI only)
```

All commands are safe to run headless (`MPLBACKEND=Agg`) and need no `.env` or secrets.

## Run / view

`main.py` is a CLI simulation (writes files, exits) — the live viewable surface is the report:

- Preview: open `output/wave_report.html` (self-contained, works from file://).
- Interactive play: `python playground.py` (matplotlib window, needs a display).

No ports, no env files, nothing to copy from the main checkout (this workspace IS the main
checkout). If artifacts are missing or stale, re-run the two commands above.
