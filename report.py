"""
Interactive HTML report — the shareable artifact.

Embeds simulation frames into a single self-contained HTML file (data as
JSON, rendering via <canvas>, zero dependencies, works offline from file://).
Anyone who receives the file can scrub time, play the animation, hover for
exact (x, u, t) readouts, and click/drag the space-time heatmap to extract
the wavefront shape at any instant — it appears as a ghost curve on the
wave chart.

Usage:
    from report import capture_run, write_report

    data = capture_run("Uniform string", x, time_history, frames,
                       energy_history=solver.energy_history)
    write_report(data, "output/wave_report.html")

Or generate a demo report straight from the CLI:

    python report.py            # writes output/wave_report.html
"""
import argparse
import json
from typing import List, Optional, Sequence

import numpy as np

import djc_theme as t

__all__ = ["capture_run", "write_report"]

# Decimation budgets keep the HTML a few hundred KB, not tens of MB.
MAX_FRAMES = 120
MAX_X = 400


# ----------------------------------------------------------------------------
# Data capture
# ----------------------------------------------------------------------------

def _decimate(arr: np.ndarray, max_len: int, axis: int = 0) -> np.ndarray:
    """Evenly sample an array along ``axis`` down to ``max_len`` entries."""
    n = arr.shape[axis]
    if n <= max_len:
        return arr
    idx = np.linspace(0, n - 1, max_len).round().astype(int)
    return np.take(arr, idx, axis=axis)


def _ramp_colors(name: str, n: int = 24) -> List[List[int]]:
    """Sample a registered colormap into n RGB triples for the JS renderer."""
    cmap = t.plt.get_cmap(name)
    return [[int(r * 255), int(g * 255), int(b * 255)]
            for r, g, b in (cmap(i / (n - 1))[:3] for i in range(n))]


def capture_run(
    label: str,
    x: np.ndarray,
    time_history: Sequence[float],
    frames: Sequence[np.ndarray],
    energy_history: Optional[Sequence[tuple]] = None,
    cfl: float = None,
    subtitle: str = "Interactive simulation report",
) -> dict:
    """
    Package one solver run for the HTML renderer.

    ``frames`` is the displacement history (num_frames × num_points). Data is
    decimated to at most 120 frames × 400 columns and rounded to 5 decimals.
    """
    u = np.asarray(frames, dtype=float)
    x = np.asarray(x, dtype=float)
    times = np.asarray(time_history[: len(u)], dtype=float)

    u_s = np.round(_decimate(u, MAX_FRAMES, axis=0)[:, :MAX_X], 5)
    x_s = np.round(x[: u_s.shape[1]], 4)
    t_s = np.round(_decimate(times, MAX_FRAMES), 5)

    data = {
        "label": label,
        "subtitle": subtitle,
        "x": x_s.tolist(),
        "t": t_s.tolist(),
        "u": u_s.tolist(),
        "umin": float(u_s.min()),
        "umax": float(u_s.max()),
        "ramp": _ramp_colors("djc_diverging"),
        "cyan": t.ACCENT.CYAN,
        "ghost": t.ACCENT.AMBER,
        "violet": t.ACCENT.VIOLET,
    }
    if cfl is not None:
        data["cfl"] = round(float(cfl), 3)
    if energy_history:
        e = np.asarray(energy_history[: len(t_s)], dtype=float)
        if e.ndim == 2 and e.shape[1] >= 3:
            data["energy"] = {
                "t": t_s[: len(e)].tolist(),
                "ke": np.round(e[:, 0], 6).tolist(),
                "pe": np.round(e[:, 1], 6).tolist(),
                "tot": np.round(e[:, 2], 6).tolist(),
            }
    return data


# ----------------------------------------------------------------------------
# HTML template — one self-contained page, no external assets
# ----------------------------------------------------------------------------

_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — DJC Wave Report</title>
<style>
  :root { --bg:#070e1a; --surface:#0a1423; --raised:#0f1d31; --line:#223246;
          --fg1:#ecf4fb; --fg2:#b8c6d8; --fg3:#7d92ab; --teal:#14b89a;
          --cyan:#3ef0e2; --amber:#f4b840; --glass:rgba(20,41,63,.55); }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--bg); color:var(--fg2);
         font:15px/1.55 system-ui,-apple-system,'Segoe UI',sans-serif; padding:24px; }
  .eyebrow { font-size:11px; letter-spacing:.14em; text-transform:uppercase;
             color:var(--teal); font-weight:600; }
  h1 { color:var(--fg1); font-size:26px; margin:4px 0 2px; font-weight:700; }
  .sub { color:var(--fg3); font-size:13px; margin-bottom:18px; }
  .panel { background:var(--glass); border:1px solid rgba(124,180,200,.18);
           border-radius:12px; padding:14px; margin-bottom:16px; }
  .row { display:flex; gap:16px; flex-wrap:wrap; }
  .row .panel { flex:1 1 380px; margin-bottom:0; }
  canvas { width:100%; display:block; border-radius:8px; cursor:crosshair; }
  #controls { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
  button { background:var(--raised); color:var(--fg1); border:1px solid var(--line);
           border-radius:8px; padding:7px 18px; font-size:13px; cursor:pointer; }
  button:hover { border-color:var(--teal); color:var(--cyan); }
  input[type=range] { flex:1; min-width:200px; accent-color:var(--teal); height:26px; }
  .tread { font:12px ui-monospace,monospace; color:var(--cyan); min-width:110px; }
  select { background:var(--raised); color:var(--fg2); border:1px solid var(--line);
           border-radius:8px; padding:6px; font-size:12px; }
  .hint { color:var(--fg3); font-size:12px; margin-top:8px; }
  #hover { position:fixed; pointer-events:none; display:none; z-index:9;
           background:rgba(10,20,35,.92); border:1px solid var(--line);
           color:var(--fg1); font:11px ui-monospace,monospace;
           padding:5px 9px; border-radius:6px; white-space:nowrap; }
  .badge { display:inline-block; font:11px ui-monospace,monospace; padding:2px 9px;
           border-radius:6px; border:1px solid var(--line); margin-left:10px; }
  .ok { color:#4ade80; } .bad { color:#ff5d73; }
  .eq { font:12px ui-monospace,monospace; color:var(--fg3); }
  .ghostnote { color:var(--amber); font:12px ui-monospace,monospace; display:none; margin-top:8px; }
</style>
</head>
<body>
<div class="eyebrow">◆ DJC WAVE SIMULATOR</div>
<h1 id="title"></h1>
<div class="sub" id="subtitle"></div>

<div class="panel">
  <div id="controls">
    <button id="play">▶ Play</button>
    <input type="range" id="scrub" min="0" value="0" step="1">
    <span class="tread" id="tread"></span>
    <select id="speed">
      <option value="66">0.5×</option>
      <option value="33" selected>1×</option>
      <option value="16">2×</option>
    </select>
  </div>
</div>

<div class="row">
  <div class="panel">
    <div class="hint" style="margin:0 0 8px">u(x,t) — hover for exact readouts</div>
    <canvas id="wave"></canvas>
  </div>
  <div class="panel">
    <div class="hint" style="margin:0 0 8px">space-time — click or drag to extract a wavefront</div>
    <canvas id="heat"></canvas>
    <div class="ghostnote" id="ghostnote"></div>
  </div>
</div>

<div class="panel">
  <div class="eq" id="eqline"></div>
  <div class="hint">Scrub below or drag on the heatmap; amber ghost = extracted wavefront.
  Keyboard: ←/→ step · space play/pause. Single file, zero dependencies.</div>
</div>

<div id="hover"></div>
<script type="application/json" id="djc-data">__DATA__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById('djc-data').textContent);
const NT = D.t.length, NX = D.x.length;
document.getElementById('title').textContent = D.label;
document.getElementById('subtitle').textContent = D.subtitle;
const eq = document.getElementById('eqline');
eq.innerHTML = '&part;&sup2;u/&part;t&sup2; = c&sup2; &part;&sup2;u/&part;x&sup2; + F(u)' +
  (D.cfl !== undefined ? ' &nbsp;<span class="badge '+(D.cfl<=1?'ok':'bad')+'">CFL = '+D.cfl.toFixed(2)+(D.cfl<=1?' OK':' UNSTABLE')+'</span>' : '');

const wave = document.getElementById('wave'), heat = document.getElementById('heat');
const wc = wave.getContext('2d'), hc = heat.getContext('2d');
const hoverBox = document.getElementById('hover');
let frame = 0, ghost = -1, playing = false, timer = null, hoverIx = -1;

const PAD = {l:52, r:14, t:14, b:26};
function sizeCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight || Math.round(w * (c===wave?0.42:0.5));
  if (c.width !== w*dpr) { c.width = w*dpr; c.height = h*dpr; }
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);
  return [w, h];
}

function ramp(stops, v) {           // v in [0,1] -> rgb string
  const n = stops.length - 1, f = Math.min(Math.max(v,0),1) * n;
  const i = Math.min(Math.floor(f), n-1), k = f - i;
  const a = stops[i], b = stops[i+1];
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*k)},${Math.round(a[1]+(b[1]-a[1])*k)},${Math.round(a[2]+(b[2]-a[2])*k)})`;
}

function drawWave() {
  const [W,H] = sizeCanvas(wave);
  wc.clearRect(0,0,W,H);
  const x0=D.x[0], x1=D.x[NX-1];
  const uLo=Math.min(D.umin,-0.1)-0.15, uHi=Math.max(D.umax,0.1)+0.15;
  const px = x => PAD.l + (x-x0)/(x1-x0)*(W-PAD.l-PAD.r);
  const py = u => H-PAD.b - (u-uLo)/(uHi-uLo)*(H-PAD.t-PAD.b);
  wc.strokeStyle='rgba(124,180,200,.12)'; wc.lineWidth=1;
  for(let g=0; g<=4; g++){ const y=PAD.t+g*(H-PAD.t-PAD.b)/4;
    wc.beginPath(); wc.moveTo(PAD.l,y); wc.lineTo(W-PAD.r,y); wc.stroke(); }
  wc.strokeStyle='rgba(124,180,200,.28)';
  wc.beginPath(); wc.moveTo(PAD.l,py(0)); wc.lineTo(W-PAD.r,py(0)); wc.stroke();
  const row = D.u[frame];
  wc.strokeStyle=D.cyan; wc.lineWidth=2; wc.shadowColor=D.cyan; wc.shadowBlur=10;
  wc.beginPath();
  for(let i=0;i<NX;i++){ const X=px(D.x[i]), Y=py(row[i]); i?wc.lineTo(X,Y):wc.moveTo(X,Y); }
  wc.stroke(); wc.shadowBlur=0;
  if(ghost>=0 && ghost!==frame){
    const gr=D.u[ghost];
    wc.strokeStyle=D.ghost; wc.globalAlpha=.6; wc.lineWidth=1.4;
    wc.beginPath();
    for(let i=0;i<NX;i++){ const X=px(D.x[i]), Y=py(gr[i]); i?wc.lineTo(X,Y):wc.moveTo(X,Y); }
    wc.stroke(); wc.globalAlpha=1;
  }
  if(hoverIx>=0){
    wc.strokeStyle='rgba(184,198,216,.4)'; wc.setLineDash([3,3]);
    wc.beginPath(); wc.moveTo(px(D.x[hoverIx]),PAD.t); wc.lineTo(px(D.x[hoverIx]),H-PAD.b); wc.stroke();
    wc.setLineDash([]);
  }
  wc.fillStyle='#ecf4fb'; wc.font='12px ui-monospace,monospace';
  wc.fillText('t = '+D.t[frame].toFixed(3)+' s', PAD.l+8, PAD.t+14);
  wc.fillStyle='#7d92ab'; wc.font='11px ui-monospace,monospace';
  wc.fillText('frame '+(frame+1)+'/'+NT, PAD.l+8, H-10);
}

function drawHeat() {
  const [W,H] = sizeCanvas(heat);
  const img = hc.createImageData(W,H);
  const d = img.data;
  for(let pyy=0; pyy<H; pyy++){
    const fi = (1 - pyy/(H-1)) * (NT-1);
    const i0 = Math.min(Math.floor(fi), NT-2), ft = fi - i0;
    const r0 = D.u[i0], r1 = D.u[i0+1];
    for(let pxx=0; pxx<W; pxx++){
      const xi = pxx/(W-1) * (NX-1);
      const j0 = Math.min(Math.floor(xi), NX-2), fx = xi - j0;
      const v = (r0[j0]*(1-fx)+r0[j0+1]*fx)*(1-ft) + (r1[j0]*(1-fx)+r1[j0+1]*fx)*ft;
      const v01 = (v-D.umin)/Math.max(D.umax-D.umin,1e-12);
      const c = rampColor(v01);
      const o = (pyy*W+pxx)*4;
      d[o]=c[0]; d[o+1]=c[1]; d[o+2]=c[2]; d[o+3]=255;
    }
  }
  hc.putImageData(img,0,0);
  hc.strokeStyle='rgba(124,180,200,.25)'; hc.lineWidth=1;
  for(let g=1; g<4; g++){ const y=g*H/4; hc.beginPath(); hc.moveTo(0,y); hc.lineTo(W,y); hc.stroke(); }
  if(ghost>=0){
    const y=(1-ghost/(NT-1))*H;
    hc.strokeStyle=D.cyan; hc.lineWidth=1.6; hc.shadowColor=D.cyan; hc.shadowBlur=8;
    hc.beginPath(); hc.moveTo(0,y); hc.lineTo(W,y); hc.stroke(); hc.shadowBlur=0;
  }
}
const rampColor = v => { const n=D.ramp.length-1, f=Math.min(Math.max(v,0),1)*n;
  const i=Math.min(Math.floor(f),n-1), k=f-i, a=D.ramp[i], b=D.ramp[i+1];
  return [a[0]+(b[0]-a[0])*k, a[1]+(b[1]-a[1])*k, a[2]+(b[2]-a[2])*k]; };

function setFrame(i, ghostAt) {
  frame = Math.max(0, Math.min(i, NT-1));
  if (ghostAt !== undefined) ghost = ghostAt;
  document.getElementById('scrub').value = frame;
  document.getElementById('tread').textContent = 't = '+D.t[frame].toFixed(3)+' s';
  const gn = document.getElementById('ghostnote');
  if (ghost >= 0) { gn.style.display='block';
    gn.textContent = 'ghost wavefront @ t = '+D.t[ghost].toFixed(3)+' s (Esc clears)'; }
  drawWave(); drawHeat();
}

function setPlaying(p) {
  playing = p;
  document.getElementById('play').textContent = p ? '❚❚ Pause' : '▶ Play';
  if (timer) { clearInterval(timer); timer = null; }
  if (p) timer = setInterval(() => setFrame((frame+1) % NT), +document.getElementById('speed').value);
}

document.getElementById('play').onclick = () => setPlaying(!playing);
document.getElementById('scrub').oninput = e => setFrame(+e.target.value);
document.getElementById('speed').onchange = () => { if (playing) { setPlaying(false); setPlaying(true); } };

wave.addEventListener('mousemove', e => {
  const r = wave.getBoundingClientRect();
  const W = r.width;
  const x0=D.x[0], x1=D.x[NX-1];
  const ix = Math.round((e.clientX-r.left-PAD.l)/(W-PAD.l-PAD.r)*(NX-1));
  if (ix>=0 && ix<NX) {
    hoverIx = ix; drawWave();
    hoverBox.style.display='block';
    hoverBox.style.left=(e.clientX+14)+'px'; hoverBox.style.top=(e.clientY-10)+'px';
    hoverBox.textContent = 'x='+D.x[ix].toFixed(1)+'  u='+D.u[frame][ix].toFixed(4)+'  t='+D.t[frame].toFixed(3);
  }
});
wave.addEventListener('mouseleave', () => { hoverIx=-1; hoverBox.style.display='none'; drawWave(); });

function heatToFrame(e) {
  const r = heat.getBoundingClientRect();
  const f = Math.round((1-(e.clientY-r.top)/r.height)*(NT-1));
  setFrame(Math.max(0,Math.min(f,NT-1)), Math.max(0,Math.min(f,NT-1)));
}
let heatDrag=false;
heat.addEventListener('mousedown', e => { heatDrag=true; heatToFrame(e); });
heat.addEventListener('mousemove', e => { if(heatDrag) heatToFrame(e); });
window.addEventListener('mouseup', () => heatDrag=false);

window.addEventListener('keydown', e => {
  if (e.key===' '){ e.preventDefault(); setPlaying(!playing); }
  if (e.key==='ArrowRight') setFrame(frame+1);
  if (e.key==='ArrowLeft') setFrame(frame-1);
  if (e.key==='Escape'){ ghost=-1; document.getElementById('ghostnote').style.display='none'; drawWave(); drawHeat(); }
});
window.addEventListener('resize', () => { drawWave(); drawHeat(); });

setFrame(0);
</script>
</body>
</html>
"""


def write_report(data: dict, out_path: str) -> str:
    """Write the self-contained interactive report. Returns the path."""
    payload = json.dumps(data, separators=(",", ":"))
    html = _HTML.replace("__TITLE__", data["label"]).replace("__DATA__", payload)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    kb = len(html) // 1024
    print(t.success(f"interactive report → {out_path}  ({kb} KB, {len(data['t'])} frames)"))
    return out_path


# ----------------------------------------------------------------------------
# Demo generator
# ----------------------------------------------------------------------------

def main() -> None:
    import os

    from solver import VerletSolver
    from string_model import String

    parser = argparse.ArgumentParser(description="Generate an interactive HTML report")
    parser.add_argument("--out", default="output/wave_report.html")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    s = String(length=50.0, num_points=400, tension=50.0,
               density_profile="uniform", density_uniform=0.01)
    s.set_initial_gaussian(center=25.0, width=5.0, amplitude=1.0)
    solver = VerletSolver(s, enable_force=False)
    dt = 0.8 * s.dx / s.get_max_wave_speed()   # auto-CFL, same rule as the playground
    solver.solve(total_time=3.0, dt=dt, save_interval=5, verbose=False)

    from main import _cfl_of  # reuse the same CFL helper
    data = capture_run(
        "Wave Propagation — Uniform String",
        s.x, solver.time_history, solver.displacement_history,
        energy_history=solver.energy_history,
        cfl=_cfl_of(s, dt),
        subtitle="Velocity-Verlet · T = 50 N · μ = 0.01 kg/m · scrub, hover, and click the heatmap",
    )
    write_report(data, args.out)


if __name__ == "__main__":
    main()
