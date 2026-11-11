TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>__TITLE__</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#0a0b0d;--bg-1:#0f1115;--bg-2:#141619;--bg-3:#1c1f24;--border:#252830;--text:#c8cdd6;--dim:#5c6370;--accent:#f0a500;--mono:'IBM Plex Mono',monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:var(--mono);font-size:12px;height:100vh;overflow:hidden}
header{height:44px;display:flex;align-items:center;gap:16px;padding:0 16px;background:var(--bg-1);border-bottom:1px solid var(--border)}
.title{color:var(--accent);font-weight:600;letter-spacing:.04em}
.meta{color:var(--dim)}
main{display:grid;grid-template-columns:290px 1fr;height:calc(100vh - 44px)}
aside{background:var(--bg-2);border-right:1px solid var(--border);overflow-y:auto;padding:12px}
.section{color:var(--accent);text-transform:uppercase;font-size:10px;letter-spacing:.08em;margin:14px 0 6px}
label{display:block;color:var(--dim);margin:6px 0 2px}
select,input{width:100%;background:var(--bg-3);color:var(--text);border:1px solid var(--border);padding:4px;font-family:var(--mono);font-size:12px}
.range{display:grid;grid-template-columns:1fr 1fr;gap:4px}
button{margin-top:12px;width:100%;background:var(--bg-3);color:var(--accent);border:1px solid var(--accent);padding:6px;cursor:pointer;font-family:var(--mono)}
.content{display:flex;flex-direction:column;min-height:0}
.readout{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;padding:10px;background:var(--bg-2);border-bottom:1px solid var(--border)}
.stat{background:var(--bg-3);padding:8px;border-left:2px solid var(--accent);color:var(--dim)}
.stat b{display:block;color:var(--text);font-size:12px;margin-top:4px;font-weight:500}
#heat{flex:1;min-height:260px}
.bottom{display:grid;grid-template-columns:1fr 1fr;height:230px;border-top:1px solid var(--border)}
</style></head>
<body>
<header><span class="title">__TITLE__</span><span class="meta" id="meta"></span></header>
<main>
<aside>
 <div class="section">Axes</div>
 <label>X axis</label><select id="xsel"></select>
 <label>Y axis</label><select id="ysel"></select>
 <div class="section">Profile</div>
 <label>Input to vary</label><select id="psel"></select>
 <label>Output to trace</label><select id="osel"></select>
 <div class="section">Scenario</div>
 <div id="scen"></div>
 <div class="section">Input filters</div>
 <div id="filters"></div>
 <button id="reset">Reset filters</button>
</aside>
<section class="content">
 <div class="readout" id="readout"></div>
 <div id="heat"></div>
 <div class="bottom"><div id="hist"></div><div id="profile"></div></div>
</section>
</main>
<script>
const D = __DATA__;
const cols = D.columns;
const colIndex = {};
cols.forEach((c, i) => colIndex[c] = i);
const rows = D.data;
const inputNames = D.inputs.map(x => x.name);
const outputNames = D.outputs.map(x => x.name);
const eventCols = cols.filter(c => c.startsWith("occurred:"));
const axisNames = cols.filter(c => !c.startsWith("occurred:"));

document.getElementById("meta").textContent =
  `${D.model} | ${D.draws.toLocaleString()} draws | seed ${D.seed} | probabilities are illustrative`;

function fill(sel, names, selected) {
  sel.innerHTML = "";
  names.forEach(n => {
    const o = document.createElement("option");
    o.value = n; o.textContent = n;
    if (n === selected) o.selected = true;
    sel.appendChild(o);
  });
}
const xsel = document.getElementById("xsel");
const ysel = document.getElementById("ysel");
const psel = document.getElementById("psel");
const osel = document.getElementById("osel");
fill(xsel, axisNames, inputNames[0]);
fill(ysel, axisNames, outputNames[0]);
fill(psel, inputNames, inputNames[0]);
fill(osel, outputNames, outputNames[0]);
[xsel, ysel, psel, osel].forEach(s => s.onchange = update);

const scenSel = {};
const scen = document.getElementById("scen");
eventCols.forEach(c => {
  const name = c.slice("occurred:".length);
  const wrap = document.createElement("div");
  wrap.innerHTML = `<label>${name} event</label>`;
  const s = document.createElement("select");
  s.innerHTML = '<option value="any">any</option><option value="1">occurred</option><option value="0">did not occur</option>';
  s.onchange = update;
  wrap.appendChild(s);
  scen.appendChild(wrap);
  scenSel[c] = s;
});

const range = {};
const filters = document.getElementById("filters");
inputNames.forEach((n, i) => {
  const col = colIndex[n];
  let mn = Infinity, mx = -Infinity;
  rows.forEach(r => { mn = Math.min(mn, r[col]); mx = Math.max(mx, r[col]); });
  range[n] = { lo: mn, hi: mx, minAll: mn, maxAll: mx };
  const wrap = document.createElement("div");
  wrap.innerHTML = `<label>${n} (${D.inputs[i].unit})</label><div class="range"><input type="number" step="any"><input type="number" step="any"></div>`;
  const [a, b] = wrap.querySelectorAll("input");
  a.value = mn; b.value = mx;
  a.oninput = b.oninput = () => { range[n].lo = +a.value; range[n].hi = +b.value; update(); };
  range[n].inputs = [a, b];
  filters.appendChild(wrap);
});

document.getElementById("reset").onclick = () => {
  inputNames.forEach(n => {
    const r = range[n];
    r.lo = r.minAll; r.hi = r.maxAll;
    r.inputs[0].value = r.minAll; r.inputs[1].value = r.maxAll;
  });
  Object.values(scenSel).forEach(s => s.value = "any");
  update();
};

function selected() {
  const out = [];
  for (const r of rows) {
    let ok = true;
    for (const n of inputNames) {
      const v = r[colIndex[n]];
      if (v < range[n].lo || v > range[n].hi) { ok = false; break; }
    }
    if (ok) {
      for (const c of eventCols) {
        const want = scenSel[c].value;
        if (want !== "any" && String(r[colIndex[c]]) !== want) { ok = false; break; }
      }
    }
    if (ok) out.push(r);
  }
  return out;
}

function quant(arr, q) {
  if (!arr.length) return NaN;
  const s = [...arr].sort((a, b) => a - b);
  const pos = q * (s.length - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

function corr(xs, ys) {
  const n = xs.length;
  if (n < 3) return NaN;
  const mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) ** 2;
    syy += (ys[i] - my) ** 2;
  }
  return sxy / Math.sqrt(sxx * syy);
}

function fmt(v, d = 2) {
  return Number.isFinite(v) ? v.toLocaleString(undefined, { maximumFractionDigits: d }) : "n/a";
}

function ext(arr) {
  let mn = Infinity, mx = -Infinity;
  arr.forEach(v => { mn = Math.min(mn, v); mx = Math.max(mx, v); });
  if (mn === mx) { mn -= 0.5; mx += 0.5; }
  return [mn, mx];
}

function stat(name, arr) {
  return `<div class="stat">${name}<b>P10 ${fmt(quant(arr, 0.1))} | P50 ${fmt(quant(arr, 0.5))} | P90 ${fmt(quant(arr, 0.9))}</b></div>`;
}

function layout(title, xt, yt) {
  return {
    title: { text: title, font: { size: 12, color: "#c8cdd6" } },
    paper_bgcolor: "#0f1115", plot_bgcolor: "#0f1115",
    font: { family: "IBM Plex Mono, monospace", color: "#c8cdd6", size: 11 },
    xaxis: { title: xt, gridcolor: "#252830", zerolinecolor: "#252830" },
    yaxis: { title: yt, gridcolor: "#252830", zerolinecolor: "#252830" },
    margin: { l: 60, r: 20, t: 36, b: 44 },
    legend: { orientation: "h", y: 1.15 },
  };
}

const AMBER = [[0, "#0f1115"], [0.5, "#a07200"], [1, "#f0a500"]];
const EMPTY = { title: { text: "Not enough draws in view", font: { color: "#5c6370" } }, paper_bgcolor: "#0f1115", plot_bgcolor: "#0f1115" };

function update() {
  const sel = selected();
  const xName = xsel.value, yName = ysel.value;
  const xi = colIndex[xName], yi = colIndex[yName];
  const xs = sel.map(r => r[xi]), ys = sel.map(r => r[yi]);

  document.getElementById("readout").innerHTML =
    `<div class="stat">Draws in view<b>${sel.length.toLocaleString()} of ${rows.length.toLocaleString()}</b></div>` +
    stat(xName, xs) + stat(yName, ys) +
    `<div class="stat">Correlation (X, Y)<b>${fmt(corr(xs, ys), 2)}</b></div>`;

  if (sel.length >= 10) {
    const bins = 40;
    const [x0, x1] = ext(xs), [y0, y1] = ext(ys);
    const dx = (x1 - x0) / bins, dy = (y1 - y0) / bins;
    const H = Array.from({ length: bins }, () => new Array(bins).fill(0));
    xs.forEach((x, k) => {
      const i = Math.min(bins - 1, Math.floor((x - x0) / dx));
      const j = Math.min(bins - 1, Math.floor((ys[k] - y0) / dy));
      H[j][i] += 1;
    });
    const z = H.map(row => row.map(c => c / (sel.length * dx * dy)));
    const xc = Array.from({ length: bins }, (_, i) => x0 + (i + 0.5) * dx);
    const yc = Array.from({ length: bins }, (_, j) => y0 + (j + 0.5) * dy);
    Plotly.react("heat", [{ type: "heatmap", x: xc, y: yc, z, colorscale: AMBER, colorbar: { title: "density" } }],
      layout(`Joint density: ${yName} vs ${xName}`, xName, yName));
    Plotly.react("hist", [{ type: "histogram", x: ys, nbinsx: 40, marker: { color: "#f0a500" } }],
      layout(`Distribution of ${yName}`, yName, "draws"));
  } else {
    Plotly.react("heat", [], EMPTY);
    Plotly.react("hist", [], EMPTY);
  }

  const pi = colIndex[psel.value], oi = colIndex[osel.value];
  const pv = sel.map(r => r[pi]);
  if (sel.length >= 50) {
    const B = 10;
    const [p0, p1] = ext(pv);
    const w = (p1 - p0) / B;
    const groups = Array.from({ length: B }, () => []);
    sel.forEach(r => groups[Math.min(B - 1, Math.floor((r[pi] - p0) / w))].push(r[oi]));
    const cx = [], p10 = [], p50 = [], p90 = [];
    groups.forEach((g, b) => {
      if (g.length < 5) return;
      cx.push(p0 + (b + 0.5) * w);
      p10.push(quant(g, 0.1)); p50.push(quant(g, 0.5)); p90.push(quant(g, 0.9));
    });
    Plotly.react("profile", [
      { x: cx, y: p10, name: "P10", mode: "lines+markers", line: { color: "#3498db" } },
      { x: cx, y: p50, name: "P50", mode: "lines+markers", line: { color: "#f0a500" } },
      { x: cx, y: p90, name: "P90", mode: "lines+markers", line: { color: "#e74c3c" } },
    ], layout(`${osel.value} quantiles across ${psel.value}`, psel.value, osel.value));
  } else {
    Plotly.react("profile", [], EMPTY);
  }
}

update();
</script>
</body></html>
"""
