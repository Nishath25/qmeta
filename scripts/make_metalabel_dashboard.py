"""Render scratch/metalabel.json into a self-contained HTML dashboard:
raw vs meta-labeled ORB equity, mode comparison, feature importances."""
import json
from pathlib import Path

SC = Path(r"C:\Users\madas\qmeta\scratch\metalabel.json")
OUT = Path(r"C:\Users\madas\qmeta\scratch\qmeta_metalabel.html")


def equity_svg(dates, raw, meta):
    W, H, ml, mr, mt, mb = 900, 320, 46, 14, 16, 34
    pw, ph = W - ml - mr, H - mt - mb
    n = len(raw)
    lo = min(min(raw), min(meta))
    hi = max(max(raw), max(meta))
    def X(i): return ml + i / (n - 1) * pw
    def Y(v): return mt + (hi - v) / (hi - lo) * ph
    def path(series):
        step = max(1, n // 1200)
        pts = [(X(i), Y(series[i])) for i in range(0, n, step)]
        pts.append((X(n - 1), Y(series[-1])))
        return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    yt = ""
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        v = lo + frac * (hi - lo)
        yt += (f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{ml+pw}" y2="{Y(v):.1f}" class="grid"/>'
               f'<text x="{ml-8}" y="{Y(v)+4:.1f}" class="tick" text-anchor="end">{v:.1f}x</text>')
    yr_ticks = ""
    for i in range(0, n, max(1, n // 6)):
        yr_ticks += f'<text x="{X(i):.1f}" y="{H-8}" class="tick" text-anchor="middle">{dates[i][:4]}</text>'
    return f'''<svg viewBox="0 0 {W} {H}" class="eq" role="img" aria-label="raw vs meta-labeled equity">
      {yt}{yr_ticks}
      <path d="{path(raw)}" class="lraw"/>
      <path d="{path(meta)}" class="lmeta"/>
    </svg>'''


def build_html(sc):
    r = sc["raw"]
    modes = sc["modes"]
    best = sc["best_mode"]
    bm = modes[best]
    eq = sc["equity"]
    imp = sc["importances"]
    rob = sc.get("robustness", {})
    wu = rob.get("with_uw", {}).get("threshold", bm["d_sharpe"])
    nu = rob.get("without_uw", {}).get("threshold", 0.0)
    maxi = max(imp.values()) or 1.0
    imp_bars = "".join(
        f'<div class="ir"><span class="iname">{k}</span>'
        f'<span class="itrack"><span class="ifill" style="width:{v/maxi*100:.0f}%"></span></span>'
        f'<span class="ival">{v:.2f}</span></div>' for k, v in imp.items())
    rows = ""
    for k, v in modes.items():
        sel = ' class="best"' if k == best else ""
        ht = "&ndash;" if v["hit_taken"] is None else f'{v["hit_taken"]*100:.0f}%'
        rows += (f'<tr{sel}><td>{k}</td><td>{v["sharpe"]:.2f}</td>'
                 f'<td>{v["d_sharpe"]:+.2f}</td><td>{v["dsr"]*100:.0f}%</td>'
                 f'<td>{v["maxdd"]*100:.1f}%</td><td>{v["taken_frac"]*100:.0f}%</td><td>{ht}</td></tr>')
    ddcut = (r["maxdd"] - bm["maxdd"]) * 100
    return f'''<style>
:root{{--bg:#f5f7fa;--surface:#fff;--ink:#182031;--muted:#5c6675;--border:#e4e8ef;
  --raw:#8a94a6;--meta:#2E9E5B;--accent:#4E79A7;--good:#2E9E5B;
  --shadow:0 1px 3px rgba(20,30,50,.07),0 8px 24px rgba(20,30,50,.05);}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0e1420;--surface:#161d2b;--ink:#e7ecf4;--muted:#93a0b4;
  --border:#25304250;--raw:#6b7688;--meta:#4CC47E;--accent:#7BA6D0;--good:#4CC47E;
  --shadow:0 1px 3px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);}}}}
:root[data-theme="light"]{{--bg:#f5f7fa;--surface:#fff;--ink:#182031;--muted:#5c6675;--border:#e4e8ef;--raw:#8a94a6;--meta:#2E9E5B;--accent:#4E79A7;--good:#2E9E5B;}}
:root[data-theme="dark"]{{--bg:#0e1420;--surface:#161d2b;--ink:#e7ecf4;--muted:#93a0b4;--border:#25304250;--raw:#6b7688;--meta:#4CC47E;--accent:#7BA6D0;--good:#4CC47E;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.5;font-variant-numeric:tabular-nums}}
.wrap{{max-width:960px;margin:0 auto;padding:40px 24px 64px}}
.eyebrow{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:600}}
h1{{font-size:29px;line-height:1.15;margin:6px 0 6px;text-wrap:balance}}
.lede{{color:var(--muted);font-size:15px;max-width:66ch;margin:0 0 22px}}
.tiles{{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:24px}}
.tile{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 20px;box-shadow:var(--shadow);flex:1;min-width:190px}}
.tile .k{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
.tile .v{{font-size:27px;font-weight:750;margin-top:3px}}
.tile .d{{font-size:14px;font-weight:650;color:var(--good)}}
.tile .s{{font-size:12px;color:var(--muted);margin-top:2px}}
.panel{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:20px 22px;box-shadow:var(--shadow);margin-bottom:22px}}
.panel h2{{font-size:16px;margin:0 0 4px}} .panel .sub{{font-size:13px;color:var(--muted);margin:0 0 12px}}
.eq{{width:100%;height:auto}}
.eq .grid{{stroke:var(--border);stroke-width:1}} .eq .tick{{fill:var(--muted);font-size:10px}}
.eq .lraw{{fill:none;stroke:var(--raw);stroke-width:1.6}}
.eq .lmeta{{fill:none;stroke:var(--meta);stroke-width:2.2}}
.legend{{display:flex;gap:18px;font-size:13px;color:var(--muted);margin-top:8px}}
.legend span{{display:inline-flex;align-items:center;gap:7px}}
.sw{{width:14px;height:3px;border-radius:2px;display:inline-block}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid var(--border);padding:8px 10px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
th{{color:var(--muted);font-weight:600;font-size:12px}}
tr.best td{{font-weight:750;color:var(--meta)}}
.ir{{display:flex;align-items:center;gap:10px;margin:6px 0}}
.iname{{width:82px;font-size:13px;color:var(--muted)}}
.itrack{{flex:1;height:11px;background:color-mix(in srgb,var(--muted) 14%,transparent);border-radius:6px;overflow:hidden}}
.ifill{{display:block;height:100%;background:var(--accent);border-radius:6px}}
.ival{{width:38px;text-align:right;font-weight:650;font-size:13px}}
footer{{color:var(--muted);font-size:12px;margin-top:24px}}
</style>
<div class="wrap">
  <div class="eyebrow">qmeta &middot; meta-labeling the ORB fund</div>
  <h1>Does a &ldquo;will this breakout win?&rdquo; model improve the fund?</h1>
  <p class="lede">A secondary random forest predicts P(win) for each ORB breakout, trained strictly
  walk-forward (no lookahead). The bet is then sized by that probability. Out-of-sample
  {sc["oos_start"]}&ndash;{sc["oos_end"]}, {sc["n_trades"]:,} trades. Verdict: skipping the breakouts the
  model flags as losers <b style="color:var(--good)">raises Sharpe and cuts drawdown</b>.</p>

  <div class="tiles">
    <div class="tile"><div class="k">Raw ORB (OOS)</div><div class="v">{r["sharpe"]:.2f}</div>
      <div class="s">Sharpe &middot; DSR {r["dsr"]*100:.0f}% &middot; maxDD {r["maxdd"]*100:.1f}%</div></div>
    <div class="tile"><div class="k">Meta-labeled ({best})</div><div class="v">{bm["sharpe"]:.2f}
      <span class="d">{bm["d_sharpe"]:+.2f}</span></div>
      <div class="s">Sharpe &middot; DSR {bm["dsr"]*100:.0f}% &middot; maxDD {bm["maxdd"]*100:.1f}%</div></div>
    <div class="tile"><div class="k">Drawdown cut</div><div class="v">{ddcut:.1f} pts</div>
      <div class="s">{r["maxdd"]*100:.1f}% &rarr; {bm["maxdd"]*100:.1f}% (vol-matched)</div></div>
  </div>

  <div class="panel">
    <h2>Equity: raw vs meta-labeled (vol-matched, OOS)</h2>
    <p class="sub">Same volatility, so the gap is shape &mdash; the model concentrates capital in the breakouts that pay.</p>
    {equity_svg(eq["dates"], eq["raw"], eq["meta"])}
    <div class="legend"><span><span class="sw" style="background:var(--raw)"></span>Raw ORB</span>
      <span><span class="sw" style="background:var(--meta)"></span>Meta-labeled ({best})</span></div>
  </div>

  <div class="panel">
    <h2>Bet-sizing modes</h2>
    <p class="sub">The win is the <b>skip</b> (threshold), not continuous fading (linear actually hurts).</p>
    <table><thead><tr><th>mode</th><th>Sharpe</th><th>&Delta;</th><th>DSR</th><th>maxDD</th><th>taken</th><th>hit (taken)</th></tr></thead>
    <tbody>{rows}</tbody></table>
  </div>

  <div class="panel">
    <h2>What the model keys on</h2>
    <p class="sub">Feature importances (random forest).</p>
    {imp_bars}
  </div>

  <div class="panel" style="border-left:3px solid var(--meta)">
    <h2>Robustness &mdash; the honest caveat</h2>
    <p class="sub" style="margin:0">The gain is <b>leakage-free</b> (walk-forward traced end-to-end; vol-match is scale-invariant),
    but <b>contingent on the uniqueness weighting</b> (uw = 1/same-day-trade-count): threshold &Delta;Sharpe is
    <b style="color:var(--good)">{wu:+.2f} with uw</b> vs <b>{nu:+.2f} without</b>. Given uw it stays positive across
    seed, refit cadence, and warmup; drop uw and meta-labeling is roughly neutral. uw is not lookahead (it weights only
    past training rows). Also: best-of-3 sizing modes selected on OOS (DSR deflates for K=19); maxDD is vol-matched.</p>
  </div>
  <footer>Trained walk-forward (2y warmup, quarterly refit); features known at entry; ATR_MAX=3.0 build.</footer>
</div>'''


if __name__ == "__main__":
    sc = json.loads(SC.read_text())
    OUT.write_text(build_html(sc), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
