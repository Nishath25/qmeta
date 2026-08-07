"""Render scratch/stress.json into a self-contained HTML stress-test dashboard."""
import json
from pathlib import Path

SC = Path(r"C:\Users\madas\qmeta\scratch\stress.json")
OUT = Path(r"C:\Users\madas\qmeta\scratch\qmeta_stress.html")
SERIES = {"ORB": "var(--orb)", "dip": "var(--dip)", "combined": "var(--comb)", "SPY": "var(--spy)"}


def friction_svg(costs, sharpes, live, be):
    W, H, ml, mr, mt, mb = 460, 250, 44, 14, 14, 34
    pw, ph = W - ml - mr, H - mt - mb
    xmax = max(costs)
    ylo, yhi = min(sharpes + [0]) - 0.1, max(sharpes) + 0.1
    def X(v): return ml + v / xmax * pw
    def Y(v): return mt + (yhi - v) / (yhi - ylo) * ph
    pts = "M" + " L".join(f"{X(c):.1f},{Y(s):.1f}" for c, s in zip(costs, sharpes))
    z0 = Y(0)
    grid = "".join(f'<text x="{ml-6}" y="{Y(v)+3:.0f}" class="tick" text-anchor="end">{v:.1f}</text>'
                   f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{ml+pw}" y2="{Y(v):.1f}" class="grid"/>'
                   for v in [0.0, 0.5, 1.0, 1.5])
    marks = (f'<line x1="{X(live):.1f}" y1="{mt}" x2="{X(live):.1f}" y2="{mt+ph}" class="mlive"/>'
             f'<text x="{X(live)+3:.1f}" y="{mt+12}" class="mlab" fill="var(--good)">live {live}R</text>')
    if be:
        marks += (f'<line x1="{X(be):.1f}" y1="{mt}" x2="{X(be):.1f}" y2="{mt+ph}" class="mbe"/>'
                  f'<text x="{X(be)+3:.1f}" y="{mt+26}" class="mlab" fill="var(--bad)">breakeven {be}R</text>')
    return (f'<svg viewBox="0 0 {W} {H}" class="ch" role="img">{grid}'
            f'<line x1="{ml}" y1="{z0:.1f}" x2="{ml+pw}" y2="{z0:.1f}" class="zero"/>'
            f'{marks}<path d="{pts}" class="fline"/>'
            f'<text x="{ml+pw/2:.0f}" y="{H-4}" class="tick" text-anchor="middle">cost per trade (R)</text></svg>')


def crisis_svg(crises):
    names = list(crises.keys())
    series = ["ORB", "dip", "combined", "SPY"]
    W, H, ml, mr, mt, mb = 620, 300, 40, 12, 16, 58
    pw, ph = W - ml - mr, H - mt - mb
    vals = [crises[c][s] for c in names for s in series if crises[c][s] is not None]
    ylo, yhi = min(vals + [0]) - 4, max(vals + [0]) + 4
    def Y(v): return mt + (yhi - v) / (yhi - ylo) * ph
    z0 = Y(0)
    gw = pw / len(names)
    bw = gw * 0.9 / len(series)
    bars, labels = "", ""
    for gi, c in enumerate(names):
        gx = ml + gi * gw + gw * 0.05
        for si, s in enumerate(series):
            v = crises[c][s]
            if v is None:
                continue
            x = gx + si * bw
            y0, y1 = min(Y(0), Y(v)), max(Y(0), Y(v))
            bars += f'<rect x="{x:.1f}" y="{y0:.1f}" width="{bw-1:.1f}" height="{max(1,y1-y0):.1f}" fill="{SERIES[s]}" rx="1"/>'
        labels += f'<text x="{ml+gi*gw+gw/2:.0f}" y="{H-40}" class="tick" text-anchor="middle">{c}</text>'
    yt = "".join(f'<text x="{ml-6}" y="{Y(v)+3:.0f}" class="tick" text-anchor="end">{v:+.0f}%</text>'
                 f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{ml+pw}" y2="{Y(v):.1f}" class="grid"/>'
                 for v in range(int(ylo // 10 * 10), int(yhi) + 10, 10))
    leg = "".join(f'<span><span class="sw" style="background:{SERIES[s]}"></span>{s}</span>' for s in series)
    return (f'<svg viewBox="0 0 {W} {H}" class="ch" role="img">{yt}{bars}'
            f'<line x1="{ml}" y1="{z0:.1f}" x2="{ml+pw}" y2="{z0:.1f}" class="zero"/>{labels}</svg>'
            f'<div class="legend">{leg}</div>')


def hist_svg(orb_dd, comb_dd):
    W, H, ml, mr, mt, mb = 620, 250, 30, 12, 12, 30
    pw, ph = W - ml - mr, H - mt - mb
    lo = min(min(orb_dd), min(comb_dd))
    bins = 32
    edges = [lo + (0 - lo) * i / bins for i in range(bins + 1)]
    def hist(data):
        h = [0] * bins
        for v in data:
            k = min(bins - 1, max(0, int((v - lo) / (0 - lo) * bins)))
            h[k] += 1
        return h
    ho, hc = hist(orb_dd), hist(comb_dd)
    hmax = max(max(ho), max(hc)) or 1
    def X(v): return ml + (v - lo) / (0 - lo) * pw
    def bars(h, color):
        s = ""
        for i in range(bins):
            x = X(edges[i]); w = X(edges[i + 1]) - x
            bh = h[i] / hmax * ph
            s += f'<rect x="{x:.1f}" y="{mt+ph-bh:.1f}" width="{max(0.5,w-0.5):.1f}" height="{bh:.1f}" fill="{color}" opacity="0.55"/>'
        return s
    xt = "".join(f'<text x="{X(v):.1f}" y="{H-4}" class="tick" text-anchor="middle">{v:.0f}%</text>'
                 for v in range(int(lo // 10 * 10), 1, 10))
    return (f'<svg viewBox="0 0 {W} {H}" class="ch" role="img">{xt}'
            f'{bars(ho, "var(--orb)")}{bars(hc, "var(--comb)")}</svg>'
            f'<div class="legend"><span><span class="sw" style="background:var(--orb)"></span>ORB</span>'
            f'<span><span class="sw" style="background:var(--comb)"></span>combined</span>'
            f'<span>2000 block-bootstrap paths &middot; distribution of max drawdown</span></div>')


def build_html(sc):
    st = sc["sharpe_table"]
    f = sc["friction"]
    mc = sc["monte_carlo"]
    wc = sc["worst_case"]
    trows = ""
    for k, v in st.items():
        raw = "&ndash;" if v["raw"] is None else f'{v["raw"]:.2f}'
        dcls = "good" if v["deflated"] >= 0.9 else ("warn" if v["deflated"] >= 0.5 else "bad")
        sel = ' class="hl"' if k.startswith("Combined") else ""
        trows += (f'<tr{sel}><td>{k}</td><td>{raw}</td><td>{v["net"]:.2f}</td>'
                  f'<td><span class="pill {dcls}">{v["deflated"]*100:.0f}%</span></td></tr>')
    crows = ""
    for k, v in wc.items():
        fat = v["maxdd"] < -v["tp_proj_maxdd"] - 0.1
        crows += (f'<tr><td>{k}</td><td>{v["maxdd"]}%</td><td class="muted">{v["tp_proj_maxdd"]}%</td>'
                  f'<td>{v["worst_day"]}%</td><td>{v["cvar5"]}%</td><td>{v["longest_underwater_days"]}d</td>'
                  f'<td>{"⚠ fat tails" if fat else "ok"}</td></tr>')
    return f'''<style>
:root{{--bg:#f4f6f9;--surface:#fff;--ink:#141b29;--muted:#5b6675;--border:#e3e7ee;
  --orb:#4E79A7;--dip:#F28E2B;--comb:#59A14F;--spy:#9aa4b2;
  --good:#2E9E5B;--warn:#C0900A;--bad:#D64550;--shadow:0 1px 3px rgba(20,30,50,.07),0 8px 24px rgba(20,30,50,.05);}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0e1420;--surface:#161d2b;--ink:#e7ecf4;--muted:#93a0b4;--border:#25304250;
  --orb:#7BA6D0;--dip:#F4A85B;--comb:#7DC46F;--spy:#79828f;--good:#4CC47E;--warn:#E4B740;--bad:#F0787F;
  --shadow:0 1px 3px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);}}}}
:root[data-theme="light"]{{--bg:#f4f6f9;--surface:#fff;--ink:#141b29;--muted:#5b6675;--border:#e3e7ee;--orb:#4E79A7;--dip:#F28E2B;--comb:#59A14F;--spy:#9aa4b2;--good:#2E9E5B;--warn:#C0900A;--bad:#D64550;}}
:root[data-theme="dark"]{{--bg:#0e1420;--surface:#161d2b;--ink:#e7ecf4;--muted:#93a0b4;--border:#25304250;--orb:#7BA6D0;--dip:#F4A85B;--comb:#7DC46F;--spy:#79828f;--good:#4CC47E;--warn:#E4B740;--bad:#F0787F;}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.5;font-variant-numeric:tabular-nums}}
.wrap{{max-width:960px;margin:0 auto;padding:40px 24px 64px}}
.eyebrow{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:600}}
h1{{font-size:28px;line-height:1.15;margin:6px 0 6px;text-wrap:balance}}
.lede{{color:var(--muted);font-size:15px;max-width:70ch;margin:0 0 22px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}@media(max-width:780px){{.grid2{{grid-template-columns:1fr}}}}
.panel{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:18px 20px;box-shadow:var(--shadow);margin-bottom:20px}}
.panel h2{{font-size:16px;margin:0 0 3px}} .panel .sub{{font-size:13px;color:var(--muted);margin:0 0 12px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid var(--border);padding:7px 9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
th{{color:var(--muted);font-weight:600;font-size:12px}} tr.hl td{{font-weight:750}} td.muted{{color:var(--muted)}}
.pill{{font-size:12px;font-weight:700;padding:2px 8px;border-radius:20px}}
.pill.good{{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good)}}
.pill.warn{{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}}
.pill.bad{{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}}
.ch{{width:100%;height:auto}} .ch .grid{{stroke:var(--border)}} .ch .zero{{stroke:var(--muted);stroke-width:1.2}}
.ch .tick{{fill:var(--muted);font-size:10px}} .ch .fline{{fill:none;stroke:var(--orb);stroke-width:2.4}}
.ch .mlive{{stroke:var(--good);stroke-width:1.4;stroke-dasharray:3 2}} .ch .mbe{{stroke:var(--bad);stroke-width:1.4;stroke-dasharray:3 2}}
.ch .mlab{{font-size:10px;font-weight:600}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:8px}}
.legend span{{display:inline-flex;align-items:center;gap:6px}}.sw{{width:12px;height:12px;border-radius:3px;display:inline-block}}
.kpis{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:22px}}
.kpi{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 18px;box-shadow:var(--shadow);flex:1;min-width:170px}}
.kpi .k{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}.kpi .v{{font-size:24px;font-weight:750;margin-top:2px}}.kpi .s{{font-size:12px;color:var(--muted)}}
.note{{font-size:13px;color:var(--muted);border-left:3px solid var(--border);padding-left:12px;margin-top:12px}}
footer{{color:var(--muted);font-size:12px;margin-top:22px}}
</style>
<div class="wrap">
  <div class="eyebrow">qmeta &middot; institutional stress-test gauntlet</div>
  <h1>How real is the edge under friction, crises, and resampled paths?</h1>
  <p class="lede">The López de Prado risk battery run on your live book, {sc["window"]["start"]}&ndash;{sc["window"]["end"]}:
  tiered friction + breakeven cost, Raw vs Net vs Deflated Sharpe, crisis &amp; volatility regimes, and a
  2000-path Monte-Carlo of the drawdown you should actually plan for.</p>

  <div class="kpis">
    <div class="kpi"><div class="k">Combined DSR</div><div class="v" style="color:var(--good)">{st["Combined (50/50)"]["deflated"]*100:.0f}%</div>
      <div class="s">near the 95% bar (ORB 75%, dip 70% alone)</div></div>
    <div class="kpi"><div class="k">ORB friction headroom</div><div class="v">{f["headroom_x"]}&times;</div>
      <div class="s">breakeven {f["breakeven_bps"]} bps vs live {f["live_cost_bps"]} bps/trade</div></div>
    <div class="kpi"><div class="k">Combined worst-path DD</div><div class="v">{mc["combined"]["maxdd_p05"]}%</div>
      <div class="s">5th-pct of 2000 resampled paths; P(loss) {mc["combined"]["p_negative"]}%</div></div>
  </div>

  <div class="panel">
    <h2>Raw vs Net vs Deflated Sharpe</h2>
    <p class="sub">Raw = frictionless, Net = after live costs, Deflated = corrected for {sc.get("n_trials",19) if isinstance(sc.get("n_trials",19),int) else 19} trials (DSR). The headline the framework demands.</p>
    <table><thead><tr><th>strategy</th><th>Raw</th><th>Net</th><th>Deflated (DSR)</th></tr></thead><tbody>{trows}</tbody></table>
    <p class="note">Neither leg clears the 95% multiple-testing bar alone, but <b>together they reach 94%</b> &mdash;
    the diversification doesn't just raise Sharpe, it makes the edge statistically credible.</p>
  </div>

  <div class="grid2">
    <div class="panel">
      <h2>Friction gauntlet (ORB)</h2>
      <p class="sub">Net Sharpe as per-trade cost rises. Breakeven is ~{f["headroom_x"]}&times; today's cost.</p>
      {friction_svg(f["costs"], f["sharpes"], f["live_cost_R"], f["breakeven_R"])}
    </div>
    <div class="panel">
      <h2>Monte-Carlo drawdown stress</h2>
      <p class="sub">Block-bootstrap resampling &mdash; the drawdown distribution, not just the one realized path.</p>
      {hist_svg(mc["ORB"]["dd_hist"], mc["combined"]["dd_hist"])}
    </div>
  </div>

  <div class="panel">
    <h2>Crisis performance &mdash; total return through each crash</h2>
    <p class="sub">ORB is a crisis <b>hedge</b> (intraday, flat overnight); the dip is a crisis <b>landmine</b>; the blend cushions.</p>
    {crisis_svg(sc["regimes"]["crises"])}
  </div>

  <div class="panel">
    <h2>Worst case &mdash; realized vs Gaussian projection</h2>
    <p class="sub">Realized max drawdown vs the Triple-Penance (Gaussian) projection. Realized &gt; projected ⇒ fat tails.</p>
    <table><thead><tr><th>strategy</th><th>realized maxDD</th><th>projected</th><th>worst day</th><th>CVaR&#8325;</th><th>longest underwater</th><th></th></tr></thead><tbody>{crows}</tbody></table>
    <p class="note">Every book drew down <b>worse than the Gaussian model predicts</b> (the dip: &minus;40% realized vs &minus;21% projected).
    Size for fat tails &mdash; the closed-form projection is a floor, not a ceiling.</p>
  </div>

  <footer>qmeta stress gauntlet &middot; friction/breakeven, DSR, regime &amp; crisis, block-bootstrap Monte-Carlo, Triple-Penance &middot; no lookahead.</footer>
</div>'''


if __name__ == "__main__":
    OUT.write_text(build_html(json.loads(SC.read_text())), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
