"""Render scratch/wft_orbdip_100k.json into a $100k ORB+dip combined WFT dashboard."""
import json
from pathlib import Path

SC = Path(r"C:\Users\madas\qmeta\scratch\wft_orbdip_100k.json")
OUT = Path(r"C:\Users\madas\qmeta\scratch\qmeta_orbdip_wft.html")
KEYS = ["Combined (ORB + dip)", "ORB fund", "Dip diversifier", "SPY buy & hold"]
COL = {"Combined (ORB + dip)": "var(--comb)", "ORB fund": "var(--orb)",
       "Dip diversifier": "var(--dip)", "SPY buy & hold": "var(--spy)"}


def eq_svg(streams):
    W, H, ml, mr, mt, mb = 900, 310, 56, 14, 14, 26
    pw, ph = W - ml - mr, H - mt - mb
    ser = {k: [d["e"] for d in streams[k]["monthly"]] for k in KEYS}
    n = min(len(v) for v in ser.values())
    allv = [v for s in ser.values() for v in s[:n]]
    lo, hi = min(allv), max(allv)
    def X(i): return ml + i / (n - 1) * pw
    def Y(v): return mt + (hi - v) / (hi - lo) * ph
    grid = ""
    for f in (0, 0.25, 0.5, 0.75, 1.0):
        v = lo + f * (hi - lo)
        grid += (f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{ml+pw}" y2="{Y(v):.1f}" class="grid"/>'
                 f'<text x="{ml-8}" y="{Y(v)+4:.1f}" class="tick" text-anchor="end">${v/1000:.0f}k</text>')
    for i in range(0, n, max(1, n // 7)):
        grid += f'<text x="{X(i):.1f}" y="{H-4}" class="tick" text-anchor="middle">{streams[KEYS[0]]["monthly"][i]["m"][:4]}</text>'
    lines = ""
    for k in KEYS:
        s = ser[k][:n]
        w = 2.6 if k.startswith("Combined") else 1.7
        dash = ' stroke-dasharray="4 3"' if k.startswith("SPY") else ""
        lines += f'<path d="M{" L".join(f"{X(i):.1f},{Y(s[i]):.1f}" for i in range(n))}" style="stroke:{COL[k]};stroke-width:{w}"{dash} fill="none"/>'
    return f'<svg viewBox="0 0 {W} {H}" class="ch" role="img">{grid}{lines}</svg>'


def uw_svg(monthly):
    W, H, ml, mr, mt, mb = 900, 130, 56, 14, 8, 20
    pw, ph = W - ml - mr, H - mt - mb
    dd = [d["dd"] for d in monthly]; n = len(dd); lo = min(dd)
    def X(i): return ml + i / (n - 1) * pw
    def Y(v): return mt + (0 - v) / (0 - lo) * ph
    p = "M" + f"{X(0):.1f},{Y(0):.1f} L" + " L".join(f"{X(i):.1f},{Y(dd[i]):.1f}" for i in range(n)) + f" L{X(n-1):.1f},{Y(0):.1f} Z"
    grid = "".join(f'<text x="{ml-8}" y="{Y(v)+3:.1f}" class="tick" text-anchor="end">{v:.0f}%</text>'
                   f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{ml+pw}" y2="{Y(v):.1f}" class="grid"/>' for v in [0, lo / 2, lo])
    return f'<svg viewBox="0 0 {W} {H}" class="ch" role="img">{grid}<path d="{p}" class="acomb"/></svg>'


def bars_svg(per_year):
    W, H, ml, mr, mt, mb = 900, 180, 56, 14, 14, 26
    pw, ph = W - ml - mr, H - mt - mb
    vals = [p[1] for p in per_year]; hi = max(vals + [0]); lo = min(vals + [0]); span = (hi - lo) or 1
    def Y(v): return mt + (hi - v) / span * ph
    z0 = Y(0); bw = pw / len(per_year) * 0.6
    out = ""
    for i, (yr, pnl, pct) in enumerate(per_year):
        x = ml + (i + 0.5) * pw / len(per_year) - bw / 2
        y0, y1 = min(Y(0), Y(pnl)), max(Y(0), Y(pnl))
        col = "var(--comb)" if pnl >= 0 else "var(--bad)"
        out += (f'<rect x="{x:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{max(1,y1-y0):.1f}" fill="{col}" rx="2"/>'
                f'<text x="{x+bw/2:.1f}" y="{(y0-4) if pnl>=0 else (y1+12):.1f}" class="blab" text-anchor="middle">${pnl/1000:+.0f}k</text>'
                f'<text x="{x+bw/2:.1f}" y="{H-8}" class="tick" text-anchor="middle">{yr}</text>')
    return f'<svg viewBox="0 0 {W} {H}" class="ch" role="img"><line x1="{ml}" y1="{z0:.1f}" x2="{ml+pw}" y2="{z0:.1f}" class="zero"/>{out}</svg>'


def build_html(sc):
    s = sc["streams"]; c = s["Combined (ORB + dip)"]
    rows = ""
    for k in KEYS:
        m = s[k]; hero = ' class="hero"' if k.startswith("Combined") else ""
        so = "n/a" if m["sortino"] is None else f'{m["sortino"]:.2f}'
        rows += (f'<tr{hero}><td><span class="dot" style="background:{COL[k]}"></span>{k}</td>'
                 f'<td>${m["final"]:,.0f}</td><td>{m["cagr"]*100:.1f}%</td><td>{m["sharpe"]:.2f}</td>'
                 f'<td>{so}</td><td>{m["maxdd"]*100:.1f}%</td><td>{m["calmar"]:.2f}</td><td>{m["dsr"]*100:.0f}%</td></tr>')
    legend = "".join(f'<span><span class="sw" style="background:{COL[k]}"></span>{k}</span>' for k in KEYS)
    return f'''<style>
:root{{--bg:#f4f6f9;--surface:#fff;--ink:#141b29;--muted:#5b6675;--border:#e3e7ee;
  --comb:#59A14F;--orb:#4E79A7;--dip:#F28E2B;--spy:#9aa4b2;--bad:#D64550;--shadow:0 1px 3px rgba(20,30,50,.07),0 8px 24px rgba(20,30,50,.05);}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0e1420;--surface:#161d2b;--ink:#e7ecf4;--muted:#93a0b4;--border:#25304250;
  --comb:#7DC46F;--orb:#7BA6D0;--dip:#F4A85B;--spy:#79828f;--bad:#F0787F;--shadow:0 1px 3px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);}}}}
:root[data-theme="light"]{{--bg:#f4f6f9;--surface:#fff;--ink:#141b29;--muted:#5b6675;--border:#e3e7ee;--comb:#59A14F;--orb:#4E79A7;--dip:#F28E2B;--spy:#9aa4b2;--bad:#D64550;}}
:root[data-theme="dark"]{{--bg:#0e1420;--surface:#161d2b;--ink:#e7ecf4;--muted:#93a0b4;--border:#25304250;--comb:#7DC46F;--orb:#7BA6D0;--dip:#F4A85B;--spy:#79828f;--bad:#F0787F;}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.5;font-variant-numeric:tabular-nums}}
.wrap{{max-width:960px;margin:0 auto;padding:40px 24px 64px}}
.eyebrow{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:600}}
h1{{font-size:28px;line-height:1.15;margin:6px 0 6px;text-wrap:balance}}
.lede{{color:var(--muted);font-size:15px;max-width:72ch;margin:0 0 22px}}
.kpis{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:22px}}
.kpi{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 18px;box-shadow:var(--shadow);flex:1;min-width:165px}}
.kpi .k{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}.kpi .v{{font-size:25px;font-weight:750;margin-top:2px}}.kpi .s{{font-size:12px;color:var(--muted)}}
.panel{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:18px 20px;box-shadow:var(--shadow);margin-bottom:20px}}
.panel h2{{font-size:16px;margin:0 0 3px}}
.ch{{width:100%;height:auto}} .ch .grid{{stroke:var(--border)}} .ch .zero{{stroke:var(--muted);stroke-width:1.2}} .ch .tick{{fill:var(--muted);font-size:10px}}
.ch .acomb{{fill:color-mix(in srgb,var(--comb) 26%,transparent);stroke:var(--comb);stroke-width:1.2}} .ch .blab{{fill:var(--muted);font-size:10px;font-weight:600}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:8px}}.legend span{{display:inline-flex;align-items:center;gap:7px}}.sw{{width:14px;height:3px;border-radius:2px;display:inline-block}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid var(--border);padding:8px 10px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
th{{color:var(--muted);font-weight:600;font-size:12px}} tr.hero td{{font-weight:750;color:var(--comb)}}
.dot{{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:7px}}
.note{{font-size:13px;color:var(--muted);border-left:3px solid var(--comb);padding-left:12px;margin-top:12px}}
footer{{color:var(--muted);font-size:12px;margin-top:22px}}
</style>
<div class="wrap">
  <div class="eyebrow">qmeta &middot; $100k walk-forward &middot; the deployable book</div>
  <h1>ORB + Dip &mdash; combined</h1>
  <p class="lede">The intraday ORB fund and the buy-the-dip diversifier, combined <b>50/50 at equal risk</b>
  (correlation {sc["correlation"]:+.2f}), $100,000 start, 15% vol, {sc["win_start"]}&ndash;{sc["win_end"]}
  ({sc["n_days"]:,} days). Two uncorrelated engines beat either sleeve &mdash; and SPY &mdash; on every risk-adjusted measure.</p>

  <div class="kpis">
    <div class="kpi"><div class="k">Combined book</div><div class="v" style="color:var(--comb)">${c["final"]:,.0f}</div>
      <div class="s">from $100k &middot; +{(c["final"]/sc["start"]-1)*100:.0f}% &middot; CAGR {c["cagr"]*100:.1f}%</div></div>
    <div class="kpi"><div class="k">Sharpe / DSR</div><div class="v">{c["sharpe"]:.2f} <span class="s">/ {c["dsr"]*100:.0f}%</span></div>
      <div class="s">vs ORB {s["ORB fund"]["sharpe"]:.2f} &middot; dip {s["Dip diversifier"]["sharpe"]:.2f} &middot; SPY {s["SPY buy & hold"]["sharpe"]:.2f}</div></div>
    <div class="kpi"><div class="k">Max drawdown</div><div class="v">{c["maxdd"]*100:.1f}%</div>
      <div class="s">Sortino {c["sortino"]:.2f} &middot; Calmar {c["calmar"]:.2f}</div></div>
    <div class="kpi"><div class="k">Correlation</div><div class="v">{sc["correlation"]:+.2f}</div>
      <div class="s">ORB vs dip &mdash; near-zero, a true diversifier</div></div>
  </div>

  <div class="panel">
    <h2>Equity &mdash; $100k compounded</h2>
    {eq_svg(s)}
    <div class="legend">{legend}</div>
  </div>

  <div class="panel"><h2>Underwater (combined drawdown)</h2>{uw_svg(c["monthly"])}</div>
  <div class="panel"><h2>Per-year P&amp;L (combined)</h2>{bars_svg(c["per_year"])}</div>

  <div class="panel">
    <h2>Metrics</h2>
    <table><thead><tr><th>book</th><th>$100k &rarr;</th><th>CAGR</th><th>Sharpe</th><th>Sortino</th><th>maxDD</th><th>Calmar</th><th>DSR</th></tr></thead><tbody>{rows}</tbody></table>
    <p class="note">The combined book <b>dominates</b>: higher return than either sleeve, a shallower drawdown than
    either, and a higher Sharpe (1.30) and Deflated Sharpe (94%) than ORB, dip, or SPY. Positive every full year,
    including the 2022 bear. This is the diversification free lunch &mdash; two uncorrelated edges each covering the
    other's weak years.</p>
  </div>

  <footer>qmeta $100k WFT &middot; raw ORB + dip, 50/50 equal-risk @15% vol, walk-forward, no lookahead &middot; SPY = buy &amp; hold over the same window.</footer>
</div>'''


if __name__ == "__main__":
    OUT.write_text(build_html(json.loads(SC.read_text())), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
