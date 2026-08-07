"""Three-way comparison over the full 2018-2026 window: raw ORB+dip (robust book)
vs meta-filtered (trained 2018, raw warmup) vs meta-filtered (trained 2016).
Reads wft_combined_full.json + wft_train2016.json."""
import json
from pathlib import Path

FULL = Path(r"C:\Users\madas\qmeta\scratch\wft_combined_full.json")
TR16 = Path(r"C:\Users\madas\qmeta\scratch\wft_train2016.json")
OUT = Path(r"C:\Users\madas\qmeta\scratch\qmeta_filter_compare.html")


def eq_svg(series):
    # series: list of (label, color, dash, monthly_e)
    W, H, ml, mr, mt, mb = 900, 320, 58, 14, 14, 26
    pw, ph = W - ml - mr, H - mt - mb
    n = min(len(s[3]) for s in series)
    allv = [v for s in series for v in s[3][:n]]
    lo, hi = min(allv), max(allv)
    def X(i): return ml + i / (n - 1) * pw
    def Y(v): return mt + (hi - v) / (hi - lo) * ph
    grid = ""
    for f in (0, 0.25, 0.5, 0.75, 1.0):
        v = lo + f * (hi - lo)
        grid += (f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{ml+pw}" y2="{Y(v):.1f}" class="grid"/>'
                 f'<text x="{ml-8}" y="{Y(v)+4:.1f}" class="tick" text-anchor="end">${v/1000:.0f}k</text>')
    months = series[0][4]
    for i in range(0, n, max(1, n // 7)):
        grid += f'<text x="{X(i):.1f}" y="{H-4}" class="tick" text-anchor="middle">{months[i][:4]}</text>'
    lines = ""
    for label, color, dash, e, _m in series:
        e = e[:n]
        p = "M" + " L".join(f"{X(i):.1f},{Y(e[i]):.1f}" for i in range(n))
        lines += f'<path d="{p}" style="stroke:{color};stroke-width:2.4"{dash} fill="none"/>'
    return f'<svg viewBox="0 0 {W} {H}" class="ch" role="img">{grid}{lines}</svg>'


def build_html(full, tr16):
    robust = full["streams"]["Combined (raw-ORB + dip)"]
    good = full["streams"]["Combined (meta-ORB + dip)"]
    bad = tr16["streams"]["Combined (meta-ORB + dip)"]
    books = [
        ("Raw ORB + dip (deploy this)", "var(--robust)", "", robust),
        ("Meta-filter, trained 2018 (raw warmup)", "var(--good)", "", good),
        ("Meta-filter, trained 2016", "var(--bad)", ' stroke-dasharray="5 3"', bad),
    ]
    series = [(lbl, col, dash, [d["e"] for d in b["monthly"]], [d["m"] for d in b["monthly"]])
              for lbl, col, dash, b in books]
    rows = ""
    for lbl, col, _dash, b in books:
        hero = ' class="hero"' if lbl.startswith("Raw") else ""
        rows += (f'<tr{hero}><td><span class="dot" style="background:{col}"></span>{lbl}</td>'
                 f'<td>${b["final"]:,.0f}</td><td>{b["cagr"]*100:.1f}%</td><td>{b["sharpe"]:.2f}</td>'
                 f'<td>{b["sortino"]:.2f}</td><td>{b["maxdd"]*100:.1f}%</td><td>{b["calmar"]:.2f}</td>'
                 f'<td><b>{b["dsr"]*100:.0f}%</b></td></tr>')
    legend = "".join(f'<span><span class="sw" style="background:{c}"></span>{l}</span>' for l, c, _d, _b in books)
    return f'''<style>
:root{{--bg:#f4f6f9;--surface:#fff;--ink:#141b29;--muted:#5b6675;--border:#e3e7ee;
  --robust:#4E79A7;--good:#59A14F;--bad:#E15759;--gd:#2E9E5B;--wn:#C0900A;--bd:#D64550;
  --shadow:0 1px 3px rgba(20,30,50,.07),0 8px 24px rgba(20,30,50,.05);}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0e1420;--surface:#161d2b;--ink:#e7ecf4;--muted:#93a0b4;--border:#25304250;
  --robust:#7BA6D0;--good:#7DC46F;--bad:#F0787F;--gd:#4CC47E;--wn:#E4B740;--bd:#F0787F;
  --shadow:0 1px 3px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);}}}}
:root[data-theme="light"]{{--bg:#f4f6f9;--surface:#fff;--ink:#141b29;--muted:#5b6675;--border:#e3e7ee;--robust:#4E79A7;--good:#59A14F;--bad:#E15759;--gd:#2E9E5B;--wn:#C0900A;--bd:#D64550;}}
:root[data-theme="dark"]{{--bg:#0e1420;--surface:#161d2b;--ink:#e7ecf4;--muted:#93a0b4;--border:#25304250;--robust:#7BA6D0;--good:#7DC46F;--bad:#F0787F;--gd:#4CC47E;--wn:#E4B740;--bd:#F0787F;}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.5;font-variant-numeric:tabular-nums}}
.wrap{{max-width:960px;margin:0 auto;padding:40px 24px 64px}}
.eyebrow{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:600}}
h1{{font-size:28px;line-height:1.15;margin:6px 0 6px;text-wrap:balance}}
.lede{{color:var(--muted);font-size:15px;max-width:72ch;margin:0 0 22px}}
.kpis{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:22px}}
.kpi{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 18px;box-shadow:var(--shadow);flex:1;min-width:180px;border-top:3px solid var(--border)}}
.kpi.r{{border-top-color:var(--robust)}}.kpi.g{{border-top-color:var(--good)}}.kpi.b{{border-top-color:var(--bad)}}
.kpi .k{{font-size:12px;color:var(--muted)}}.kpi .v{{font-size:23px;font-weight:750;margin-top:2px}}.kpi .s{{font-size:12px;color:var(--muted)}}
.panel{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:18px 20px;box-shadow:var(--shadow);margin-bottom:20px}}
.panel h2{{font-size:16px;margin:0 0 3px}}
.ch{{width:100%;height:auto}} .ch .grid{{stroke:var(--border)}} .ch .tick{{fill:var(--muted);font-size:10px}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:8px}}.legend span{{display:inline-flex;align-items:center;gap:7px}}.sw{{width:14px;height:3px;border-radius:2px;display:inline-block}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid var(--border);padding:8px 10px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
th{{color:var(--muted);font-weight:600;font-size:12px}} tr.hero td{{font-weight:750;color:var(--robust)}}
.dot{{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:7px}}
.note{{font-size:13px;color:var(--muted);border-left:3px solid var(--robust);padding-left:12px;margin-top:12px}}
footer{{color:var(--muted);font-size:12px;margin-top:22px}}
</style>
<div class="wrap">
  <div class="eyebrow">qmeta &middot; does the meta-filter survive?</div>
  <h1>Raw ORB + dip vs the meta-filter, tested honestly</h1>
  <p class="lede">All three books are the ORB + dip diversifier, 50/50 equal-risk, $100k, 15% vol, over the
  <b>full 2018&ndash;2026</b> window. The only difference is the breakout filter: none, trained on 2018+ data,
  or trained back to 2016. The verdict: <b>the plain book is the one to trade</b> &mdash; forcing the filter
  onto the full history reverses its edge.</p>

  <div class="kpis">
    <div class="kpi r"><div class="k">Raw ORB + dip &mdash; deploy this</div><div class="v">${robust["final"]:,.0f}</div>
      <div class="s">Sharpe {robust["sharpe"]:.2f} &middot; DSR {robust["dsr"]*100:.0f}% &middot; maxDD {robust["maxdd"]*100:.0f}%</div></div>
    <div class="kpi g"><div class="k">Meta-filter (trained 2018)</div><div class="v">${good["final"]:,.0f}</div>
      <div class="s">Sharpe {good["sharpe"]:.2f} &middot; DSR {good["dsr"]*100:.0f}% &middot; only acts from 2020</div></div>
    <div class="kpi b"><div class="k">Meta-filter (trained 2016)</div><div class="v">${bad["final"]:,.0f}</div>
      <div class="s">Sharpe {bad["sharpe"]:.2f} &middot; DSR {bad["dsr"]*100:.0f}% &middot; <b style="color:var(--bd)">worse than raw</b></div></div>
  </div>

  <div class="panel">
    <h2>Equity &mdash; $100k, full 2018&ndash;2026</h2>
    {eq_svg(series)}
    <div class="legend">{legend}</div>
  </div>

  <div class="panel">
    <h2>Metrics</h2>
    <table><thead><tr><th>book</th><th>$100k &rarr;</th><th>CAGR</th><th>Sharpe</th><th>Sortino</th><th>maxDD</th><th>Calmar</th><th>DSR</th></tr></thead><tbody>{rows}</tbody></table>
    <p class="note"><b>The finding:</b> the meta-filter helps only when trained on 2018+ data (with the HAR weight
    feature) and applied to recent years &mdash; there it nudges the blend to Sharpe {good["sharpe"]:.2f}/DSR {good["dsr"]*100:.0f}%.
    Trained back to 2016 (which forces dropping that feature) and made to act over the whole period, it
    <b>reverses</b> to {bad["sharpe"]:.2f}/DSR {bad["dsr"]*100:.0f}% &mdash; below the unfiltered book. So the filter's edge is
    recent and fragile (a paper-trade candidate), while <b>raw ORB + dip (Sharpe {robust["sharpe"]:.2f}, DSR {robust["dsr"]*100:.0f}%)
    is the robust, deployable book.</b></p>
  </div>

  <footer>qmeta &middot; ORB + dip, 50/50 equal-risk, $100k @15% vol, walk-forward, no lookahead &middot; DSR deflated for K=19 trials.</footer>
</div>'''


if __name__ == "__main__":
    OUT.write_text(build_html(json.loads(FULL.read_text()), json.loads(TR16.read_text())), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
