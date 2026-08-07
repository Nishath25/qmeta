"""Render the combined $100k WFT (wft_combined_100k.json) + overfitting audit
(overfit.json) into one self-contained dashboard."""
import json
from pathlib import Path

CB = Path(r"C:\Users\madas\qmeta\scratch\wft_combined_100k.json")
OF = Path(r"C:\Users\madas\qmeta\scratch\overfit.json")
OUT = Path(r"C:\Users\madas\qmeta\scratch\qmeta_combined.html")
KEYS = ["Combined (meta-ORB + dip)", "ORB + meta-filter", "Dip diversifier", "Combined (raw-ORB + dip)"]
COL = {"Combined (meta-ORB + dip)": "var(--comb)", "ORB + meta-filter": "var(--orb)",
       "Dip diversifier": "var(--dip)", "Combined (raw-ORB + dip)": "var(--raw)"}


def eq_svg(streams):
    W, H, ml, mr, mt, mb = 900, 310, 56, 14, 14, 26
    pw, ph = W - ml - mr, H - mt - mb
    series = {k: [d["e"] for d in streams[k]["monthly"]] for k in KEYS}
    n = len(next(iter(series.values())))
    allv = [v for s in series.values() for v in s]
    lo, hi = min(allv), max(allv)
    def X(i): return ml + i / (n - 1) * pw
    def Y(v): return mt + (hi - v) / (hi - lo) * ph
    def poly(s): return "M" + " L".join(f"{X(i):.1f},{Y(s[i]):.1f}" for i in range(len(s)))
    grid = ""
    for f in (0, 0.25, 0.5, 0.75, 1.0):
        v = lo + f * (hi - lo)
        grid += (f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{ml+pw}" y2="{Y(v):.1f}" class="grid"/>'
                 f'<text x="{ml-8}" y="{Y(v)+4:.1f}" class="tick" text-anchor="end">${v/1000:.0f}k</text>')
    for i in range(0, n, max(1, n // 6)):
        grid += f'<text x="{X(i):.1f}" y="{H-4}" class="tick" text-anchor="middle">{streams[KEYS[0]]["monthly"][i]["m"][:4]}</text>'
    lines = ""
    for k in KEYS:
        dash = ' stroke-dasharray="4 3"' if k == "Combined (raw-ORB + dip)" else ""
        wdt = 2.6 if k == "Combined (meta-ORB + dip)" else 1.7
        lines += f'<path d="{poly(series[k])}" style="stroke:{COL[k]};stroke-width:{wdt}"{dash} fill="none"/>'
    return f'<svg viewBox="0 0 {W} {H}" class="ch" role="img">{grid}{lines}</svg>'


def pill(cls, txt):
    return f'<span class="pill {cls}">{txt}</span>'


def build_html(cb, of):
    s = cb["streams"]
    comb = s["Combined (meta-ORB + dip)"]
    mrows = ""
    for k in KEYS:
        m = s[k]
        hero = ' class="hero"' if k == "Combined (meta-ORB + dip)" else ""
        so = "&ndash;" if m["sortino"] is None else f'{m["sortino"]:.2f}'
        ca = "&ndash;" if m["calmar"] is None else f'{m["calmar"]:.2f}'
        mrows += (f'<tr{hero}><td><span class="dot" style="background:{COL[k]}"></span>{k}</td>'
                  f'<td>${m["final"]:,.0f}</td><td>{m["cagr"]*100:.1f}%</td><td>{m["sharpe"]:.2f}</td>'
                  f'<td>{so}</td><td>{m["maxdd"]*100:.1f}%</td><td>{ca}</td><td>{m["dsr"]*100:.0f}%</td></tr>')
    sub = of["subperiod"]
    uw = of["uw_sensitivity"]
    audit = [
        ("Deflated Sharpe (combined)", f'{of["dsr_combined"]*100:.0f}%',
         "warn" if of["dsr_combined"] < 0.95 else "good", "strong; just under the 95% cert bar"),
        ("PBO &mdash; config overfitting", f'{of["pbo"]*100:.0f}%', "good" if of["pbo"] < 0.5 else "warn",
         "threshold/blend interchangeable &mdash; no lucky-parameter pick"),
        ("MinBTL vs track record", f'{of["minbtl_years"]}y / {of["track_years"]}y',
         "good" if of["minbtl_years"] <= of["track_years"] else "bad", "backtest is long enough for K=19 trials"),
        ("Sub-period Sharpe (H1 / H2)", f'{sub["comb_sharpe_h1"]} / {sub["comb_sharpe_h2"]}', "good",
         "strong in both halves &mdash; not one-period luck"),
        ("Filter lift by half (&Delta;Sharpe)", f'{sub["filter_dsharpe_h1"]:+.2f} / {sub["filter_dsharpe_h2"]:+.2f}',
         "warn", "the filter's edge is concentrated in recent data"),
        ("Uniqueness-weight sensitivity", f'{uw["with_uw"]:+.2f} / {uw["without_uw"]:+.2f}', "warn",
         "the filter's +0.21 is contingent on the uw weighting"),
    ]
    arows = "".join(f'<tr><td>{n}</td><td>{v}</td><td>{pill(c, {"good":"ok","warn":"caution","bad":"fail"}[c])}</td>'
                    f'<td class="muted">{note}</td></tr>' for n, v, c, note in audit)
    return f'''<style>
:root{{--bg:#f4f6f9;--surface:#fff;--ink:#141b29;--muted:#5b6675;--border:#e3e7ee;
  --comb:#59A14F;--orb:#4E79A7;--dip:#F28E2B;--raw:#9aa4b2;--good:#2E9E5B;--warn:#C0900A;--bad:#D64550;
  --shadow:0 1px 3px rgba(20,30,50,.07),0 8px 24px rgba(20,30,50,.05);}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0e1420;--surface:#161d2b;--ink:#e7ecf4;--muted:#93a0b4;--border:#25304250;
  --comb:#7DC46F;--orb:#7BA6D0;--dip:#F4A85B;--raw:#79828f;--good:#4CC47E;--warn:#E4B740;--bad:#F0787F;
  --shadow:0 1px 3px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);}}}}
:root[data-theme="light"]{{--bg:#f4f6f9;--surface:#fff;--ink:#141b29;--muted:#5b6675;--border:#e3e7ee;--comb:#59A14F;--orb:#4E79A7;--dip:#F28E2B;--raw:#9aa4b2;--good:#2E9E5B;--warn:#C0900A;--bad:#D64550;}}
:root[data-theme="dark"]{{--bg:#0e1420;--surface:#161d2b;--ink:#e7ecf4;--muted:#93a0b4;--border:#25304250;--comb:#7DC46F;--orb:#7BA6D0;--dip:#F4A85B;--raw:#79828f;--good:#4CC47E;--warn:#E4B740;--bad:#F0787F;}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.5;font-variant-numeric:tabular-nums}}
.wrap{{max-width:960px;margin:0 auto;padding:40px 24px 64px}}
.eyebrow{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:600}}
h1{{font-size:28px;line-height:1.15;margin:6px 0 6px;text-wrap:balance}}
.lede{{color:var(--muted);font-size:15px;max-width:72ch;margin:0 0 22px}}
.kpis{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:22px}}
.kpi{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 18px;box-shadow:var(--shadow);flex:1;min-width:160px}}
.kpi .k{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}.kpi .v{{font-size:25px;font-weight:750;margin-top:2px}}.kpi .s{{font-size:12px;color:var(--muted)}}
.panel{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:18px 20px;box-shadow:var(--shadow);margin-bottom:20px}}
.panel h2{{font-size:16px;margin:0 0 3px}} .panel .sub{{font-size:13px;color:var(--muted);margin:0 0 12px}}
.ch{{width:100%;height:auto}} .ch .grid{{stroke:var(--border)}} .ch .tick{{fill:var(--muted);font-size:10px}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:8px}}.legend span{{display:inline-flex;align-items:center;gap:7px}}.sw{{width:14px;height:3px;border-radius:2px;display:inline-block}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid var(--border);padding:7px 9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
th{{color:var(--muted);font-weight:600;font-size:12px}} tr.hero td{{font-weight:750;color:var(--comb)}} td.muted{{color:var(--muted);font-weight:400}}
.dot{{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:7px}}
.pill{{font-size:11px;font-weight:700;padding:2px 8px;border-radius:20px}}
.pill.good{{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good)}}
.pill.warn{{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}}
.pill.bad{{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}}
.note{{font-size:13px;color:var(--muted);border-left:3px solid var(--comb);padding-left:12px;margin-top:12px}}
footer{{color:var(--muted);font-size:12px;margin-top:22px}}
</style>
<div class="wrap">
  <div class="eyebrow">qmeta &middot; $100k combined book + overfitting audit</div>
  <h1>Meta-filtered ORB + dip diversifier</h1>
  <p class="lede">The improved (meta-filtered) ORB fund combined 50/50 equal-risk with the dip diversifier,
  $100,000 start, all streams at 15% vol, OOS {cb["oos_start"]}&ndash;{cb["oos_end"]}. Correlation just
  <b>{cb["correlation"]:+.2f}</b> &mdash; still a clean diversifier.</p>

  <div class="kpis">
    <div class="kpi"><div class="k">Combined book</div><div class="v" style="color:var(--comb)">${comb["final"]:,.0f}</div>
      <div class="s">from $100k &middot; +{comb["total"]*100:.0f}% &middot; CAGR {comb["cagr"]*100:.1f}%</div></div>
    <div class="kpi"><div class="k">Sharpe / DSR</div><div class="v">{comb["sharpe"]:.2f} <span class="s">/ {comb["dsr"]*100:.0f}%</span></div>
      <div class="s">vs raw-ORB blend {s["Combined (raw-ORB + dip)"]["sharpe"]:.2f} / {s["Combined (raw-ORB + dip)"]["dsr"]*100:.0f}%</div></div>
    <div class="kpi"><div class="k">Max drawdown</div><div class="v">{comb["maxdd"]*100:.1f}%</div>
      <div class="s">Sortino {comb["sortino"]:.2f} &middot; Calmar {comb["calmar"]:.2f}</div></div>
    <div class="kpi"><div class="k">Correlation</div><div class="v">{cb["correlation"]:+.2f}</div>
      <div class="s">meta-ORB vs dip &mdash; near-zero</div></div>
  </div>

  <div class="panel">
    <h2>Equity &mdash; $100k compounded</h2>
    {eq_svg(s)}
    <div class="legend">
      <span><span class="sw" style="background:var(--comb)"></span>Combined (meta-ORB + dip)</span>
      <span><span class="sw" style="background:var(--orb)"></span>meta-ORB</span>
      <span><span class="sw" style="background:var(--dip)"></span>dip</span>
      <span><span class="sw" style="background:var(--raw)"></span>Combined (raw-ORB + dip)</span></div>
  </div>

  <div class="panel">
    <h2>Metrics &mdash; all at 15% vol</h2>
    <table><thead><tr><th>book</th><th>$100k &rarr;</th><th>CAGR</th><th>Sharpe</th><th>Sortino</th><th>maxDD</th><th>Calmar</th><th>DSR</th></tr></thead><tbody>{mrows}</tbody></table>
    <p class="note">The dip is a return-diversifier but a drawdown-adder: the blend maximizes <b>Sharpe and terminal wealth</b>
    (1.38, ${comb["final"]:,.0f}) while meta-ORB alone keeps the tightest drawdown (best Calmar).</p>
  </div>

  <div class="panel">
    <h2>Overfitting audit</h2>
    <p class="sub">Is the edge real or curve-fit? The López de Prado battery on the combined book + the filter.</p>
    <table><thead><tr><th>check</th><th>value</th><th>verdict</th><th>read</th></tr></thead><tbody>{arows}</tbody></table>
    <p class="note"><b>Bottom line:</b> the <b>core is robust</b> &mdash; the ORB+dip diversification (corr {cb["correlation"]:+.2f},
    DSR 90%, strong in both sub-periods, MinBTL passes) is not curve-fit. The <b>meta-filter's extra lift is promising but
    not certified</b>: it's contingent on the uniqueness weighting and concentrated in recent data. Trade the diversification
    with confidence; treat the filter's +0.21 as a paper-trade candidate, not banked alpha.</p>
  </div>

  <footer>qmeta $100k combined WFT + overfitting audit &middot; walk-forward, no lookahead &middot; DSR / PBO / MinBTL / sub-period / uw-sensitivity.</footer>
</div>'''


if __name__ == "__main__":
    OUT.write_text(build_html(json.loads(CB.read_text()), json.loads(OF.read_text())), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
