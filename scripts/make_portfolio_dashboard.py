"""Render scratch/portfolio.json into a self-contained HTML dashboard:
the ORB-vs-dip principled split, and the rolling-OOS allocator contest."""
import json
from pathlib import Path

SC = Path(r"C:\Users\madas\qmeta\scratch\portfolio.json")
OUT = Path(r"C:\Users\madas\qmeta\scratch\qmeta_portfolio.html")
COLORS = {"equal": "var(--c-eq)", "inverse-variance": "var(--c-iv)",
          "min-variance": "var(--c-mv)", "HRP": "var(--c-hrp)", "NCO": "var(--c-nco)"}


def equity_svg(dates, equity):
    W, H, ml, mr, mt, mb = 900, 320, 46, 14, 14, 30
    pw, ph = W - ml - mr, H - mt - mb
    n = len(dates)
    allv = [v for s in equity.values() for v in s]
    lo, hi = min(allv), max(allv)
    def X(i): return ml + i / (n - 1) * pw
    def Y(v): return mt + (hi - v) / (hi - lo) * ph
    def poly(s):
        step = max(1, n // 1200)
        pts = [(X(i), Y(s[i])) for i in range(0, n, step)] + [(X(n - 1), Y(s[-1]))]
        return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    grid = ""
    for f in (0, 0.5, 1.0):
        v = lo + f * (hi - lo)
        grid += (f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{ml+pw}" y2="{Y(v):.1f}" class="grid"/>'
                 f'<text x="{ml-8}" y="{Y(v)+4:.1f}" class="tick" text-anchor="end">{v:.2f}x</text>')
    for i in range(0, n, max(1, n // 6)):
        grid += f'<text x="{X(i):.1f}" y="{H-6}" class="tick" text-anchor="middle">{dates[i][:4]}</text>'
    lines = "".join(f'<path d="{poly(equity[me])}" style="stroke:{COLORS[me]}" class="ln"/>' for me in equity)
    return f'<svg viewBox="0 0 {W} {H}" class="eq" role="img">{grid}{lines}</svg>'


def build_html(sc):
    a, b = sc["panel_a"], sc["panel_b"]
    best_split = max(a["splits"], key=lambda k: a["splits"][k]["sharpe"])
    naive = a["splits"]["50/50"]["sharpe"]
    arows = ""
    for k, v in a["splits"].items():
        ws = " &middot; ".join(f'{c} {v["weights"][c]*100:.0f}%' for c in a["cols"])
        sel = ' class="best"' if k == best_split else ""
        arows += f'<tr{sel}><td>{k}</td><td>{ws}</td><td>{v["sharpe"]:.2f}</td></tr>'
    best_b = max(b["methods"], key=lambda k: b["methods"][k]["sharpe"])
    brows = ""
    for me, v in b["methods"].items():
        sel = ' class="best"' if me == best_b else ""
        brows += (f'<tr{sel}><td><span class="dot" style="background:{COLORS[me]}"></span>{me}</td>'
                  f'<td>{v["sharpe"]:.2f}</td><td>{v["vol"]*100:.1f}%</td>'
                  f'<td>{v["maxdd"]*100:.1f}%</td><td>{v["eff_n"]:.1f}</td></tr>')
    legend = "".join(f'<span><span class="sw" style="background:{COLORS[me]}"></span>{me}</span>' for me in b["methods"])
    return f'''<style>
:root{{--bg:#f5f7fa;--surface:#fff;--ink:#182031;--muted:#5c6675;--border:#e4e8ef;--good:#2E9E5B;
  --c-eq:#8a94a6;--c-iv:#4E79A7;--c-mv:#E15759;--c-hrp:#59A14F;--c-nco:#B07AA1;
  --shadow:0 1px 3px rgba(20,30,50,.07),0 8px 24px rgba(20,30,50,.05);}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0e1420;--surface:#161d2b;--ink:#e7ecf4;--muted:#93a0b4;--border:#25304250;--good:#4CC47E;
  --c-eq:#7c8698;--c-iv:#7BA6D0;--c-mv:#F0787F;--c-hrp:#7DC46F;--c-nco:#C89BC0;
  --shadow:0 1px 3px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);}}}}
:root[data-theme="light"]{{--bg:#f5f7fa;--surface:#fff;--ink:#182031;--muted:#5c6675;--border:#e4e8ef;--good:#2E9E5B;--c-eq:#8a94a6;--c-iv:#4E79A7;--c-mv:#E15759;--c-hrp:#59A14F;--c-nco:#B07AA1;}}
:root[data-theme="dark"]{{--bg:#0e1420;--surface:#161d2b;--ink:#e7ecf4;--muted:#93a0b4;--border:#25304250;--good:#4CC47E;--c-eq:#7c8698;--c-iv:#7BA6D0;--c-mv:#F0787F;--c-hrp:#7DC46F;--c-nco:#C89BC0;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.5;font-variant-numeric:tabular-nums}}
.wrap{{max-width:940px;margin:0 auto;padding:40px 24px 64px}}
.eyebrow{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:600}}
h1{{font-size:28px;line-height:1.15;margin:6px 0 6px;text-wrap:balance}}
.lede{{color:var(--muted);font-size:15px;max-width:68ch;margin:0 0 22px}}
.panel{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:20px 22px;box-shadow:var(--shadow);margin-bottom:22px}}
.panel h2{{font-size:16px;margin:0 0 4px}} .panel .sub{{font-size:13px;color:var(--muted);margin:0 0 14px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid var(--border);padding:8px 10px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
th{{color:var(--muted);font-weight:600;font-size:12px}} tr.best td{{font-weight:750;color:var(--good)}}
.dot{{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:7px}}
.eq{{width:100%;height:auto}} .eq .grid{{stroke:var(--border)}} .eq .tick{{fill:var(--muted);font-size:10px}}
.eq .ln{{fill:none;stroke-width:1.8}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:10px}}
.legend span{{display:inline-flex;align-items:center;gap:6px}} .sw{{width:14px;height:3px;border-radius:2px;display:inline-block}}
.note{{font-size:13px;color:var(--muted);margin-top:12px;border-left:3px solid var(--border);padding-left:12px}}
footer{{color:var(--muted);font-size:12px;margin-top:24px}}
</style>
<div class="wrap">
  <div class="eyebrow">qmeta &middot; portfolio construction (HRP / NCO)</div>
  <h1>Does principled allocation beat the hand-tuned 50/50?</h1>
  <p class="lede">Hierarchical Risk Parity and Nested-Clustered Optimization vs the baselines, on your real
  streams. The honest answer: <b>a risk-based ORB/dip tilt beats flat 50/50</b>, but on a well-conditioned
  multi-sleeve panel naive equal-weight is hard to beat &mdash; HRP's edge is for ill-conditioned problems.</p>

  <div class="panel">
    <h2>Panel A &middot; ORB vs dip &mdash; the principled split</h2>
    <p class="sub">ORB vol {a["vol"][a["cols"][0]]*100:.0f}%, dip vol {a["vol"][a["cols"][1]]*100:.0f}%.
    Full-period Sharpe of each fixed split. Your 50/50 scores {naive:.2f}.</p>
    <table><thead><tr><th>method</th><th>weights</th><th>Sharpe</th></tr></thead><tbody>{arows}</tbody></table>
    <p class="note">Take-away: the flat 50/50 leaves a little on the table &mdash; tilting toward the lower-vol
    ORB sleeve (risk-parity {a["splits"]["inverse-vol (risk parity)"]["weights"][a["cols"][0]]*100:.0f}/{a["splits"]["inverse-vol (risk parity)"]["weights"][a["cols"][1]]*100:.0f})
    lifts Sharpe to {a["splits"][best_split]["sharpe"]:.2f}.</p>
  </div>

  <div class="panel">
    <h2>Panel B &middot; {len(b["cols"])} sleeves, rolling out-of-sample</h2>
    <p class="sub">252-day trailing covariance, monthly rebalance, {b["n_rebal"]} rebalances,
    OOS {b["oos_start"]}&ndash;{b["oos_end"]}. effN = effective number of positions (diversification).</p>
    {equity_svg(b["dates"], b["equity"])}
    <div class="legend">{legend}</div>
    <table style="margin-top:16px"><thead><tr><th>method</th><th>Sharpe</th><th>vol</th><th>maxDD</th><th>effN</th></tr></thead><tbody>{brows}</tbody></table>
    <p class="note">HRP and NCO are correct and stable, but here they don't beat equal-weight or Markowitz
    min-variance: with 9 well-behaved sleeves and a full year of data the covariance is well-conditioned, so
    HRP's robustness (its advantage when the matrix is near-singular) has nothing to fix. The risk-based
    methods do cut volatility to ~1.4% (from equal-weight's 3.1%) &mdash; useful if low vol is the goal.</p>
  </div>

  <footer>qmeta Phase 3 &middot; HRP (LdP 2016) + NCO (LdP 2019) + baselines &middot; rolling OOS, no lookahead.</footer>
</div>'''


if __name__ == "__main__":
    OUT.write_text(build_html(json.loads(SC.read_text())), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
