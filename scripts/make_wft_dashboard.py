"""Render scratch/wft_metalabel_100k.json into a self-contained $100k WFT dashboard:
raw ORB vs ORB+meta-filter equity, underwater, per-year P&L, and full metrics."""
import json
from pathlib import Path

SC = Path(r"C:\Users\madas\qmeta\scratch\wft_metalabel_100k.json")
OUT = Path(r"C:\Users\madas\qmeta\scratch\qmeta_wft_metalabel.html")


def eq_svg(raw_m, meta_m):
    W, H, ml, mr, mt, mb = 900, 300, 56, 14, 14, 26
    pw, ph = W - ml - mr, H - mt - mb
    er = [d["e"] for d in raw_m]; em = [d["e"] for d in meta_m]
    n = len(er)
    lo = min(min(er), min(em)); hi = max(max(er), max(em))
    def X(i): return ml + i / (n - 1) * pw
    def Y(v): return mt + (hi - v) / (hi - lo) * ph
    def poly(s): return "M" + " L".join(f"{X(i):.1f},{Y(s[i]):.1f}" for i in range(n))
    grid = ""
    for f in (0, 0.25, 0.5, 0.75, 1.0):
        v = lo + f * (hi - lo)
        grid += (f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{ml+pw}" y2="{Y(v):.1f}" class="grid"/>'
                 f'<text x="{ml-8}" y="{Y(v)+4:.1f}" class="tick" text-anchor="end">${v/1000:.0f}k</text>')
    for i in range(0, n, max(1, n // 6)):
        grid += f'<text x="{X(i):.1f}" y="{H-4}" class="tick" text-anchor="middle">{raw_m[i]["m"][:4]}</text>'
    return (f'<svg viewBox="0 0 {W} {H}" class="ch" role="img">{grid}'
            f'<path d="{poly(er)}" class="lraw"/><path d="{poly(em)}" class="lmeta"/></svg>')


def uw_svg(raw_m, meta_m):
    W, H, ml, mr, mt, mb = 900, 130, 56, 14, 8, 20
    pw, ph = W - ml - mr, H - mt - mb
    dr = [d["dd"] for d in raw_m]; dm = [d["dd"] for d in meta_m]
    n = len(dr); lo = min(min(dr), min(dm))
    def X(i): return ml + i / (n - 1) * pw
    def Y(v): return mt + (0 - v) / (0 - lo) * ph
    def poly(s): return "M" + f"{X(0):.1f},{Y(0):.1f} L" + " L".join(f"{X(i):.1f},{Y(s[i]):.1f}" for i in range(n)) + f" L{X(n-1):.1f},{Y(0):.1f} Z"
    grid = "".join(f'<text x="{ml-8}" y="{Y(v)+3:.1f}" class="tick" text-anchor="end">{v:.0f}%</text>'
                   f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{ml+pw}" y2="{Y(v):.1f}" class="grid"/>'
                   for v in [0, lo / 2, lo])
    return (f'<svg viewBox="0 0 {W} {H}" class="ch" role="img">{grid}'
            f'<path d="{poly(dr)}" class="araw"/><path d="{poly(dm)}" class="ameta"/></svg>')


def bars_svg(per_year):
    W, H, ml, mr, mt, mb = 900, 180, 56, 14, 14, 26
    pw, ph = W - ml - mr, H - mt - mb
    vals = [p[1] for p in per_year]
    hi = max(vals + [0]); lo = min(vals + [0]); span = (hi - lo) or 1
    def Y(v): return mt + (hi - v) / span * ph
    z0 = Y(0); bw = pw / len(per_year) * 0.6
    bars, labels = "", ""
    for i, (yr, pnl, pct) in enumerate(per_year):
        x = ml + (i + 0.5) * pw / len(per_year) - bw / 2
        y0, y1 = min(Y(0), Y(pnl)), max(Y(0), Y(pnl))
        col = "var(--meta)" if pnl >= 0 else "var(--bad)"
        bars += (f'<rect x="{x:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{max(1,y1-y0):.1f}" fill="{col}" rx="2"/>'
                 f'<text x="{x+bw/2:.1f}" y="{(y0-4) if pnl>=0 else (y1+12):.1f}" class="blab" text-anchor="middle">${pnl/1000:+.0f}k</text>')
        labels += f'<text x="{x+bw/2:.1f}" y="{H-8}" class="tick" text-anchor="middle">{yr}</text>'
    return f'<svg viewBox="0 0 {W} {H}" class="ch" role="img"><line x1="{ml}" y1="{z0:.1f}" x2="{ml+pw}" y2="{z0:.1f}" class="zero"/>{bars}{labels}</svg>'


def build_html(sc):
    r, m = sc["raw"], sc["meta"]
    dpl = m["final"] - r["final"]
    def row(label, rv, mv, better_hi=True):
        win = "meta" if ((mv > rv) == better_hi) else "raw"
        return f'<tr><td>{label}</td><td>{rv}</td><td class="{ "win" if win=="meta" else ""}">{mv}</td></tr>'
    rows = "".join([
        row("Final equity", f"${r['final']:,.0f}", f"${m['final']:,.0f}"),
        row("Total return", f"{r['total']*100:+.0f}%", f"{m['total']*100:+.0f}%"),
        row("CAGR", f"{r['cagr']*100:.1f}%", f"{m['cagr']*100:.1f}%"),
        row("Sharpe", f"{r['sharpe']:.2f}", f"{m['sharpe']:.2f}"),
        row("Sortino", f"{r['sortino']:.2f}", f"{m['sortino']:.2f}"),
        row("Max drawdown", f"{r['maxdd']*100:.1f}%", f"{m['maxdd']*100:.1f}%", better_hi=False),
        row("Calmar", f"{r['calmar']:.2f}", f"{m['calmar']:.2f}"),
        row("Volatility", f"{r['vol']*100:.0f}%", f"{m['vol']*100:.0f}%", better_hi=False),
        row("Deflated Sharpe", f"{r['dsr']*100:.0f}%", f"{m['dsr']*100:.0f}%"),
    ])
    return f'''<style>
:root{{--bg:#f4f6f9;--surface:#fff;--ink:#141b29;--muted:#5b6675;--border:#e3e7ee;
  --raw:#8a94a6;--meta:#2E9E5B;--bad:#D64550;--shadow:0 1px 3px rgba(20,30,50,.07),0 8px 24px rgba(20,30,50,.05);}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0e1420;--surface:#161d2b;--ink:#e7ecf4;--muted:#93a0b4;--border:#25304250;
  --raw:#6b7688;--meta:#4CC47E;--bad:#F0787F;--shadow:0 1px 3px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);}}}}
:root[data-theme="light"]{{--bg:#f4f6f9;--surface:#fff;--ink:#141b29;--muted:#5b6675;--border:#e3e7ee;--raw:#8a94a6;--meta:#2E9E5B;--bad:#D64550;}}
:root[data-theme="dark"]{{--bg:#0e1420;--surface:#161d2b;--ink:#e7ecf4;--muted:#93a0b4;--border:#25304250;--raw:#6b7688;--meta:#4CC47E;--bad:#F0787F;}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.5;font-variant-numeric:tabular-nums}}
.wrap{{max-width:960px;margin:0 auto;padding:40px 24px 64px}}
.eyebrow{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:600}}
h1{{font-size:28px;line-height:1.15;margin:6px 0 6px;text-wrap:balance}}
.lede{{color:var(--muted);font-size:15px;max-width:70ch;margin:0 0 22px}}
.kpis{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:22px}}
.kpi{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 18px;box-shadow:var(--shadow);flex:1;min-width:165px}}
.kpi .k{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}.kpi .v{{font-size:25px;font-weight:750;margin-top:2px}}.kpi .s{{font-size:12px;color:var(--muted)}}
.panel{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:18px 20px;box-shadow:var(--shadow);margin-bottom:20px}}
.panel h2{{font-size:16px;margin:0 0 3px}} .panel .sub{{font-size:13px;color:var(--muted);margin:0 0 12px}}
.ch{{width:100%;height:auto}} .ch .grid{{stroke:var(--border)}} .ch .zero{{stroke:var(--muted);stroke-width:1.2}} .ch .tick{{fill:var(--muted);font-size:10px}}
.ch .lraw{{fill:none;stroke:var(--raw);stroke-width:1.8}} .ch .lmeta{{fill:none;stroke:var(--meta);stroke-width:2.4}}
.ch .araw{{fill:color-mix(in srgb,var(--raw) 30%,transparent);stroke:var(--raw);stroke-width:1}}
.ch .ameta{{fill:color-mix(in srgb,var(--meta) 26%,transparent);stroke:var(--meta);stroke-width:1.2}}
.ch .blab{{fill:var(--muted);font-size:10px;font-weight:600}}
.legend{{display:flex;gap:16px;font-size:13px;color:var(--muted);margin-top:8px}}.legend span{{display:inline-flex;align-items:center;gap:7px}}.sw{{width:14px;height:3px;border-radius:2px;display:inline-block}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid var(--border);padding:8px 10px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
th{{color:var(--muted);font-weight:600;font-size:12px}} td.win{{font-weight:750;color:var(--meta)}}
.note{{font-size:13px;color:var(--muted);border-left:3px solid var(--meta);padding-left:12px;margin-top:12px}}
footer{{color:var(--muted);font-size:12px;margin-top:22px}}
</style>
<div class="wrap">
  <div class="eyebrow">qmeta &middot; $100k walk-forward</div>
  <h1>ORB fund with the meta-label breakout filter</h1>
  <p class="lede">A secondary model predicts P(a breakout wins), trained strictly walk-forward, and the fund
  <b>skips the breakouts it flags as losers</b> (takes {sc["taken_frac"]*100:.0f}%). $100,000 start, out-of-sample
  {sc["oos_start"]}&ndash;{sc["oos_end"]}, {sc["n_trades"]:,} trades. Same volatility as the raw fund, so the gap is skill, not leverage.</p>

  <div class="kpis">
    <div class="kpi"><div class="k">Filtered fund</div><div class="v" style="color:var(--meta)">${m["final"]:,.0f}</div>
      <div class="s">from $100k &middot; +{m["total"]*100:.0f}% &middot; vs raw ${r["final"]:,.0f}</div></div>
    <div class="kpi"><div class="k">Extra profit</div><div class="v">+${dpl:,.0f}</div>
      <div class="s">the filter's contribution over raw</div></div>
    <div class="kpi"><div class="k">Sharpe / DSR</div><div class="v">{m["sharpe"]:.2f} <span class="s">/ {m["dsr"]*100:.0f}%</span></div>
      <div class="s">raw {r["sharpe"]:.2f} / {r["dsr"]*100:.0f}%</div></div>
    <div class="kpi"><div class="k">Max drawdown</div><div class="v">{m["maxdd"]*100:.1f}%</div>
      <div class="s">raw {r["maxdd"]*100:.1f}% &middot; Calmar {m["calmar"]:.2f}</div></div>
  </div>

  <div class="panel">
    <h2>Equity &mdash; $100k compounded</h2>
    {eq_svg(sc["raw"]["monthly"], sc["meta"]["monthly"])}
    <div class="legend"><span><span class="sw" style="background:var(--raw)"></span>Raw ORB</span>
      <span><span class="sw" style="background:var(--meta)"></span>ORB + meta-filter</span></div>
  </div>

  <div class="panel">
    <h2>Underwater (drawdown)</h2>
    {uw_svg(sc["raw"]["monthly"], sc["meta"]["monthly"])}
  </div>

  <div class="panel">
    <h2>Per-year P&amp;L &mdash; filtered fund</h2>
    {bars_svg(sc["meta"]["per_year"])}
  </div>

  <div class="panel">
    <h2>Metrics &mdash; raw vs filtered</h2>
    <table><thead><tr><th>metric</th><th>Raw ORB</th><th>ORB + meta-filter</th></tr></thead><tbody>{rows}</tbody></table>
    <p class="note">Honest caveat: the filter's edge (&Delta;Sharpe {sc["d_sharpe_with_uw"]:+.2f}) is <b>leakage-free</b>
    (walk-forward, features known at entry) but <b>contingent on the uniqueness weighting</b> &mdash; without it,
    &Delta;Sharpe is {sc["d_sharpe_without_uw"]:+.2f} (roughly neutral). Given the weighting it's robust across seed/refit/warmup.</p>
  </div>

  <footer>qmeta $100k WFT &middot; meta-label skip-gate on ORB &middot; walk-forward (2y warmup, quarterly refit), no lookahead &middot; vol-matched to raw.</footer>
</div>'''


if __name__ == "__main__":
    OUT.write_text(build_html(json.loads(SC.read_text())), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
