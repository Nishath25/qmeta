"""Render scratch/wft_dip_100k.json into a self-contained $100k dip WFT dashboard:
dip vs SPY equity, underwater, per-year P&L, and metrics."""
import json
from pathlib import Path

SC = Path(r"C:\Users\madas\qmeta\scratch\wft_dip_100k.json")
OUT = Path(r"C:\Users\madas\qmeta\scratch\qmeta_dip_wft.html")


def eq_svg(dip_m, spy_m):
    W, H, ml, mr, mt, mb = 900, 300, 56, 14, 14, 26
    pw, ph = W - ml - mr, H - mt - mb
    ed = [d["e"] for d in dip_m]; es = [d["e"] for d in spy_m]
    n = min(len(ed), len(es))
    lo = min(min(ed[:n]), min(es[:n])); hi = max(max(ed[:n]), max(es[:n]))
    def X(i): return ml + i / (n - 1) * pw
    def Y(v): return mt + (hi - v) / (hi - lo) * ph
    def poly(s): return "M" + " L".join(f"{X(i):.1f},{Y(s[i]):.1f}" for i in range(n))
    grid = ""
    for f in (0, 0.25, 0.5, 0.75, 1.0):
        v = lo + f * (hi - lo)
        grid += (f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{ml+pw}" y2="{Y(v):.1f}" class="grid"/>'
                 f'<text x="{ml-8}" y="{Y(v)+4:.1f}" class="tick" text-anchor="end">${v/1000:.0f}k</text>')
    for i in range(0, n, max(1, n // 7)):
        grid += f'<text x="{X(i):.1f}" y="{H-4}" class="tick" text-anchor="middle">{dip_m[i]["m"][:4]}</text>'
    return (f'<svg viewBox="0 0 {W} {H}" class="ch" role="img">{grid}'
            f'<path d="{poly(es)}" class="lspy"/><path d="{poly(ed)}" class="ldip"/></svg>')


def uw_svg(dip_m):
    W, H, ml, mr, mt, mb = 900, 130, 56, 14, 8, 20
    pw, ph = W - ml - mr, H - mt - mb
    dd = [d["dd"] for d in dip_m]; n = len(dd); lo = min(dd)
    def X(i): return ml + i / (n - 1) * pw
    def Y(v): return mt + (0 - v) / (0 - lo) * ph
    p = "M" + f"{X(0):.1f},{Y(0):.1f} L" + " L".join(f"{X(i):.1f},{Y(dd[i]):.1f}" for i in range(n)) + f" L{X(n-1):.1f},{Y(0):.1f} Z"
    grid = "".join(f'<text x="{ml-8}" y="{Y(v)+3:.1f}" class="tick" text-anchor="end">{v:.0f}%</text>'
                   f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{ml+pw}" y2="{Y(v):.1f}" class="grid"/>'
                   for v in [0, lo / 2, lo])
    return f'<svg viewBox="0 0 {W} {H}" class="ch" role="img">{grid}<path d="{p}" class="adip"/></svg>'


def bars_svg(per_year):
    W, H, ml, mr, mt, mb = 900, 180, 56, 14, 14, 26
    pw, ph = W - ml - mr, H - mt - mb
    vals = [p[1] for p in per_year]; hi = max(vals + [0]); lo = min(vals + [0]); span = (hi - lo) or 1
    def Y(v): return mt + (hi - v) / span * ph
    z0 = Y(0); bw = pw / len(per_year) * 0.6
    bars, labels = "", ""
    for i, (yr, pnl, pct) in enumerate(per_year):
        x = ml + (i + 0.5) * pw / len(per_year) - bw / 2
        y0, y1 = min(Y(0), Y(pnl)), max(Y(0), Y(pnl))
        col = "var(--dip)" if pnl >= 0 else "var(--bad)"
        bars += (f'<rect x="{x:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{max(1,y1-y0):.1f}" fill="{col}" rx="2"/>'
                 f'<text x="{x+bw/2:.1f}" y="{(y0-4) if pnl>=0 else (y1+12):.1f}" class="blab" text-anchor="middle">${pnl/1000:+.0f}k</text>')
        labels += f'<text x="{x+bw/2:.1f}" y="{H-8}" class="tick" text-anchor="middle">{yr}</text>'
    return f'<svg viewBox="0 0 {W} {H}" class="ch" role="img"><line x1="{ml}" y1="{z0:.1f}" x2="{ml+pw}" y2="{z0:.1f}" class="zero"/>{bars}{labels}</svg>'


def build_html(sc):
    d, s = sc["dip"], sc["spy"]
    def row(label, dv, sv, better_hi=True):
        win = "dip" if ((dv > sv) == better_hi) else "spy"
        return f'<tr><td>{label}</td><td class="{ "win" if win=="dip" else ""}">{dv}</td><td class="{ "win" if win=="spy" else ""}">{sv}</td></tr>'
    rows = "".join([
        row("Final equity", f"${d['final']:,.0f}", f"${s['final']:,.0f}"),
        row("Total return", f"{(d['final']/sc['start']-1)*100:+.0f}%", f"{(s['final']/sc['start']-1)*100:+.0f}%"),
        row("CAGR", f"{d['cagr']*100:.1f}%", f"{s['cagr']*100:.1f}%"),
        row("Sharpe", f"{d['sharpe']:.2f}", f"{s['sharpe']:.2f}"),
        row("Sortino", f"{d['sortino']:.2f}", f"{s['sortino']:.2f}"),
        row("Volatility", f"{d['vol']*100:.0f}%", f"{s['vol']*100:.0f}%", better_hi=False),
        row("Max drawdown", f"{d['maxdd']*100:.1f}%", f"{s['maxdd']*100:.1f}%", better_hi=False),
        row("Calmar", f"{d['calmar']:.2f}", f"{s['calmar']:.2f}"),
    ])
    return f'''<style>
:root{{--bg:#f4f6f9;--surface:#fff;--ink:#141b29;--muted:#5b6675;--border:#e3e7ee;
  --dip:#F28E2B;--spy:#8a94a6;--bad:#D64550;--good:#2E9E5B;--shadow:0 1px 3px rgba(20,30,50,.07),0 8px 24px rgba(20,30,50,.05);}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0e1420;--surface:#161d2b;--ink:#e7ecf4;--muted:#93a0b4;--border:#25304250;
  --dip:#F4A85B;--spy:#79828f;--bad:#F0787F;--good:#4CC47E;--shadow:0 1px 3px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);}}}}
:root[data-theme="light"]{{--bg:#f4f6f9;--surface:#fff;--ink:#141b29;--muted:#5b6675;--border:#e3e7ee;--dip:#F28E2B;--spy:#8a94a6;--bad:#D64550;--good:#2E9E5B;}}
:root[data-theme="dark"]{{--bg:#0e1420;--surface:#161d2b;--ink:#e7ecf4;--muted:#93a0b4;--border:#25304250;--dip:#F4A85B;--spy:#79828f;--bad:#F0787F;--good:#4CC47E;}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.5;font-variant-numeric:tabular-nums}}
.wrap{{max-width:960px;margin:0 auto;padding:40px 24px 64px}}
.eyebrow{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:600}}
h1{{font-size:28px;line-height:1.15;margin:6px 0 6px;text-wrap:balance}}
.lede{{color:var(--muted);font-size:15px;max-width:70ch;margin:0 0 22px}}
.kpis{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:22px}}
.kpi{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 18px;box-shadow:var(--shadow);flex:1;min-width:165px}}
.kpi .k{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}.kpi .v{{font-size:25px;font-weight:750;margin-top:2px}}.kpi .s{{font-size:12px;color:var(--muted)}}
.panel{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:18px 20px;box-shadow:var(--shadow);margin-bottom:20px}}
.panel h2{{font-size:16px;margin:0 0 3px}}
.ch{{width:100%;height:auto}} .ch .grid{{stroke:var(--border)}} .ch .zero{{stroke:var(--muted);stroke-width:1.2}} .ch .tick{{fill:var(--muted);font-size:10px}}
.ch .ldip{{fill:none;stroke:var(--dip);stroke-width:2.4}} .ch .lspy{{fill:none;stroke:var(--spy);stroke-width:1.8}}
.ch .adip{{fill:color-mix(in srgb,var(--dip) 26%,transparent);stroke:var(--dip);stroke-width:1.2}} .ch .blab{{fill:var(--muted);font-size:10px;font-weight:600}}
.legend{{display:flex;gap:16px;font-size:13px;color:var(--muted);margin-top:8px}}.legend span{{display:inline-flex;align-items:center;gap:7px}}.sw{{width:14px;height:3px;border-radius:2px;display:inline-block}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid var(--border);padding:8px 10px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
th{{color:var(--muted);font-weight:600;font-size:12px}} td.win{{font-weight:750;color:var(--good)}}
.note{{font-size:13px;color:var(--muted);border-left:3px solid var(--dip);padding-left:12px;margin-top:12px}}
footer{{color:var(--muted);font-size:12px;margin-top:22px}}
</style>
<div class="wrap">
  <div class="eyebrow">qmeta &middot; $100k walk-forward</div>
  <h1>Dip strategy &mdash; standalone</h1>
  <p class="lede">Buy-the-recoverer mean reversion, $100,000 start, natural sizing, {sc["win_start"]}&ndash;{sc["win_end"]}
  ({sc["n_days"]:,} days). The rule is fixed a priori, so every year is out-of-sample. Benchmarked against SPY buy &amp; hold.</p>

  <div class="kpis">
    <div class="kpi"><div class="k">Dip strategy</div><div class="v" style="color:var(--dip)">${d["final"]:,.0f}</div>
      <div class="s">+{(d["final"]/sc["start"]-1)*100:.0f}% &middot; CAGR {d["cagr"]*100:.1f}% &middot; vs SPY ${s["final"]:,.0f}</div></div>
    <div class="kpi"><div class="k">Sharpe (risk-adjusted)</div><div class="v">{d["sharpe"]:.2f}</div>
      <div class="s">SPY {s["sharpe"]:.2f} &mdash; SPY edges it on risk-adjusted return</div></div>
    <div class="kpi"><div class="k">Volatility / Max DD</div><div class="v">{d["vol"]*100:.0f}% <span class="s">/ {d["maxdd"]*100:.0f}%</span></div>
      <div class="s">SPY {s["vol"]*100:.0f}% / {s["maxdd"]*100:.0f}% &mdash; the dip runs hotter</div></div>
  </div>

  <div class="panel">
    <h2>Equity &mdash; $100k compounded</h2>
    {eq_svg(sc["dip"]["monthly"], sc["spy"]["monthly"])}
    <div class="legend"><span><span class="sw" style="background:var(--dip)"></span>Dip strategy</span>
      <span><span class="sw" style="background:var(--spy)"></span>SPY buy &amp; hold</span></div>
  </div>

  <div class="panel"><h2>Underwater (dip drawdown)</h2>{uw_svg(sc["dip"]["monthly"])}</div>
  <div class="panel"><h2>Per-year P&amp;L (dip)</h2>{bars_svg(sc["dip"]["per_year"])}</div>

  <div class="panel">
    <h2>Metrics &mdash; dip vs SPY</h2>
    <table><thead><tr><th>metric</th><th>Dip strategy</th><th>SPY buy &amp; hold</th></tr></thead><tbody>{rows}</tbody></table>
    <p class="note">The dip <b>out-earns SPY (+{(d["final"]/sc["start"]-1)*100:.0f}% vs +{(s["final"]/sc["start"]-1)*100:.0f}%)</b>
    but at higher volatility and a deeper drawdown, so its Sharpe ({d["sharpe"]:.2f}) sits <i>below</i> SPY's ({s["sharpe"]:.2f}).
    It's essentially leveraged long-beta &mdash; two big down years (2018, 2022). Its real value is as an <b>uncorrelated
    diversifier to the ORB fund</b> (corr &asymp; 0), not as a standalone book.</p>
  </div>

  <footer>qmeta $100k dip WFT &middot; natural sizing, next-open fills, 15bps/fill, delisted names force-liquidated &middot; no lookahead.</footer>
</div>'''


if __name__ == "__main__":
    OUT.write_text(build_html(json.loads(SC.read_text())), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
