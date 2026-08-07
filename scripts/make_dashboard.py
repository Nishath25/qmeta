"""Render scratch/scorecard.json into a self-contained HTML scorecard dashboard.
Theme-aware (light/dark), inline SVG charts, no external assets.
Run: python scripts/make_dashboard.py"""
import json
from pathlib import Path

SC = Path(r"C:\Users\madas\qmeta\scratch\scorecard.json")
OUT = Path(r"C:\Users\madas\qmeta\scratch\qmeta_scorecard.html")


def _status(kind, c):
    """Return (cls, icon, note) for a metric's health."""
    ty = c["track_years"]
    if kind == "psr":
        v = c["psr"]
        return ("good" if v >= .95 else "warn" if v >= .90 else "bad", None,
                "confident SR>0" if v >= .95 else "borderline")
    if kind == "dsr":
        v = c["dsr"]
        return ("good" if v >= .95 else "warn" if v >= .50 else "bad", None,
                "beats K-trial luck bar" if v >= .95 else "not past the 95% multiple-testing bar")
    if kind == "mintrl":
        ok = c["mintrl_years"] <= ty
        return ("good" if ok else "bad", None, "track record sufficient" if ok else "need more history")
    if kind == "minbtl":
        ok = c["min_backtest_years"] <= ty
        return ("good" if ok else "bad", None, "backtest long enough" if ok else "too few years for K trials")
    if kind == "pbo":
        v = c["pbo"]
        if v is None:
            return ("na", None, "single aggregated stream")
        return ("good" if v <= .20 else "warn" if v <= .50 else "bad", None,
                "instrument choice holds OOS" if v <= .20 else "some selection fragility")
    return ("na", None, "")


ICON = {"good": "✓", "warn": "!", "bad": "✗", "na": "–"}


def pill(cls, text):
    return f'<span class="pill {cls}">{ICON.get(cls, "")} {text}</span>'


def bar_have_need(have, need, accent):
    """Small SVG: track record (have) vs required (need)."""
    m = max(have, need, 0.1)
    hw, nw = 100 * have / m, 100 * need / m
    return (f'<svg class="hn" viewBox="0 0 100 26" preserveAspectRatio="none" role="img">'
            f'<rect x="0" y="2" width="{hw:.1f}" height="9" rx="2" fill="{accent}"></rect>'
            f'<rect x="0" y="15" width="{nw:.1f}" height="9" rx="2" fill="var(--muted)" opacity="0.55"></rect>'
            f'</svg>')


def metric_row(label, value, cls, note, extra=""):
    return (f'<div class="mrow"><div class="mlab">{label}</div>'
            f'<div class="mval">{value}{extra}</div>'
            f'<div class="mstat">{pill(cls, note)}</div></div>')


def card(c, accent):
    dd = c["drawdown"]
    ps, _, pn = _status("psr", c)
    ds, _, dn = _status("dsr", c)
    ms, _, mn = _status("mintrl", c)
    bs, _, bn = _status("minbtl", c)
    os_, _, on = _status("pbo", c)
    pbo_txt = "n/a" if c["pbo"] is None else f'{c["pbo"]*100:.0f}%'
    rows = "".join([
        metric_row("Sharpe (annualized)", f'{c["sharpe_ann"]:.2f}', "na", "realized"),
        metric_row("PSR &mdash; P(SR&gt;0)", f'{c["psr"]*100:.1f}%', ps, pn),
        metric_row("DSR &mdash; deflated for K trials", f'{c["dsr"]*100:.1f}%', ds, dn,
                   f'<span class="sub">E[maxSR]={c["expected_max_sr_ann"]:.2f}</span>'),
        metric_row("MinTRL vs track record",
                   f'{c["mintrl_years"]:.1f}y {bar_have_need(c["track_years"], c["mintrl_years"], accent)}',
                   ms, mn, f'<span class="sub">have {c["track_years"]:.1f}y</span>'),
        metric_row("MinBTL (needed history)", f'{c["min_backtest_years"]:.1f}y', bs, bn),
        metric_row("PBO &mdash; overfit probability", pbo_txt, os_, on),
    ])
    dd_note = "fatter-than-Gaussian tails" if dd["max_dd"] < c["empirical_maxdd"] else "within model"
    return f'''
    <section class="card">
      <div class="chead"><span class="dot" style="background:{accent}"></span>
        <h2>{c["name"]}</h2><span class="years">{c["track_years"]:.1f} years &middot; {c["n_obs"]} days</span></div>
      <div class="metrics">{rows}</div>
      <div class="ddblock">
        <div class="ddlab">Projected max drawdown <span class="sub">(95%, Triple-Penance)</span></div>
        <div class="ddbars">
          {_ddbar("projected", dd["max_dd"], accent)}
          {_ddbar("empirical (WFT)", c["empirical_maxdd"], "var(--muted)")}
        </div>
        <div class="ddfoot">Expected time underwater <b>{dd["max_tuw_years"]:.1f}y</b>
          (bottom at {dd["time_to_maxdd_years"]:.1f}y, then 3&times; to recover) &middot; {dd_note}</div>
      </div>
    </section>'''


def _ddbar(label, frac, color):
    w = min(100, frac / 0.45 * 100)  # scale so a 45% drawdown ~ full width
    return (f'<div class="ddrow"><span class="ddname">{label}</span>'
            f'<span class="ddtrack"><span class="ddfill" style="width:{w:.0f}%;background:{color}"></span></span>'
            f'<span class="ddpct">{frac*100:.0f}%</span></div>')


def approval_svg(ap, dip_accent, orb_accent):
    """Sharpe-indifference chart: x=correlation, y=required candidate Sharpe.
    Boundary sr_new = rho*S_a; region above = adding improves the book."""
    W, H = 460, 300
    ml, mr, mt, mb = 46, 16, 20, 40
    pw, ph = W - ml - mr, H - mt - mb
    xmin, xmax = -1.0, 1.0
    ymin, ymax = -1.0, 1.6

    def X(x): return ml + (x - xmin) / (xmax - xmin) * pw
    def Y(y): return mt + (ymax - y) / (ymax - ymin) * ph

    corr = ap["indifference_curve"]["corr"]
    srn = ap["indifference_curve"]["sr_new"]
    pts = [(X(c), Y(s)) for c, s in zip(corr, srn) if s is not None]
    line = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    # approve region = above the boundary line up to top
    region = (f'{line} L{X(xmax):.1f},{Y(ymax):.1f} L{X(xmin):.1f},{Y(ymax):.1f} Z')
    # axes ticks
    xticks = "".join(
        f'<line x1="{X(t):.1f}" y1="{mt}" x2="{X(t):.1f}" y2="{mt+ph}" class="grid"/>'
        f'<text x="{X(t):.1f}" y="{mt+ph+16}" class="tick" text-anchor="middle">{t:+.1f}</text>'
        for t in [-1.0, -0.5, 0.0, 0.5, 1.0])
    yticks = "".join(
        f'<line x1="{ml}" y1="{Y(t):.1f}" x2="{ml+pw}" y2="{Y(t):.1f}" class="grid"/>'
        f'<text x="{ml-8}" y="{Y(t)+4:.1f}" class="tick" text-anchor="end">{t:.1f}</text>'
        for t in [-1.0, 0.0, 0.5, 1.0, 1.5])
    px_, py_ = X(ap["correlation"]), Y(ap["sr_candidate"])
    verdict = "APPROVE" if ap["improves"] else "REJECT"
    return f'''<svg viewBox="0 0 {W} {H}" class="approve" role="img" aria-label="Sharpe indifference curve">
      {yticks}{xticks}
      <path d="{region}" class="approvefill"/>
      <path d="{line}" class="boundary"/>
      <line x1="{ml}" y1="{Y(0):.1f}" x2="{ml+pw}" y2="{Y(0):.1f}" class="axis0"/>
      <circle cx="{px_:.1f}" cy="{py_:.1f}" r="6" fill="{dip_accent}" stroke="var(--surface)" stroke-width="2"/>
      <text x="{px_-10:.1f}" y="{py_-12:.1f}" class="ptlab" text-anchor="middle">dip &rarr; {verdict}</text>
      <text x="{ml+pw/2:.0f}" y="{H-6}" class="axtitle" text-anchor="middle">correlation to the ORB book</text>
      <text x="14" y="{mt+ph/2:.0f}" class="axtitle" transform="rotate(-90 14 {mt+ph/2:.0f})" text-anchor="middle">candidate Sharpe needed</text>
    </svg>'''


def build_html(sc):
    orb, dip = sc["streams"]["orb"], sc["streams"]["dip"]
    ap = sc["approval"]
    ORB_A, DIP_A = "var(--orb)", "var(--dip)"
    # headline verdict
    def verdict(c):
        strong = c["psr"] >= .95 and c["mintrl_years"] <= c["track_years"]
        deflated = c["dsr"] >= .95
        if strong and deflated:
            return "real edge, survives multiple-testing"
        if strong:
            return "real & long enough, but not past the K-trial deflation bar"
        return "unconfirmed"
    combined = ap["combined_max_sharpe"]
    lift = combined - ap["sr_approved"]
    cards = card(orb, ORB_A) + card(dip, DIP_A)
    return f'''<style>
:root{{
  --bg:#f5f7fa; --surface:#ffffff; --ink:#182031; --muted:#5c6675; --border:#e4e8ef;
  --orb:#4E79A7; --dip:#F28E2B; --good:#2E9E5B; --warn:#C0900A; --bad:#D64550;
  --shadow:0 1px 3px rgba(20,30,50,.07),0 8px 24px rgba(20,30,50,.05);
}}
@media (prefers-color-scheme:dark){{:root{{
  --bg:#0e1420; --surface:#161d2b; --ink:#e7ecf4; --muted:#93a0b4; --border:#25304250;
  --orb:#7BA6D0; --dip:#F4A85B; --good:#4CC47E; --warn:#E4B740; --bad:#F0787F;
  --shadow:0 1px 3px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
}}}}
:root[data-theme="light"]{{--bg:#f5f7fa;--surface:#fff;--ink:#182031;--muted:#5c6675;--border:#e4e8ef;--orb:#4E79A7;--dip:#F28E2B;--good:#2E9E5B;--warn:#C0900A;--bad:#D64550;}}
:root[data-theme="dark"]{{--bg:#0e1420;--surface:#161d2b;--ink:#e7ecf4;--muted:#93a0b4;--border:#25304250;--orb:#7BA6D0;--dip:#F4A85B;--good:#4CC47E;--warn:#E4B740;--bad:#F0787F;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.5;-webkit-font-smoothing:antialiased;font-variant-numeric:tabular-nums;}}
.wrap{{max-width:980px;margin:0 auto;padding:40px 24px 64px}}
.eyebrow{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:600}}
h1{{font-size:30px;line-height:1.15;margin:6px 0 4px;text-wrap:balance;font-weight:700}}
.lede{{color:var(--muted);font-size:15px;margin:0 0 8px;max-width:64ch}}
.headverdict{{display:flex;gap:22px;flex-wrap:wrap;margin:20px 0 30px}}
.hv{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 18px;box-shadow:var(--shadow);flex:1;min-width:210px}}
.hv .k{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}}
.hv .v{{font-size:22px;font-weight:700;margin-top:3px}}
.hv .s{{font-size:13px;color:var(--muted);margin-top:2px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}
@media(max-width:760px){{.grid2{{grid-template-columns:1fr}}}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:20px 20px 18px;box-shadow:var(--shadow)}}
.chead{{display:flex;align-items:center;gap:9px;margin-bottom:14px}}
.chead h2{{font-size:17px;margin:0;font-weight:700;flex:1}}
.chead .years{{font-size:12px;color:var(--muted)}}
.dot{{width:11px;height:11px;border-radius:50%;flex:none}}
.metrics{{display:flex;flex-direction:column;gap:2px}}
.mrow{{display:grid;grid-template-columns:1.5fr 1.2fr auto;gap:10px;align-items:center;padding:7px 0;border-top:1px solid var(--border)}}
.mrow:first-child{{border-top:none}}
.mlab{{font-size:13px;color:var(--muted)}}
.mval{{font-size:15px;font-weight:650;display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
.sub{{font-size:11px;color:var(--muted);font-weight:500}}
.hn{{width:70px;height:16px}}
.pill{{font-size:11px;font-weight:600;padding:2px 8px;border-radius:20px;white-space:nowrap;display:inline-block}}
.pill.good{{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good)}}
.pill.warn{{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}}
.pill.bad{{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}}
.pill.na{{background:color-mix(in srgb,var(--muted) 14%,transparent);color:var(--muted)}}
.ddblock{{margin-top:16px;border-top:1px solid var(--border);padding-top:12px}}
.ddlab{{font-size:13px;color:var(--muted);margin-bottom:8px}}
.ddrow{{display:flex;align-items:center;gap:10px;margin:5px 0}}
.ddname{{font-size:12px;width:98px;color:var(--muted)}}
.ddtrack{{flex:1;height:10px;background:color-mix(in srgb,var(--muted) 14%,transparent);border-radius:5px;overflow:hidden}}
.ddfill{{display:block;height:100%;border-radius:5px}}
.ddpct{{font-size:13px;font-weight:650;width:38px;text-align:right}}
.ddfoot{{font-size:12px;color:var(--muted);margin-top:8px}}
.panel{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:20px;box-shadow:var(--shadow);margin-top:22px}}
.panel h2{{font-size:17px;margin:0 0 4px}}
.panel .lede{{margin-bottom:12px}}
.approvewrap{{display:flex;gap:20px;align-items:center;flex-wrap:wrap}}
.approve{{width:100%;max-width:480px;height:auto}}
.approve .grid{{stroke:var(--border);stroke-width:1}}
.approve .axis0{{stroke:var(--muted);stroke-width:1;opacity:.5;stroke-dasharray:3 3}}
.approve .tick{{fill:var(--muted);font-size:10px}}
.approve .axtitle{{fill:var(--muted);font-size:11px}}
.approve .boundary{{fill:none;stroke:var(--orb);stroke-width:2.5}}
.approve .approvefill{{fill:color-mix(in srgb,var(--good) 12%,transparent)}}
.approve .ptlab{{fill:var(--ink);font-size:11px;font-weight:700}}
.appfacts{{flex:1;min-width:210px;font-size:14px}}
.appfacts .big{{font-size:26px;font-weight:750;margin:2px 0}}
.appfacts .row{{display:flex;justify-content:space-between;padding:5px 0;border-top:1px solid var(--border)}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:12px}}
.legend span{{display:inline-flex;align-items:center;gap:6px}}
.sw{{width:12px;height:12px;border-radius:3px;display:inline-block}}
footer{{color:var(--muted);font-size:12px;margin-top:26px;text-align:center}}
details{{margin-top:18px}} summary{{cursor:pointer;color:var(--muted);font-size:13px}}
table{{border-collapse:collapse;width:100%;margin-top:10px;font-size:13px}}
th,td{{border:1px solid var(--border);padding:6px 9px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
</style>

<div class="wrap">
  <div class="eyebrow">qmeta &middot; strategy-selection scorecard</div>
  <h1>Is the edge real &mdash; and is it worth combining?</h1>
  <p class="lede">Every number below is a L&oacute;pez de Prado meta-strategy statistic, computed on your
  two live return streams. Deflated Sharpe, track-record length, overfitting probability, and the
  projected drawdown are the honest tests of whether an edge survives scrutiny.</p>

  <div class="headverdict">
    <div class="hv"><div class="k">ORB fund</div><div class="v">Sharpe {orb["sharpe_ann"]:.2f}</div>
      <div class="s">{verdict(orb)}</div></div>
    <div class="hv"><div class="k">Dip diversifier</div><div class="v">Sharpe {dip["sharpe_ann"]:.2f}</div>
      <div class="s">{verdict(dip)}</div></div>
    <div class="hv"><div class="k">Combined book</div><div class="v">Sharpe {combined:.2f}
      <span class="sub" style="color:var(--good)">+{lift:.2f}</span></div>
      <div class="s">optimal ORB+dip, corr {ap["correlation"]:+.2f}</div></div>
  </div>

  <div class="grid2">{cards}</div>

  <div class="panel">
    <h2>Strategy-Approval decision</h2>
    <p class="lede">The theorem: a candidate improves the book's Sharpe iff its Sharpe exceeds
    (correlation &times; the book's Sharpe). The shaded region is where adding the candidate helps.</p>
    <div class="approvewrap">
      {approval_svg(ap, DIP_A, ORB_A)}
      <div class="appfacts">
        <div class="sub">combined max-Sharpe</div>
        <div class="big">{combined:.2f}</div>
        <div class="row"><span>ORB (approved)</span><b>{ap["sr_approved"]:.2f}</b></div>
        <div class="row"><span>dip (candidate)</span><b>{ap["sr_candidate"]:.2f}</b></div>
        <div class="row"><span>correlation</span><b>{ap["correlation"]:+.2f}</b></div>
        <div class="row"><span>helps up to corr</span><b>{ap["max_corr_for_approval"]:.2f}</b></div>
        <div class="row"><span>verdict</span><b style="color:var(--good)">{"ADD IT" if ap["improves"] else "REJECT"}</b></div>
      </div>
    </div>
    <div class="legend">
      <span><span class="sw" style="background:var(--orb)"></span>ORB fund</span>
      <span><span class="sw" style="background:var(--dip)"></span>Dip diversifier</span>
      <span><span class="sw" style="background:color-mix(in srgb,var(--good) 40%,transparent)"></span>approve region</span>
    </div>
  </div>

  <details><summary>Show the numbers as a table</summary>
  <table><thead><tr><th>metric</th><th>ORB fund</th><th>Dip diversifier</th></tr></thead><tbody>
    <tr><td>Sharpe (ann)</td><td>{orb["sharpe_ann"]:.2f}</td><td>{dip["sharpe_ann"]:.2f}</td></tr>
    <tr><td>PSR (SR&gt;0)</td><td>{orb["psr"]*100:.1f}%</td><td>{dip["psr"]*100:.1f}%</td></tr>
    <tr><td>DSR (deflated, K={sc["n_trials"]})</td><td>{orb["dsr"]*100:.1f}%</td><td>{dip["dsr"]*100:.1f}%</td></tr>
    <tr><td>E[max SR] under null</td><td>{orb["expected_max_sr_ann"]:.2f}</td><td>{dip["expected_max_sr_ann"]:.2f}</td></tr>
    <tr><td>MinTRL (years)</td><td>{orb["mintrl_years"]:.1f}</td><td>{dip["mintrl_years"]:.1f}</td></tr>
    <tr><td>track record (years)</td><td>{orb["track_years"]:.1f}</td><td>{dip["track_years"]:.1f}</td></tr>
    <tr><td>MinBTL (years)</td><td>{orb["min_backtest_years"]:.1f}</td><td>{dip["min_backtest_years"]:.1f}</td></tr>
    <tr><td>PBO</td><td>{"n/a" if orb["pbo"] is None else f'{orb["pbo"]*100:.0f}%'}</td><td>{"n/a" if dip["pbo"] is None else f'{dip["pbo"]*100:.0f}%'}</td></tr>
    <tr><td>proj. MaxDD 95%</td><td>{orb["drawdown"]["max_dd"]*100:.1f}%</td><td>{dip["drawdown"]["max_dd"]*100:.1f}%</td></tr>
    <tr><td>empirical MaxDD</td><td>{orb["empirical_maxdd"]*100:.0f}%</td><td>{dip["empirical_maxdd"]*100:.0f}%</td></tr>
    <tr><td>proj. time underwater (y)</td><td>{orb["drawdown"]["max_tuw_years"]:.1f}</td><td>{dip["drawdown"]["max_tuw_years"]:.1f}</td></tr>
  </tbody></table></details>

  <footer>qmeta Phase 1 &middot; PSR / MinTRL / DSR / MinBTL / PBO / Strategy-Approval / Triple-Penance
  &middot; computed on realized returns, no lookahead.</footer>
</div>'''


if __name__ == "__main__":
    sc = json.loads(SC.read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html(sc), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
