"""Smoke test for the dashboard generator: build_html must run on a real
scorecard.json and produce well-formed HTML. Skips if no scorecard cached."""
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, r"C:\Users\madas\qmeta\scripts")
SC = Path(r"C:\Users\madas\qmeta\scratch\scorecard.json")

pytestmark = pytest.mark.skipif(not SC.exists(), reason="no scorecard.json cached")


def test_build_html_runs_and_has_content():
    from make_dashboard import build_html
    html = build_html(json.loads(SC.read_text()))
    assert isinstance(html, str) and len(html) > 2000
    for token in ("ORB fund", "Dip diversifier", "Strategy-Approval", "combined max-Sharpe"):
        assert token in html
    assert "None%" not in html and "nan%" not in html  # no unformatted values leaked
