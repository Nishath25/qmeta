import math
import numpy as np
from qmeta.selection.approval import (
    combined_max_sharpe, improves_max_sharpe,
    max_correlation_for_approval, indifference_curve,
)


def test_combined_max_sharpe_reference():
    # S_a=1, S_n=1, rho=0.5 -> sqrt((1-1+1)/0.75) = 1.15470
    assert abs(combined_max_sharpe(1.0, 1.0, 0.5) - math.sqrt(1 / 0.75)) < 1e-12


def test_improves_condition():
    assert improves_max_sharpe(0.95, 0.76, -0.02) is True    # ORB + dip
    assert improves_max_sharpe(1.0, 0.4, 0.5) is False       # 0.4 < 0.5*1.0
    assert improves_max_sharpe(1.0, 0.6, 0.5) is True


def test_max_correlation_for_approval():
    assert abs(max_correlation_for_approval(0.95, 0.76) - 0.76 / 0.95) < 1e-12
    assert max_correlation_for_approval(0.5, 1.0) == 1.0     # capped


def test_indifference_boundary_is_rho_times_Sa():
    # At level == sr_app the boundary is sr_new = rho * sr_app
    corr, srn = indifference_curve(1.0, level=1.0, n=41)
    ok = ~np.isnan(srn)
    assert np.allclose(srn[ok], corr[ok], atol=1e-9)


def test_orb_dip_combined_beats_orb():
    assert combined_max_sharpe(0.95, 0.76, -0.02) > 0.95
