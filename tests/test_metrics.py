import numpy as np
import pytest
from scipy.special import i0, i1

from flyloop.metrics import (
    angular_difference,
    direction_selectivity_index,
    peak_response,
    preferred_direction,
)

ANGLES = np.arange(0, 360, 45)


def von_mises(pref, kappa, angles=ANGLES):
    return np.exp(kappa * np.cos(np.radians(angles - pref)))


@pytest.mark.parametrize("pref", [0, 90, 180, 270, 30, 333])
def test_preferred_direction_recovers_von_mises_peak(pref):
    pd = preferred_direction(von_mises(pref, 2.0), ANGLES)
    assert abs(angular_difference(pd, pref)) < 0.5


@pytest.mark.parametrize("kappa", [0.5, 1.0, 2.0, 3.0])
def test_dsi_of_von_mises_matches_bessel_ratio(kappa):
    # for a von Mises curve the vector-sum DSI is I1(kappa)/I0(kappa); 8 samples
    # are enough for the discrete sum to match closely up to kappa ~ 3
    dsi = direction_selectivity_index(von_mises(90, kappa), ANGLES)
    assert dsi == pytest.approx(i1(kappa) / i0(kappa), abs=1e-3)


def test_dsi_limits():
    assert direction_selectivity_index(np.ones(8), ANGLES) == pytest.approx(0, abs=1e-12)
    single = np.zeros(8)
    single[3] = 2.0
    assert direction_selectivity_index(single, ANGLES) == pytest.approx(1.0)
    assert preferred_direction(single, ANGLES) == pytest.approx(135.0)
    assert direction_selectivity_index(np.zeros(8), ANGLES) == 0
    assert np.isnan(preferred_direction(np.zeros(8), ANGLES))


def test_rotation_equivariance_and_batching():
    base = np.clip(np.cos(np.radians(ANGLES - 90)), 0, None) + 0.1
    curves = np.stack([np.roll(base, k) for k in range(8)])
    pds = preferred_direction(curves, ANGLES)
    np.testing.assert_allclose(angular_difference(pds, 90 + 45 * np.arange(8)), 0,
                               atol=1e-9)
    dsis = direction_selectivity_index(curves, ANGLES)
    np.testing.assert_allclose(dsis, dsis[0])
    assert 0 < dsis[0] < 1


def test_norm_across_polarities_like_flyvis():
    on = von_mises(0, 2.0)
    off = 0.25 * von_mises(180, 1.0)
    norm = max(on.sum(), off.sum())
    assert direction_selectivity_index(on, ANGLES, norm) == pytest.approx(
        direction_selectivity_index(on, ANGLES))
    # the weaker polarity's DSI shrinks by its share of the larger response sum
    assert direction_selectivity_index(off, ANGLES, norm) == pytest.approx(
        direction_selectivity_index(off, ANGLES) * off.sum() / norm)


def test_negative_responses_rejected():
    with pytest.raises(ValueError):
        direction_selectivity_index(np.array([1, -1, 0, 0, 0, 0, 0, 0.0]), ANGLES)


def test_angular_difference_wraps():
    np.testing.assert_allclose(angular_difference([350, 10, 180], [10, 350, 0]),
                               [-20, 20, 180])


def test_peak_response_window_baseline_rectify():
    time = np.round(np.arange(-10, 20) * 0.05, 10)
    trace = np.where(time < 0, 2.0, 2.0 + np.sin(2 * np.pi * time))
    assert peak_response(trace, time) == pytest.approx(3.0, abs=1e-2)
    assert peak_response(trace, time, baseline="pre") == pytest.approx(1.0, abs=1e-2)
    assert peak_response(trace, time, 0.55, 0.95, baseline="pre") == 0.0
    assert peak_response(trace, time, 0.55, 0.95, baseline="pre", rectify=False) < 0
    batch = np.stack([trace, 2 * trace])
    np.testing.assert_allclose(peak_response(batch, time, baseline="pre"), [1.0, 2.0],
                               atol=1e-2)
