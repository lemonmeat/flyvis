import numpy as np
import pytest

from flyloop.eyes import EYE_ORDER, Eyes
from flyloop.sim import Drum


@pytest.fixture(scope="module")
def eyes():
    return Eyes(axis_azimuth_deg=80, n_sub=1)


def test_eyes_are_mirror_images(eyes):
    left, right = (EYE_ORDER.index(s) for s in ("left", "right"))
    np.testing.assert_allclose(eyes.azimuth[left], -eyes.azimuth[right], atol=1e-9)
    np.testing.assert_allclose(eyes.elevation[left], eyes.elevation[right], atol=1e-9)


def test_lattice_x_points_forward_and_y_up_on_both_eyes(eyes):
    x, y = eyes.lattice.x, eyes.lattice.y
    for e in range(2):
        az, el = eyes.azimuth[e, :, 0], eyes.elevation[e, :, 0]
        front = np.abs(az) < 90  # columns looking forward of the eye's lateral axis
        assert x[front].mean() > 0 and x[~front].mean() < 0
        assert np.corrcoef(y, el)[0, 1] > 0.95
    # left eye looks left (positive azimuth), right eye right
    assert eyes.azimuth[0].mean() > 0 > eyes.azimuth[1].mean()


def test_drum_rotation_moves_pattern_counterclockwise():
    drum = Drum.grating(period_deg=40)
    az = np.linspace(-170, 170, 50)
    el = np.zeros_like(az)
    before = drum.luminance(az, el)
    drum.rotate(omega_deg_s=100.0, dt=0.1)  # 10 deg
    np.testing.assert_allclose(drum.luminance(az + 10.0, el), before, atol=1e-6)


def test_turning_fly_equals_counter_rotating_drum(eyes):
    drum = Drum.texture(seed=1)
    turned = eyes.sample(drum, heading_deg=12.0)
    drum.phase_deg = -12.0
    np.testing.assert_allclose(eyes.sample(drum, heading_deg=0.0), turned, atol=1e-6)


def test_sample_shape_and_range(eyes):
    frame = eyes.sample(Drum.grating(), heading_deg=0.0)
    assert frame.shape == (2, 1, 721) and frame.dtype == np.float32
    assert 0 <= frame.min() and frame.max() <= 1
