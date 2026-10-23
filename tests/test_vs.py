import numpy as np
import pytest

from flyloop.eyes import Eyes
from flyloop.lptc import LPTCLayer, LPTCWeights
from flyloop.lptc.analysis import matched_filter_response, preferred_axis, rotation_flow
from flyloop.sim import (
    PITCH_AXIS,
    ROLL_AXIS,
    YAW_AXIS,
    Drum,
    SphereWorld,
    VerticalGrating,
    horizontal_axis,
)


@pytest.fixture(scope="module")
def eyes():
    return Eyes(n_sub=1)


def test_sphere_yaw_matches_drum():
    sphere = SphereWorld.texture(seed=2)
    drum = Drum(sphere.panorama)
    az = np.linspace(-170, 170, 40)
    el = np.zeros_like(az)
    sphere.rotate(YAW_AXIS, 50.0, 0.2)
    drum.rotate(50.0, 0.2)
    np.testing.assert_allclose(sphere.luminance(az, el), drum.luminance(az, el), atol=1e-5)


@pytest.mark.parametrize("axis, at, moves_to", [
    (PITCH_AXIS, (0.0, 0.0), (0.0, -10.0)),    # pitch +: front moves down
    (ROLL_AXIS, (-90.0, 0.0), (-90.0, -10.0)),  # roll +: right side moves down
    (ROLL_AXIS, (90.0, 0.0), (90.0, 10.0)),     # ... and left side up
])
def test_sphere_rotation_directions(axis, at, moves_to):
    w = SphereWorld.checkerboard(7)
    before = w.luminance(*map(np.array, at))
    w.rotate(axis, 10.0, 1.0)
    np.testing.assert_allclose(w.luminance(*map(np.array, moves_to)), before, atol=1e-5)


def test_vertical_grating_drifts_down():
    g = VerticalGrating(period_deg=40)
    el = np.linspace(-60, 60, 25)
    before = g.luminance(np.zeros_like(el), el)
    g.drift(-50.0, 0.1)  # 5 deg down
    np.testing.assert_allclose(g.luminance(np.zeros_like(el), el - 5.0), before)


def test_rotation_flow_vertical_component_is_independent_of_elevation(eyes):
    theta = 30.0
    v_az, v_el = rotation_flow(eyes, horizontal_axis(theta))
    phi = np.radians(eyes.azimuth.mean(-1))
    el = np.radians(eyes.elevation.mean(-1))
    np.testing.assert_allclose(v_el, np.sin(phi - np.radians(theta)), atol=1e-9)
    np.testing.assert_allclose(v_az, -np.sin(el) * np.cos(phi - np.radians(theta)), atol=1e-9)


def test_matched_filter_simple_vs_prefers_axis_90_deg_from_eye(eyes):
    w = LPTCWeights.simple()
    grid = np.arange(0, 360, 10.0)
    r = np.stack([matched_filter_response(w, eyes, horizontal_axis(t)) for t in grid], -1)
    pref = preferred_axis(r[:, w.index("VS")[0]], grid)
    # eye axes at +-80 deg -> preferred roll-like axes at -80+90 and 80+90
    np.testing.assert_allclose(pref, [170.0, 10.0], atol=3.0)  # left, right


@pytest.mark.flyvis
def test_vs_responds_to_downward_motion_and_roll():
    from flyloop.sim import VisualRig
    from flyloop.vision import FlyvisModel

    dt = 0.01
    w = LPTCWeights.simple()
    rig = VisualRig(Eyes(n_sub=2), FlyvisModel(dt=dt), LPTCLayer(w, dt))
    vs = w.index("VS")[0]

    def run(world, move, n=60):
        rig.reset(world, t_settle=0.3)
        out = []
        for _ in range(n):
            move(world)
            out.append(rig.step(world).lptc)
        return np.concatenate(out, 1)[:, n // 2:, vs].mean(1)  # (left, right)

    down = run(VerticalGrating(30), lambda g: g.drift(-60, dt))
    up = run(VerticalGrating(30), lambda g: g.drift(+60, dt))
    assert (down > up + 0.1).all()
    roll_pos = run(SphereWorld.checkerboard(20), lambda g: g.rotate(ROLL_AXIS, 90, dt))
    roll_neg = run(SphereWorld.checkerboard(20), lambda g: g.rotate(ROLL_AXIS, -90, dt))
    # roll +: right side moves down -> right VS up, left VS down
    assert roll_pos[1] - roll_pos[0] > 0.05 and roll_neg[1] - roll_neg[0] < -0.05
