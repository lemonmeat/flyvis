import numpy as np
import pytest
import torch

from flyloop.stimuli import (
    HexLattice,
    drifting_grating,
    frames_to_hex,
    moving_edge,
    rotating_grating,
    rotating_texture,
    texture_1f,
)

DT = 1 / 100


def _centroid(w, lat):
    w = w / w.sum()
    return np.array([(w * lat.x).sum(), (w * lat.y).sum()])


def test_lattice_matches_flyvis_size_and_center():
    lat = HexLattice()
    assert lat.n_hexals == 721 and lat.x.shape == (721,)
    assert lat.x[lat.central_index] == 0 and lat.y[lat.central_index] == 0


def test_to_sphere_axis_and_mirror():
    lat = HexLattice()
    az, el = lat.to_sphere(np.array([0.0, 10.0]), np.array([0.0, 0.0]), 90.0, 0.0)
    np.testing.assert_allclose(az, [90.0, 100.0], atol=1e-9)
    np.testing.assert_allclose(el, [0.0, 0.0], atol=1e-9)
    az, _ = lat.to_sphere(np.array([10.0]), np.array([0.0]), -90.0, 0.0, mirror=True)
    np.testing.assert_allclose(az, [-100.0], atol=1e-9)


@pytest.mark.parametrize("direction", [0, 45, 90, 225])
@pytest.mark.parametrize("polarity", ["on", "off"])
def test_moving_edge_sweeps_in_direction(direction, polarity):
    lat = HexLattice(n_sub=1)
    stim = moving_edge(direction, polarity, t_pre=0.1, t_post=0.1, dt=DT, lattice=lat)
    f = stim.frames.cpu().numpy()[:, 0]
    assert f.shape[1] == 721 and stim.frames.dtype == torch.float32
    assert f.min() >= 0 and f.max() <= 1
    np.testing.assert_allclose(f[0], 0.5)  # grey before onset
    final = 1.0 if polarity == "on" else 0.0
    np.testing.assert_allclose(f[-1], final, atol=1e-6)  # sweep covers the lattice
    # each column switches when the edge passes; the gradient of switch time over
    # (x, y) points along the motion direction with magnitude 1 / speed
    t_switch = stim.time[np.argmax(f != 0.5, axis=0)]
    A = np.stack([lat.x, lat.y, np.ones_like(lat.x)], axis=1)
    gx, gy, _ = np.linalg.lstsq(A, t_switch, rcond=None)[0]
    err = (np.degrees(np.arctan2(gy, gx)) - direction + 180) % 360 - 180
    assert abs(err) < 1
    assert np.isclose(1 / np.hypot(gx, gy), stim.params["speed_deg_s"], rtol=0.02)


def test_drifting_grating_contrast_and_speed():
    lat = HexLattice(n_sub=1)
    s = drifting_grating(0, spatial_period_deg=40, temporal_freq_hz=2, contrast=0.4,
                         t_pre=0, duration_s=0.5, dt=DT, lattice=lat)
    f = s.frames.cpu().numpy()[:, 0]
    np.testing.assert_allclose(np.unique(f.round(6)), [0.3, 0.7])
    # one temporal period (0.5 s) later the pattern repeats
    np.testing.assert_allclose(drifting_grating(0, 40, 2, 0.4, t_pre=0, duration_s=0.6,
                                                dt=DT, lattice=lat).frames[50].cpu(),
                               s.frames[0].cpu())


def test_rotating_grating_moves_toward_increasing_azimuth():
    lat = HexLattice(n_sub=1)
    omega, period = 60.0, 30.0
    s = rotating_grating(omega, period, t_pre=0, duration_s=0.2, dt=DT, lattice=lat,
                         eye_azimuth_deg=90)
    f = s.frames.cpu().numpy()[:, 0]
    az, el = lat.to_sphere(lat.x, lat.y, 90, 0)
    ref = rotating_grating(0.0, period, t_pre=0, duration_s=DT, dt=DT, lattice=lat,
                           eye_azimuth_deg=90).frames.cpu().numpy()[0, 0]
    # after t seconds the pattern equals the static drum shifted by omega * t
    t = 0.1
    shifted = 0.5 * (1 + np.where(np.sin(2 * np.pi * (az - omega * t) / period) >= 0,
                                  1, -1))
    np.testing.assert_allclose(f[int(t / DT)], shifted, atol=1e-6)
    assert not np.allclose(f[int(t / DT)], ref)


def test_texture_spectrum_contrast_density():
    img = texture_1f((256, 256), alpha=2.0, contrast=0.8, seed=1)
    assert img.dtype == np.float32
    assert img.min() >= 0.5 * (1 - 0.8) - 1e-6 and img.max() <= 0.5 * (1 + 0.8) + 1e-6
    # fit the radially averaged power spectrum slope
    p = np.abs(np.fft.fft2(img - img.mean())) ** 2
    f = np.hypot(*np.meshgrid(np.fft.fftfreq(256), np.fft.fftfreq(256)))
    bins = np.logspace(np.log10(2 / 256), np.log10(0.4), 15)
    idx = np.digitize(f.ravel(), bins)
    fm = [f.ravel()[idx == i].mean() for i in range(1, len(bins))]
    pm = [p.ravel()[idx == i].mean() for i in range(1, len(bins))]
    slope = np.polyfit(np.log(fm), np.log(pm), 1)[0]
    assert -2.3 < slope < -1.7
    sparse = texture_1f((256, 256), alpha=2.0, density=0.3, seed=1)
    assert abs(np.mean(sparse != 0.5) - 0.3) < 0.02


def test_rotating_texture_and_boxeye_shapes():
    pano = texture_1f((90, 180), seed=0)
    s = rotating_texture(pano, 90.0, duration_s=0.05, t_pre=0.0, dt=DT,
                         lattice=HexLattice(n_sub=2))
    assert s.frames.shape == (5, 1, 721)
    hexed = frames_to_hex(np.random.rand(3, 64, 64))
    assert hexed.shape == (3, 1, 721)


@pytest.mark.flyvis
def test_lattice_angle_convention_matches_flyvis_hexeye():
    """A spot offset along flyvis angle theta moves toward theta on our lattice."""
    from flyvis.datasets.rendering import HexEye

    lat = HexLattice()
    eye = HexEye(721, ppo=9, dtype=torch.float32)
    for theta in [0, 60, 135, 300]:
        c = [
            _centroid(
                eye.render_bar(np.radians(12), np.radians(12), np.radians(90 + off),
                               np.radians(90), 1, 1.0, 0.0, theta).cpu().numpy().ravel(),
                lat,
            )
            for off in (-25, 25)
        ]
        d = c[1] - c[0]
        err = (np.degrees(np.arctan2(d[1], d[0])) - theta + 180) % 360 - 180
        assert abs(err) < 1
        assert abs(np.hypot(*d) - 50) < 2
