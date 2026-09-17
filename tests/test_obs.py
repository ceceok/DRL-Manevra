"""Gözlem kodlayıcıları — AbsoluteObs boyutu + LOSRelativeObs dönme değişmezliği.

LOS gözleminin ASIL amacı rotational symmetry: tüm sahneyi bir açı kadar döndürmek
gözlemi değiştirmemeli. Bu, LOS-göreli çerçevenin doğru kurulduğunu kanıtlar
(RL'nin dönme simetrisini sömürebilmesinin ön koşulu).
"""
import numpy as np
from tma.obs import AbsoluteObs, LOSRelativeObs
from tma.util import wrap


class _StubFilter:
    def __init__(self, m, P):
        self._m = m; self._P = P
    @property
    def estimate(self): return self._m
    @property
    def covariance(self): return self._P


class _StubOwnship:
    def __init__(self, pos, hdg, filt):
        self.pos = pos; self.hdg = hdg; self.filter = filt


class _StubWorld:
    def __init__(self, m, P, own, hdg, t=300.0, total_time=1200.0):
        self.ownship = _StubOwnship(own, hdg, _StubFilter(m, P))
        self.t = t; self.total_time = total_time


def _rotate_vec(v, phi):
    # bearing atan2(x,y) konvansiyonuyla tutarlı dönme (bearing -> bearing+phi)
    c, s = np.cos(phi), np.sin(phi)
    return np.array([v[0] * c + v[1] * s, -v[0] * s + v[1] * c])


def _rotate_cov(P2, phi):
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, s], [-s, c]])
    return R @ P2 @ R.T


def test_los_rotation_invariance():
    rng = np.random.default_rng(3)
    enc = LOSRelativeObs()
    for _ in range(300):
        m = np.concatenate([rng.uniform(-15000, 15000, 2), rng.uniform(-8, 8, 2)])
        A = rng.uniform(-1, 1, (4, 4)); P = A @ A.T + np.eye(4)     # PD kovaryans
        own = rng.uniform(-5000, 5000, 2)
        hdg = rng.uniform(0, 2 * np.pi)
        obs0 = enc(_StubWorld(m, P, own, hdg))
        phi = rng.uniform(0, 2 * np.pi)
        m_r = np.concatenate([_rotate_vec(m[:2], phi), _rotate_vec(m[2:], phi)])
        own_r = _rotate_vec(own, phi)
        P_r = P.copy()
        P_r[:2, :2] = _rotate_cov(P[:2, :2], phi)
        P_r[2:, 2:] = _rotate_cov(P[2:, 2:], phi)
        obs_r = enc(_StubWorld(m_r, P_r, own_r, wrap(hdg + phi)))
        assert np.allclose(obs0, obs_r, atol=1e-9), "LOS gözlemi dönme altında değişti"


def test_absolute_obs_dim():
    enc = AbsoluteObs()
    from tma.registry import build_world
    w = build_world(seed=1)
    w.reset()
    o = enc(w)
    assert o.shape == (15,)
    assert np.all(np.isfinite(o))


def test_los_obs_dim():
    from tma.registry import build_world
    w = build_world(seed=1, obs="los")
    o = w.reset()
    assert o.shape == (9,)
    assert np.all(np.isfinite(o))


if __name__ == "__main__":
    test_los_rotation_invariance()
    test_absolute_obs_dim()
    test_los_obs_dim()
    print("Gözlem: LOS dönme-değişmez (300 örnek, atol=1e-9); AbsoluteObs=15D, LOS=9D. GEÇTI.")
