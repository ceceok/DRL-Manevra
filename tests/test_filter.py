"""Adım 2 regresyonu — Filtre bit-birebir.

Yeni RPEKFBankFilter ile mevcut kodun (tma_full.py) gömülü bankasının birebir
kopyası, AYNI (z, own) dizisi üzerinde lockstep koşturulur; her adımda
xs/Ps/wb/estimate/covariance bit-birebir eşit olmalı.

OldBank, tma_full.py'deki _init_bank/_bank_update/_moment_match'in birebir
kopyasıdır (geçici referans; refactor bitince silinir). Nihai doğrulama Adım 3'te
golden.npz'ye karşı tam-tablo regresyonuyla yapılacak.
"""
import numpy as np
from tma.sensor import BearingOnlySensor
from tma.filter import RPEKFBankFilter

SIGMA = np.deg2rad(1.0); N_HYP = 6; DT = 10.0; OWN_SPD = 8.0
def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi


# ---- eski gömülü banka (tma_full.py birebir) ----
class OldBank:
    def init(self, z0, own):
        rh = np.geomspace(3000, 18000, N_HYP)
        ul = np.array([np.sin(z0), np.cos(z0)]); up = np.array([np.cos(z0), -np.sin(z0)])
        dl = np.log(rh[1]/rh[0])
        self.xs = np.zeros((N_HYP, 4)); self.Ps = np.zeros((N_HYP, 4, 4))
        for i, ri in enumerate(rh):
            self.xs[i, :2] = own + ri*ul
            Rm = np.outer(ul, ul)*(ri*dl/2)**2 + np.outer(up, up)*(ri*SIGMA*3)**2
            self.Ps[i][:2, :2] = Rm; self.Ps[i][2, 2] = self.Ps[i][3, 3] = 25.0
        self.wb = np.full(N_HYP, 1.0/N_HYP)
        self.F = np.eye(4); self.F[0, 2] = self.F[1, 3] = DT
        q = 1e-3
        self.Q = q*np.array([[DT**3/3, 0, DT**2/2, 0], [0, DT**3/3, 0, DT**2/2],
                             [DT**2/2, 0, DT, 0], [0, DT**2/2, 0, DT]])

    def step(self, z, own):
        lw = np.zeros(N_HYP); I4 = np.eye(4)
        for i in range(N_HYP):
            x = self.F @ self.xs[i]; Pm = self.F @ self.Ps[i] @ self.F.T + self.Q
            dx, dy = x[0]-own[0], x[1]-own[1]; r2 = dx*dx + dy*dy
            H = np.array([dy/r2, -dx/r2, 0, 0])
            S = H @ Pm @ H + SIGMA**2
            nu = wrap(z - np.arctan2(dx, dy)); K = Pm @ H / S
            self.xs[i] = x + K*nu
            IKH = I4 - np.outer(K, H)
            self.Ps[i] = IKH @ Pm @ IKH.T + np.outer(K, K)*SIGMA**2
            lw[i] = -0.5*nu*nu/S - 0.5*np.log(S)
        w = self.wb*np.exp(lw - lw.max()); s = w.sum()
        self.wb = np.full(N_HYP, 1.0/N_HYP) if (not np.isfinite(s) or s <= 0) else w/s

    def estimate(self): return self.wb @ self.xs

    def cov(self):
        m = self.estimate()
        return sum(self.wb[i]*(self.Ps[i] + np.outer(self.xs[i]-m, self.xs[i]-m))
                   for i in range(N_HYP))


def _one_scenario(seed):
    g = np.random.default_rng(seed)
    # tma_full.reset ile aynı domain randomization sırası
    R0 = g.uniform(5000, 15000); b0 = g.uniform(0, 2*np.pi)
    spd = g.uniform(2, 8); hdg = g.uniform(0, 2*np.pi)
    tgt = np.array([R0*np.sin(b0), R0*np.cos(b0), spd*np.sin(hdg), spd*np.cos(hdg)])
    own = np.zeros(2); ohdg = g.uniform(0, 2*np.pi)
    sensor = BearingOnlySensor(1.0)

    z0 = sensor.sample(own, tgt, 0.0, g)
    old = OldBank(); old.init(z0, own)
    new = RPEKFBankFilter(sensor, dt=DT); new.init(z0, own)

    for k in range(1, 121):
        if (k - 1) % 6 == 0:                       # 60 s'de bir rota değişimi
            ohdg = g.uniform(0, 2*np.pi)
        own = own + OWN_SPD*np.array([np.sin(ohdg), np.cos(ohdg)])*DT
        t = k*DT
        z = sensor.sample(own, tgt, t, g)          # AYNI z ikisine de
        old.step(z, own); new.step(z, own)
        assert np.array_equal(old.xs, new.xs),        f"xs ayrıştı (seed={seed}, k={k})"
        assert np.array_equal(old.Ps, new.Ps),        f"Ps ayrıştı (seed={seed}, k={k})"
        assert np.array_equal(old.wb, new.wb),        f"wb ayrıştı (seed={seed}, k={k})"
        assert np.array_equal(old.estimate(), new.estimate),   f"estimate (seed={seed}, k={k})"
        assert np.array_equal(old.cov(), new.covariance),      f"covariance (seed={seed}, k={k})"


def test_filter_bitexact():
    for seed in range(30):
        _one_scenario(seed)


def test_propagate_state_matches_closed_form():
    sensor = BearingOnlySensor(1.0)
    f = RPEKFBankFilter(sensor, dt=DT)
    g = np.random.default_rng(1)
    for _ in range(200):
        x = np.concatenate([g.uniform(-15000, 15000, 2), g.uniform(-8, 8, 2)])
        dts = g.uniform(0, 1200, 7)
        got = f.propagate_state(x, dts)
        want = np.column_stack([x[:2] + np.outer(dts, x[2:]),
                                np.tile(x[2:], (len(dts), 1))])
        assert np.array_equal(got[:, :2], x[:2] + np.outer(dts, x[2:]))
        assert np.array_equal(got[:, 2], np.full(len(dts), x[2]))
        assert np.array_equal(got[:, 3], np.full(len(dts), x[3]))


def test_planning_states():
    sensor = BearingOnlySensor(1.0)
    f = RPEKFBankFilter(sensor, dt=DT)
    f.init(0.3, np.zeros(2))
    ps = f.planning_states()
    assert len(ps) == 6
    assert abs(sum(w for _, w in ps) - 1.0) < 1e-12
    assert all(len(s) == 4 for s, _ in ps)


if __name__ == "__main__":
    test_filter_bitexact()
    test_propagate_state_matches_closed_form()
    test_planning_states()
    print("Filtre bit-birebir: 30 senaryo x 120 adım — xs/Ps/wb/estimate/covariance TÜM adımlarda eşit.")
    print("propagate_state kapalı-form ile eşit; planning_states 6 hipotez, ağırlık toplamı 1. GEÇTI.")
