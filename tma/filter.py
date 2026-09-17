"""Menzil-parametreli EKF bankası — Filter sözleşmesine uyarlanmış.

Tek filtre; içeride menzil-hipotezli bir EKF karışımı tutar ama dışarıya TEK
(estimate, covariance) raporlar (moment-eşleme). FIM planlayıcının ihtiyacı olan
ağırlıklı hipotezleri `planning_states()` ile ayrı bir pencereden verir (Karar A-i).

Ölçüm modeli (h, H, R) filtreye AIT DEĞİL — Sensör'den gelir. Böylece sensörü
değiştirince (ör. bearing+range) filtre güncellemesi kendiliğinden değişir.

Bit-birebir referans: tma_full.py'deki gömülü banka (_init_bank / _bank_update /
_moment_match). Sensör Adım 1'de bit-birebir doğrulandığı için buradaki
sensor.h/H/R çağrıları da eski satır-içi matematikle aynı sayıyı üretir.
"""
import numpy as np
from .util import wrap


class RPEKFBankFilter:
    privileged = False   # gerçeği kullanmaz; yalnızca ölçümlerden belief kurar

    def __init__(self, sensor, dt, n_hyp=6, r_range=(3000.0, 18000.0),
                 q=1e-3, vel_var0=25.0, cross_sigma_scale=3.0):
        self.sensor = sensor
        self.dt = dt
        self.n_hyp = n_hyp
        self.r_range = r_range
        self.vel_var0 = vel_var0
        self.cross_sigma_scale = cross_sigma_scale
        self.F = np.eye(4); self.F[0, 2] = self.F[1, 3] = dt
        self.Q = q * np.array([[dt**3/3, 0, dt**2/2, 0],
                               [0, dt**3/3, 0, dt**2/2],
                               [dt**2/2, 0, dt, 0],
                               [0, dt**2/2, 0, dt]])
        self.xs = self.Ps = self.wb = None

    # ---- ilk hipotez bankası (ilk kerterizden) ----
    def init(self, z0, own_pos):
        rh = np.geomspace(self.r_range[0], self.r_range[1], self.n_hyp)
        ul = np.array([np.sin(z0), np.cos(z0)])
        up = np.array([np.cos(z0), -np.sin(z0)])
        dl = np.log(rh[1] / rh[0])
        self.xs = np.zeros((self.n_hyp, 4))
        self.Ps = np.zeros((self.n_hyp, 4, 4))
        for i, ri in enumerate(rh):
            self.xs[i, :2] = own_pos + ri * ul
            Rm = (np.outer(ul, ul) * (ri * dl / 2)**2
                  + np.outer(up, up) * (ri * self.sensor.sigma * self.cross_sigma_scale)**2)
            self.Ps[i][:2, :2] = Rm
            self.Ps[i][2, 2] = self.Ps[i][3, 3] = self.vel_var0
        self.wb = np.full(self.n_hyp, 1.0 / self.n_hyp)

    # ---- tek adım: predict + (varsa) update ----
    def step(self, z, own_pos):
        # z None ise (görüş yok) yalnızca predict — bankayı ileri taşı, ağırlık dokunma
        if z is None:
            for i in range(self.n_hyp):
                self.xs[i] = self.F @ self.xs[i]
                self.Ps[i] = self.F @ self.Ps[i] @ self.F.T + self.Q
            return

        lw = np.zeros(self.n_hyp)
        I4 = np.eye(4)
        R = self.sensor.R[0, 0]
        for i in range(self.n_hyp):
            x = self.F @ self.xs[i]
            Pm = self.F @ self.Ps[i] @ self.F.T + self.Q
            H = self.sensor.H(x, own_pos)                 # ölçüm modeli sensörden
            S = H @ Pm @ H + R
            nu = wrap(z - self.sensor.h(x, own_pos))
            K = Pm @ H / S
            self.xs[i] = x + K * nu
            IKH = I4 - np.outer(K, H)
            self.Ps[i] = IKH @ Pm @ IKH.T + np.outer(K, K) * R   # Joseph
            lw[i] = -0.5 * nu * nu / S - 0.5 * np.log(S)
        w = self.wb * np.exp(lw - lw.max())
        s = w.sum()
        self.wb = (np.full(self.n_hyp, 1.0 / self.n_hyp)
                   if (not np.isfinite(s) or s <= 0) else w / s)

    # ---- raporlanan kestirim (tek Gaussian, moment-eşlenmiş) ----
    @property
    def estimate(self):
        return self.wb @ self.xs

    @property
    def covariance(self):
        m = self.wb @ self.xs
        return sum(self.wb[i] * (self.Ps[i] + np.outer(self.xs[i] - m, self.xs[i] - m))
                   for i in range(self.n_hyp))

    # ---- deterministik ileri-model (planlayıcı için; Q yok, cov yok) ----
    def propagate_state(self, x, dts):
        dts = np.atleast_1d(np.asarray(dts, dtype=float))
        return np.column_stack([x[0] + x[2] * dts, x[1] + x[3] * dts,
                                np.full_like(dts, x[2]), np.full_like(dts, x[3])])

    # ---- planlama görünüşü: ağırlıklı hipotezler (Karar A-i) ----
    def planning_states(self):
        return list(zip(self.xs, self.wb))
