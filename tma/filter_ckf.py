"""Cubature Kalman Filter (CKF) — Filter sözleşmesi (Ristic & Arulampalam 2026).

Türevsiz: yalnızca sensor.h'ye ihtiyaç duyar (sensor.H'ye değil). Mutlak hedef
durumu izlenir ([x,y,vx,vy]); gözlemci deterministik bilindiği için mutlak-hedef
kovaryansı = makalenin göreli-durum kovaryansına denktir. Üçüncü-derece küresel-
radyal kübatür: n=4, L=2n=8 nokta, eşit ağırlık 1/L.
"""
import numpy as np
from .util import wrap


class CubatureKF:
    privileged = False

    def __init__(self, sensor, dt, q=1e-6, R0=23.0, sigma_R=5.0, vmax=3.0):
        self.sensor = sensor
        self.dt = dt
        self.R0 = R0
        self.sigma_R = sigma_R
        self.sigma_v = vmax / 3.0
        self.F = np.eye(4); self.F[0, 2] = self.F[1, 3] = dt
        self.Q = q * np.array([[dt**3/3, 0, dt**2/2, 0],
                               [0, dt**3/3, 0, dt**2/2],
                               [dt**2/2, 0, dt, 0],
                               [0, dt**2/2, 0, dt]])
        self.n = 4
        self.x = None
        self.P = None

    def init(self, z0, own_pos):
        ul = np.array([np.sin(z0), np.cos(z0)])       # LOS yönü (menzil belirsizliği)
        up = np.array([np.cos(z0), -np.sin(z0)])      # dik (kerteriz belirsizliği)
        pos = own_pos + self.R0 * ul
        self.x = np.array([pos[0], pos[1], 0.0, 0.0])
        Ppos = (np.outer(ul, ul) * self.sigma_R**2
                + np.outer(up, up) * (self.R0 * self.sensor.sigma)**2)
        self.P = np.zeros((4, 4))
        self.P[:2, :2] = Ppos
        self.P[2, 2] = self.P[3, 3] = self.sigma_v**2

    def _cubature_points(self, x, P):
        try:
            S = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            S = np.linalg.cholesky(P + 1e-9 * np.eye(4))
        pts = np.empty((2 * self.n, 4))
        f = np.sqrt(self.n)
        for i in range(self.n):
            pts[i] = x + f * S[:, i]
            pts[self.n + i] = x - f * S[:, i]
        return pts

    def step(self, z, own_pos):
        xp = self.F @ self.x
        Pp = self.F @ self.P @ self.F.T + self.Q
        if z is None:
            self.x, self.P = xp, Pp
            return
        L = 2 * self.n
        pts = self._cubature_points(xp, Pp)
        Zi = np.array([self.sensor.h(p, own_pos) for p in pts])
        zbar = np.arctan2(np.mean(np.sin(Zi)), np.mean(np.cos(Zi)))   # açısal ortalama
        dZ = wrap(Zi - zbar)
        Pzz = np.mean(dZ**2) + self.sensor.sigma**2
        dX = pts - xp
        Pxz = (dX * dZ[:, None]).mean(axis=0)
        W = Pxz / Pzz
        nu = wrap(z - zbar)
        self.x = xp + W * nu
        self.P = Pp - np.outer(W, W) * Pzz

    @property
    def estimate(self):
        return self.x

    @property
    def covariance(self):
        return self.P

    def propagate_state(self, x, dts):
        dts = np.atleast_1d(np.asarray(dts, dtype=float))
        return np.column_stack([x[0] + x[2]*dts, x[1] + x[3]*dts,
                                np.full_like(dts, x[2]), np.full_like(dts, x[3])])

    def planning_states(self):
        return [(self.x, 1.0)]          # tek Gaussian -> certainty-equivalent
