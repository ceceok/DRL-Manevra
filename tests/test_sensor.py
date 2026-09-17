"""Adım 1 regresyonu — Sensör bit-birebir.

Yeni BearingOnlySensor'ın h/H/sample çıktılarının, mevcut koddaki (tma_full.py)
satır-içi matematikle BIT-BIREBIR eşit olduğunu doğrular. Bu, Faz 1'in temel
güvencesi: bileşeni çıkardık ama sayısal olarak hiçbir şey değişmedi.

Aşağıdaki old_* fonksiyonları, tma_full.py'deki ilgili satırların birebir
kopyasıdır (geçici referans; refactor bitince silinir).
"""
import numpy as np
from tma.sensor import BearingOnlySensor

SIGMA = np.deg2rad(1.0)   # tma_full.py'deki modül sabiti


# ---- eski (satır-içi) referans, tma_full.py'den birebir ----
def old_measure(tgt, own, t, rng):
    d = tgt[:2] + tgt[2:] * t - own
    return np.arctan2(d[0], d[1]) + SIGMA * rng.standard_normal()

def old_h(x, own):
    dx, dy = x[0] - own[0], x[1] - own[1]
    return np.arctan2(dx, dy)

def old_H(x, own):
    dx, dy = x[0] - own[0], x[1] - own[1]
    r2 = dx * dx + dy * dy
    return np.array([dy / r2, -dx / r2, 0, 0])


def test_h_H_bitexact():
    s = BearingOnlySensor(sigma_deg=1.0)
    g = np.random.default_rng(12345)
    for _ in range(2000):
        own = g.uniform(-5000, 5000, 2)
        x = np.concatenate([g.uniform(-15000, 15000, 2), g.uniform(-8, 8, 2)])
        assert s.h(x, own) == old_h(x, own)                     # atan2 bit-birebir
        assert np.array_equal(s.H(x, own), old_H(x, own))       # Jacobian bit-birebir


def test_sample_bitexact():
    s = BearingOnlySensor(sigma_deg=1.0)
    g = np.random.default_rng(12345)
    # AYNI tohumlu iki rng: lockstep'te aynı çekme -> aynı ölçüm
    rng_old = np.random.default_rng(7)
    rng_new = np.random.default_rng(7)
    for _ in range(2000):
        own = g.uniform(-5000, 5000, 2)
        tgt = np.concatenate([g.uniform(-15000, 15000, 2), g.uniform(-8, 8, 2)])
        t = g.uniform(0, 1200)
        z_old = old_measure(tgt, own, t, rng_old)
        z_new = s.sample(own, tgt, t, rng_new)
        assert z_old == z_new                                   # gürültü dahil bit-birebir


def test_sigma_and_R():
    s = BearingOnlySensor(sigma_deg=1.0)
    assert s.sigma == SIGMA
    assert s.R[0, 0] == SIGMA ** 2


if __name__ == "__main__":
    test_h_H_bitexact()
    test_sample_bitexact()
    test_sigma_and_R()
    print("Sensör bit-birebir: h, H, sample, sigma/R — TÜM testler GEÇTI (4000 örnek).")
