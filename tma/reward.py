"""Pluggable ödül fonksiyonları.

İki yüz: baseline(world) karar aralığı BAŞINDAKI referansı yakalar; __call__ aralık
SONUNDAKI ödülü verir. Böylece potansiyel-tabanlı şaping / terminal terim eklemek
arayüzü değiştirmez.

BeliefLogDetReward: mevcut davranış (belief bilgi matrisinin logdet artışı),
tma_full._belief_logdet ile birebir.
"""
import numpy as np


class BeliefLogDetReward:
    privileged = False   # yalnızca filtre kovaryansını kullanır; gerçeği kullanmaz

    def _logdet(self, world):
        P = world.ownship.filter.covariance
        sgn, ld = np.linalg.slogdet(np.linalg.inv(P + 1e-9 * np.eye(4)))
        return ld if sgn > 0 else -50.0

    def baseline(self, world):
        return self._logdet(world)

    def __call__(self, world, prev, action):
        return self._logdet(world) - prev


class ParetoTerminalReward:
    """Makale ödülü: r = -dE^beta * dM^(1-beta), yalnızca terminalde (ara adım 0).
    dE=Euclidean (doğruluk), dM=Mahalanobis (tutarlılık). GERÇEĞI kullanır -> privileged."""
    privileged = True

    def __init__(self, beta=0.7):
        self.beta = beta

    def baseline(self, world):
        return 0.0

    def __call__(self, world, prev, action):
        if world.t < world.total_time:
            return 0.0
        true = world.true_target_state()
        pe = world.ownship.filter.estimate[:2]
        diff = pe - true[:2]
        dE = float(np.linalg.norm(diff))
        Ppos = world.ownship.filter.covariance[:2, :2]
        dM = float(np.sqrt(max(diff @ np.linalg.solve(Ppos + 1e-12*np.eye(2), diff), 0.0)))
        return -(dE**self.beta) * (dM**(1.0 - self.beta))
