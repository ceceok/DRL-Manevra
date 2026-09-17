"""Gözlem kodlayıcıları (pluggable).

AbsoluteObs   : 15 boyut, mutlak çerçeve — mevcut _obs ile birebir (Faz 1/2 tabanı).
LOSRelativeObs: 9 boyut, filtre-agnostik, LOS çerçevesinde — dönme değişmez
                (rotational symmetry). Faz 3'te RL gözlemi olarak kullanılır.
"""
import numpy as np
from .util import wrap


class AbsoluteObs:
    """Mevcut tma_full._obs ile birebir (RP-EKF ağırlıklarını da içerir)."""

    def __call__(self, world):
        f = world.ownship.filter
        m = f.estimate
        P = f.covariance
        rel = m[:2] - world.ownship.pos
        return np.concatenate([
            f.wb,
            [np.linalg.norm(rel) / 20000, np.arctan2(rel[0], rel[1]) / np.pi],
            m[2:] / 10,
            [np.sqrt(np.trace(P[:2, :2])) / 5000, np.sqrt(np.trace(P[2:, 2:])) / 5],
            [np.sin(world.ownship.hdg), np.cos(world.ownship.hdg),
             world.t / world.total_time]])


def _to_los(v, bearing):
    """Bir vektörü LOS çerçevesine döndür: [along (radyal), cross (enine)]."""
    sb, cb = np.sin(bearing), np.cos(bearing)
    return np.array([v[0] * sb + v[1] * cb, v[0] * cb - v[1] * sb])


class LOSRelativeObs:
    """Her şey o anki LOS kerterizine göre ifade edilir → sahneyi döndürmek gözlemi
    değiştirmez (rotational symmetry). Filtre-agnostik (wb kullanmaz)."""

    def __call__(self, world):
        f = world.ownship.filter
        m = f.estimate
        P = f.covariance
        own = world.ownship.pos
        rel = m[:2] - own
        bearing = np.arctan2(rel[0], rel[1])
        r = np.linalg.norm(rel)
        vel_los = _to_los(m[2:], bearing)
        Rm = np.array([[np.sin(bearing), np.cos(bearing)],       # satır 0: along
                       [np.cos(bearing), -np.sin(bearing)]])     # satır 1: cross
        Ppos_los = Rm @ P[:2, :2] @ Rm.T
        along_std = np.sqrt(max(Ppos_los[0, 0], 0.0))
        cross_std = np.sqrt(max(Ppos_los[1, 1], 0.0))
        hdg_rel = wrap(world.ownship.hdg - bearing)
        return np.concatenate([
            [r / 20000],
            vel_los / 10,
            [along_std / 5000, cross_std / 5000, np.sqrt(np.trace(P[2:, 2:])) / 5],
            [np.sin(hdg_rel), np.cos(hdg_rel), world.t / world.total_time]])


class Paper12Obs:
    """Ristic 2026 eq (16): 12-eleman gözlem.
    [rel_pos(2), rel_vel(2), P11,P12,P22, P33,P34,P44, cos z, sin z].
    rel = hedef - gözlemci (mutlak izlemeden türetilir). DQN kararlılığı için sabit
    normalizasyon (makaleden sapma; ölçek O(1)'e çekilir)."""

    def __init__(self, pos_scale=25.0, vel_scale=3.0, pvar_scale=100.0, vvar_scale=4.0):
        self.ps, self.vs, self.pv, self.vv = pos_scale, vel_scale, pvar_scale, vvar_scale

    def __call__(self, world):
        f = world.ownship.filter
        m = f.estimate
        P = f.covariance
        own = world.ownship.pos
        own_vel = world.ownship.speed * np.array([np.sin(world.ownship.hdg),
                                                  np.cos(world.ownship.hdg)])
        p = (m[:2] - own) / self.ps
        v = (m[2:] - own_vel) / self.vs
        z = world.last_z if world.last_z is not None else 0.0
        return np.array([p[0], p[1], v[0], v[1],
                         P[0, 0]/self.pv, P[0, 1]/self.pv, P[1, 1]/self.pv,
                         P[2, 2]/self.vv, P[2, 3]/self.vv, P[3, 3]/self.vv,
                         np.cos(z), np.sin(z)])
