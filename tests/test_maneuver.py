"""Adım 4-5 regresyonu — pluggable World + ctx manevralar tam-tablo bit-birebir.

build_world (RandomizedScenario + BeliefLogDetReward + AbsoluteObs, opening_leg=False,
turn_rate=inf) ile kurulan World, ctx tabanlı manevralarla golden.npz'ye karşı
bit-birebir. Ödül ve initializer'ın pluggable'a taşınması davranışı değiştirmedi.
"""
import numpy as np
from tma.registry import build_world
from tma.maneuver import (OracleManeuver, RH2Maneuver, RHManeuver,
                          RandomManeuver, StraightManeuver)

MANEUVERS = {"kahin": OracleManeuver, "belief2b": RH2Maneuver, "belief1b": RHManeuver,
             "rastgele": RandomManeuver, "duz": StraightManeuver}


def evaluate_new(make_maneuver, n_ep=20, seed0=100):
    errs, rews = [], []
    for e in range(n_ep):
        man = make_maneuver()
        w = build_world(seed=seed0 + e, maneuver=man)
        w.reset()
        done, R, info = False, 0.0, {}
        while not done:
            a = int(man.decide(w.maneuver_context()))
            _, r, done, info = w.apply(a)
            R += r
        errs.append(info["pos_err"]); rews.append(R)
    return np.array(errs), np.array(rews)


def test_pluggable_full_table_bitexact():
    g = np.load("golden.npz")
    for name, mk in MANEUVERS.items():
        errs, rews = evaluate_new(mk)
        assert np.array_equal(errs, g[f"{name}_err"]), f"{name}: err ayrıştı"
        assert np.array_equal(rews, g[f"{name}_rew"]), f"{name}: rew ayrıştı"


if __name__ == "__main__":
    g = np.load("golden.npz")
    print(f"{'Manevra':10s} | {'medyan':>7s} | {'ort':>7s} | >1km | {'odul':>5s} | bit-birebir")
    ok_all = True
    for name, mk in MANEUVERS.items():
        errs, rews = evaluate_new(mk)
        ok = np.array_equal(errs, g[f"{name}_err"]) and np.array_equal(rews, g[f"{name}_rew"])
        ok_all &= ok
        print(f"{name:10s} | {np.median(errs):6.0f}m | {np.mean(errs):6.0f}m |"
              f" {(errs>1000).sum():2d}/20 | {np.mean(rews):5.1f} | {'EVET' if ok else 'HAYIR***'}")
    print("\nPluggable World + ctx manevralar golden ile bit-birebir:", "GEÇTI" if ok_all else "HATA")
    assert ok_all
