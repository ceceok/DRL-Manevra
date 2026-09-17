"""Faz 2 — açılış bacağı (+90° kerterize dik) + sonlu dönüş oranı (3°/s).

Davranışı BILEREK değiştirir; golden.npz (Faz 1) TUTMAZ. Yeni referans tablosu
üretilir (golden_phase2.npz) ve Faz 1 ile öncesi/sonrası kıyaslanır.
Kahin açılış bacağından muaftır (privileged); dönüş oranı ona da uygulanır.
"""
import numpy as np
from tma.registry import build_world
from tma.initializer import FixedScenario
from tma.maneuver import (OracleManeuver, RH2Maneuver, RHManeuver,
                          RandomManeuver, StraightManeuver)

MANEUVERS = {"kahin": OracleManeuver, "belief2b": RH2Maneuver, "belief1b": RHManeuver,
             "rastgele": RandomManeuver, "duz": StraightManeuver}
TURN_RATE_DEG = 3.0

# Açılış ofseti testi için sabit geometri. Kerteriz 30° seçildi: 30°'lik aksiyon
# ızgarasının tam ortasında değil, yani 1°'lik ölçüm gürültüsü yuvarlamayı kaydırmaz
# (45° seçilseydi 30 ile 60 arasında knife-edge olurdu).
OPENING_SPEC = {"target": {"bearing": 30.0, "range": 10000.0, "course": 270.0, "speed": 5.0},
                "ownship": {"x": 0.0, "y": 0.0, "course": 0.0, "speed": 8.0}}


def evaluate_phase2(make_maneuver, n_ep=20, seed0=100):
    errs, rews = [], []
    for e in range(n_ep):
        man = make_maneuver()
        w = build_world(seed=seed0 + e, maneuver=man,
                        opening_leg=True, turn_rate_deg=TURN_RATE_DEG)
        w.reset()
        done, R, info = False, 0.0, {}
        while not done:
            a = int(man.decide(w.maneuver_context()))
            _, r, done, info = w.apply(a)
            R += r
        errs.append(info["pos_err"]); rews.append(R)
    return np.array(errs), np.array(rews)


def test_phase2_runs_and_is_deterministic():
    # aynı tohumla iki koşu -> birebir aynı (determinizm)
    e1, _ = evaluate_phase2(RHManeuver)
    e2, _ = evaluate_phase2(RHManeuver)
    assert np.array_equal(e1, e2)


# ---------------- açılış bacağı ofseti ----------------
def _opening_hdg_deg(seed=7, **kw):
    """reset() sonrası gözlemcinin rotası (derece). turn_rate_max=inf (varsayılan)
    olduğu için platform komut edilen rotaya anında oturur -> açılış bacağının
    komutunu doğrudan okuyabiliriz."""
    w = build_world(seed=seed, maneuver=StraightManeuver(),
                    initializer=FixedScenario(OPENING_SPEC),
                    opening_leg=True, **kw)
    w.reset(seed=seed)
    return round(float(np.degrees(w.ownship.hdg)) % 360.0, 6)


def test_opening_offset_perpendicular_is_default():
    # varsayılan +90°: kerteriz 30° -> 120° (ayrık ızgarada tam nokta)
    assert _opening_hdg_deg() == 120.0
    assert _opening_hdg_deg(opening_offset_deg=90.0) == 120.0


def test_opening_offset_zero_steers_to_contact_bearing():
    # ofset 0 -> ilk temas kerterizine git: kerteriz 30° -> 30°
    assert _opening_hdg_deg(opening_offset_deg=0.0) == 30.0


def test_opening_offset_deg_matches_radians_and_overrides():
    # derece ve radyan yolu aynı sonucu vermeli; ikisi birden verilirse DERECE kazanır
    assert (_opening_hdg_deg(opening_offset_deg=0.0)
            == _opening_hdg_deg(opening_offset=0.0))
    assert _opening_hdg_deg(opening_offset=np.pi / 2, opening_offset_deg=0.0) == 30.0


def test_opening_offset_changes_outcome():
    # dik açılış ile temasa gitme aynı tohumda farklı kestirim hatası vermeli
    # (aksi halde ofset hiçbir yere bağlanmamış demektir)
    def err(offset_deg):
        man = RHManeuver()
        w = build_world(seed=11, maneuver=man, initializer=FixedScenario(OPENING_SPEC),
                        opening_leg=True, turn_rate_deg=TURN_RATE_DEG,
                        opening_offset_deg=offset_deg)
        w.reset(seed=11)
        done, info = False, {}
        while not done:
            _, _, done, info = w.apply(int(man.decide(w.maneuver_context())))
        return info["pos_err"]

    assert err(90.0) != err(0.0)


if __name__ == "__main__":
    g1 = np.load("golden.npz")
    data = {}
    print(f"{'Manevra':10s} | {'Faz1 med':>8s} | {'Faz2 med':>8s} | {'Faz1 >1km':>9s} | {'Faz2 >1km':>9s}")
    for name, mk in MANEUVERS.items():
        errs, rews = evaluate_phase2(mk)
        data[f"{name}_err"] = errs; data[f"{name}_rew"] = rews
        m1 = np.median(g1[f"{name}_err"]); d1 = int((g1[f"{name}_err"] > 1000).sum())
        m2 = np.median(errs); d2 = int((errs > 1000).sum())
        print(f"{name:10s} | {m1:7.0f}m | {m2:7.0f}m | {d1:6d}/20 | {d2:6d}/20")
    np.savez("golden_phase2.npz", **data)
    print("\ngolden_phase2.npz kaydedildi (Faz 2 yeni referansı).")
