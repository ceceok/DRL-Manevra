"""CKF + makale bileşenleri dumanı: build_paper_world çalışır, CKF sonlu kestirim
üretir, PTB açılış bacağı menzil belirsizliğini azaltır (dE makul)."""
import numpy as np


def test_paper_world_runs():
    from paper_repro import build_paper_world, _metrics
    from tma.maneuver import PTBManeuver
    dEs = []
    for e in range(50):
        man = PTBManeuver()
        w = build_paper_world(0.7, maneuver=man, seed=e)
        w.reset()
        w.apply(man.decide(w.maneuver_context()))
        dE, dM = _metrics(w)
        assert np.isfinite(dE) and np.isfinite(dM)
        dEs.append(dE)
    assert np.median(dEs) < 15.0    # PTB makul (menzil bir miktar çözülür)


if __name__ == "__main__":
    test_paper_world_runs()
    print("CKF + makale dünyası: 50 episode sonlu, PTB medyan dE makul. GEÇTI.")
