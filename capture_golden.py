"""Adım 0 — altın referansı çivile (bit-birebir regresyon için).

Mevcut kodu (tma_full.py) 20 tohumda (seed 100-119) koşturur ve her politikanın
TAM sonuç vektörünü (medyanı değil, 20 değerin tümünü) golden.npz'ye kaydeder.
Sonraki her adım bu vektörlere karşı np.array_equal ile doğrulanır.

Kullanım: tma_full.py importlanabilir olmalı (repo kökünden ya da PYTHONPATH).
    python capture_golden.py
"""
import numpy as np
from tma_full import (evaluate, oracle_policy, rh2_policy, rh_policy,
                      random_policy, straight_policy)

POLICIES = {
    "kahin":    oracle_policy,
    "belief2b": rh2_policy,
    "belief1b": rh_policy,
    "rastgele": random_policy,
    "duz":      straight_policy,
}


def main():
    data = {}
    print(f"{'Politika':10s} | {'medyan':>7s} | {'ort':>7s} | >1km  | {'odul':>5s}")
    for name, pol in POLICIES.items():
        errs, rews, _ = evaluate(pol, n_ep=20, seed0=100)
        data[f"{name}_err"] = errs
        data[f"{name}_rew"] = rews
        print(f"{name:10s} | {np.median(errs):6.0f}m | {np.mean(errs):6.0f}m |"
              f" {(errs > 1000).sum():2d}/20 | {np.mean(rews):5.1f}")
    np.savez("golden.npz", **data)
    print("\ngolden.npz kaydedildi (her politika icin 20 tohumluk err+rew vektorleri).")


if __name__ == "__main__":
    main()
