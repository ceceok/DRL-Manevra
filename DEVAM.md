# DEVAM — VS Code / Claude Code'a Aktarım Notu

Bu dosya, `tma_refactor/` paketini VS Code'a taşıyıp Claude Code ile devam etmek için
bağlam belgesidir. Tasarım kararlarının tamamı `MIMARI_KARARLAR.md`'de; bu dosya
"ne yapıldı, nasıl çalıştırılır, sırada ne var" özetidir.

---

## 1. Ne yapıldı (durum)

**Faz 1 — pluggable refactor (bit-birebir).** Eski tek-parça ortam, tak-çıkar bir
mimariye taşındı ve mevcut davranışla BIT-BIREBIR doğrulandı (`golden.npz`):
sensor → filter → world → maneuver → reward/initializer/obs → registry → gym.

**Faz 2 — açılış bacağı (+90° dik) + sonlu dönüş oranı.** Yeni referans
`golden_phase2.npz`. Medyan iyileşti, kuyruk (>1km) bazı politikalarda arttı.

**Faz 3 — LOS-göreli gözlem.** `LOSRelativeObs` (9D, dönme-değişmez, kanıtlandı).

**Makale reprodüksiyonu (Ristic & Arulampalam 2026).** Bizim mimaride, SB3 KULLANMADAN
sıfırdan numpy DQN ile. CKF filtresi + tek-karar episode + PTB açılış + Pareto terminal
ödül + PTB/ITO baseline'ları. Makalenin Fig 1, Fig 2 ve Table III'ü üretildi;
merkezi bulgular reprodüklendi (ITO en iyi ortalama ama katastrofik kuyruk + en kötü
tutarlılık; DQN β=0.7 en iyi denge; β=0.9 kuyruk patlar; PTB en kötü ortalama).

---

## 2. Paket düzeni

```
tma_refactor/
  tma/
    sensor.py        BearingOnlySensor — ölçüm modelinin tek kaynağı (h, H, R, sample)
    filter.py        RPEKFBankFilter — menzil-param. EKF bankası (init/step/estimate/cov/
                       propagate_state/planning_states)
    filter_ckf.py    CubatureKF — türevsiz CKF (makale filtresi); sensor.h kullanır
    world.py         Platform + World — döngü, gerçek, açılış bacağı, dönüş oranı, last_z
    maneuver.py      ManeuverContext + Oracle/RH/RH2/Random/Straight + PTB/ITO (decide(ctx))
    reward.py        BeliefLogDetReward, ParetoTerminalReward (baseline + __call__)
    initializer.py   RandomizedScenario / FixedScenario / PaperInitializer
    obs.py           AbsoluteObs (15D) / LOSRelativeObs (9D) / Paper12Obs (12D)
    registry.py      tip->sınıf + build_world / build_world_from_json
    gym_env.py       TMAGym — ince Gymnasium adaptörü (tembel import)
    dqn.py           Sıfırdan numpy DQN (MLP+Adam+Huber+replay+eps-greedy)
    util.py          wrap()
  tests/             test_sensor / filter / maneuver / obs / gym / phase2 / ckf
  capture_golden.py  altın referans üretici (tma_full.py gerekir — bir kez)
  paper_repro.py     makale eğitim/değerlendirme (train/eval mod)
  paper_figs.py      Fig 1, Fig 2, Table III üretimi
  scenarios.json     FixedScenario örneği
  golden.npz         Faz 1 bit-birebir referansı
  golden_phase2.npz  Faz 2 referansı
  models/agent_b0{1,3,5,7,9}.npz  eğitilmiş DQN ağları (β=0.1..0.9)
  MIMARI_KARARLAR.md tasarım karar dosyası
  README.md          paket özeti
  DEVAM.md           bu dosya
```

---

## 3. Kurulum

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install numpy matplotlib gymnasium
# SB3 ile denemek istersen (Faz 3 RL): pip install stable-baselines3 torch
```
Not: Windows/Python 3.13'te torch `c10.dll` sorunu için 3.12 venv önerilir.
Çekirdek (tma/, DQN, makale reprodüksiyonu) SALT numpy — torch/SB3 gerekmez.

---

## 4. Çalıştırma

### Testler (hepsi geçmeli)
```bash
cd tma_refactor
PYTHONPATH=. python tests/test_sensor.py     # sensör bit-birebir
PYTHONPATH=. python tests/test_filter.py     # RP-EKF bit-birebir
PYTHONPATH=. python tests/test_maneuver.py   # pluggable World + manevralar -> golden bit-birebir
PYTHONPATH=. python tests/test_obs.py        # LOS dönme-değişmezliği
PYTHONPATH=. python tests/test_gym.py        # gymnasium check_env
PYTHONPATH=. python tests/test_phase2.py     # açılış bacağı + dönüş oranı (öncesi/sonrası)
PYTHONPATH=. python tests/test_ckf.py        # CKF + makale dünyası dumanı
```
`golden.npz` yoksa önce üret (eski `tma_full.py` repoda olmalı):
`PYTHONPATH=.:. python capture_golden.py`

### Makale reprodüksiyonu
```bash
# IDE'den: paper_repro.py başındaki AYARLAR bloğunu düzenleyip Run'a bas.
# Terminalden pozisyonel biçim AYARLAR'ın üzerine yazar:

# eğitim (β başına ~2.5 dk / 30k episode; makale ölçeği 50k)
PYTHONPATH=. python paper_repro.py train 0.7 50000 models/agent_b07.npz
# değerlendirme (birden çok β + PTB + ITO)
PYTHONPATH=. python paper_repro.py eval 5000 0.7,models/agent_b07.npz 0.9,models/agent_b09.npz
# figürler + Table III (eğitilmiş 5 β ajanı gerekir: models/agent_b0{1,3,5,7,9}.npz)
PYTHONPATH=. python paper_figs.py 5000
```
NOT: ajan dosyaları `models/` altında. Yolu vermezsen `agent_path(beta)` ile
`models/agent_b0{β×10}.npz` olarak türetilir (paper_figs.py ile aynı kural).

---

## 5. Kilitli kararlar (özet)

- **Filtre dışa tek (mean, cov) raporlar**; FIM planlayıcı ağırlıklı hipotezleri
  `planning_states()` ile alır (RP-EKF: 6; CKF: `[(mean,1.0)]`).
- **Bit-birebir regresyon**: RNG çekme sırası korunur; Faz 1 golden'a `array_equal`.
- **Sensör = ölçüm modelinin tek kaynağı**; filtre `sensor.h/H/R` kullanır.
- **Kahin/GT-ödül gerçeği** `ctx`'ten değil `world.true_target_state()` ayrıcalıklı
  kanalından alır; `ctx` diğerlerine dürüst (truth=None).
- **`propagate_state`** filtrede (ileri-model); planlayıcı onu kullanır.
- **TMAGym**: çok-karar zaman-limiti → `truncated=done, terminated=False` (GAE doğru).
  Makale tek-karar episode → görev-bitimi terminal (paper_repro içinde ele alındı).
- **DQN (makale)**: tek-karar terminal → Bellman hedefi = r (bootstrap yok, γ etkisiz).
  Huber kaybı + global-norm gradyan kırpma; obs'a sabit normalizasyon (kararlılık).
- **CKF mutlak durumda** kurulur (göreli ile denk; observer deterministik).

---

## 6. Bilinen sapmalar / açık uçlar

- **Makale sayıları birebir DEĞİL** (RL varyansı, 30k vs 50k, tohum). Niteliksel
  bulgular reprodüklendi. Daha yakın için 50k eğitim + 5000 eval.
- **PTB ortalamam makaleden yüksek** (5.66 vs 3.34) — PTB yön-seçim kuralı makaleyle
  daha sıkı hizalanabilir (leg2 yön tercihi).
- **β=0.3'te tek ıraksama episode'u** maks'ı şişirdi (RL/geometri gürültüsü).
- **Faz 2 planlayıcı** ani-dönüş varsayıyor (Karar D-a); sonlu dönüş oranıyla küçük
  uyumsuzluk kuyrukta bedel yaratıyor olabilir — `_own_future`'a dönüş yayı modellemek
  (Karar D-b) açık iyileştirme.
- **Compat katmanı kaldırıldı**; `capture_golden.py` yalnızca eski `tma_full.py`'ye
  bağlı (bir kez golden üretmek için).

---

## 7. Sırada ne var (öneriler)

1. **Makaleyi tam ölçekte kapat**: 50k eğitim × 5 β + 5000 eval; Fig/Table'ı yenile.
2. **Tez bağlantısı — filtre-agnostik test**: ParetoTerminalReward + DQN'i CKF yerine
   **RP-EKF bankasına** tak (tek satır: filter=rp_ekf). Çok-hipotezli filtrenin
   ITO'nun katastrofik kuyruğunu daha da bastırıp bastırmadığını ölç. Faz 2'de
   gözlediğin >1km ıraksamanın Mahalanobis terimiyle çözülüp çözülmediğinin doğrudan testi.
3. **SB3 karşılaştırması**: aynı `TMAGym`'de SB3 PPO/DQN eğit; sıfırdan DQN ile kıyasla
   (API doğrulaması + hız).
4. **Pareto-front grafiği**: ort dE vs ort dM, β boyunca — tezin görsel özeti.
5. **Çok-bacaklı genişleme** (makalenin "future work"): episode'da >1 karar; bizim
   çok-karar World zaten hazır (`plan_every`, truncation doğru).
6. **PTB yön-seçimi** ve **β=0.3 kuyruğu**nu incele (yukarıdaki sapmalar).

---

## 8. Claude Code'a ilk komut önerisi

Repoyu açıp şunu vermen yeterli: "MIMARI_KARARLAR.md ve DEVAM.md'yi oku; önce tüm
testleri koştur (bit-birebir tutmalı), sonra madde 1/2'den başla." Böylece bağlam
ve regresyon güvencesi korunarak devam eder.
