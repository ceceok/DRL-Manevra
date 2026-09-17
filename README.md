# TMA — pluggable mimari (yeniden yapılandırma)

Bearings-only TMA'nın sensör / filtre / manevra / ödül / senaryo-başlatıcı /
gözlem katmanlarının tak-çıkar olduğu temiz-sayfa yeniden yapılandırması.
Tasarım kararları: `MIMARI_KARARLAR.md`.

## Paket
```
tma/
  sensor.py       BearingOnlySensor — ölçüm modelinin tek kaynağı (h, H, R, sample)
  filter.py       RPEKFBankFilter — init/step/estimate/covariance/propagate_state/planning_states
  world.py        Platform + World — döngü, gerçek durum, açılış bacağı, dönüş oranı
  maneuver.py     ManeuverContext + Oracle/RH/RH2/Random/Straight (decide(ctx))
  reward.py       BeliefLogDetReward (pluggable, baseline + __call__)
  initializer.py  RandomizedScenario / FixedScenario
  obs.py          AbsoluteObs (15D) / LOSRelativeObs (9D, dönme-değişmez)
  registry.py     tip->sınıf + build_world / build_world_from_json
  gym_env.py      TMAGym — ince Gymnasium adaptörü (tembel import)
```

## Fazlar
- **Faz 1** (Adım 1-6): saf refactor, mevcut davranışla BIT-BIREBIR (`golden.npz`).
- **Faz 2**: açılış bacağı (+90° kerterize dik) + sonlu dönüş oranı → yeni referans
  (`golden_phase2.npz`).
- **Faz 3**: LOS-göreli gözlem (RL için); `TMAGym` bunu kullanır. RL eğitimi
  (SB3) kullanıcının sıradaki adımı.

## Testler
```
python capture_golden.py            # altın referansı üret (tma_full.py gerekir)
python tests/test_sensor.py         # sensör bit-birebir
python tests/test_filter.py         # filtre bit-birebir
python tests/test_maneuver.py       # pluggable World + manevralar, golden bit-birebir
python tests/test_obs.py            # LOS dönme-değişmezliği + boyutlar
python tests/test_gym.py            # gymnasium check_env
python tests/test_phase2.py         # açılış bacağı + dönüş oranı, öncesi/sonrası
```

## RL (sıradaki adım — SB3)
```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from tma.gym_env import make_tma_gym
env = make_tma_gym(obs="los", opening_leg=True, turn_rate_deg=3.0,
                   opening_offset_deg=90.0)   # 90 = kerterize dik, 0 = ilk temasa git
model = PPO("MlpPolicy", env, n_steps=2048).learn(1_000_000)   # torch gerekir
```
`n_steps` KARAR sayar (ölçüm değil): 2048 karar ≈ 100 episode.
`opening_offset_deg` açılış bacağının kerterize ofseti (bkz. `MIMARI_KARARLAR.md §11.G′`);
ajanın başlangıç durumunu belirlediği için eğitim ve değerlendirmede AYNI olmalı.
Windows/Python 3.13 torch (`c10.dll`) sorunu için 3.12 venv önerilir.
